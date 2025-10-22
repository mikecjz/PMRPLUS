import copy
from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from mri_utils import normalized_l1_loss
torch.set_float32_matmul_precision('high')


class TestTimeTrainingCallback(L.Callback):
    """
    Test-Time Training with an ADAPTIVE & HYBRID robust self-supervised early stopping strategy.

    - Robust Huber Delta (per-modality, MAD-based): Estimate residual scale via MAD and set Huber delta
      with modality-specific multipliers (e.g., larger for LGE).
    - Target-Ratio TV Weighting: Choose lambda_tv so that its penalty around the tv-threshold
      contributes ~r fraction of the validation metric scale (val_dc). r can be modality-specific.
    - Optional Residual Normalization: Normalize k-space residual by a robust amplitude scale of measured k,
      for more comparable logging across studies/centers.
    - Enhanced Logging.
    """

    def __init__(
        # --- Basic TTT settings ---
        self,
        lr: float = 1e-7,
        inner_steps: int = 15,
        log_every_n_steps: int = 1,

        # --- Validation split and early stopping ---
        val_num_low_lines: Optional[int] = None,
        min_val_lines: int = 8,
        min_delta: float = 0.0,
        stop_on_first_worse: bool = True,

        # --- Huber settings (robust & modality-specific) ---
        use_robust_huber_delta: bool = True,
        huber_k_default: float = 1.0,     # delta = k * sigma_rob (sigma_rob from MAD)
        huber_k_lge: float = 2.0,         # larger for LGE
        huber_delta_map: Dict[str, float] = None,  # optional hard-coded delta overrides per modality
        robust_delta_recompute_each_iter: bool = False,  # if True, re-estimate delta on each iter's val residual

        # --- Residual normalization (for Huber only) ---
        residual_norm_mode: str = "none",  # ["none", "median_meas"]
        residual_norm_eps: float = 1e-8,

        # --- Target-ratio TV strategy ---
        tv_target_ratio_map: Dict[str, float] = None,  # e.g., {"lge": 0.07, "default": 0.20}
        tv_tol_ratio: float = 0.10,                    # TV threshold = tv0 * (1 + tv_tol_ratio)
        tv_eps: float = 1e-6,

        # --- Optional dc weighting per modality ---
        dc_weight_map: Dict[str, float] = None,  # e.g., {"lge": 0.5, "default": 1.0}

        # --- Optim / misc ---
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = False,
        enable_wandb_logging: bool = True,
    ):
        super().__init__()
        # Basic
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

        # Val split / early stop
        self.val_num_low_lines = val_num_low_lines
        self.min_val_lines = int(min_val_lines)
        self.min_delta = float(min_delta)
        self.stop_on_first_worse = bool(stop_on_first_worse)

        # Huber config
        self.use_robust_huber_delta = bool(use_robust_huber_delta)
        self.huber_k_default = float(huber_k_default)
        self.huber_k_lge = float(huber_k_lge)
        self.huber_delta_map = huber_delta_map or {"default": 1e-3}
        if "default" not in self.huber_delta_map:
            self.huber_delta_map["default"] = 1e-3
        self.robust_delta_recompute_each_iter = bool(robust_delta_recompute_each_iter)

        # Residual normalization config
        assert residual_norm_mode in ("none", "median_meas")
        self.residual_norm_mode = residual_norm_mode
        self.residual_norm_eps = float(residual_norm_eps)

        # TV config (target-ratio)
        self.tv_target_ratio_map = tv_target_ratio_map or {"lge": 0.07, "default": 0.20}
        if "default" not in self.tv_target_ratio_map:
            self.tv_target_ratio_map["default"] = 0.20
        self.tv_tol_ratio = float(tv_tol_ratio)
        self.tv_eps = float(tv_eps)

        # DC weight per modality
        self.dc_weight_map = dc_weight_map or {"default": 1.0}
        if "default" not in self.dc_weight_map:
            self.dc_weight_map["default"] = 1.0

        # Optim/misc
        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.enable_wandb_logging = bool(enable_wandb_logging)
        self.wandb_logger: Optional[WandbLogger] = None

        # Project-specific loss
        self.normalized_l1 = normalized_l1_loss()

        # Cache for per-modality robust deltas (estimated at baseline)
        self._delta_cache: Dict[str, float] = {}

    # ------------------- Utility Functions -------------------
    @staticmethod
    def _get_modality_from_fname(fname: str) -> str:
        fname_lower = fname.lower()
        known_modalities = ['t1w', 't2w', 'lge', 'cine', 't1map', 't2map', 'flow2d']
        for mod in known_modalities:
            if f'/{mod}/' in fname_lower or f'_{mod}_' in fname_lower:
                return mod
        path_parts = fname.split('/')
        if len(path_parts) > 0 and path_parts[0].lower() in known_modalities:
            return path_parts[0].lower()
        return 'default'

    @staticmethod
    def _center_slice_index(module) -> int:
        num_adj = getattr(module, "num_adj_slices", 1)
        return int(num_adj // 2)

    @staticmethod
    def _center_crop_fraction(img: torch.Tensor, frac_w: float = 1/3, frac_h: float = 1/2) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(1)
        B, C, H, W = img.shape
        tw, th = max(1, round(W*frac_w)), max(1, round(H*frac_h))
        x0, y0 = (W - int(tw)) // 2, (H - int(th)) // 2
        return img[:, :, y0:y0+int(th), x0:x0+int(tw)]

    @staticmethod
    def _total_variation(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if img.dim() == 3:
            img = img.unsqueeze(1)
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        tv = torch.sqrt(dx[:, :, :-1, :]**2 + dy[:, :, :, :-1]**2 + eps)
        return tv.mean()

    @staticmethod
    def _is_complex_last(x: torch.Tensor) -> bool:
        return x.dim() >= 5 and x.size(-1) == 2

    @staticmethod
    def _broadcast_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return x.expand_as(target) if list(x.shape) != list(target.shape) else x

    @torch.no_grad()
    def _center_mask(self, full_mask: torch.Tensor, cidx: int) -> torch.Tensor:
        if full_mask.dim() < 5:
            return full_mask
        chunks = torch.chunk(full_mask, chunks=full_mask.shape[1], dim=1)
        return chunks[cidx]

    @torch.no_grad()
    def _build_val_train_masks_lowfreq(self, center_mask: torch.Tensor, num_low_lines: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = center_mask.shape[:4]
        val_mask = torch.zeros_like(center_mask)
        tr_mask  = torch.zeros_like(center_mask)
        k = max(self.min_val_lines, int(num_low_lines))
        k = min(k, H)
        if self._is_complex_last(center_mask):
            acq = (center_mask[..., 0] > 0)
        else:
            acq = center_mask if center_mask.dim() == 4 else center_mask.squeeze(-1)
            acq = (acq > 0)
        cy = H // 2
        for b in range(B):
            rows = torch.where(acq[b, 0].any(dim=1))[0]
            if rows.numel() == 0:
                continue
            dist = torch.abs(rows - cy)
            order = torch.argsort(dist)
            chosen_val_rows = rows[order[:k]]
            chosen_tr_rows = rows[order[k:]]

            if self._is_complex_last(center_mask):
                val_mask[b, 0, chosen_val_rows, :, :] = center_mask[b, 0, chosen_val_rows, :, :]
                tr_mask[b, 0, chosen_tr_rows, :, :] = center_mask[b, 0, chosen_tr_rows, :, :]
            else:
                val_mask[b, 0, chosen_val_rows] = center_mask[b, 0, chosen_val_rows]
                tr_mask[b, 0, chosen_tr_rows] = center_mask[b, 0, chosen_tr_rows]
        return tr_mask, val_mask

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    # ------------------- Robust Huber helpers -------------------
    @staticmethod
    def _complex_abs(x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 2:
            return torch.linalg.vector_norm(x, dim=-1)
        return torch.abs(x)

    @staticmethod
    def _mad_scale(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Compute robust scale via MAD: sigma_rob = 1.4826 * median(|x - median(x)|)
        """
        x_flat = x.reshape(-1)
        med = torch.median(x_flat)
        mad = torch.median(torch.abs(x_flat - med)) + eps
        return 1.4826 * mad

    def _choose_huber_delta(
        self,
        modality: str,
        resid_for_scale: torch.Tensor,
    ) -> float:
        """
        Choose delta per modality:
        - If explicit map has modality key, use it.
        - Else if robust mode, estimate via MAD with modality-specific k.
        - Else fallback to default map.
        """
        # explicit override
        if modality in self.huber_delta_map and self.huber_delta_map[modality] > 0:
            return float(self.huber_delta_map[modality])

        if self.use_robust_huber_delta:
            abs_r = self._complex_abs(resid_for_scale)
            sigma_rob = float(self._mad_scale(abs_r).item())
            k = self.huber_k_lge if modality == "lge" else self.huber_k_default
            delta = max(k * sigma_rob, 1e-12)
            return float(delta)

        # fallback
        return float(self.huber_delta_map.get("default", 1e-3))

    @staticmethod
    def _huber_on_complex_residual(residual: torch.Tensor, delta: float) -> torch.Tensor:
        """
        Standard Huber on |residual|. residual could be complex-last (..., 2).
        """
        x = TestTimeTrainingCallback._complex_abs(residual)
        # Piecewise
        delta = max(delta, 1e-12)
        abs_x = torch.abs(x)
        quadratic = torch.clamp(abs_x, max=delta)
        linear = abs_x - quadratic
        loss = 0.5 * (quadratic**2) / delta + linear
        return loss.mean()

    def _apply_residual_normalization(
        self,
        resid: torch.Tensor,
        k_meas_center: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Optionally normalize residual by a robust amplitude scale from measured center k-space.
        Returns (resid_scaled, scale_used).
        """
        if self.residual_norm_mode == "none":
            return resid, 1.0

        # median amplitude of measured k-space as scale
        meas_abs = self._complex_abs(k_meas_center)
        scale = float(torch.median(meas_abs).item()) + self.residual_norm_eps
        return resid / scale, scale

    # ------------------- Lightning Hooks -------------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        if self.enable_wandb_logging:
            loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
            self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = batch.masked_kspace.device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)
        fname = getattr(batch, 'fname', ['unknown'])[0]
        slice_num = getattr(batch, 'slice_num', [0])[0]

        modality = self._get_modality_from_fname(fname)

        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())
        adapt_model = copy.deepcopy(pl_module).to(device).train()
        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)

        for param in adapt_model.parameters():
            param.requires_grad_(True)

        opt = torch.optim.Adam(
            (p for p in adapt_model.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay
        )

        cidx = self._center_slice_index(adapt_model)
        center_mask = self._center_mask(batch.mask, cidx)

        num_lf = int(self.val_num_low_lines or getattr(batch, "num_low_frequencies", 16) or 16)
        train_center, val_center = self._build_val_train_masks_lowfreq(center_mask, num_lf)

        # Expand masks to full multi-slice shape
        mask_train_full = batch.mask.clone()
        mask_val_full = torch.zeros_like(batch.mask)
        mask_train_full[:, cidx] = train_center
        mask_val_full[:, cidx] = val_center

        k_train = batch.masked_kspace * (mask_train_full > 0).float()

        # ---- Baseline forward (no-grad) ----
        with torch.no_grad():
            out0 = adapt_model(
                k_train,
                mask_train_full,
                batch.num_low_frequencies,
                batch.mask_type,
                compute_sens_per_coil=getattr(pl_module, 'compute_sens_per_coil', False)
            )

            img0 = out0["img_pred"]
            img0_crop = self._center_crop_fraction(img0)
            tv0 = float(self._total_variation(img0_crop, self.tv_eps).item())

            k_pred_center = out0["pred_kspace"][:, cidx]
            k_meas_center = batch.masked_kspace[:, cidx]

            resid0_raw = (k_pred_center - k_meas_center) * (val_center > 0).float()
            resid0, scale_used = self._apply_residual_normalization(resid0_raw, k_meas_center)

            # --- Choose Huber delta ---
            # if cached delta exists for modality and robust mode off-each-iter, use it; else estimate now
            if (self.use_robust_huber_delta and not self.robust_delta_recompute_each_iter) or (modality in self.huber_delta_map):
                huber_delta = self._choose_huber_delta(modality, resid0)
                # cache for later reuse
                if self.use_robust_huber_delta and not self.robust_delta_recompute_each_iter:
                    self._delta_cache[modality] = float(huber_delta)
            else:
                # fallback to cache or default
                huber_delta = self._delta_cache.get(modality, self._choose_huber_delta(modality, resid0))

            # --- DC weight per modality ---
            dc_weight = float(self.dc_weight_map.get(modality, self.dc_weight_map.get("default", 1.0)))

            baseline_dc_loss = float(self._huber_on_complex_residual(resid0, huber_delta).item())
            baseline_dc_loss_w = dc_weight * baseline_dc_loss

            # --- Target-ratio lambda_tv ---
            r = float(self.tv_target_ratio_map.get(modality, self.tv_target_ratio_map["default"]))
            # When TV just crosses threshold: penalty ≈ r * baseline_dc_loss_w  => solve lambda_tv
            tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
            # We approximate over-TH amount around threshold by tv0 * tv_tol_ratio
            denom = tv0 * self.tv_tol_ratio + 1e-8
            adaptive_lambda_tv = (r * baseline_dc_loss_w) / denom

            val_metric_best = baseline_dc_loss_w  # baseline metric without TV penalty
            best_state = copy.deepcopy(adapt_model.state_dict())

            print(f"[{device_name}] TTT starting for {fname}/slice{slice_num} (modality: {modality})")
            print(f"[{device_name}] Residual norm mode: {self.residual_norm_mode} (scale_used={scale_used:.3e})")
            print(f"[{device_name}] Huber delta: {huber_delta:.3e} (robust={self.use_robust_huber_delta}, k={'lge' if modality=='lge' else 'default'})")
            print(f"[{device_name}] DC weight: {dc_weight:.3f}")
            print(f"[{device_name}] TV target ratio r={r:.3f}, tv0={tv0:.3f}, tv_tol_ratio={self.tv_tol_ratio:.3f} -> lambda_tv={adaptive_lambda_tv:.4e}")
            print(f"[{device_name}] Baseline: val_dc={baseline_dc_loss_w:.4f} (raw={baseline_dc_loss:.4f}), METRIC={val_metric_best:.4f}")

        # ---- Iterative inner-loop ----
        for it in range(self.inner_steps):
            opt.zero_grad()
            with torch.enable_grad():
                out = adapt_model(
                    k_train,
                    mask_train_full,
                    batch.num_low_frequencies,
                    batch.mask_type,
                    compute_sens_per_coil=getattr(pl_module, 'compute_sens_per_coil', False)
                )
                k_pred_center_train = out["pred_kspace"][:, cidx]
                k_meas_center_train = batch.masked_kspace[:, cidx]
                pred_masked = k_pred_center_train * (train_center > 0).float()
                target_masked = k_meas_center_train * (train_center > 0).float()
                loss_tr = self.normalized_l1(pred_masked, target_masked.detach())

                if torch.isnan(loss_tr):
                    print(f"[{device_name}] WARNING: TTT iter {it+1} got NaN loss, skipping.")
                else:
                    loss_tr.backward()
                    opt.step()

            # ---- Validation metric (no grad) ----
            with torch.no_grad():
                out_eval = adapt_model(
                    k_train,
                    mask_train_full,
                    batch.num_low_frequencies,
                    batch.mask_type,
                    compute_sens_per_coil=getattr(pl_module, 'compute_sens_per_coil', False)
                )

                k_pred_center_val = out_eval["pred_kspace"][:, cidx]
                k_meas_center_val = batch.masked_kspace[:, cidx]
                resid_val_raw = (k_pred_center_val - k_meas_center_val) * (val_center > 0).float()
                resid_val, _ = self._apply_residual_normalization(resid_val_raw, k_meas_center_val)

                # optionally re-estimate delta each iter
                if self.robust_delta_recompute_each_iter:
                    huber_delta_iter = self._choose_huber_delta(modality, resid_val)
                else:
                    huber_delta_iter = huber_delta  # reuse baseline

                val_dc_raw = float(self._huber_on_complex_residual(resid_val, huber_delta_iter).item())
                val_dc = dc_weight * val_dc_raw

                img_cur = out_eval["img_pred"]
                img_cur_crop = self._center_crop_fraction(img_cur)
                tv_cur = float(self._total_variation(img_cur_crop, self.tv_eps).item())

                tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
                tv_penalty_val = max(0.0, tv_cur - tv_thresh)

                penalty_term_contrib = adaptive_lambda_tv * tv_penalty_val
                val_metric = val_dc + penalty_term_contrib

                log_line = (
                    f"[{device_name}] iter {it+1:02d}: "
                    f"val_dc={val_dc:.4f} (raw={val_dc_raw:.4f}) | "
                    f"tv={tv_cur:.3f} (thr={tv_thresh:.3f}, +={tv_penalty_val:.3f}) | "
                    f"pen_term={penalty_term_contrib:.4f} | "
                    f"METRIC={val_metric:.4f} (best={val_metric_best:.4f}) | "
                    f"delta={huber_delta_iter:.2e}"
                )
                print(log_line)

            if val_metric < (val_metric_best - self.min_delta):
                val_metric_best = val_metric
                best_state = copy.deepcopy(adapt_model.state_dict())
                print(f"[{device_name}]     ✓ New best found.")
            elif self.stop_on_first_worse:
                print(f"[{device_name}]     ✗ Early stopping triggered.")
                break

        # ---- Commit best adapted weights for this batch's predict ----
        adapt_model.load_state_dict(best_state)
        pl_module.load_state_dict(adapt_model.state_dict())
        pl_module.eval()
        print(f"[{device_name}] TTT completed for {fname}/slice{slice_num}: final best metric={val_metric_best:.4f}")

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, prediction, batch, dataloader_idx: int = 0, **kwargs):
        if hasattr(pl_module, "_orig_state"):
            pl_module.load_state_dict(pl_module._orig_state)
            pl_module.eval()
            del pl_module._orig_state
