# callbacks/ttt_callback_ssdu_adaptive_barron.py
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
    Test-Time Training with Barron (CVPR'19) robust validation metric and TV target-ratio penalty.
    Includes three critical fixes for stability on LGE:
      (1) Residual normalization uses ONLY the acquired (masked) samples.
      (2) Robust MAD/c estimation uses ONLY the acquired (masked) samples.
      (3) Barron loss clamps extreme residual magnitudes and keeps stable limits near a -> 2.

    Training inner-loop keeps your original normalized L1 DC loss on the train-center mask.
    Validation metric uses Barron loss on the val-center residual + TV target-ratio penalty.
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

        # --- Barron settings (robust & modality-specific) ---
        use_barron_adaptive: bool = True,
        barron_alpha_default: float = 1.0,     # ~between L1 and L2
        barron_alpha_lge: float = 0.6,         # more robust for LGE
        barron_alpha_map: Dict[str, float] = None,   # explicit α per modality

        barron_c_k_default: float = 1.0,       # c = k * sigma_rob (from MAD)
        barron_c_k_lge: float = 1.5,           # larger for LGE
        barron_c_k_map: Dict[str, float] = None,

        # α / c adaptation & smoothing
        alpha_min: float = -4.0,
        alpha_max: float = 2.0,
        c_eps: float = 1e-8,
        ema_beta_alpha: float = 0.8,
        ema_beta_c: float = 0.8,
        recompute_params_each_iter: bool = False,
        alpha_decay: float = 0.8,  # used only in heuristic mode

        # --- Residual normalization (pre-Barron) ---
        residual_norm_mode: str = "median_meas",  # ["none", "median_meas"]
        residual_norm_eps: float = 1e-8,

        # --- TV target-ratio ---
        tv_target_ratio_map: Dict[str, float] = None,  # e.g., {"lge": 0.07, "default": 0.20}
        tv_tol_ratio: float = 0.10,                    # TV threshold = tv0 * (1 + tv_tol_ratio)
        tv_eps: float = 1e-6,

        # --- Optional dc weighting per modality ---
        dc_weight_map: Dict[str, float] = None,  # e.g., {"lge": 0.5, "default": 1.0}

        # --- Optim / misc ---
        weight_decay: float = 1e-6,
        freeze_bn_stats: bool = False,
        enable_wandb_logging: bool = False,
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

        # Barron config
        self.use_barron_adaptive = bool(use_barron_adaptive)
        self.barron_alpha_default = float(barron_alpha_default)
        self.barron_alpha_lge = float(barron_alpha_lge)
        self.barron_alpha_map = barron_alpha_map or {"default": self.barron_alpha_default}

        self.barron_c_k_default = float(barron_c_k_default)
        self.barron_c_k_lge = float(barron_c_k_lge)
        self.barron_c_k_map = barron_c_k_map or {"default": self.barron_c_k_default}

        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.c_eps = float(c_eps)
        self.ema_beta_alpha = float(ema_beta_alpha)
        self.ema_beta_c = float(ema_beta_c)
        self.recompute_params_each_iter = bool(recompute_params_each_iter)
        self.alpha_decay = float(alpha_decay)

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

        # Project-specific loss (your training proxy in inner-loop)
        self.normalized_l1 = normalized_l1_loss()

        # Caches (per-modality EMA params)
        self._alpha_cache: Dict[str, float] = {}
        self._c_cache: Dict[str, float] = {}

    # ------------------- Utilities -------------------
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

    # ------------------- Robust helpers -------------------
    @staticmethod
    def _complex_abs(x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 2:
            return torch.linalg.vector_norm(x, dim=-1)
        return torch.abs(x)

    def _align_mask_to(self, ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Make 'mask' broadcastable to 'ref' for indexing.
        e.g. ref: [B,H,W] or [B,1,H,W], mask: [B,1,H,W,1] / [B,1,H,W] / [B,H,W]
        Returns a bool mask with same ndim as ref and same batch/space sizes.
        """
        m = mask.clone()
        # drop trailing singleton dims (e.g. complex dim [B,1,H,W,1] -> [B,1,H,W])
        while m.dim() > ref.dim() and m.size(-1) == 1:
            m = m.squeeze(-1)
        # drop channel singleton if ref is 3D (e.g. [B,1,H,W] -> [B,H,W] when ref is [B,H,W])
        if ref.dim() == 3 and m.dim() == 4 and m.size(1) == 1:
            m = m.squeeze(1)
        # add channel dim if ref is 4D and m is 3D
        elif ref.dim() == 4 and m.dim() == 3:
            m = m.unsqueeze(1)
        return (m > 0)

    @staticmethod
    def _mad_sigma_from_flat(x_flat: torch.Tensor, eps: float = 1e-8) -> float:
        """Robust sigma via MAD on a 1-D vector: 1.4826 * median(|x - median(x)|)."""
        if x_flat.numel() == 0:
            return 0.0
        med = torch.median(x_flat)
        mad = torch.median(torch.abs(x_flat - med)) + eps
        return float(1.4826 * mad.item())

    @staticmethod
    def _percentile_from_flat(x_flat: torch.Tensor, q: float) -> float:
        """Percentile on a 1-D vector (no grad)."""
        n = x_flat.numel()
        if n == 0:
            return 0.0
        k = max(0, min(int(round((q/100.0) * (n-1))), n-1))
        vals, _ = torch.sort(x_flat)
        return float(vals[k].item())

    def _choose_barron_params_from_masked_abs(
        self,
        modality: str,
        abs_r_masked_flat: torch.Tensor,  # magnitude residuals on acquired positions only (1-D)
        prev_alpha: Optional[float],
        prev_c: Optional[float],
    ) -> Tuple[float, float]:
        """Choose (alpha, c) per modality using masked residual magnitudes only."""
        # α base by modality
        if modality in (self.barron_alpha_map or {}):
            alpha_base = float(self.barron_alpha_map[modality])
        else:
            alpha_base = self.barron_alpha_lge if modality == "lge" else self.barron_alpha_default

        # c base multiplier by modality
        if modality in (self.barron_c_k_map or {}):
            c_k = float(self.barron_c_k_map[modality])
        else:
            c_k = self.barron_c_k_lge if modality == "lge" else self.barron_c_k_default

        # robust sigma from masked residuals
        sigma_rob = self._mad_sigma_from_flat(abs_r_masked_flat, self.c_eps)
        c_new = max(c_k * sigma_rob, 1e-12)

        # tail-heaviness heuristic for alpha
        p50 = self._percentile_from_flat(abs_r_masked_flat, 50.0) + 1e-12
        p95 = self._percentile_from_flat(abs_r_masked_flat, 95.0)
        tail_ratio = max(p95 / p50, 1.0)
        alpha_new = alpha_base - self.alpha_decay * (tail_ratio - 1.0)
        alpha_new = float(max(self.alpha_min, min(self.alpha_max, alpha_new)))

        # EMA smoothing if previous exist
        if prev_alpha is not None:
            alpha_new = self.ema_beta_alpha * prev_alpha + (1.0 - self.ema_beta_alpha) * alpha_new
        if prev_c is not None:
            c_new = self.ema_beta_c * prev_c + (1.0 - self.ema_beta_c) * c_new

        # final clamp
        alpha_new = float(max(self.alpha_min, min(self.alpha_max, alpha_new)))
        c_new = float(max(1e-12, c_new))
        return alpha_new, c_new

    @staticmethod
    def _barron_loss_residual(residual: torch.Tensor, alpha: float, c: float, eps: float = 1e-12) -> torch.Tensor:
        """
        Barron general robust loss on complex residual magnitude.
        f = |a-2|/a * ( ((r^2)/|a-2| + 1)^(a/2) - 1 ),  r = |x|/c
        Includes gentle clamping and stable L2 limit.
        """
        x = TestTimeTrainingCallback._complex_abs(residual)
        # Gentle clamp to avoid extreme overflow after normalization
        x = torch.clamp(x, max=1e6)
        r = x / max(c, 1e-12)
        a = alpha

        # near L2: 0.5 r^2
        if abs(a - 2.0) < 1e-3:
            loss = 0.5 * (r * r)
            return loss.mean()

        abs_a2 = max(abs(a - 2.0), 1e-6)
        base = (r * r) / abs_a2 + 1.0
        loss = (abs_a2 / (a + eps)) * (torch.pow(base, a / 2.0) - 1.0)
        return loss.mean()

    def _apply_residual_normalization(
        self,
        resid: torch.Tensor,
        k_meas_center: torch.Tensor,
        mask_center: torch.Tensor,  # center slice mask
    ) -> Tuple[torch.Tensor, float]:
        """
        Normalize residual by a robust amplitude scale computed ONLY over acquired samples.
        """
        if self.residual_norm_mode == "none":
            return resid, 1.0

        meas_abs = self._complex_abs(k_meas_center)   # [B,1,H,W] or [B,H,W]
        # Align mask to meas_abs for safe indexing
        m = self._align_mask_to(meas_abs, mask_center)

        if m.any():
            scale_vals = meas_abs[m]
            scale = float(scale_vals.median().item()) + self.residual_norm_eps
        else:
            scale = float(meas_abs.median().item()) + self.residual_norm_eps

        return resid / scale, scale

    # ------------------- Lightning hooks -------------------
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

        # Snapshot original & working copy
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())
        adapt_model = copy.deepcopy(pl_module).to(device).train()
        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)
        for p in adapt_model.parameters():
            p.requires_grad_(True)

        opt = torch.optim.Adam(
            (p for p in adapt_model.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay
        )

        cidx = self._center_slice_index(adapt_model)
        center_mask = self._center_mask(batch.mask, cidx)

        # build LF val/train split
        num_lf = int(self.val_num_low_lines or getattr(batch, "num_low_frequencies", 16) or 16)
        train_center, val_center = self._build_val_train_masks_lowfreq(center_mask, num_lf)

        # expand to all slices
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

            # residuals on validation subset only
            resid0_raw = (k_pred_center - k_meas_center) * (val_center > 0).float()
            # normalized using acquired samples only
            resid0, scale_used = self._apply_residual_normalization(resid0_raw, k_meas_center, val_center)

            # build masked flat magnitudes for MAD/alpha estimation
            abs_r0 = self._complex_abs(resid0)                  # [B,1,H,W] or [B,H,W]
            m = self._align_mask_to(abs_r0, val_center)         # bool mask aligned to abs_r0
            abs_r0_masked_flat = abs_r0[m] if m.any() else abs_r0.reshape(-1)

            # choose α, c with EMA using caches if any
            prev_a = self._alpha_cache.get(modality, None)
            prev_c = self._c_cache.get(modality, None)
            alpha_b, c_b = self._choose_barron_params_from_masked_abs(modality, abs_r0_masked_flat, prev_a, prev_c)
            # update caches
            self._alpha_cache[modality] = alpha_b
            self._c_cache[modality] = c_b

            # DC weight per modality
            dc_weight = float(self.dc_weight_map.get(modality, self.dc_weight_map.get("default", 1.0)))

            baseline_dc_loss = float(self._barron_loss_residual(resid0, alpha_b, c_b).item())
            baseline_dc_loss_w = dc_weight * baseline_dc_loss
            # Optional clamp to avoid exploding lambda_tv if something goes wrong
            baseline_dc_loss_w = float(min(baseline_dc_loss_w, 1e12))

            # TV target-ratio
            r = float(self.tv_target_ratio_map.get(modality, self.tv_target_ratio_map["default"]))
            tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
            denom = tv0 * self.tv_tol_ratio + 1e-8
            adaptive_lambda_tv = (r * baseline_dc_loss_w) / denom

            val_metric_best = baseline_dc_loss_w
            best_state = copy.deepcopy(adapt_model.state_dict())

            print(f"[{device_name}] TTT starting for {fname}/slice{slice_num} (modality: {modality})")
            print(f"[{device_name}] Residual norm: {self.residual_norm_mode} (scale_used={scale_used:.3e})")
            print(f"[{device_name}] Barron α={alpha_b:.3f}, c={c_b:.3e} (EMA prev a={prev_a}, c={prev_c})")
            print(f"[{device_name}] DC weight: {dc_weight:.3f}")
            print(f"[{device_name}] TV target ratio r={r:.3f}, tv0={tv0:.3f}, tv_tol_ratio={self.tv_tol_ratio:.3f} -> lambda_tv={adaptive_lambda_tv:.4e}")
            print(f"[{device_name}] Baseline: val_dc={baseline_dc_loss_w:.4f} (raw={baseline_dc_loss:.4f}), METRIC={val_metric_best:.4f}")

        # ---- Iterative inner-loop ----
        alpha_cur, c_cur = alpha_b, c_b
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
                # training proxy: DC on train-center with normalized L1 (your original)
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

            # ---- Validation metric (Barron on val-center residual, no grad) ----
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
                resid_val, _ = self._apply_residual_normalization(resid_val_raw, k_meas_center_val, val_center)

                # optionally re-estimate α/c each iter from masked residuals
                if self.recompute_params_each_iter:
                    abs_rv = self._complex_abs(resid_val)
                    m = self._align_mask_to(abs_rv, val_center)
                    abs_rv_masked_flat = abs_rv[m] if m.any() else abs_rv.reshape(-1)
                    alpha_iter, c_iter = self._choose_barron_params_from_masked_abs(modality, abs_rv_masked_flat, alpha_cur, c_cur)
                    alpha_cur, c_cur = alpha_iter, c_iter

                val_dc_raw = float(self._barron_loss_residual(resid_val, alpha_cur, c_cur).item())
                val_dc = dc_weight * val_dc_raw

                img_cur = out_eval["img_pred"]
                img_cur_crop = self._center_crop_fraction(img_cur)
                tv_cur = float(self._total_variation(img_cur_crop, self.tv_eps).item())

                tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
                tv_penalty_val = max(0.0, tv_cur - tv_thresh)
                penalty_term_contrib = adaptive_lambda_tv * tv_penalty_val
                val_metric = val_dc + penalty_term_contrib

                print(
                    f"[{device_name}] iter {it+1:02d}: "
                    f"val_dc={val_dc:.4f} (raw={val_dc_raw:.4f}) | "
                    f"tv={tv_cur:.3f} (thr={tv_thresh:.3f}, +={tv_penalty_val:.3f}) | "
                    f"pen_term={penalty_term_contrib:.4f} | "
                    f"METRIC={val_metric:.4f} (best={val_metric_best:.4f}) | "
                    f"α={alpha_cur:.3f}, c={c_cur:.2e}"
                )

            if val_metric < (val_metric_best - self.min_delta):
                val_metric_best = val_metric
                best_state = copy.deepcopy(adapt_model.state_dict())
                # update modality caches with latest α/c as well
                self._alpha_cache[modality] = alpha_cur
                self._c_cache[modality] = c_cur
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
