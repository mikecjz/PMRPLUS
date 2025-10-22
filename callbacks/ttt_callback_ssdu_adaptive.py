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
    
    - Adaptive TV Guard: The TV penalty weight is automatically scaled based on the 
      baseline k-space loss of each individual slice.
    - Modality-Specific Huber Delta: The delta for the robust Huber loss can be specified 
      per modality via a dictionary.
    - Enhanced Logging: Provides detailed console output of all internal metrics 
      for easier parameter tuning and debugging.
    """

    def __init__(
        self,
        # --- Basic TTT settings ---
        lr: float = 1e-7,
        inner_steps: int = 15,
        log_every_n_steps: int = 1,

        # --- Validation split and early stopping ---
        val_num_low_lines: Optional[int] = None,
        min_val_lines: int = 8,
        min_delta: float = 0.0,
        stop_on_first_worse: bool = True,
        
        # --- Hybrid Adaptive Strategy Parameters ---
        huber_delta_map: Dict[str, float] = None,
        relative_lambda_tv: float = 0.01,
        adaptive_tv_scale_factor: float = 0.001,
        tv_tol_ratio: float = 0.10,
        
        # --- Technical details ---
        tv_eps: float = 1e-6,
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = False,
        enable_wandb_logging: bool = True,
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)
        self.val_num_low_lines = val_num_low_lines
        self.min_val_lines = int(min_val_lines)
        self.min_delta = float(min_delta)
        self.stop_on_first_worse = bool(stop_on_first_worse)
        
        if huber_delta_map is None:
            self.huber_delta_map = {'default': 1e-3}
        else:
            self.huber_delta_map = huber_delta_map
            if 'default' not in self.huber_delta_map:
                self.huber_delta_map['default'] = 1e-3
        
        self.relative_lambda_tv = float(relative_lambda_tv)
        self.adaptive_tv_scale_factor = float(adaptive_tv_scale_factor)
        
        self.tv_tol_ratio = float(tv_tol_ratio)
        self.tv_eps = float(tv_eps)
        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.enable_wandb_logging = bool(enable_wandb_logging)
        self.wandb_logger: Optional[WandbLogger] = None
        self.normalized_l1 = normalized_l1_loss()

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
            return path_parts[0]
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
    def _huber_on_complex_residual(residual: torch.Tensor, delta: float) -> torch.Tensor:
        if residual.size(-1) == 2:
            x = torch.linalg.vector_norm(residual, dim=-1)
        else:
            x = torch.abs(residual)
        
        delta = max(delta, 1e-12)
        abs_x = torch.abs(x)
        quadratic = torch.clamp(abs_x, max=delta)
        linear = abs_x - quadratic
        loss = 0.5 * (quadratic**2) / delta + linear
        return loss.mean()
    
    # ... (Other utility functions from your original file like _is_complex_last, _broadcast_like, etc. can be kept as they were) ...
    # ... I will include them here for a complete, runnable file ...

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
            if rows.numel() == 0: continue
            dist = torch.abs(rows - cy)
            order = torch.argsort(dist)
            chosen_val_rows = rows[order[:k]]
            chosen_tr_rows = rows[order[k:]]
            
            # This logic correctly handles different mask formats
            if self._is_complex_last(center_mask):
                val_mask[b, 0, chosen_val_rows, :, :] = center_mask[b, 0, chosen_val_rows, :, :]
                tr_mask[b, 0, chosen_tr_rows, :, :] = center_mask[b, 0, chosen_tr_rows, :, :]
            else: # Real mask format
                val_mask[b, 0, chosen_val_rows] = center_mask[b, 0, chosen_val_rows]
                tr_mask[b, 0, chosen_tr_rows] = center_mask[b, 0, chosen_tr_rows]
        return tr_mask, val_mask

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

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
        huber_delta = self.huber_delta_map.get(modality, self.huber_delta_map['default'])
        
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

        num_lf = int(self.val_num_low_lines or batch.num_low_frequencies or 16)
        train_center, val_center = self._build_val_train_masks_lowfreq(center_mask, num_lf)
        
        # Expand masks to full multi-slice shape
        mask_train_full = batch.mask.clone()
        mask_val_full = torch.zeros_like(batch.mask)
        mask_train_full[:, cidx] = train_center
        mask_val_full[:, cidx] = val_center
        
        k_train = batch.masked_kspace * (mask_train_full > 0).float()
        
        with torch.no_grad():
            # <--- [FIX] Replaced placeholder '...' with the full, correct arguments ---
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
            resid0 = (k_pred_center - k_meas_center) * (val_center > 0).float()
            baseline_dc_loss = float(self._huber_on_complex_residual(resid0, huber_delta).item())
            
            adaptive_lambda_tv = self.relative_lambda_tv * baseline_dc_loss * self.adaptive_tv_scale_factor
            
            val_metric_best = baseline_dc_loss
            best_state = copy.deepcopy(adapt_model.state_dict())
            
            print(f"[{device_name}] TTT starting for {fname}/slice{slice_num} (modality: {modality})")
            print(f"[{device_name}] TTT Params: huber_delta={huber_delta:.1e}, adaptive_lambda_tv={adaptive_lambda_tv:.6f} (rel_lambda={self.relative_lambda_tv})")
            print(f"[{device_name}] TTT Baseline: baseline_dc={baseline_dc_loss:.4f}, tv0={tv0:.1f}, val_metric={val_metric_best:.4f}")

        for it in range(self.inner_steps):
            opt.zero_grad()
            with torch.enable_grad():
                # <--- [FIX] Ensuring all forward calls have the correct arguments ---
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
                    continue
                loss_tr.backward()
            opt.step()
            
            with torch.no_grad():
                # <--- [FIX] Ensuring all forward calls have the correct arguments ---
                out_eval = adapt_model(
                    k_train, 
                    mask_train_full, 
                    batch.num_low_frequencies, 
                    batch.mask_type,
                    compute_sens_per_coil=getattr(pl_module, 'compute_sens_per_coil', False)
                )
                
                k_pred_center_val = out_eval["pred_kspace"][:, cidx]
                k_meas_center_val = batch.masked_kspace[:, cidx]
                resid_val = (k_pred_center_val - k_meas_center_val) * (val_center > 0).float()
                val_dc = float(self._huber_on_complex_residual(resid_val, huber_delta).item())
                
                img_cur = out_eval["img_pred"]
                img_cur_crop = self._center_crop_fraction(img_cur)
                tv_cur = float(self._total_variation(img_cur_crop, self.tv_eps).item())

                tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
                tv_penalty_val = max(0.0, tv_cur - tv_thresh)
                
                penalty_term_contrib = adaptive_lambda_tv * tv_penalty_val
                val_metric = val_dc + penalty_term_contrib
                
                log_line = (
                    f"[{device_name}] iter {it+1:02d}: "
                    f"val_dc={val_dc:.4f} | "
                    f"tv={tv_cur:.1f} (penalty={tv_penalty_val:.1f}) | "
                    f"pen_term={penalty_term_contrib:.4f} | "
                    f"METRIC={val_metric:.4f} (best={val_metric_best:.4f})"
                )
                print(log_line)

            if val_metric < (val_metric_best - self.min_delta):
                val_metric_best = val_metric
                best_state = copy.deepcopy(adapt_model.state_dict())
                print(f"[{device_name}]     ✓ New best found.")
            elif self.stop_on_first_worse:
                print(f"[{device_name}]     ✗ Early stopping triggered.")
                break
        
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