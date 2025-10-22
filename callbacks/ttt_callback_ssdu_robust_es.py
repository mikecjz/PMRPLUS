# callbacks/ttt_callback_ssdu_robust_es.py
import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# from your project
from mri_utils import normalized_l1_loss
torch.set_float32_matmul_precision('high')  # 或 'medium'

class TestTimeTrainingCallback(L.Callback):
    """
    Test-Time Training with robust self-supervised early stopping (NO GT).
    - Validation set Ω_val = low-frequency (LF) acquired lines on the CENTER slice only.
    - Train only on Ω_tr = acquired \ Ω_val (k-space DC loss).
    - Validation metric = Huber(k-space residual on Ω_val) + lambda_tv * max(0, TV_growth - tv_tol)
      where TV_growth is computed on the center crop (W/3 x H/2), following your MATLAB convention.

    Rationale:
      * Using only LF for validation prevents a misleading monotonic decrease caused by fitting HF noise.
      * Huber is robust to outliers/noise.
      * TV-growth guard catches the typical "noise chasing" (TV tends to explode when overfitting noise).
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 7,
        log_every_n_steps: int = 1,

        # ---- LF validation split ----
        val_num_low_lines: Optional[int] = None,  # if None, use batch.num_low_frequencies
        min_val_lines: int = 8,                   # ensure at least this many lines if possible

        # ---- early stopping ----
        min_delta: float = 0.0,           # stop if val_metric > best + min_delta
        stop_on_first_worse: bool = True, # stop immediately on first degradation

        # ---- robust validation ----
        huber_delta: float = 1e-3,        # Huber threshold (in k-space units, after normalization)
                                          # smaller -> more L1-like; larger -> more L2-like

        # ---- TV guard (image domain) ----
        lambda_tv: float = 0.01,          # weight for TV-growth penalty in validation metric
        tv_tol_ratio: float = 0.10,       # allow up to +10% TV growth before penalizing
        tv_eps: float = 1e-6,             # numerical epsilon for TV

        # ---- optimization hygiene ----
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = False,
        
        # ---- logging control ----
        enable_wandb_logging: bool = True, # whether to enable W&B logging
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

        self.val_num_low_lines = val_num_low_lines
        self.min_val_lines = int(min_val_lines)

        self.min_delta = float(min_delta)
        self.stop_on_first_worse = bool(stop_on_first_worse)

        self.huber_delta = float(huber_delta)

        self.lambda_tv = float(lambda_tv)
        self.tv_tol_ratio = float(tv_tol_ratio)
        self.tv_eps = float(tv_eps)

        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.enable_wandb_logging = bool(enable_wandb_logging)

        self.wandb_logger: Optional[WandbLogger] = None
        self.normalized_l1 = normalized_l1_loss()
        
        # ---- File-level reset tracking (方案2) ----
        self.current_fname: Optional[str] = None  # Track current file being processed
        self._true_initial_state: Optional[dict] = None  # True initial checkpoint state (saved in setup)
        self.slice_count = 0  # Count slices processed in current file

    # ---------------- utils ----------------
    @staticmethod
    def _center_slice_index(module) -> int:
        num_adj = getattr(module, "num_adj_slices", 1)
        return int(num_adj // 2)

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
        return chunks[cidx]  # (B,1,H,W,[2?])

    @staticmethod
    def _center_crop_fraction(img: torch.Tensor, frac_w: float = 1/3, frac_h: float = 1/2) -> torch.Tensor:
        """Center crop to (H*frac_h, W*frac_w); supports (B,1,H,W) or (B,H,W)."""
        was_3d = False
        if img.dim() == 3:
            img = img.unsqueeze(1)
            was_3d = True
        B, C, H, W = img.shape
        tw, th = max(1, round(W*frac_w)), max(1, round(H*frac_h))
        x0, y0 = (W - int(tw)) // 2, (H - int(th)) // 2
        out = img[:, :, y0:y0+int(th), x0:x0+int(tw)]
        return out.squeeze(1) if was_3d else out

    @staticmethod
    def _total_variation(img: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """TV on (B,1,H,W) or (B,H,W). Returns scalar mean per-batch."""
        if img.dim() == 3:
            img = img.unsqueeze(1)
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]    # (B,C,H,W-1)
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]    # (B,C,H-1,W)
        # align to common interior (H-1, W-1)
        dx_i = dx[:, :, :-1, :]
        dy_i = dy[:, :, :, :-1]
        tv = torch.sqrt(dx_i*dx_i + dy_i*dy_i + eps).mean()
        return tv

    @staticmethod
    def _huber_on_complex_residual(residual: torch.Tensor, delta: float) -> torch.Tensor:
        """
        residual: (..., 2) complex (real, imag) OR any real tensor.
        Apply Huber to magnitude if complex-last, else to values directly.
        """
        if residual.dim() >= 1 and residual.size(-1) == 2:
            mag = torch.linalg.vector_norm(residual, dim=-1)
            x = mag
        else:
            x = torch.abs(residual)
        # huber
        quad = torch.clamp(x, max=delta)
        lin  = x - quad
        loss = 0.5 * (quad**2) / max(delta, 1e-12) + lin
        return loss.mean()

    @torch.no_grad()
    def _build_val_train_masks_lowfreq(
        self, center_mask: torch.Tensor, num_low_lines: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split center-slice acquired mask into contiguous LF band for validation (Ω_val)
        and the rest for train (Ω_tr). Works even if LF band isn't fully dense by
        selecting the nearest acquired lines to DC.
        """
        B, _, H, W = center_mask.shape[:4]
        val_mask = torch.zeros_like(center_mask)
        tr_mask  = torch.zeros_like(center_mask)

        # choose target number of LF lines
        k = max(self.min_val_lines, int(num_low_lines))
        k = min(k, H)  # clamp

        # decide acquisition logic channel
        if self._is_complex_last(center_mask):
            acq = (center_mask[..., 0] > 0)  # (B,1,H,W)
        else:
            acq = center_mask if center_mask.dim() == 4 else center_mask.squeeze(-1)
            acq = (acq > 0)  # (B,1,H,W)

        cy = H // 2
        for b in range(B):
            # acquired rows
            rows = torch.where(acq[b, 0].any(dim=1))[0]  # indices of acquired ky rows
            if rows.numel() == 0:
                continue
            # sort rows by proximity to DC (cy)
            dist = torch.abs(rows - cy)
            order = torch.argsort(dist)
            chosen = rows[order[:k]]  # LF band rows
            # mark val on chosen rows, train on the rest of acquired
            if self._is_complex_last(center_mask):
                val_mask[b, 0, chosen, :, :] = center_mask[b, 0, chosen, :, :]
                tr_rows = rows[order[k:]]
                tr_mask[b, 0, tr_rows, :, :] = center_mask[b, 0, tr_rows, :, :]
            else:
                if center_mask.dim() == 5:
                    val_mask[b, 0, chosen, :, 0] = center_mask[b, 0, chosen, :, 0]
                    tr_rows = rows[order[k:]]
                    tr_mask[b, 0, tr_rows, :, 0] = center_mask[b, 0, tr_rows, :, 0]
                else:
                    val_mask[b, 0, chosen, :] = center_mask[b, 0, chosen, :]
                    tr_rows = rows[order[k:]]
                    tr_mask[b, 0, tr_rows, :] = center_mask[b, 0, tr_rows, :]
        return tr_mask, val_mask

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    # --------------- Lightning hooks ---------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        # 在setup阶段保存真正的原始状态（方案2）
        if self._true_initial_state is None:
            self._true_initial_state = copy.deepcopy(pl_module.state_dict())
            print(f"[TTT-SSDU] 🔵 Saved TRUE initial checkpoint state in setup")
        
        if self.enable_wandb_logging:
            loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
            self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
            if self.wandb_logger is None:
                print("[TTT-SSDU] WARNING: WandbLogger not found but enable_wandb_logging=True. Proceeding without W&B logging.")
        else:
            print("[TTT-SSDU] W&B logging disabled by enable_wandb_logging=False")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = batch.masked_kspace.device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)
        fname = getattr(batch, 'fname', ['unknown'])[0] if hasattr(batch, 'fname') else 'unknown'
        slice_num = getattr(batch, 'slice_num', [0])[0] if hasattr(batch, 'slice_num') else 0
        
        # ========== FILE-LEVEL RESET LOGIC (方案2) ==========
        # Detect file change -> reset to TRUE initial state
        if fname != self.current_fname:
            print(f"[{device_name}] 🔄 New file detected: '{self.current_fname}' -> '{fname}'")
            print(f"[{device_name}] 🔄 Resetting model to TRUE initial checkpoint state (processed {self.slice_count} slices in previous file)")
            # 使用真正的原始状态进行重置
            if self._true_initial_state is not None:
                pl_module.load_state_dict(self._true_initial_state)
                print(f"[{device_name}] ✅ Model reset to TRUE initial state")
            else:
                print(f"[{device_name}] ⚠️ No TRUE initial state available, using current state")
            self.current_fname = fname
            self.slice_count = 0
        
        self.slice_count += 1
        print(f"[{device_name}] 📍 Processing: {fname}/slice{slice_num} (slice #{self.slice_count} in this file)")
        # ============================================
        
        # save original (for backward compatibility)
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())

        # working copy
        adapt_model = copy.deepcopy(pl_module).to(device).train()
        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)
        
        # Ensure parameters require grad
        for param in adapt_model.parameters():
            param.requires_grad_(True)

        opt = torch.optim.Adam(
            (p for p in adapt_model.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay
        )

        # ---- build LF val / HF train masks on center slice ----
        cidx = self._center_slice_index(adapt_model)
        center_mask = self._center_mask(batch.mask, cidx)  # (B,1,H,W,[2?])

        num_lf = (
            int(batch.num_low_frequencies)
            if hasattr(batch, "num_low_frequencies") and batch.num_low_frequencies is not None
            else 16
        )
        if self.val_num_low_lines is not None:
            num_lf = int(self.val_num_low_lines)

        train_center, val_center = self._build_val_train_masks_lowfreq(center_mask, num_lf)

        # expand to full multi-slice masks by replacing only center slice
        if batch.mask.dim() >= 5:
            chunks_tr = list(torch.chunk(batch.mask.clone(), chunks=batch.mask.shape[1], dim=1))
            chunks_va = list(torch.chunk(batch.mask.clone(), chunks=batch.mask.shape[1], dim=1))
            chunks_tr[cidx] = train_center
            chunks_va[cidx] = val_center
            mask_train_full = torch.cat(chunks_tr, dim=1)
            mask_val_full   = torch.cat(chunks_va, dim=1)
        else:
            mask_train_full = train_center
            mask_val_full   = val_center

        # masked_kspace for training: zero out Ω_val on center slice
        if self._is_complex_last(batch.masked_kspace):
            keep = (train_center > 0).float()
            keep = self._broadcast_like(keep, batch.masked_kspace)
        else:
            keep = (train_center > 0).float()
            if keep.dim() == 5: keep = keep.squeeze(-1)
            keep = self._broadcast_like(keep, batch.masked_kspace)
        k_train = batch.masked_kspace * keep

        # ---- baseline recon & metrics ----
        with torch.no_grad():
            out0 = adapt_model(
                k_train, mask_train_full,
                batch.num_low_frequencies, batch.mask_type,
                compute_sens_per_coil=pl_module.compute_sens_per_coil,
            )
            img0 = out0["img_pred"]
            if img0.dim() == 3: img0 = img0.unsqueeze(1)
            img0_crop = self._center_crop_fraction(img0, 1/3, 1/2)
            tv0 = float(self._total_variation(img0_crop, self.tv_eps).item())

            # validation residual on Ω_val (align to center-slice multi-coil shapes)
            if self._is_complex_last(out0["pred_kspace"]):
                mv = self._broadcast_like(val_center, out0["pred_kspace"]).float()
                # center-slice of measured k-space to match pred_kspace
                ks_chunks = torch.chunk(batch.masked_kspace, chunks=adapt_model.num_adj_slices, dim=1)
                ks_center = ks_chunks[cidx]
                mv_y = self._broadcast_like(val_center, ks_center).float()
                yv = mv_y * ks_center
            else:
                mv = val_center
                if mv.dim() == 5: mv = mv.squeeze(-1)
                mv = self._broadcast_like(mv, out0["pred_kspace"]).float()
                ks_chunks = torch.chunk(batch.masked_kspace, chunks=adapt_model.num_adj_slices, dim=1)
                ks_center = ks_chunks[cidx]
                mv_y = self._broadcast_like(mv, ks_center).float()
                yv = mv_y * ks_center

            resid0 = (out0["pred_kspace"] - yv) * mv
            val_dc0 = float(self._huber_on_complex_residual(resid0, self.huber_delta).item())

            val_metric_best = val_dc0  # tv-penalty is zero at baseline by definition
            best_state = copy.deepcopy(adapt_model.state_dict())

            # Check for NaN and skip TTT if baseline is invalid
            if torch.isnan(torch.tensor([val_dc0, tv0])).any():
                print(f"[{device_name}] WARNING: TTT baseline contains NaN for {fname}/slice{slice_num}, skipping TTT")
                return
            
            print(f"[{device_name}] TTT starting for {fname}/slice{slice_num} (batch {batch_idx})")
            print(f"[{device_name}] TTT baseline: val_dc={val_dc0:.6f}, tv={tv0:.6f}, val_metric={val_metric_best:.6f}")
            
            if trainer.is_global_zero and self.enable_wandb_logging and self.wandb_logger is not None:
                self.wandb_logger.experiment.log({
                    "TTT/val_dc_huber/baseline": val_dc0,
                    "TTT/tv_center/baseline": tv0,
                    "TTT/val_metric/best": val_metric_best,
                    "TTT/inner_iter": -1,
                })

        # ---- inner loop ----
        for it in range(self.inner_steps):
            opt.zero_grad()

            # Enable gradients for training
            with torch.enable_grad():
                out = adapt_model(
                    k_train, mask_train_full,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )

                # train DC loss on Ω_tr (standard normalized L1, align to center-slice)
                if self._is_complex_last(out["pred_kspace"]):
                    mt = self._broadcast_like(train_center, out["pred_kspace"]).float()
                    ks_chunks = torch.chunk(batch.masked_kspace, chunks=adapt_model.num_adj_slices, dim=1)
                    ks_center = ks_chunks[cidx]
                    mt_y = self._broadcast_like(train_center, ks_center).float()
                    yt = mt_y * ks_center
                else:
                    mt = train_center
                    if mt.dim() == 5: mt = mt.squeeze(-1)
                    mt = self._broadcast_like(mt, out["pred_kspace"]).float()
                    ks_chunks = torch.chunk(batch.masked_kspace, chunks=adapt_model.num_adj_slices, dim=1)
                    ks_center = ks_chunks[cidx]
                    mt_y = self._broadcast_like(mt, ks_center).float()
                    yt = mt_y * ks_center
                
                pred_masked = out["pred_kspace"] * mt
                target_masked = yt.detach()  # ensure target doesn't need gradients
                
                loss_tr = self.normalized_l1(pred_masked, target_masked)
                
                # Check for NaN in training loss
                if torch.isnan(loss_tr):
                    print(f"[{device_name}] WARNING: TTT iter {it} got NaN loss, skipping this iteration")
                    continue
                    
                loss_tr.backward()
            opt.step()
            
            # Print training info every step (参考原版格式)
            print(f"[{device_name}] TTT iter {it+1}/{self.inner_steps}: train_loss={loss_tr.item():.6f}")

            # ---- validation metric on Ω_val (no grad) ----
            with torch.no_grad():
                out_eval = adapt_model(
                    k_train, mask_train_full,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )
                if self._is_complex_last(out_eval["pred_kspace"]):
                    mv = self._broadcast_like(val_center, out_eval["pred_kspace"]).float()
                    ks_chunks = torch.chunk(batch.masked_kspace, chunks=adapt_model.num_adj_slices, dim=1)
                    ks_center = ks_chunks[cidx]
                    mv_y = self._broadcast_like(val_center, ks_center).float()
                    yv = mv_y * ks_center
                else:
                    mv = val_center
                    if mv.dim() == 5: mv = mv.squeeze(-1)
                    mv = self._broadcast_like(mv, out_eval["pred_kspace"]).float()
                    ks_chunks = torch.chunk(batch.masked_kspace, chunks=adapt_model.num_adj_slices, dim=1)
                    ks_center = ks_chunks[cidx]
                    mv_y = self._broadcast_like(mv, ks_center).float()
                    yv = mv_y * ks_center

                resid = (out_eval["pred_kspace"] - yv) * mv
                val_dc = float(self._huber_on_complex_residual(resid, self.huber_delta).item())

                img = out_eval["img_pred"]
                if img.dim() == 3: img = img.unsqueeze(1)
                img_crop = self._center_crop_fraction(img, 1/3, 1/2)
                tv_cur = float(self._total_variation(img_crop, self.tv_eps).item())

                # TV-growth penalty (only penalize growth beyond tv0*(1+tv_tol_ratio))
                tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
                tv_penalty = max(0.0, tv_cur - tv_thresh)
                val_metric = val_dc + self.lambda_tv * tv_penalty
                
                # Check for NaN in validation metrics
                if torch.isnan(torch.tensor([val_dc, tv_cur, val_metric])).any():
                    print(f"[{device_name}] WARNING: TTT iter {it} got NaN validation metrics, skipping")
                    continue
                
                # Print validation info (简化格式，类似原版)
                print(f"[{device_name}]     val_loss={val_dc:.4f}, tv={tv_cur:.1f} (↑{tv_penalty:.1f}), "
                      f"metric={val_metric:.4f} (best={val_metric_best:.4f})")

            # logging (参考原版的per-subject格式)
            if trainer.is_global_zero and self.enable_wandb_logging and self.wandb_logger is not None and \
               (it % self.log_every_n_steps == 0 or it == self.inner_steps - 1):
                key = f"TTT/batch{batch_idx:03d}"
                self.wandb_logger.experiment.log({
                    f"{key}/train_loss": float(loss_tr.item()),
                    f"{key}/val_loss": val_dc,
                    f"{key}/val_metric": val_metric,
                    f"{key}/tv_current": tv_cur,
                    f"{key}/tv_penalty": tv_penalty,
                    "TTT/global_step": int(it),
                })

            # early stopping
            if val_metric < (val_metric_best - self.min_delta):
                val_metric_best = val_metric
                best_state = copy.deepcopy(adapt_model.state_dict())
                print(f"[{device_name}]     ✓ New best: {val_metric_best:.4f}")
            elif self.stop_on_first_worse:
                print(f"[{device_name}]     ✗ Early stop at iter {it+1} (worse: {val_metric:.4f} > {val_metric_best:.4f})")
                break

        # restore best and copy back
        adapt_model.load_state_dict(best_state)
        pl_module.load_state_dict(adapt_model.state_dict())
        pl_module.eval()
        
        print(f"[{device_name}] TTT completed for {fname}/slice{slice_num}: final best={val_metric_best:.4f}")
        
        # 
        if trainer.is_global_zero and self.enable_wandb_logging and self.wandb_logger is not None:
            key = f"TTT/batch{batch_idx:03d}"
            self.wandb_logger.experiment.log({
                f"{key}/final_best_metric": val_metric_best,
                f"{key}/fname": fname,
                f"{key}/slice_num": slice_num,
            })

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, prediction, batch, dataloader_idx: int = 0, **kwargs):
        if hasattr(pl_module, "_orig_state"):
            pl_module.load_state_dict(pl_module._orig_state)
            pl_module.eval()
            del pl_module._orig_state
