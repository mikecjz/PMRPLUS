# callbacks/ttt_callback_mae.py

import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# Provided by your project
from mri_utils import normalized_l1_loss


class TestTimeTrainingCallback(L.Callback):
    """
    MAE-style Test-Time Training (TTT) without early stopping.

    Design:
      - Teacher: forward pass with the original measurements (no grad) to produce a stable target.
      - Student: forward pass with an "extra-sparsified" k-space on the center slice + image-domain patch masking.
      - Loss: L = alpha_dc * DC_loss + beta_mae * MAE_patch_loss
      - Parameter updates: by default, ALL trainable parameters are updated (no restriction).
        (BatchNorm running stats can optionally be frozen; this does not block affine parameters.)

    Notes:
      - Works with both (B, C, H, W) and (B, C, H, W, 2) complex-last layouts for k-space/masks.
      - No early stopping; uses fixed number of inner steps.
      - The model state is restored after each batch to avoid cross-sample drift.
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 6,
        log_every_n_steps: int = 1,
        # MAE/TTT hyperparameters (safe defaults)
        kdrop_frac: float = 0.10,       # fraction of additionally dropped k-lines on the center slice
        mae_mask_ratio: float = 0.5,    # image patch masking ratio for MAE loss
        mae_patch: int = 16,            # patch size for MAE masking (H,W should be multiples; we auto-pad)
        alpha_dc: float = 1.0,          # weight of k-space data-consistency loss
        beta_mae: float = 0.2,          # weight of image-domain MAE patch loss
        weight_decay: float = 0.0,      # optional L2 on all parameters during TTT
        freeze_bn_stats: bool = True,   # freeze BatchNorm running means/vars during TTT (affine params still updated)
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

        self.kdrop_frac = float(kdrop_frac)
        self.mae_mask_ratio = float(mae_mask_ratio)
        self.mae_patch = int(mae_patch)
        self.alpha_dc = float(alpha_dc)
        self.beta_mae = float(beta_mae)
        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)

        self.wandb_logger: Optional[WandbLogger] = None
        self._ttt_step = 0
        self.normalized_l1 = normalized_l1_loss()

    # -------------------- small helpers --------------------
    @staticmethod
    def _center_slice_index(module) -> int:
        """Return the center slice index based on module.num_adj_slices."""
        num_adj = getattr(module, "num_adj_slices", 1)
        return int(num_adj // 2)

    @staticmethod
    def _is_complex_last_dim(x: torch.Tensor) -> bool:
        """Check if the last dimension is the 2-channel complex representation (real/imag)."""
        return x.dim() >= 5 and x.size(-1) == 2

    @staticmethod
    def _broadcast_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Broadcast tensor x to the shape of target if needed."""
        return x.expand_as(target) if list(x.shape) != list(target.shape) else x

    @torch.no_grad()
    def _extract_center_mask(self, full_mask: torch.Tensor, center_idx: int) -> torch.Tensor:
        """
        Extract the center-slice mask from a multi-slice mask.
        Returns shape (B, 1, H, W, [2?]).
        """
        if full_mask.dim() < 5:
            # Already (B,1,H,W) or (B,1,H,W,2)
            return full_mask
        chunks = torch.chunk(full_mask, chunks=full_mask.shape[1], dim=1)
        return chunks[center_idx]  # (B,1,H,W,[2])

    def _rand_kspace_drop_center(self, center_mask: torch.Tensor, drop_frac: float) -> torch.Tensor:
        """
        Randomly drop a fraction of acquired k-lines on the center slice mask along the H dimension (rows).
        Input/Output: (B,1,H,W,[2?]) with values in {0,1}.
        """
        assert 0.0 <= drop_frac < 1.0
        B = center_mask.shape[0]
        aug = center_mask.clone()

        # Use a single logical channel to determine acquisition (if last dim == 2, use the first channel)
        if self._is_complex_last_dim(center_mask):
            cm = center_mask[..., 0]  # (B,1,H,W)
        else:
            cm = center_mask if center_mask.dim() == 4 else center_mask.squeeze(-1)  # (B,1,H,W)

        for b in range(B):
            acquired = (cm[b, 0] > 0)  # (H,W)
            lines = torch.where(acquired.any(dim=1))[0]  # indices of acquired rows
            if lines.numel() <= 1:
                continue
            n_drop = max(1, int(lines.numel() * drop_frac))
            drop_idx = lines[torch.randperm(lines.numel(), device=center_mask.device)[:n_drop]]
            if self._is_complex_last_dim(aug):
                aug[b, 0, drop_idx, :, :] = 0
            else:
                aug[b, 0, drop_idx, :, ...] = 0
        return aug

    @staticmethod
    def _pad_to_multiple(img: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad (B,C,H,W) to multiples of 'multiple' on H and W; return padded tensor and original (H,W)."""
        B, C, H, W = img.shape
        Hm = (H + multiple - 1) // multiple * multiple
        Wm = (W + multiple - 1) // multiple * multiple
        pad_h, pad_w = Hm - H, Wm - W
        # Pad bottom and right using 'replicate' to avoid introducing zeros
        img_pad = F.pad(img, (0, pad_w, 0, pad_h), mode="replicate")
        return img_pad, (H, W)

    @staticmethod
    def _unpad(img_pad: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
        """Crop padded (B,C,H,W) back to original (H,W)."""
        H, W = orig_hw
        return img_pad[..., :H, :W]

    def _make_patch_mask(self, img: torch.Tensor, mask_ratio: float, patch: int) -> torch.Tensor:
        """
        Create an image-domain patch mask for MAE:
          - Pad to multiples of patch, create a (B,1,ph,pw) binary grid mask, upsample to (B,1,H,W), then unpad.
          - Returns (B,1,H,W) with values in {0,1}, where 1 marks masked (to-be-predicted) regions.
        """
        B, C, H, W = img.shape
        img_pad, orig_hw = self._pad_to_multiple(img, patch)
        _, _, Hp, Wp = img_pad.shape
        ph, pw = Hp // patch, Wp // patch

        num_total = ph * pw
        num_mask = max(1, int(num_total * mask_ratio))

        # Choose masked cells per sample
        mask_cells = torch.zeros(B, 1, num_total, device=img.device)
        rand_idx = torch.rand(B, num_total, device=img.device).argsort(dim=1)[:, :num_mask]
        mask_cells.scatter_(2, rand_idx.unsqueeze(1), 1.0)
        mask_cells = mask_cells.view(B, 1, ph, pw)  # (B,1,ph,pw)

        patch_mask = F.interpolate(mask_cells, scale_factor=patch, mode="nearest")  # (B,1,Hp,Wp)
        patch_mask = self._unpad(patch_mask, orig_hw)  # (B,1,H,W)
        return patch_mask

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        """Set BatchNorm modules to eval mode to freeze running stats during TTT (affine params still trainable)."""
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    # -------------------- Lightning hooks --------------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        """Bind WandbLogger instance for logging."""
        loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                self.wandb_logger = lg
                break
        if self.wandb_logger is None:
            print("[TTT-MAE] WARNING: WandbLogger not found. Proceeding without W&B logging.")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = batch.masked_kspace.device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)
        
        # Extract metadata for logging
        fname = getattr(batch, 'fname', ['unknown'])
        fname = fname[0] if isinstance(fname, (list, tuple)) else str(fname)
        slice_num = getattr(batch, 'slice_num', [0])
        slice_num = int(slice_num[0] if isinstance(slice_num, (list, tuple)) else slice_num)

        # Save original weights to restore after this batch
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())

        # Create an adaptive copy for TTT
        adapt_model = copy.deepcopy(pl_module).to(device).train()
        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)
        
        print(f"[{device_name}] TTT(MAE) starting for {fname}/slice{slice_num} (batch {batch_idx})")

        # Optimizer on ALL trainable parameters (no restriction)
        opt = torch.optim.Adam(
            (p for p in adapt_model.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Get center-slice mask (B,1,H,W,[2?])
        center_idx = self._center_slice_index(adapt_model)
        center_mask_orig = self._extract_center_mask(batch.mask, center_idx)

        # -------------- Teacher forward (no grad) on original measurements --------------
        with torch.no_grad():
            out_teacher = adapt_model(
                batch.masked_kspace, batch.mask,
                batch.num_low_frequencies, batch.mask_type,
                compute_sens_per_coil=pl_module.compute_sens_per_coil,
            )
            img_teacher = out_teacher["img_pred"]  # (B,H,W) or (B,1,H,W)
            if img_teacher.dim() == 3:
                img_teacher = img_teacher.unsqueeze(1)  # -> (B,1,H,W)

        # -------------- Student inner-loop updates --------------
        for it in range(self.inner_steps):
            # Enable gradients explicitly (Lightning disables them during predict)
            with torch.enable_grad():
                opt.zero_grad()

                # (1) Extra k-space sparsification on the center slice
                center_mask_aug = self._rand_kspace_drop_center(center_mask_orig, self.kdrop_frac)  # (B,1,H,W,[2?])

                # (2) Build full augmented mask by replacing the center slice only
                if batch.mask.dim() >= 5:
                    mask_chunks = list(torch.chunk(batch.mask, chunks=batch.mask.shape[1], dim=1))
                    mask_chunks[center_idx] = center_mask_aug
                    mask_aug_full = torch.cat(mask_chunks, dim=1)
                else:
                    mask_aug_full = center_mask_aug  # already a single-slice mask

                # (3) Create augmented masked_kspace for the center slice
                keep_center = (center_mask_aug > 0).float()  # (B,1,H,W,[2?])
                if self._is_complex_last_dim(batch.masked_kspace):
                    keep_center = self._broadcast_like(keep_center, batch.masked_kspace)
                else:
                    keep_center = keep_center.squeeze(-1) if keep_center.dim() == 5 else keep_center
                    keep_center = self._broadcast_like(keep_center, batch.masked_kspace)
                masked_kspace_aug = batch.masked_kspace * keep_center

                # (4) Student forward on augmented inputs
                out_student = adapt_model(
                    masked_kspace_aug, mask_aug_full,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )
                img_student = out_student["img_pred"]
                if img_student.dim() == 3:
                    img_student = img_student.unsqueeze(1)  # (B,1,H,W)

                # (5) k-space DC loss using the augmented center mask
                if self._is_complex_last_dim(out_student["pred_kspace"]):
                    cm_for_dc = self._broadcast_like(center_mask_aug, out_student["pred_kspace"]).float()
                else:
                    cm_for_dc = center_mask_aug.squeeze(-1) if center_mask_aug.dim() == 5 else center_mask_aug
                    cm_for_dc = self._broadcast_like(cm_for_dc, out_student["pred_kspace"]).float()

                loss_dc = self.normalized_l1(
                    out_student["pred_kspace"] * cm_for_dc,
                    out_student["original_kspace"] * cm_for_dc,
                )

                # (6) Image-domain MAE patch loss: only on masked patches (student -> teacher)
                patch_mask = self._make_patch_mask(img_student, mask_ratio=self.mae_mask_ratio, patch=self.mae_patch)
                if img_student.shape[1] != patch_mask.shape[1]:
                    patch_mask = patch_mask.expand(-1, img_student.shape[1], -1, -1)
                loss_mae = F.l1_loss(img_student * patch_mask, img_teacher.detach() * patch_mask)

                # (7) Total loss and update
                loss = self.alpha_dc * loss_dc + self.beta_mae * loss_mae
                
                # NaN guard
                if torch.isnan(loss) or torch.isnan(loss_dc) or torch.isnan(loss_mae):
                    print(f"[{device_name}] WARNING: TTT iter {it+1} got NaN loss (total={loss.item():.6f}, "
                          f"dc={loss_dc.item():.6f}, mae={loss_mae.item():.6f}), skipping this iteration")
                    continue
                
                loss.backward()
                opt.step()
                
                # Save loss values for logging (detach to avoid keeping computation graph)
                loss_total = loss.item()
                loss_dc_val = loss_dc.item()
                loss_mae_val = loss_mae.item()

            # Console logging every step
            print(f"[{device_name}] TTT(MAE) iter {it+1}/{self.inner_steps}: "
                  f"loss={loss_total:.6f} (dc={loss_dc_val:.6f}, mae={loss_mae_val:.6f})")

            # W&B Logging
            if (it % self.log_every_n_steps == 0) or (it == self.inner_steps - 1):
                if trainer.is_global_zero and self.wandb_logger is not None:
                    self.wandb_logger.experiment.log({
                        f"TTT/batch{batch_idx:03d}/loss_total": float(loss_total),
                        f"TTT/batch{batch_idx:03d}/loss_dc": float(loss_dc_val),
                        f"TTT/batch{batch_idx:03d}/loss_mae": float(loss_mae_val),
                        f"TTT/batch{batch_idx:03d}/fname": fname,
                        f"TTT/batch{batch_idx:03d}/slice_num": slice_num,
                        "TTT/inner_iter": int(it),
                    })
                self._ttt_step += 1

        # Load adapted weights back to the original module for this batch's prediction
        pl_module.load_state_dict(adapt_model.state_dict())
        pl_module.eval()
        
        print(f"[{device_name}] TTT(MAE) completed for {fname}/slice{slice_num}: ran {self.inner_steps} iterations (no early stopping)")

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, prediction, batch, dataloader_idx: int = 0, **kwargs):
        """Restore the original weights after prediction to avoid cross-sample drift."""
        if hasattr(pl_module, "_orig_state"):
            pl_module.load_state_dict(pl_module._orig_state)
            pl_module.eval()
            del pl_module._orig_state
