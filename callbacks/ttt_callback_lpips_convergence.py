import copy
from typing import Optional

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info

# Your project's loss function and metrics
from mri_utils import normalized_l1_loss, LPIPS
torch.set_float32_matmul_precision('high')

class LPIPSConvergenceTTTCallback(L.Callback):
    """
    Test-Time Training with early stopping based on perceptual convergence.

    Strategy:
    - Train on ALL acquired k-space points for data consistency.
    - At each step, compare the current output image with the previous one using LPIPS.
    - Stop when the perceptual change between images falls below a threshold.
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 25,
        log_every_n_steps: int = 1,
        
        # ---- LPIPS early stopping ----
        stopping_threshold: float = 1e-4, # Key hyperparameter: stop if LPIPS(t, t-1) < threshold

        # ---- optimization hygiene ----
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = True,
        
        # ---- logging control ----
        enable_wandb_logging: bool = True,
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)
        self.stopping_threshold = float(stopping_threshold)
        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        self.enable_wandb_logging = bool(enable_wandb_logging)

        self.wandb_logger: Optional[WandbLogger] = None
        self.lpips_metric: Optional[LPIPS] = None
        self.data_consistency_loss = normalized_l1_loss()

    @staticmethod
    def _broadcast_like(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Broadcast tensor x to the shape of target if needed."""
        return x.expand_as(target) if list(x.shape) != list(target.shape) else x

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: Optional[str] = None):
        # Initialize the LPIPS metric on the correct device
        if self.lpips_metric is None:
            rank_zero_info("[TTT-LPIPS] Initializing LPIPS model for convergence metric.")
            self.lpips_metric = LPIPS().to(pl_module.device)
            self.lpips_metric.eval()

        if self.enable_wandb_logging:
            loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
            self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)

    def on_predict_batch_start(self, trainer: L.Trainer, pl_module: L.LightningModule, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        device = pl_module.device
        fname = getattr(batch, 'fname', ['unknown'])[0]
        slice_num = getattr(batch, 'slice_num', [0])[0]
        
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())
        adapt_model = copy.deepcopy(pl_module).to(device).train()

        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)
        
        for param in adapt_model.parameters():
            param.requires_grad_(True)

        opt = torch.optim.Adam(
            [p for p in adapt_model.parameters() if p.requires_grad],
            lr=self.lr, weight_decay=self.weight_decay
        )

        # Use the full acquired k-space and mask for training
        k_train = batch.masked_kspace
        mask_train_full = batch.mask
        
        # Extract center slice index (for multi-slice data)
        num_adj_slices = getattr(pl_module, 'num_adj_slices', 5)
        cidx = num_adj_slices // 2

        previous_img = None
        best_loss = float('inf')
        best_state = copy.deepcopy(adapt_model.state_dict()) # Start with baseline

        rank_zero_info(f"[TTT-LPIPS] Starting for {fname}/slice{slice_num} (batch {batch_idx})...")

        for it in range(self.inner_steps):
            opt.zero_grad()

            with torch.enable_grad():
                out = adapt_model(
                    k_train, mask_train_full,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )
                
                # Training loss is data consistency on ALL acquired points
                # Extract center slice from measurements (chunk multi-slice data)
                ks_chunks = torch.chunk(batch.masked_kspace, chunks=num_adj_slices, dim=1)
                ks_center = ks_chunks[cidx]
                mask_chunks = torch.chunk(batch.mask, chunks=num_adj_slices, dim=1)
                mask_center = mask_chunks[cidx]
                
                # Apply mask with broadcast
                mask_binary = (mask_center > 0).float()
                mask_bc = self._broadcast_like(mask_binary, out["pred_kspace"])
                pred_masked = out["pred_kspace"] * mask_bc
                target_masked = ks_center * mask_bc
                loss_tr = self.data_consistency_loss(pred_masked, target_masked)
                
                if torch.isnan(loss_tr):
                    rank_zero_info(f"WARNING: NaN loss at TTT step {it}. Stopping.")
                    break
                
                loss_tr.backward()
            opt.step()

            # --- LPIPS Convergence & Early Stopping Logic ---
            with torch.no_grad():
                current_img = out["img_pred"].detach()
                if current_img.dim() == 3:
                    current_img = current_img.unsqueeze(1) # Ensure B,C,H,W
                
                perceptual_change = torch.tensor(float('inf'), device=device)
                if previous_img is not None:
                    # LPIPS expects 3 channels, so repeat the grayscale channel
                    current_img_3ch = current_img.repeat(1, 3, 1, 1)
                    previous_img_3ch = previous_img.repeat(1, 3, 1, 1)
                    perceptual_change = self.lpips_metric(current_img_3ch, previous_img_3ch).mean()

                if (it + 1) % self.log_every_n_steps == 0:
                    rank_zero_info(f"  [Iter {it+1:02d}/{self.inner_steps}] DC Loss: {loss_tr.item():.6f}, LPIPS Change: {perceptual_change.item():.6f}")
                
                # Save best state based on lowest training loss
                if loss_tr.item() < best_loss:
                    best_loss = loss_tr.item()
                    best_state = copy.deepcopy(adapt_model.state_dict())

                # Check stopping criterion (after first iteration)
                if it > 0 and perceptual_change.item() < self.stopping_threshold:
                    rank_zero_info(f"  ✓ Stopping early at iter {it+1}. Perceptual change ({perceptual_change.item():.6f}) < threshold ({self.stopping_threshold}).")
                    break
                
                previous_img = current_img

        pl_module.load_state_dict(best_state)
        pl_module.eval()
        rank_zero_info(f"[TTT-LPIPS] Completed for {fname}/slice{slice_num}. Best DC Loss: {best_loss:.6f}")

    def on_predict_batch_end(self, trainer: L.Trainer, pl_module: L.LightningModule, outputs, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        if hasattr(pl_module, "_orig_state"):
            pl_module.load_state_dict(pl_module._orig_state)
            pl_module.eval()
            del pl_module._orig_state