# callbacks/ttt_callback_ssdu_half_kspace_freeze_v3.py
import copy
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# from your project
from mri_utils import normalized_l1_loss
torch.set_float32_matmul_precision('high')  # or 'medium'

class TestTimeTrainingCallback(L.Callback):
    """
    Test-Time Training with robust self-supervised early stopping (no ground truth).
    
    Strategy: Split k-space vertically (upper/lower half) for train/val.
    - Train on upper (or lower) half of center slice k-space
    - Validate on the complementary half
    - Validation metric: Huber(k-space residual) + lambda_tv * max(0, TV_growth - tv_tol)
    
    Robust features: Huber loss for outliers, TV penalty for noise overfitting.
    
    **V3: Encoder/Decoder level freezing**
    - freeze_encoder: Freeze all encoder layers (enc_level1, enc_level2, enc_level3)
    - freeze_decoder: Freeze all decoder layers (dec_level1, dec_level2, dec_level3)
    - freeze_bottleneck: Freeze bottleneck layers
    - Also supports V2 features: cascade freezing, sens_net freezing, prompt freezing
    
    Note: freeze_encoder and freeze_decoder can be combined, but usually you'd choose one.
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 7,
        log_every_n_steps: int = 1,

        # ---- k-space split control ----
        use_upper_half_for_train: bool = True,  # True: upper half for train, lower for val; False: vice versa

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
        
        # ---- V2: Bi-directional Freezing options (cascade-level) ----
        train_first_n_cascades: int = 0,       # Train ONLY first N cascades (0 = disabled)
        freeze_first_n_cascades: int = 0,      # Freeze first N cascades (0 = disabled)
        freeze_sens_net: bool = False,         # Freeze sensitivity estimation network
        freeze_prompts: bool = False,          # Freeze all prompt modules in cascades
        
        # ---- V3: Encoder/Decoder level freezing (NEW) ----
        freeze_encoder: bool = False,          # Freeze all encoder layers
        freeze_decoder: bool = False,          # Freeze all decoder layers
        freeze_bottleneck: bool = False,       # Freeze bottleneck layers
        
        # ---- logging control ----
        enable_wandb_logging: bool = True, # whether to enable W&B logging
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

        self.use_upper_half_for_train = bool(use_upper_half_for_train)

        self.min_delta = float(min_delta)
        self.stop_on_first_worse = bool(stop_on_first_worse)

        self.huber_delta = float(huber_delta)

        self.lambda_tv = float(lambda_tv)
        self.tv_tol_ratio = float(tv_tol_ratio)
        self.tv_eps = float(tv_eps)

        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)
        
        # V2: Bi-directional freezing options
        self.train_first_n_cascades = int(train_first_n_cascades)
        self.freeze_first_n_cascades = int(freeze_first_n_cascades)
        self.freeze_sens_net = bool(freeze_sens_net)
        self.freeze_prompts = bool(freeze_prompts)
        
        # V3: Encoder/Decoder freezing options
        self.freeze_encoder = bool(freeze_encoder)
        self.freeze_decoder = bool(freeze_decoder)
        self.freeze_bottleneck = bool(freeze_bottleneck)
        
        # Validate mutually exclusive options
        if self.train_first_n_cascades > 0 and self.freeze_first_n_cascades > 0:
            print("[TTT-V3] WARNING: Both train_first_n_cascades and freeze_first_n_cascades are set.")
            print("[TTT-V3]          train_first_n_cascades takes priority, ignoring freeze_first_n_cascades.")
            self.freeze_first_n_cascades = 0
        
        if self.freeze_encoder and self.freeze_decoder:
            print("[TTT-V3] WARNING: Both freeze_encoder and freeze_decoder are True.")
            print("[TTT-V3]          This will significantly limit trainable parameters.")
        
        self.enable_wandb_logging = bool(enable_wandb_logging)

        self.wandb_logger: Optional[WandbLogger] = None
        self.normalized_l1 = normalized_l1_loss()

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
    def _build_val_train_masks_half_kspace(
        self, center_mask: torch.Tensor, use_upper_for_train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split center-slice acquired mask vertically (upper/lower half).
        
        Args:
            center_mask: (B,1,H,W,[2?]) - center slice mask
            use_upper_for_train: if True, upper half for train, lower for val; else vice versa
            
        Returns:
            tr_mask, val_mask: two masks with same shape as center_mask
        """
        B, _, H, W = center_mask.shape[:4]
        val_mask = torch.zeros_like(center_mask)
        tr_mask  = torch.zeros_like(center_mask)

        # k-space center line position
        cy = H // 2
        
        # Define upper and lower half row ranges
        # Upper: [0, cy)
        # Lower: [cy, H)
        if use_upper_for_train:
            train_rows = list(range(0, cy))      # upper half
            val_rows = list(range(cy, H))        # lower half
        else:
            train_rows = list(range(cy, H))      # lower half
            val_rows = list(range(0, cy))        # upper half

        # Handle complex or real mask
        if self._is_complex_last(center_mask):
            # Complex format: (B,1,H,W,2)
            for b in range(B):
                # Get actually sampled rows in this batch (check if any column has nonzero)
                acq = (center_mask[b, 0, :, :, 0] > 0).any(dim=1)  # (H,)
                acq_rows = torch.where(acq)[0].tolist()
                
                # Only set masks on actually sampled rows
                train_acq = [r for r in train_rows if r in acq_rows]
                val_acq = [r for r in val_rows if r in acq_rows]
                
                if train_acq:
                    tr_mask[b, 0, train_acq, :, :] = center_mask[b, 0, train_acq, :, :]
                if val_acq:
                    val_mask[b, 0, val_acq, :, :] = center_mask[b, 0, val_acq, :, :]
        else:
            # Real format or other formats
            for b in range(B):
                if center_mask.dim() == 5:
                    # (B,1,H,W,1) format
                    acq = (center_mask[b, 0, :, :, 0] > 0).any(dim=1)
                    acq_rows = torch.where(acq)[0].tolist()
                    
                    train_acq = [r for r in train_rows if r in acq_rows]
                    val_acq = [r for r in val_rows if r in acq_rows]
                    
                    if train_acq:
                        tr_mask[b, 0, train_acq, :, 0] = center_mask[b, 0, train_acq, :, 0]
                    if val_acq:
                        val_mask[b, 0, val_acq, :, 0] = center_mask[b, 0, val_acq, :, 0]
                else:
                    # (B,1,H,W) format
                    acq = (center_mask[b, 0, :, :] > 0).any(dim=1)
                    acq_rows = torch.where(acq)[0].tolist()
                    
                    train_acq = [r for r in train_rows if r in acq_rows]
                    val_acq = [r for r in val_rows if r in acq_rows]
                    
                    if train_acq:
                        tr_mask[b, 0, train_acq, :] = center_mask[b, 0, train_acq, :]
                    if val_acq:
                        val_mask[b, 0, val_acq, :] = center_mask[b, 0, val_acq, :]
        
        return tr_mask, val_mask

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    def _apply_freezing_strategy(self, model: torch.nn.Module) -> Tuple[int, int]:
        """
        Apply freezing strategy to the model.
        
        V3: Supports both cascade-level and encoder/decoder-level freezing.
        
        Returns:
            (total_params, trainable_params): tuple of parameter counts
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0
        
        # Freeze sens_net if requested
        if self.freeze_sens_net and hasattr(model, 'promptmr') and hasattr(model.promptmr, 'sens_net'):
            for param in model.promptmr.sens_net.parameters():
                param.requires_grad_(False)
                frozen_params += param.numel()
        
        # V2: Handle bi-directional cascade freezing
        if hasattr(model, 'promptmr') and hasattr(model.promptmr, 'cascades'):
            num_cascades = len(model.promptmr.cascades)
            
            if self.train_first_n_cascades > 0:
                # Train ONLY first N cascades, freeze the rest
                train_up_to = min(self.train_first_n_cascades, num_cascades)
                
                # Freeze cascades after train_up_to
                for i in range(train_up_to, num_cascades):
                    for param in model.promptmr.cascades[i].parameters():
                        param.requires_grad_(False)
                        frozen_params += param.numel()
                        
            elif self.freeze_first_n_cascades > 0:
                # Freeze first N cascades, train the rest
                freeze_up_to = min(self.freeze_first_n_cascades, num_cascades)
                
                for i in range(freeze_up_to):
                    for param in model.promptmr.cascades[i].parameters():
                        param.requires_grad_(False)
                        frozen_params += param.numel()
            
            # V3: Encoder/Decoder level freezing
            # Apply to each cascade's model (if not already frozen by cascade-level freezing)
            for i, cascade in enumerate(model.promptmr.cascades):
                # Skip if this cascade is already frozen by cascade-level freezing
                cascade_frozen = False
                if self.train_first_n_cascades > 0 and i >= self.train_first_n_cascades:
                    cascade_frozen = True
                elif self.freeze_first_n_cascades > 0 and i < self.freeze_first_n_cascades:
                    cascade_frozen = True
                
                if not cascade_frozen and hasattr(cascade, 'model'):
                    # Freeze encoder layers
                    if self.freeze_encoder:
                        for enc_name in ['enc_level1', 'enc_level2', 'enc_level3']:
                            if hasattr(cascade.model, enc_name):
                                enc_module = getattr(cascade.model, enc_name)
                                for param in enc_module.parameters():
                                    if param.requires_grad:
                                        param.requires_grad_(False)
                                        frozen_params += param.numel()
                    
                    # Freeze decoder layers
                    if self.freeze_decoder:
                        for dec_name in ['dec_level1', 'dec_level2', 'dec_level3']:
                            if hasattr(cascade.model, dec_name):
                                dec_module = getattr(cascade.model, dec_name)
                                for param in dec_module.parameters():
                                    if param.requires_grad:
                                        param.requires_grad_(False)
                                        frozen_params += param.numel()
                    
                    # Freeze bottleneck
                    if self.freeze_bottleneck and hasattr(cascade.model, 'bottleneck'):
                        for param in cascade.model.bottleneck.parameters():
                            if param.requires_grad:
                                param.requires_grad_(False)
                                frozen_params += param.numel()
        
        # Freeze prompts if requested
        if self.freeze_prompts and hasattr(model, 'promptmr') and hasattr(model.promptmr, 'cascades'):
            for cascade in model.promptmr.cascades:
                if hasattr(cascade, 'model'):
                    # Freeze prompt_level1, prompt_level2, prompt_level3
                    for prompt_name in ['prompt_level1', 'prompt_level2', 'prompt_level3']:
                        if hasattr(cascade.model, prompt_name):
                            prompt_module = getattr(cascade.model, prompt_name)
                            for param in prompt_module.parameters():
                                if param.requires_grad:  # Only count if not already frozen
                                    param.requires_grad_(False)
                                    frozen_params += param.numel()
        
        # Count total and trainable parameters
        for param in model.parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        return total_params, trainable_params

    # --------------- Lightning hooks ---------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        if self.enable_wandb_logging:
            loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
            self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
            if self.wandb_logger is None:
                print("[TTT-V3] WARNING: WandbLogger not found but enable_wandb_logging=True. Proceeding without W&B logging.")
        else:
            print("[TTT-V3] W&B logging disabled by enable_wandb_logging=False")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = batch.masked_kspace.device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)
        fname = getattr(batch, 'fname', ['unknown'])[0] if hasattr(batch, 'fname') else 'unknown'
        slice_num = getattr(batch, 'slice_num', [0])[0] if hasattr(batch, 'slice_num') else 0
        
        # Save original state
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())

        # Create working copy
        adapt_model = copy.deepcopy(pl_module).to(device).train()
        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)
        
        # Apply freezing strategy
        total_params, trainable_params = self._apply_freezing_strategy(adapt_model)
        frozen_params = total_params - trainable_params
        
        # Determine freezing mode for display
        freeze_mode_parts = []
        
        if self.train_first_n_cascades > 0:
            freeze_mode_parts.append(f"train first {self.train_first_n_cascades} cascades")
        elif self.freeze_first_n_cascades > 0:
            freeze_mode_parts.append(f"freeze first {self.freeze_first_n_cascades} cascades")
        
        if self.freeze_encoder:
            freeze_mode_parts.append("freeze encoder")
        if self.freeze_decoder:
            freeze_mode_parts.append("freeze decoder")
        if self.freeze_bottleneck:
            freeze_mode_parts.append("freeze bottleneck")
        if self.freeze_sens_net:
            freeze_mode_parts.append("freeze sens_net")
        if self.freeze_prompts:
            freeze_mode_parts.append("freeze prompts")
        
        freeze_mode = ", ".join(freeze_mode_parts) if freeze_mode_parts else "all parameters trainable"
        
        # Print freezing summary
        print(f"[{device_name}] 🔒 Freezing Strategy Applied (V3):")
        print(f"  - Strategy: {freeze_mode}")
        print(f"  - Total params: {total_params:,} | Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%) | Frozen: {frozen_params:,} ({100*frozen_params/total_params:.1f}%)")

        # Create optimizer only for trainable parameters
        trainable_param_list = [p for p in adapt_model.parameters() if p.requires_grad]
        if len(trainable_param_list) == 0:
            print(f"[{device_name}] WARNING: No trainable parameters! Skipping TTT.")
            return
        
        opt = torch.optim.Adam(
            trainable_param_list,
            lr=self.lr, weight_decay=self.weight_decay
        )

        # Build half k-space train/val masks on center slice
        cidx = self._center_slice_index(adapt_model)
        center_mask = self._center_mask(batch.mask, cidx)  # (B,1,H,W,[2?])

        train_center, val_center = self._build_val_train_masks_half_kspace(
            center_mask, self.use_upper_half_for_train
        )

        # Expand to full multi-slice masks by replacing only center slice
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

        # Masked k-space for training: zero out validation region on center slice
        if self._is_complex_last(batch.masked_kspace):
            keep = (train_center > 0).float()
            keep = self._broadcast_like(keep, batch.masked_kspace)
        else:
            keep = (train_center > 0).float()
            if keep.dim() == 5: keep = keep.squeeze(-1)
            keep = self._broadcast_like(keep, batch.masked_kspace)
        k_train = batch.masked_kspace * keep

        # Baseline reconstruction & metrics
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

            # Validation residual on validation region (align to center-slice multi-coil shapes)
            if self._is_complex_last(out0["pred_kspace"]):
                mv = self._broadcast_like(val_center, out0["pred_kspace"]).float()
                # Center-slice of measured k-space to match pred_kspace
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

            val_metric_best = val_dc0  # TV-penalty is zero at baseline by definition
            best_state = copy.deepcopy(adapt_model.state_dict())

            # Check for NaN and skip TTT if baseline is invalid
            if torch.isnan(torch.tensor([val_dc0, tv0])).any():
                print(f"[{device_name}] WARNING: TTT baseline contains NaN for {fname}/slice{slice_num}, skipping TTT")
                return
            
            split_mode = "upper-train/lower-val" if self.use_upper_half_for_train else "lower-train/upper-val"
            print(f"[{device_name}] TTT starting for {fname}/slice{slice_num} (batch {batch_idx}) - mode: {split_mode}")
            print(f"[{device_name}] TTT baseline: val_dc={val_dc0:.6f}, tv={tv0:.6f}, val_metric={val_metric_best:.6f}")
            
            if trainer.is_global_zero and self.enable_wandb_logging and self.wandb_logger is not None:
                self.wandb_logger.experiment.log({
                    "TTT/val_dc_huber/baseline": val_dc0,
                    "TTT/tv_center/baseline": tv0,
                    "TTT/val_metric/best": val_metric_best,
                    "TTT/inner_iter": -1,
                    "TTT/freeze/total_params": total_params,
                    "TTT/freeze/trainable_params": trainable_params,
                    "TTT/freeze/frozen_params": frozen_params,
                    "TTT/freeze/trainable_ratio": trainable_params / total_params,
                })

        # Inner training loop
        for it in range(self.inner_steps):
            opt.zero_grad()

            # Enable gradients for training
            with torch.enable_grad():
                out = adapt_model(
                    k_train, mask_train_full,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )

                # Training DC loss on training region (standard normalized L1, align to center-slice)
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
                target_masked = yt.detach()  # Ensure target doesn't need gradients
                
                loss_tr = self.normalized_l1(pred_masked, target_masked)
                
                # Check for NaN in training loss
                if torch.isnan(loss_tr):
                    print(f"[{device_name}] WARNING: TTT iter {it} got NaN loss, skipping this iteration")
                    continue
                    
                loss_tr.backward()
            opt.step()
            
            # Print training info every step
            print(f"[{device_name}] TTT iter {it+1}/{self.inner_steps}: train_loss={loss_tr.item():.6f}")

            # Validation metric on validation region (no grad)
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
                
                # Print validation info
                print(f"[{device_name}]     val_loss={val_dc:.4f}, tv={tv_cur:.1f} (↑{tv_penalty:.1f}), "
                      f"metric={val_metric:.4f} (best={val_metric_best:.4f})")

            # Logging
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

            # Early stopping
            if val_metric < (val_metric_best - self.min_delta):
                val_metric_best = val_metric
                best_state = copy.deepcopy(adapt_model.state_dict())
                print(f"[{device_name}]     ✓ New best: {val_metric_best:.4f}")
            elif self.stop_on_first_worse:
                print(f"[{device_name}]     ✗ Early stop at iter {it+1} (worse: {val_metric:.4f} > {val_metric_best:.4f})")
                break

        # Restore best state and copy back
        adapt_model.load_state_dict(best_state)
        pl_module.load_state_dict(adapt_model.state_dict())
        pl_module.eval()
        
        print(f"[{device_name}] TTT completed for {fname}/slice{slice_num}: final best={val_metric_best:.4f}")
        
        # Per-subject logging
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


