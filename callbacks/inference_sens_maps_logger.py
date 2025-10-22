"""
Callback to log sensitivity maps to wandb during inference stage (predict mode).
"""
import torch
from typing import Optional, Dict, Any, List
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from lightning.pytorch.core import LightningModule


class InferenceSensMapsLogger(Callback):
    """
    Callback to log sensitivity maps to wandb during inference stage.
    
    This callback logs sensitivity maps along with reconstructions during the predict phase,
    allowing visualization of coil sensitivity patterns in wandb.
    """
    
    def __init__(
        self,
        log_every_n_batches: int = 10,
        log_indices: Optional[List[int]] = None,
        enable_wandb_logging: bool = True,
    ):
        """
        Args:
            log_every_n_batches: Log every N batches (default: 10)
            log_indices: Specific batch indices to log (if None, uses log_every_n_batches)
            enable_wandb_logging: Whether to enable wandb logging
        """
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        self.log_indices = log_indices or []
        self.enable_wandb_logging = enable_wandb_logging
        self.wandb_logger: Optional[WandbLogger] = None
        self.batch_count = 0
        print(f"[InferenceSensMapsLogger] Initialized with log_every_n_batches={log_every_n_batches}, log_indices={self.log_indices}, enable_wandb_logging={enable_wandb_logging}")
        
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """Setup the callback and find wandb logger."""
        print(f"[InferenceSensMapsLogger] Setup called with stage='{stage}', enable_wandb_logging={self.enable_wandb_logging}")
        if stage == "predict" and self.enable_wandb_logging:
            loggers = trainer.loggers
            print(f"[InferenceSensMapsLogger] Found {len(loggers)} loggers: {[type(lg).__name__ for lg in loggers]}")
            self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
            if self.wandb_logger is None:
                print("[InferenceSensMapsLogger] WARNING: WandbLogger not found but enable_wandb_logging=True. Proceeding without W&B logging.")
            else:
                print("[InferenceSensMapsLogger] WandbLogger found. Sensitivity maps will be logged during inference.")
        elif stage == "predict":
            print("[InferenceSensMapsLogger] W&B logging disabled by enable_wandb_logging=False")
        else:
            print(f"[InferenceSensMapsLogger] Skipping setup for stage '{stage}' (only handles 'predict' stage)")
    
    def on_predict_batch_end(
        self, 
        trainer: Trainer, 
        pl_module: LightningModule, 
        outputs: Dict[str, Any], 
        batch: Any, 
        batch_idx: int, 
        dataloader_idx: int = 0
    ) -> None:
        """Log sensitivity maps after each prediction batch."""
        if not self.enable_wandb_logging or self.wandb_logger is None:
            return
            
        # Only log on global zero process in distributed training
        if not trainer.is_global_zero:
            return
            
        self.batch_count += 1
        
        # Determine if we should log this batch
        should_log = False
        if self.log_indices:
            should_log = batch_idx in self.log_indices
        else:
            should_log = (self.batch_count % self.log_every_n_batches == 0)
            
        if not should_log:
            return
            
        # Extract data from outputs
        if 'sens_maps' not in outputs:
            print(f"[InferenceSensMapsLogger] WARNING: No 'sens_maps' found in outputs for batch {batch_idx}")
            return
            
        try:
            # Handle batch outputs - outputs might be a list if batch_size > 1
            if isinstance(outputs['sens_maps'], list):
                # Multiple items in batch
                for i, sens_map in enumerate(outputs['sens_maps']):
                    self._log_single_sens_map(
                        batch_idx, i, sens_map, outputs, batch
                    )
            else:
                # Single item in batch
                self._log_single_sens_map(
                    batch_idx, 0, outputs['sens_maps'], outputs, batch
                )
                
        except Exception as e:
            print(f"[InferenceSensMapsLogger] Error logging batch {batch_idx}: {e}")
    
    def _log_single_sens_map(
        self, 
        batch_idx: int, 
        item_idx: int, 
        sens_map: torch.Tensor, 
        outputs: Dict[str, Any], 
        batch: Any
    ) -> None:
        """Log a single sensitivity map following the same pattern as validation stage."""
        try:
            # Get all required images
            sens_map_norm = sens_map / sens_map.max() if sens_map.max() > 0 else sens_map
            
            # Get reconstruction (output)
            if 'output' in outputs:
                if isinstance(outputs['output'], list):
                    reconstruction = outputs['output'][item_idx]
                else:
                    reconstruction = outputs['output']
                reconstruction_norm = reconstruction / reconstruction.max() if reconstruction.max() > 0 else reconstruction
            else:
                reconstruction_norm = None
                
            # Get zero-filled image (img_zf)
            if 'img_zf' in outputs:
                if isinstance(outputs['img_zf'], list):
                    img_zf = outputs['img_zf'][item_idx]
                else:
                    img_zf = outputs['img_zf']
                img_zf_norm = img_zf / img_zf.max() if img_zf.max() > 0 else img_zf
            else:
                img_zf_norm = None
                
            # Get filename for caption
            fname = "unknown"
            if 'fname' in outputs:
                if isinstance(outputs['fname'], list):
                    fname = outputs['fname'][item_idx]
                else:
                    fname = outputs['fname']
                # Extract just the filename without path
                if isinstance(fname, tuple):
                    fname = fname[0]
                fname = fname.split('/')[-1] if '/' in str(fname) else str(fname)
            
            # Get slice number for caption
            slice_num = "unknown"
            if 'slice_num' in outputs:
                if isinstance(outputs['slice_num'], list):
                    slice_num = outputs['slice_num'][item_idx].item()
                else:
                    slice_num = outputs['slice_num'].item()
            
            # Prepare images for logging - following validation pattern exactly
            # Validation logs: [mask, sens_maps, img_zf**alpha, output**alpha, target**alpha, error]
            # For inference, we don't have target or error, so we'll log: [sens_maps, img_zf**alpha, output**alpha]
            images_to_log = []
            captions = []
            
            # Add sensitivity maps
            images_to_log.append(sens_map_norm)
            captions.append(f"Sensitivity Map - {fname} - Slice {slice_num}")
            
            # Add zero-filled image with alpha adjustment (same as validation)
            if img_zf_norm is not None:
                alpha = 0.2  # Same alpha as validation stage
                images_to_log.append(img_zf_norm ** alpha)
                captions.append(f"Zero-filled (α={alpha}) - {fname} - Slice {slice_num}")
            
            # Add reconstruction with alpha adjustment (same as validation)
            if reconstruction_norm is not None:
                alpha = 0.2  # Same alpha as validation stage
                images_to_log.append(reconstruction_norm ** alpha)
                captions.append(f"Reconstruction (α={alpha}) - {fname} - Slice {slice_num}")
            
            # Log to wandb using the same pattern as validation
            key = f"inference/batch_{batch_idx:03d}_item_{item_idx}"
            self.wandb_logger.experiment.log({
                key: [self.wandb_logger.experiment.Image(img, caption=caption) 
                      for img, caption in zip(images_to_log, captions)]
            })
            
            print(f"[InferenceSensMapsLogger] Logged images for batch {batch_idx}, item {item_idx}: {fname} (sens_maps, zf, reconstruction)")
            
        except Exception as e:
            print(f"[InferenceSensMapsLogger] Error logging single sens map for batch {batch_idx}, item {item_idx}: {e}")
