import copy, os, time
import numpy as np
import torch
import lightning as L
import torchvision
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import wandb
from mri_utils import SSIMLoss,normalized_l1_loss
from lightning.pytorch.loggers import WandbLogger
# at the top of your callback class, ensure init once:
#if not wandb.run:
 #   wandb.init(project="cmr-ttt", name="ttt-test")


#does it make sense to loop over batches?
class TestTimeTrainingCallback(L.Callback):
    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 20,
        # holdout_frac: float = 0.05,
        # patience_window: int = 20,
        log_every_n_steps: int = 50,
        # save_dir: str = "./ttt_saved_tensors",
    ):
        super().__init__()
        self.lr = lr
        self.inner_steps = inner_steps
        #self.holdout_frac = holdout_frac
        #self.patience_window = patience_window
        self.log_every_n_steps = log_every_n_steps
        # self.save_dir = save_dir
        # os.makedirs(self.save_dir, exist_ok=True)
        self.before_recons = {}
        self._ttt_step = 0
        self.normalized_l1=normalized_l1_loss()
    
    def setup(self, trainer, pl_module, stage: str | None = None):
        # grab the WandbLogger Lightning created from trainer.logger
        loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
        for lg in loggers:
            if isinstance(lg, WandbLogger):
                self.wandb_logger = lg
                break
        else:
            raise RuntimeError("TestTimeTrainingCallback requires a WandbLogger under trainer.logger")



    # #@staticmethod
    # def _choose_holdout(mask2d: np.ndarray, frac: float):
 
    #     # Find which kx lines (rows) have acquired data
    #     acquired_lines_indices = np.unique(np.where(mask2d == 1)[0])
    
    #     # Determine how many lines to hold out for validation
    #     num_val_lines = max(1, int(len(acquired_lines_indices) * frac))
    
    #     # Randomly choose which lines to hold out
    #     val_lines = np.random.choice(acquired_lines_indices, num_val_lines, replace=False)
    
    #     # Get all coordinates that fall on these validation lines
    #     val_coords = np.where(np.isin(np.arange(mask2d.shape[0])[:, None], val_lines), mask2d, 0)
    
    #     return np.array(np.where(val_coords == 1))


    # def _log_scalar(self, trainer, name, value, step):
    #     if trainer.is_global_zero and trainer.logger:
    #         loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
    #         for logger in loggers:
    #             if hasattr(logger.experiment, 'log'):
    #                 logger.experiment.log({name: value}, step=step)
    #             else:
    #                 logger.experiment.add_scalar(name, value, step)

    # def _log_image(self, trainer, name, image_tensor, step):
    #     if trainer.is_global_zero and trainer.logger:
    #         loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
    #         for logger in loggers:
    #             if hasattr(logger.experiment, 'log'):
    #                 logger.experiment.log(
    #                     {name: [logger.experiment.Image(image_tensor.cpu(), caption=name)]}, step=step
    #                 )
    #             else:
    #                 logger.experiment.add_image(name, image_tensor, step)

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        device = batch.masked_kspace.device
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())

        # with torch.no_grad():
        #     out_before = pl_module(
        #         batch.masked_kspace, batch.mask,
        #         batch.num_low_frequencies, batch.mask_type,
        #         compute_sens_per_coil=pl_module.compute_sens_per_coil,
        #     ) 
        #     recon_before = out_before["img_pred"]
           # print(batch)
          #  gt_img = getattr(batch, "target", None)
           # self.before_recons[batch_idx] = recon_before
           # print('check recon before:',recon_before.shape)
           # print('check gt:',gt_img.shape)

        adapt_model = copy.deepcopy(pl_module).train().to(device)
        opt = torch.optim.Adam(adapt_model.parameters(), lr=self.lr)
       # center_slice=5//2#need to change
        center_slice = adapt_model.num_adj_slices//2
        current_mask= torch.chunk(batch.mask, adapt_model.num_adj_slices, dim=1)[center_slice]  # mask for the center slice

       # mask=torch.chunk(batch.mask,5,dim=1)[center_slice] # [1,1,448,204,2]
        #masked_kspace=batch.masked_kspace #[1,50,448,204,2]
        #print('original masked kspace:', masked_kspace.shape)

       # _,_,nx,ny,_ = masked_kspace.shape
        #if current_mask.shape[2]==1: 
         #   current_mask = current_mask.repeat(1, 1, nx, 1, 1) # add nx dimension

           # print('mask shape:',mask.shape)
       # mask2d = mask[0,0,:,:,0].cpu().numpy()
        #print('mask2d:',mask2d.shape) #kx,ky directions
        #ids_val = self._choose_holdout(mask2d, self.holdout_frac)
        #verrors = []

        tic = time.time()
        with torch.enable_grad():
            for it in range(self.inner_steps):
                opt.zero_grad()
                out = adapt_model(
                    batch.masked_kspace, batch.mask,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )

                k_pred = out["pred_kspace"] * current_mask.float() #what is the size of batch.mask here? 
                #print('k pred:',k_pred.shape)

                k_true = out["original_kspace"]* current_mask.float() #what is the size of batch.mask here?
              #  k_pred = k_pred.clone()
               # k_true = k_true.clone()
              #  k_pred_train[:, :, ids_val[0], ids_val[1], :] = 0 #mask out the nx,ny dimensions
               # k_true_train[:, :, ids_val[0], ids_val[1], :] = 0
                loss = self.normalized_l1(k_pred, k_true)
                print('step:',it)
                print('train loss:',loss)
                loss.backward()
                opt.step()

                #if it % self.log_every_n_steps == 0 or it == self.inner_steps - 1:
                  #  print(f"[TTT] batch {batch_idx}, iter {it+1}/{self.inner_steps}, loss={loss.item():.6f}")
                    #self._log_scalar(trainer, "TTT/loss_train", loss.item(), self._ttt_step)
                  #  wandb.log({"TTT/loss_train": loss.item(), "ttt_step": self._ttt_step})
                   # self._ttt_step += 1

                #with torch.no_grad():
                 #   verror = torch.nn.functional.l1_loss(
                  #      k_pred[:,:,ids_val[0], ids_val[1], :],
                   #     k_true[:,:,ids_val[0], ids_val[1], :]
                    #).item()
                    #verrors.append(verror)
                    #if it > 3*self.patience_window and np.mean(verrors[-self.patience_window:]) > np.mean(verrors[-2*self.patience_window:-self.patience_window]):
                       # break
                 # --- compute validation loss, SSIM, and log all three to W&B via pl_module.log() ---
                # with torch.no_grad():
                #      # 1) validation L1 on held‐out k‐lines
                #     verror = self.normalized_l1(
                #         k_pred[:,:,ids_val[0], ids_val[1], :],
                #         k_true[:,:,ids_val[0], ids_val[1], :]
                #     )
                #     verrors.append(verror)

                #     # 2) SSIM between predicted image and GT
                #     recon_img = out["img_pred"]           # (B, C, H, W)
                #     gt_img    = batch.target.to(device)   # same shape
                #    # print('shape of recon image:', recon_img.shape)
                #    # print('GT image:', gt_img.shape)
                #     recon_img = recon_img.unsqueeze(1)
                #     gt_img = gt_img.unsqueeze(1)
                #     ssim_val  = ssim_fn(recon_img, gt_img).item()


                     # 3) log scalars under a per‐subject namespace
                key = f"TTT/sub{batch_idx:02d}"
                if trainer.is_global_zero:
                    self.wandb_logger.experiment.log(
                        {f"{key}/train_loss": loss.item()})
                       # f"{key}/val_loss":   verror,
                       # f"{key}/ssim":       ssim_val,})
            

                  #  loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]

                     # 4) early‐stop check
                #if it > 3*self.patience_window and \
                 #   np.mean(verrors[-self.patience_window:]) > \
                  #  np.mean(verrors[-2*self.patience_window:-self.patience_window]):
                   # break

        pl_module.load_state_dict(adapt_model.state_dict())
        pl_module.eval()
       # print(f"[TTT] batch {batch_idx}: {it+1} steps, {(time.time()-tic)/60:.1f} min")

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, prediction, batch, dataloader_idx=0, **kwargs):
       # print('prediction:',prediction)
        batch_idx = prediction.get("batch_idx")
        fname = prediction.get("fname")
        slice_num = prediction.get("slice_num")
        #recon_before = self.before_recons.pop(batch_idx)
        #recon_after  = prediction["output"]
        #target=batch.target
       # batch        = kwargs.get("batch")
      #  gt_img       = getattr(batch, "target", None)

        #print('recon before:',recon_before.shape)
        #print('recon_after:',recon_after.shape)
       # print('target:',target.shape)
      #  print('GT img:',gt_img.shape)

        # Save tensors to disk
        # save_dict = {
        #     "before": recon_before.cpu(),
        #     "after": recon_after.cpu(),
        #     "target:":  target.cpu()           #"gt": gt_img.cpu() if gt_img is not None else None,
        # }
        # torch.save(save_dict, os.path.join(self.save_dir, f"batch{batch_idx:04d}_tensors.pt"))
        # print(f"Saved tensors for batch {batch_idx}")

        # Logging images to WandB
        #grid = torchvision.utils.make_grid([recon_before[0].cpu(), recon_after[0].cpu()], 
         #   normalize=True, scale_each=True
        #)
       # self._log_image(trainer, "TTT/recon_comparison", grid, trainer.global_step)

        # Restore original pl_module state
        pl_module.load_state_dict(pl_module._orig_state)
        pl_module.eval()
        del pl_module._orig_state
