import torch
from data import transforms
from pl_modules import MriModule
from typing import List
import copy 
from mri_utils import SSIMLoss, normalized_l1_loss
import torch.nn.functional as F
import importlib
from typing import Optional

def get_model_class(module_name, class_name="PromptMR"):
    """
    Dynamically imports the specified module and retrieves the class.

    Args:
        module_name (str): The module to import (e.g., 'model.m1', 'model.m2').
        class_name (str): The class to retrieve from the module (default: 'PromptMR').

    Returns:
        type: The imported class.
    """
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    return model_class

class ParallelPromptMrModule(MriModule):
    """
    并行Cascade架构的PromptMR模块
    
    架构设计:
    Branch A: Cascade1_A → Cascade2_A → Cascade3_A → ... → CascadeN_A
    Branch B: Cascade1_B → Cascade2_B → Cascade3_B → ... → CascadeN_B
    
    Tensor传递路径:
    Cascade1_A → Cascade1_B → Cascade2_A → Cascade2_B → Cascade3_A → Cascade3_B → ...
    """

    def __init__(
        self,
        num_cascades: int = 12,
        num_adj_slices: int = 5,
        n_feat0: int = 48,
        feature_dim: List[int] = [72,96,120],
        prompt_dim: List[int] = [24,48,72],
        sens_n_feat0: int = 24,
        sens_feature_dim: List[int] = [36,48,60],
        sens_prompt_dim: List[int] = [12,24,36],
        len_prompt: List[int] = [5,5,5],
        prompt_size: List[int] = [64,32,16],
        n_enc_cab: List[int] = [2,3,3],
        n_dec_cab: List[int] = [2,2,3],
        n_skip_cab: List[int] = [1,1,1],
        n_bottleneck_cab: int = 3,
        no_use_ca: bool = False,
        learnable_prompt: bool = False,
        adaptive_input: bool = True,
        n_buffer: int = 4,
        n_history: int = 0,  # 与promptmr_module一致
        use_sens_adj: bool = True,
        model_version: str = "promptmr_v2",
        lr: float = 0.0002,
        lr_step_size: int = 11,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.01,
        use_checkpoint: bool = False,
        compute_sens_per_coil: bool = False,
        pretrain: bool = False,
        pretrain_weights_path: str = None,
        kspace_loss_weight: float = 0.01,
        parallel_mode: bool = True,  # 强制使用并行架构
        # parallel-only knobs
        chain_mode: str = "AB",   # "A_only" or "AB"
        b_apply_dc: bool = True,
        b_model_scale: float = 1.0,
        b_use_buffer: bool = True,
        b_use_history: bool = True,
        ab_indices: Optional[list] = None,
        inference_log_frequency: int = 50,
        **kwargs,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational network.
            num_adj_slices: Number of adjacent slices.
            n_feat0: Number of top-level feature channels for PromptUnet.
            feature_dim: feature dim for each level in PromptUnet.
            prompt_dim: prompt dim for each level in PromptUnet.
            sens_n_feat0: Number of top-level feature channels for sense map
                estimation PromptUnet in PromptMR.
            sens_feature_dim: feature dim for each level in PromptUnet for
                sensitivity map estimation (SME) network.
            sens_prompt_dim: prompt dim for each level in PromptUnet in
                sensitivity map estimation (SME) network.
            len_prompt: number of prompt component in each level.
            prompt_size: prompt spatial size.
            n_enc_cab: number of CABs (channel attention Blocks) in DownBlock.
            n_dec_cab: number of CABs (channel attention Blocks) in UpBlock.
            n_skip_cab: number of CABs (channel attention Blocks) in SkipBlock.
            n_bottleneck_cab: number of CABs (channel attention Blocks) in BottleneckBlock.
            no_use_ca: not using channel attention.
            learnable_prompt: whether to set the prompt as learnable parameters.
            adaptive_input: whether to use adaptive input.
            n_buffer: number of buffer in adaptive input.
            n_history: number of historical feature aggregation, should be less than num_cascades.
            use_sens_adj: whether to use adjacent sensitivity map estimation.
            model_version: model version. Default is "promptmr_v2".
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            use_checkpoint: Whether to use checkpointing to trade compute for GPU memory.
            compute_sens_per_coil: (bool) whether to compute sensitivity maps per coil for memory saving
            pretrain: whether to load pretrain weights
            pretrain_weights_path: path to pretrain weights
            kspace_loss_weight: weight for k-space loss in combined loss function
            parallel_mode: whether to use parallel cascade architecture (always True for this module)
        """
        super().__init__(**kwargs)
        # just to have it for logging / debugging
        try:
            self.save_hyperparameters()
        except KeyError as e:
            print(f"⚠️  save_hparams failed: {e}")

        self.num_cascades = num_cascades
        self.num_adj_slices = num_adj_slices

        self.n_feat0 = n_feat0
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim

        self.sens_n_feat0 = sens_n_feat0
        self.sens_feature_dim = sens_feature_dim
        self.sens_prompt_dim = sens_prompt_dim

        self.len_prompt = len_prompt
        self.prompt_size = prompt_size
        self.n_enc_cab = n_enc_cab
        self.n_dec_cab = n_dec_cab
        self.n_skip_cab = n_skip_cab
        self.n_bottleneck_cab = n_bottleneck_cab

        self.no_use_ca = no_use_ca

        self.learnable_prompt = learnable_prompt
        self.adaptive_input = adaptive_input
        self.n_buffer = n_buffer
        self.n_history = n_history
        self.use_sens_adj = use_sens_adj
        # two flags for reducing memory usage
        self.use_checkpoint = use_checkpoint
        self.compute_sens_per_coil = compute_sens_per_coil
        
        self.pretrain = pretrain
        self.pretrain_weights_path = pretrain_weights_path
        self.kspace_loss_weight = kspace_loss_weight
        self.inference_log_frequency = inference_log_frequency
        self.parallel_mode = True  # 强制使用并行架构
        # save parallel knobs
        self.chain_mode = chain_mode
        self.b_apply_dc = b_apply_dc
        self.b_model_scale = b_model_scale
        self.b_use_buffer = b_use_buffer
        self.b_use_history = b_use_history
        self.ab_indices = ab_indices if ab_indices is not None else []
        
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.model_version = model_version
        
        # 强制使用并行架构，失败则终止
        try:
            from models.promptmr_parallel import ParallelPromptMR
            self.promptmr = ParallelPromptMR(
                parallel_mode=True,
                num_cascades=self.num_cascades,
                num_adj_slices=self.num_adj_slices,
                n_feat0=self.n_feat0,
                feature_dim=self.feature_dim,
                prompt_dim=self.prompt_dim,
                sens_n_feat0=self.sens_n_feat0,
                sens_feature_dim=self.sens_feature_dim,
                sens_prompt_dim=self.sens_prompt_dim,
                len_prompt=self.len_prompt,
                prompt_size=self.prompt_size,
                n_enc_cab=self.n_enc_cab,
                n_dec_cab=self.n_dec_cab,
                n_skip_cab=self.n_skip_cab,
                n_bottleneck_cab=self.n_bottleneck_cab,
                no_use_ca=self.no_use_ca,
                learnable_prompt=learnable_prompt,
                n_history=self.n_history,
                n_buffer=self.n_buffer,
                adaptive_input=self.adaptive_input,
                use_sens_adj=self.use_sens_adj,
                chain_mode=self.chain_mode,
                b_apply_dc=self.b_apply_dc,
                b_model_scale=self.b_model_scale,
                b_use_buffer=self.b_use_buffer,
                b_use_history=self.b_use_history,
                ab_indices=self.ab_indices,
            )
            print("✅ 并行PromptMR架构创建成功")
        except Exception as e:
            print(f"❌ 并行PromptMR架构创建失败: {e}")
            raise RuntimeError(f"ParallelPromptMrModule必须使用并行架构，创建失败: {e}")
        
        self._load_pretrain_weights()
        self.loss = SSIMLoss()
        self.normalized_l1 = normalized_l1_loss()
    
    def load_state_dict(self, state_dict, strict=True):
        # always do a non-strict load, so any extra keys are simply skipped
        # let PyTorch ignore any keys that don't match (e.g. domain_adapt.*)
        return super().load_state_dict(state_dict, strict=False)
   
    def _load_pretrain_weights(self):
        # load pretrain weights
        if not self.pretrain:
            print('Train from scratch, no pretrain weights loaded')
            return
        
        print(f"loading pretrain weights from {self.pretrain_weights_path}")
        checkpoint = torch.load(self.pretrain_weights_path)
        upd_state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            if 'loss.w' in k:  # Skip this key
                continue
            new_key = k.replace('promptmr.', '')  # or just remove 'model.'
            upd_state_dict[new_key] = v
        
        if hasattr(self.promptmr, 'copy_weights_from_original'):
            # 对于并行架构，需要特殊处理权重复制
            print("Detected parallel architecture, copying weights to both branches...")
            # 先创建一个临时的原始模型来加载权重
            from models.promptmr_v2 import PromptMR
            temp_model = PromptMR(
                num_cascades=self.num_cascades,
                num_adj_slices=self.num_adj_slices,
                n_feat0=self.n_feat0,
                feature_dim=self.feature_dim,
                prompt_dim=self.prompt_dim,
                sens_n_feat0=self.sens_n_feat0,
                sens_feature_dim=self.sens_feature_dim,
                sens_prompt_dim=self.sens_prompt_dim,
                len_prompt=self.len_prompt,
                prompt_size=self.prompt_size,
                n_enc_cab=self.n_enc_cab,
                n_dec_cab=self.n_dec_cab,
                n_skip_cab=self.n_skip_cab,
                n_bottleneck_cab=self.n_bottleneck_cab,
                no_use_ca=self.no_use_ca,
                learnable_prompt=self.learnable_prompt,
                n_history=self.n_history,
                n_buffer=self.n_buffer,
                adaptive_input=self.adaptive_input,
                use_sens_adj=self.use_sens_adj,
            )
            temp_model.load_state_dict(upd_state_dict)
            # 复制权重到并行架构
            self.promptmr.copy_weights_from_original(temp_model)
        else:
            self.promptmr.load_state_dict(upd_state_dict)

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # step lr scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )
        return [optim], [scheduler]
    
    def _override_lr_and_log(self):
        """Override optimizer LR with self.lr and log scheduler state; tolerate env differences."""
        # Ensure trainer/optimizer exist
        if not hasattr(self, "trainer") or self.trainer is None:
            return False
        if not hasattr(self.trainer, "optimizers") or not self.trainer.optimizers:
            return False

        optimizer = self.trainer.optimizers[0]

        # 1) Override optimizer param group learning rates
        if hasattr(optimizer, "param_groups"):
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr

        # 2) Sync scheduler base_lrs so future steps use the new LR as base
        schedulers = []
        if hasattr(self.trainer, "lr_scheduler_configs") and self.trainer.lr_scheduler_configs:
            for cfg in self.trainer.lr_scheduler_configs:
                scheduler = getattr(cfg, "scheduler", None)
                if scheduler is not None:
                    schedulers.append(scheduler)
        elif hasattr(self.trainer, "lr_schedulers") and self.trainer.lr_schedulers:
            for s in self.trainer.lr_schedulers:
                scheduler = s.get("scheduler") if isinstance(s, dict) else getattr(s, "scheduler", None)
                if scheduler is not None:
                    schedulers.append(scheduler)

        # Prepare debug info
        group_lrs = [pg.get("lr", None) for pg in getattr(optimizer, "param_groups", [])]
        sched_infos = []

        for scheduler in schedulers:
            # 覆盖调度策略参数（若支持）：step_size 与 gamma
            try:
                if hasattr(scheduler, "step_size") and hasattr(self, "lr_step_size"):
                    scheduler.step_size = self.lr_step_size
                if hasattr(scheduler, "gamma") and hasattr(self, "lr_gamma"):
                    scheduler.gamma = self.lr_gamma
            except Exception:
                pass

            if hasattr(scheduler, "base_lrs") and getattr(optimizer, "param_groups", None):
                scheduler.base_lrs = [self.lr for _ in optimizer.param_groups]
            # Collect scheduler state for logging
            info = {
                "type": type(scheduler).__name__,
                "last_epoch": getattr(scheduler, "last_epoch", None),
                "step_size": getattr(scheduler, "step_size", None),
                "gamma": getattr(scheduler, "gamma", None),
                "get_last_lr": None,
            }
            try:
                if hasattr(scheduler, "get_last_lr"):
                    info["get_last_lr"] = scheduler.get_last_lr()
            except Exception:
                pass
            sched_infos.append(info)

        try:
            self.print(
                f"[ParallelPromptMrModule] Start training with LR={self.lr}. Param group LRs={group_lrs}. "
                f"Schedulers={sched_infos} (states preserved from ckpt)."
            )
        except Exception:
            pass

        return True

    def on_fit_start(self):
        """
        After resuming from a checkpoint, Lightning restores optimizer and LR scheduler states.
        Here we override ONLY the learning rate to `self.lr` while preserving other states.
        """
        try:
            self._override_lr_and_log()
        except Exception as e:
            try:
                self.print(f"[ParallelPromptMrModule][WARN] on_fit_start override failed: {e}")
            except Exception:
                pass

    def on_train_start(self):
        """Fallback point: ensure LR override/logging happens after optimizers are fully set up."""
        try:
            self._override_lr_and_log()
        except Exception as e:
            try:
                self.print(f"[ParallelPromptMrModule][WARN] on_train_start override failed: {e}")
            except Exception:
                pass
    
    def forward(self, masked_kspace, mask, num_low_frequencies, mask_type="cartesian", use_checkpoint=False, compute_sens_per_coil=False):
        return self.promptmr(masked_kspace, mask, num_low_frequencies, mask_type, use_checkpoint=use_checkpoint, compute_sens_per_coil=compute_sens_per_coil)   

    def training_step(self, batch, batch_idx):
        mask = batch.mask
        kspace_data = batch.masked_kspace
        if mask.shape[1] == 1:
            mask = mask.repeat(1, kspace_data.shape[1], 1, 1, 1)
        if mask.shape[2] == 1:
            mask = mask.repeat(1, 1, kspace_data.shape[2], 1, 1)
        
        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           use_checkpoint=self.use_checkpoint, compute_sens_per_coil=self.compute_sens_per_coil)
        center_slice = self.num_adj_slices // 2
        current_mask = torch.chunk(mask, self.num_adj_slices, dim=1)[center_slice]
        output = output_dict['img_pred']
        pred_kspace = output_dict['pred_kspace']
        GT_kspace = output_dict['original_kspace']
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        GT_kspace, pred_kspace = transforms.center_crop_to_smallest(GT_kspace, pred_kspace)
        
        # 扩展mask到复数维度以匹配kspace
        current_mask_expanded = current_mask.expand_as(pred_kspace)
        
        pred_kspace1 = pred_kspace * current_mask_expanded
        GT_kspace1 = GT_kspace * current_mask_expanded
        
        L_sup = self.loss(
            output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
        )
        L_self = self.normalized_l1(pred_kspace1, GT_kspace1)
        loss = L_sup + self.kspace_loss_weight * L_self
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("sup_loss", L_sup, prog_bar=True)
        self.log("self_loss", L_self, prog_bar=True)

        ##! raise error if loss is nan
        if torch.isnan(loss):
            raise ValueError(f'nan loss on {batch.fname} of slice {batch.slice_num}')
        return loss


    def validation_step(self, batch, batch_idx):
        mask = batch.mask
        kspace_data = batch.masked_kspace
        if mask.shape[1] == 1:
            mask = mask.repeat(1, kspace_data.shape[1], 1, 1, 1)
        if mask.shape[2] == 1:
            mask = mask.repeat(1, 1, kspace_data.shape[2], 1, 1)

        output_dict = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        center_slice = self.num_adj_slices // 2
        current_mask = torch.chunk(mask, self.num_adj_slices, dim=1)[center_slice]
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']
        pred_kspace = output_dict['pred_kspace']
        GT_kspace = output_dict['original_kspace']
        
        target, output = transforms.center_crop_to_smallest(batch.target, output)
        _, img_zf = transforms.center_crop_to_smallest(batch.target, img_zf)
        GT_kspace, pred_kspace = transforms.center_crop_to_smallest(GT_kspace, pred_kspace)
        
        pred_kspace1 = pred_kspace * current_mask
        GT_kspace1 = GT_kspace * current_mask
        
        val_loss_sup = self.loss(
                output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
            )
        val_loss_kspace = self.normalized_l1(pred_kspace1, GT_kspace1)
        
        self.log("val_loss_sup", val_loss_sup, prog_bar=True)
        self.log("val_loss_kspace", val_loss_kspace, prog_bar=True)
        
        cc = batch.masked_kspace.shape[1]
        centered_coil_visual = torch.log(1e-10+torch.view_as_complex(batch.masked_kspace[:,cc//2]).abs())
        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "img_zf":   img_zf,
            "mask": centered_coil_visual, 
            "sens_maps": output_dict['sens_maps'][:,0].abs(),
            "output": output,
            "target": target,
            "loss": val_loss_sup,  # 主要验证损失仍然是SSIM
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output_dict = self(batch.masked_kspace.float(), batch.mask, batch.num_low_frequencies, batch.mask_type,
                           compute_sens_per_coil=self.compute_sens_per_coil)
        output = output_dict['img_pred']
        img_zf = output_dict['img_zf']

        crop_size = batch.crop_size 
        crop_size = [crop_size[0][0], crop_size[1][0]] # if batch_size>1
        # detect FLAIR 203
        if output.shape[-1] < crop_size[1]:
            crop_size = (output.shape[-1], output.shape[-1])
        output = transforms.center_crop(output, crop_size)
        img_zf = transforms.center_crop(img_zf, crop_size)

        num_slc = batch.num_slc
        sense_map = output_dict['sens_maps'][:,0].abs()
        
        # Log sensitivity maps to wandb during inference (similar to validation logging)
        # Only log if logger exists and batch_idx matches the logging frequency
        if hasattr(self, 'logger') and self.logger is not None and batch_idx % self.inference_log_frequency == 0:
            try:
                # Normalize images for visualization
                sens_map_norm = sense_map / sense_map.max() if sense_map.max() > 0 else sense_map
                img_zf_norm = img_zf / img_zf.max() if img_zf.max() > 0 else img_zf
                output_norm = output / output.max() if output.max() > 0 else output
                
                # Apply alpha adjustment like in validation
                alpha = 0.2
                img_zf_alpha = img_zf_norm ** alpha
                output_alpha = output_norm ** alpha
                
                # Get filename for caption
                fname = batch.fname[0] if isinstance(batch.fname, list) else batch.fname
                if isinstance(fname, tuple):
                    fname = fname[0]
                fname = fname.split('/')[-1] if '/' in str(fname) else str(fname)
                
                # Get slice number
                slice_num = batch.slice_num[0].item() if isinstance(batch.slice_num, list) else batch.slice_num.item()
                
                # Log images to wandb
                key = f"inference/batch_{batch_idx:03d}"
                images = [sens_map_norm, img_zf_alpha, output_alpha]
                captions = [
                    f"Sensitivity Map - {fname} - Slice {slice_num}",
                    f"Zero-filled (α={alpha}) - {fname} - Slice {slice_num}",
                    f"Reconstruction (α={alpha}) - {fname} - Slice {slice_num}"
                ]
                
                self.logger.log_image(key, images, caption=captions, step=self.global_step)
                print(f"[PredictStep] Logged sensitivity maps for batch {batch_idx}: {fname}")
                
            except Exception as e:
                print(f"[PredictStep] Error logging sensitivity maps for batch {batch_idx}: {e}")
        
        return {
            'output': output.cpu(), 
            'img_zf': img_zf.cpu(),  # Add zero-filled images
            'slice_num': batch.slice_num, 
            'fname': batch.fname,
            'num_slc':  num_slc,
            'batch_idx': batch_idx,
            'time_frame': batch.num_t,
            'has_fake_time_dim': batch.has_fake_time_dim,
            'sens_maps': sense_map.cpu()  # Add sensitivity maps to return dict
        }

