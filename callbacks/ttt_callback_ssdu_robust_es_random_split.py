# callbacks/ttt_callback_ssdu_robust_es_random_split.py
import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# from your project
from mri_utils import normalized_l1_loss
torch.set_float32_matmul_precision('high')

class TestTimeTrainingCallback(L.Callback):
    """
    Test-Time Training with robust self-supervised early stopping (NO GT).
    
    采用随机划分策略 + ACS保护：
    - 固定保留中心 ACS lines 用于训练（提供稳定的低频信息）
    - 剩余采样点按比例随机划分为训练集和验证集
    - 适用于所有mask类型：radial, gaussian, uniform
    
    优势：
    1. 对不同mask类型具有普适性
    2. 保证训练集有足够的低频信息（通过ACS保护）
    3. 验证集采样分布与训练集相似（随机划分）
    4. 可以灵活调整训练/验证比例
    
    Validation metric = Huber(k-space residual on Ω_val) + lambda_tv * max(0, TV_growth - tv_tol)
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 7,
        log_every_n_steps: int = 1,

        # ---- k-space split control ----
        val_ratio: float = 0.15,          # 验证集占采样点的比例（推荐10-15%）
        keep_acs_lines: int = 8,          # 保留在训练集的ACS中心线数量（您有16条，保留8条给训练）
        random_seed: Optional[int] = None, # 随机种子，None表示每次随机

        # ---- early stopping ----
        min_delta: float = 0.0,           
        stop_on_first_worse: bool = True, 

        # ---- robust validation ----
        huber_delta: float = 1e-3,        

        # ---- TV guard (image domain) ----
        lambda_tv: float = 0.01,          
        tv_tol_ratio: float = 0.10,       
        tv_eps: float = 1e-6,             

        # ---- optimization hygiene ----
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = False,
        
        # ---- logging control ----
        enable_wandb_logging: bool = True,
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

        self.val_ratio = float(val_ratio)
        self.keep_acs_lines = int(keep_acs_lines)
        self.random_seed = random_seed

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
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
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
    def _build_val_train_masks_random_split(
        self, 
        center_mask: torch.Tensor, 
        val_ratio: float,
        keep_acs_lines: int,
        random_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        随机划分k-space采样点为训练集和验证集，同时保护ACS区域。
        
        策略：
        1. 将中心 keep_acs_lines 条线固定分配给训练集（保证低频稳定性）
        2. 剩余采样点随机划分：(1-val_ratio)给训练集，val_ratio给验证集
        
        Args:
            center_mask: (B,1,H,W,[2?]) - center slice的mask
            val_ratio: 验证集占比
            keep_acs_lines: 固定保留在训练集的中心线数量
            random_seed: 随机种子
            
        Returns:
            tr_mask, val_mask: 两个mask，形状与center_mask相同
        """
        B, _, H, W = center_mask.shape[:4]
        val_mask = torch.zeros_like(center_mask)
        tr_mask  = torch.zeros_like(center_mask)

        # k-space中心位置
        cy = H // 2
        
        # 定义ACS区域：中心 keep_acs_lines 条线
        acs_start = max(0, cy - keep_acs_lines // 2)
        acs_end = min(H, cy + (keep_acs_lines + 1) // 2)
        acs_rows = set(range(acs_start, acs_end))
        
        # 设置随机种子（如果提供）
        if random_seed is not None:
            rng_state = torch.random.get_rng_state()
            torch.manual_seed(random_seed)
        
        # 处理每个batch
        for b in range(B):
            # 获取所有采样的k-space位置
            if self._is_complex_last(center_mask):
                # 复数格式: (B,1,H,W,2)
                acq = (center_mask[b, 0, :, :, 0] > 0)  # (H,W)
            else:
                # 实数格式
                if center_mask.dim() == 5:
                    acq = (center_mask[b, 0, :, :, 0] > 0)
                else:
                    acq = (center_mask[b, 0, :, :] > 0)
            
            # 获取所有采样点的坐标
            acq_indices = torch.nonzero(acq, as_tuple=False)  # (N, 2) - [row, col]
            
            if acq_indices.numel() == 0:
                continue
            
            # 分离ACS区域和非ACS区域的采样点
            acs_mask = torch.tensor([idx[0].item() in acs_rows for idx in acq_indices], 
                                   dtype=torch.bool, device=acq.device)
            
            acs_indices = acq_indices[acs_mask]
            non_acs_indices = acq_indices[~acs_mask]
            
            # 1. ACS区域全部分配给训练集
            if acs_indices.numel() > 0:
                if self._is_complex_last(center_mask):
                    tr_mask[b, 0, acs_indices[:, 0], acs_indices[:, 1], :] = \
                        center_mask[b, 0, acs_indices[:, 0], acs_indices[:, 1], :]
                elif center_mask.dim() == 5:
                    tr_mask[b, 0, acs_indices[:, 0], acs_indices[:, 1], 0] = \
                        center_mask[b, 0, acs_indices[:, 0], acs_indices[:, 1], 0]
                else:
                    tr_mask[b, 0, acs_indices[:, 0], acs_indices[:, 1]] = \
                        center_mask[b, 0, acs_indices[:, 0], acs_indices[:, 1]]
            
            # 2. 非ACS区域随机划分
            if non_acs_indices.numel() > 0:
                n_non_acs = non_acs_indices.size(0)
                n_val = max(1, int(round(n_non_acs * val_ratio)))
                
                # 随机打乱
                perm = torch.randperm(n_non_acs, device=acq.device)
                val_indices = non_acs_indices[perm[:n_val]]
                train_indices = non_acs_indices[perm[n_val:]]
                
                # 分配到验证集
                if val_indices.numel() > 0:
                    if self._is_complex_last(center_mask):
                        val_mask[b, 0, val_indices[:, 0], val_indices[:, 1], :] = \
                            center_mask[b, 0, val_indices[:, 0], val_indices[:, 1], :]
                    elif center_mask.dim() == 5:
                        val_mask[b, 0, val_indices[:, 0], val_indices[:, 1], 0] = \
                            center_mask[b, 0, val_indices[:, 0], val_indices[:, 1], 0]
                    else:
                        val_mask[b, 0, val_indices[:, 0], val_indices[:, 1]] = \
                            center_mask[b, 0, val_indices[:, 0], val_indices[:, 1]]
                
                # 分配到训练集
                if train_indices.numel() > 0:
                    if self._is_complex_last(center_mask):
                        tr_mask[b, 0, train_indices[:, 0], train_indices[:, 1], :] = \
                            center_mask[b, 0, train_indices[:, 0], train_indices[:, 1], :]
                    elif center_mask.dim() == 5:
                        tr_mask[b, 0, train_indices[:, 0], train_indices[:, 1], 0] = \
                            center_mask[b, 0, train_indices[:, 0], train_indices[:, 1], 0]
                    else:
                        tr_mask[b, 0, train_indices[:, 0], train_indices[:, 1]] = \
                            center_mask[b, 0, train_indices[:, 0], train_indices[:, 1]]
        
        # 恢复随机状态
        if random_seed is not None:
            torch.random.set_rng_state(rng_state)
        
        return tr_mask, val_mask

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    # --------------- Lightning hooks ---------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        if self.enable_wandb_logging:
            loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
            self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
            if self.wandb_logger is None:
                print("[TTT-RandomSplit] WARNING: WandbLogger not found but enable_wandb_logging=True. Proceeding without W&B logging.")
        else:
            print("[TTT-RandomSplit] W&B logging disabled by enable_wandb_logging=False")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = batch.masked_kspace.device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)
        fname = getattr(batch, 'fname', ['unknown'])[0] if hasattr(batch, 'fname') else 'unknown'
        slice_num = getattr(batch, 'slice_num', [0])[0] if hasattr(batch, 'slice_num') else 0
        
        # save original
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

        # ---- build random split train/val masks on center slice ----
        cidx = self._center_slice_index(adapt_model)
        center_mask = self._center_mask(batch.mask, cidx)  # (B,1,H,W,[2?])

        train_center, val_center = self._build_val_train_masks_random_split(
            center_mask, 
            self.val_ratio, 
            self.keep_acs_lines,
            self.random_seed
        )

        # 统计划分信息
        train_points = (train_center > 0).sum().item()
        val_points = (val_center > 0).sum().item()
        total_points = train_points + val_points

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

            # validation residual on Ω_val
            if self._is_complex_last(out0["pred_kspace"]):
                mv = self._broadcast_like(val_center, out0["pred_kspace"]).float()
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

            val_metric_best = val_dc0
            best_state = copy.deepcopy(adapt_model.state_dict())

            # Check for NaN
            if torch.isnan(torch.tensor([val_dc0, tv0])).any():
                print(f"[{device_name}] WARNING: TTT baseline contains NaN for {fname}/slice{slice_num}, skipping TTT")
                return
            
            print(f"[{device_name}] TTT starting for {fname}/slice{slice_num} (batch {batch_idx})")
            print(f"[{device_name}] 划分策略: 随机split (train={train_points}/{total_points}={train_points/total_points*100:.1f}%, "
                  f"val={val_points}/{total_points}={val_points/total_points*100:.1f}%, ACS保护={self.keep_acs_lines}条)")
            print(f"[{device_name}] TTT baseline: val_dc={val_dc0:.6f}, tv={tv0:.6f}, val_metric={val_metric_best:.6f}")
            
            if trainer.is_global_zero and self.enable_wandb_logging and self.wandb_logger is not None:
                self.wandb_logger.experiment.log({
                    "TTT/val_dc_huber/baseline": val_dc0,
                    "TTT/tv_center/baseline": tv0,
                    "TTT/val_metric/best": val_metric_best,
                    "TTT/split_train_points": train_points,
                    "TTT/split_val_points": val_points,
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

                # train DC loss on Ω_tr
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
                target_masked = yt.detach()
                
                loss_tr = self.normalized_l1(pred_masked, target_masked)
                
                # Check for NaN
                if torch.isnan(loss_tr):
                    print(f"[{device_name}] WARNING: TTT iter {it} got NaN loss, skipping this iteration")
                    continue
                    
                loss_tr.backward()
            opt.step()
            
            print(f"[{device_name}] TTT iter {it+1}/{self.inner_steps}: train_loss={loss_tr.item():.6f}")

            # ---- validation metric on Ω_val ----
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

                # TV-growth penalty
                tv_thresh = tv0 * (1.0 + self.tv_tol_ratio)
                tv_penalty = max(0.0, tv_cur - tv_thresh)
                val_metric = val_dc + self.lambda_tv * tv_penalty
                
                # Check for NaN
                if torch.isnan(torch.tensor([val_dc, tv_cur, val_metric])).any():
                    print(f"[{device_name}] WARNING: TTT iter {it} got NaN validation metrics, skipping")
                    continue
                
                print(f"[{device_name}]     val_loss={val_dc:.4f}, tv={tv_cur:.1f} (↑{tv_penalty:.1f}), "
                      f"metric={val_metric:.4f} (best={val_metric_best:.4f})")

            # logging
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
        
        # per-subject logging
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

