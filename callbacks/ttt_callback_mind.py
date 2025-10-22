# callbacks/ttt_callback_mind.py
import copy
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only

# from your project
from mri_utils import normalized_l1_loss


class TestTimeTrainingCallback(L.Callback):
    """
    Test-Time Training with MIND-based self-supervised early stopping (NO GT, NO MAE).

    Train loss:
      - Standard k-space DC (normalized L1) on the acquired mask (center slice broadcast).

    Validation metric (no grad):
      - Build a low-frequency pseudo-reference (keep central 'num_low_frequencies' ky lines).
      - On center crop (W/3 x H/2), compute MIND descriptors for current recon and LF reference.
      - Mean L1 distance between descriptors is the metric to MINIMIZE.

    Early-stop modes (choose via `early_stop_mode`):
      - "first_drop": stop immediately when metric > best + min_delta_abs AND > best*(1+min_delta_rel).
      - "robust": filter single spikes with EMA + (abs/rel) thresholds + patience.
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 7,
        log_every_n_steps: int = 1,

        # ---- early stop mode ----
        early_stop_mode: str = "first_drop",  # "first_drop" | "robust"

        # thresholds used in BOTH modes
        min_delta_abs: float = 0.0,      # absolute minimal change to consider worsening
        min_delta_rel: float = 0.0,      # relative minimal change (e.g., 0.001 = 0.1%)

        # extras for "robust" mode
        patience: int = 1,               # require consecutive worsens before stop
        use_ema: bool = True,            # smooth metric with EMA before comparing
        ema_beta: float = 0.7,           # EMA decay (0.6~0.8 works well)

        # ---- MIND settings ----
        mind_patch: int = 7,             # odd
        mind_nonlocal: int = 9,          # odd, defines non-local window
        mind_sigma: float = 2.0,         # Gaussian sigma
        mind_stride: int = 2,            # subsampling of non-local offsets

        # ---- crop to match your ranking region ----
        crop_frac_w: float = 1.0/3.0,
        crop_frac_h: float = 1.0/2.0,

        # ---- optimization hygiene ----
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = False,
    ):
        super().__init__()
        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

        assert early_stop_mode in ("first_drop", "robust")
        self.early_stop_mode = early_stop_mode
        self.min_delta_abs = float(min_delta_abs)
        self.min_delta_rel = float(min_delta_rel)

        self.patience = int(patience)
        self.use_ema = bool(use_ema)
        self.ema_beta = float(ema_beta)

        self.mind_patch = int(mind_patch)
        self.mind_nonlocal = int(mind_nonlocal)
        self.mind_sigma = float(mind_sigma)
        self.mind_stride = int(mind_stride)

        self.crop_frac_w = float(crop_frac_w)
        self.crop_frac_h = float(crop_frac_h)

        self.weight_decay = float(weight_decay)
        self.freeze_bn_stats = bool(freeze_bn_stats)

        self.wandb_logger: Optional[WandbLogger] = None
        self.normalized_l1 = normalized_l1_loss()

        # runtime buffers
        self._gauss_kernel: Optional[torch.Tensor] = None
        self._offsets: Optional[List[Tuple[int, int]]] = None

    # ---------------- utils ----------------
    @staticmethod
    def _center_slice_index(module) -> int:
        num_adj = getattr(module, "num_adj_slices", 1)
        return int(num_adj // 2)

    @staticmethod
    def _is_complex_last(x: torch.Tensor) -> bool:
        return x.dim() >= 5 and x.size(-1) == 2

    @staticmethod
    def _broadcast_like(x: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        return x.expand_as(tgt) if list(x.shape) != list(tgt.shape) else x

    @torch.no_grad()
    def _center_mask(self, full_mask: torch.Tensor, cidx: int) -> torch.Tensor:
        if full_mask.dim() < 5:
            return full_mask
        chunks = torch.chunk(full_mask, chunks=full_mask.shape[1], dim=1)
        return chunks[cidx]

    @staticmethod
    def _center_crop_fraction(img: torch.Tensor, fw: float, fh: float) -> torch.Tensor:
        was_3d = False
        if img.dim() == 3:
            img = img.unsqueeze(1)
            was_3d = True
        B, C, H, W = img.shape
        tw = max(1, int(round(W * fw)))
        th = max(1, int(round(H * fh)))
        x0 = (W - tw) // 2
        y0 = (H - th) // 2
        out = img[:, :, y0:y0+th, x0:x0+tw]
        return out.squeeze(1) if was_3d else out

    @staticmethod
    def _set_bn_eval(module: torch.nn.Module):
        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
                m.eval()

    # ---- LF reference from k-space ----
    @staticmethod
    @torch.no_grad()
    def _to_complex(x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == 2
        return torch.view_as_complex(x.contiguous())

    @torch.no_grad()
    def _lf_reference_from_kspace(self, masked_kspace: torch.Tensor, num_low_freq: int) -> torch.Tensor:
        B, C, H, W, _ = masked_kspace.shape
        k_lf = torch.zeros_like(masked_kspace)
        half = int(num_low_freq // 2)
        y0 = max(0, int(H // 2 - half))
        y1 = min(H, int(y0 + num_low_freq))
        k_lf[:, :, y0:y1, :, :] = masked_kspace[:, :, y0:y1, :, :]
        img_c = torch.fft.ifft2(self._to_complex(k_lf), dim=(-2, -1), norm="ortho")
        mag = torch.abs(img_c)
        rss = torch.sqrt(torch.clamp((mag * mag).sum(dim=1, keepdim=True), min=1e-12))
        return rss  # (B,1,H,W)

    # ---- MIND ----
    def _build_gaussian_kernel(self, device, dtype) -> torch.Tensor:
        if self._gauss_kernel is not None and self._gauss_kernel.device == device and self._gauss_kernel.dtype == dtype:
            return self._gauss_kernel
        k = self.mind_patch
        assert k % 2 == 1, "mind_patch must be odd."
        r = k // 2
        yy, xx = torch.meshgrid(torch.arange(-r, r+1, device=device, dtype=dtype),
                                torch.arange(-r, r+1, device=device, dtype=dtype), indexing='ij')
        g = torch.exp(-(xx*xx + yy*yy) / (2.0 * (self.mind_sigma ** 2)))
        g = g / g.sum()
        self._gauss_kernel = g[None, None, :, :]
        return self._gauss_kernel

    def _build_offsets(self) -> List[Tuple[int, int]]:
        if self._offsets is not None:
            return self._offsets
        nl = self.mind_nonlocal
        assert nl % 2 == 1, "mind_nonlocal must be odd."
        r = nl // 2
        stride = max(1, self.mind_stride)
        offs = []
        for dy in range(-r, r+1, stride):
            for dx in range(-r, r+1, stride):
                if dy == 0 and dx == 0:
                    continue
                offs.append((dy, dx))
        self._offsets = offs
        return offs

    @staticmethod
    def _shift2d_reflect(img: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        B, C, H, W = img.shape
        pad_t, pad_b = max(dy, 0), max(-dy, 0)
        pad_l, pad_r = max(dx, 0), max(-dx, 0)
        x = F.pad(img, (pad_l, pad_r, pad_t, pad_b), mode="reflect")
        y0, y1 = pad_t - min(dy, 0), pad_t - min(dy, 0) + H
        x0, x1 = pad_l - min(dx, 0), pad_l - min(dx, 0) + W
        return x[:, :, y0:y1, x0:x1]

    def _dp_conv(self, img: torch.Tensor, dy: int, dx: int, gk: torch.Tensor) -> torch.Tensor:
        shifted = self._shift2d_reflect(img, dy, dx)
        diff2 = (img - shifted) ** 2
        return F.conv2d(diff2, gk, padding=self.mind_patch // 2)

    def _mind(self, img: torch.Tensor) -> torch.Tensor:
        assert img.dim() == 4 and img.size(1) == 1
        gk = self._build_gaussian_kernel(img.device, img.dtype)
        offsets = self._build_offsets()
        # local variance from 4-neighbors
        dp4 = [self._dp_conv(img, 1, 0, gk), self._dp_conv(img, -1, 0, gk),
               self._dp_conv(img, 0, 1, gk), self._dp_conv(img, 0, -1, gk)]
        V = (dp4[0] + dp4[1] + dp4[2] + dp4[3]) * 0.25
        V = torch.clamp(V, min=1e-6)
        feats = []
        for (dy, dx) in offsets:
            dp = self._dp_conv(img, dy, dx, gk)
            feats.append(torch.exp(- dp / V))
        Fstack = torch.cat(feats, dim=1)           # (B,K,H,W)
        Fstack = Fstack / (Fstack.amax(dim=1, keepdim=True) + 1e-8)
        return Fstack

    # ---------------- Lightning hooks ----------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
        self.wandb_logger = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)
        if self.wandb_logger is None:
            raise RuntimeError("TestTimeTrainingCallback requires a WandbLogger in trainer.logger")

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = batch.masked_kspace.device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)
        fname = getattr(batch, 'fname', ['unknown'])[0] if hasattr(batch, 'fname') else 'unknown'
        slice_num = getattr(batch, 'slice_num', [0])[0] if hasattr(batch, 'slice_num') else 0

        # save & working copy
        pl_module._orig_state = copy.deepcopy(pl_module.state_dict())
        adapt_model = copy.deepcopy(pl_module).to(device).train()

        # make absolutely sure params require grad (predict loop may disable grads globally)
        for p in adapt_model.parameters():
            p.requires_grad_(True)
        if self.freeze_bn_stats:
            self._set_bn_eval(adapt_model)

        opt = torch.optim.Adam(
            (p for p in adapt_model.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay
        )

        # masks
        cidx = self._center_slice_index(adapt_model)
        center_mask = self._center_mask(batch.mask, cidx)

        # LF pseudo-reference (no grad)
        with torch.no_grad():
            num_lf = int(batch.num_low_frequencies) if hasattr(batch, "num_low_frequencies") else 16
            ref_lf = self._lf_reference_from_kspace(batch.masked_kspace, num_lf)  # (B,1,H,W)

        # baseline (no grad)
        with torch.no_grad():
            out0 = adapt_model(
                batch.masked_kspace, batch.mask,
                batch.num_low_frequencies, batch.mask_type,
                compute_sens_per_coil=pl_module.compute_sens_per_coil,
            )
            img0 = out0["img_pred"]
            if img0.dim() == 3: img0 = img0.unsqueeze(1)
            img0c = self._center_crop_fraction(img0, self.crop_frac_w, self.crop_frac_h)
            refc  = self._center_crop_fraction(ref_lf, self.crop_frac_w, self.crop_frac_h)
            F_img0 = self._mind(img0c)
            F_ref  = self._mind(refc)
            best_metric = float(torch.mean(torch.abs(F_img0 - F_ref)).item())
            best_state = copy.deepcopy(adapt_model.state_dict())
            # NaN guard for baseline
            if torch.isnan(torch.tensor(best_metric)):
                print(f"[{device_name}] WARNING: MIND baseline is NaN for {fname}/slice{slice_num}, skipping TTT")
                return
            # Console baseline
            print(f"[{device_name}] TTT(MIND) starting for {fname}/slice{slice_num} (batch {batch_idx})")
            print(f"[{device_name}] TTT(MIND) baseline: mind_dist={best_metric:.6f}")
            # W&B baseline
            if trainer.is_global_zero and self.wandb_logger is not None:
                key = f"TTT/batch{batch_idx:03d}"
                self.wandb_logger.experiment.log({
                    f"{key}/mind_baseline": best_metric,
                    f"{key}/fname": fname,
                    f"{key}/slice_num": slice_num,
                    "TTT/inner_iter": -1,
                })

        # robust-mode state
        bad_count = 0
        ema_metric = best_metric if self.use_ema else None

        # ------------- inner loop (ensure grads enabled) -------------
        with torch.enable_grad():
            for it in range(self.inner_steps):
                opt.zero_grad(set_to_none=True)

                out = adapt_model(
                    batch.masked_kspace, batch.mask,
                    batch.num_low_frequencies, batch.mask_type,
                    compute_sens_per_coil=pl_module.compute_sens_per_coil,
                )

                # sanity checks: outputs must require grad
                pred_k = out["pred_kspace"]
                if not (isinstance(pred_k, torch.Tensor) and pred_k.requires_grad):
                    raise RuntimeError(
                        "pred_kspace.requires_grad=False. "
                        "Make sure adapt_model.train() is set and forward has no torch.no_grad()/detach()."
                    )

                # DC loss on acquired (broadcast center mask)
                if self._is_complex_last(pred_k):
                    cm = self._broadcast_like(center_mask, pred_k).float()
                else:
                    cm = center_mask
                    if cm.dim() == 5: cm = cm.squeeze(-1)
                    cm = self._broadcast_like(cm, pred_k).float()

                loss_dc = self.normalized_l1(pred_k * cm, out["original_kspace"] * cm)
                if torch.isnan(loss_dc):
                    print(f"[{device_name}] WARNING: iter {it+1} got NaN train loss, skip")
                    continue

                if not (isinstance(loss_dc, torch.Tensor) and loss_dc.requires_grad):
                    raise RuntimeError(
                        "loss_dc.requires_grad=False. "
                        "Ensure normalized_l1_loss returns a Tensor (not .item())."
                    )

                loss_dc.backward()
                opt.step()

                # metric (no grad)
                with torch.no_grad():
                    img = out["img_pred"]
                    if img.dim() == 3: img = img.unsqueeze(1)
                    imgc = self._center_crop_fraction(img, self.crop_frac_w, self.crop_frac_h)
                    F_img = self._mind(imgc)
                    metric_cur = float(torch.mean(torch.abs(F_img - F_ref)).item())
                    if torch.isnan(torch.tensor(metric_cur)):
                        print(f"[{device_name}] WARNING: iter {it+1} got NaN val metric, skip")
                        continue

                # EMA (robust mode)
                metric_for_judge = metric_cur
                if self.early_stop_mode == "robust" and self.use_ema:
                    if ema_metric is None:
                        ema_metric = metric_cur
                    else:
                        ema_metric = self.ema_beta * ema_metric + (1.0 - self.ema_beta) * metric_cur
                    metric_for_judge = ema_metric

                # compare with thresholds
                worse = metric_for_judge > (best_metric * (1.0 + self.min_delta_rel) + self.min_delta_abs)

                # console print (multi-GPU friendly)
                print(f"[{device_name}] TTT(MIND) iter {it+1}/{self.inner_steps}: train_loss={loss_dc.item():.6f}")
                print(f"[{device_name}]     mind_cur={metric_cur:.6f}, judged={metric_for_judge:.6f}, best={best_metric:.6f}")

                # W&B logging (per-batch namespace)
                if trainer.is_global_zero and self.wandb_logger is not None and \
                   (it % self.log_every_n_steps == 0 or it == self.inner_steps - 1):
                    key = f"TTT/batch{batch_idx:03d}"
                    self.wandb_logger.experiment.log({
                        f"{key}/loss_dc": float(loss_dc.item()),
                        f"{key}/mind_cur": metric_cur,
                        f"{key}/mind_judged": metric_for_judge,
                        f"{key}/mind_best": best_metric,
                        "TTT/inner_iter": int(it),
                    })

                if not worse:
                    best_metric = metric_for_judge
                    best_state = copy.deepcopy(adapt_model.state_dict())
                    bad_count = 0
                    print(f"[{device_name}]     ✓ New best: {best_metric:.6f}")
                else:
                    if self.early_stop_mode == "first_drop":
                        print(f"[{device_name}]     ✗ Early stop at iter {it+1} (worse: {metric_for_judge:.6f} > {best_metric:.6f})")
                        break
                    else:
                        bad_count += 1
                        print(f"[{device_name}]     worse #{bad_count}: judged={metric_for_judge:.6f} > best={best_metric:.6f}")
                        if bad_count > self.patience:
                            print(f"[{device_name}]     ✗ Early stop (patience exceeded)")
                            break

        # restore best and copy back
        adapt_model.load_state_dict(best_state)
        pl_module.load_state_dict(adapt_model.state_dict())
        pl_module.eval()

        print(f"[{device_name}] TTT(MIND) completed for {fname}/slice{slice_num}: final best={best_metric:.6f}")
        if trainer.is_global_zero and self.wandb_logger is not None:
            key = f"TTT/batch{batch_idx:03d}"
            self.wandb_logger.experiment.log({
                f"{key}/mind_final_best": best_metric,
                f"{key}/fname": fname,
                f"{key}/slice_num": slice_num,
            })

    @rank_zero_only
    def on_predict_batch_end(self, trainer, pl_module, prediction, batch, dataloader_idx: int = 0, **kwargs):
        if hasattr(pl_module, "_orig_state"):
            pl_module.load_state_dict(pl_module._orig_state)
            pl_module.eval()
            del pl_module._orig_state
