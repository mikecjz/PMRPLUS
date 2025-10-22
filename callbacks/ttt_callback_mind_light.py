# callbacks/ttt_callback_mind_light.py
import copy
import contextlib
from typing import Optional, Tuple, List, Any, Dict

import torch
import torch.nn.functional as F
import lightning as L

# ---- from your project ----
from mri_utils import normalized_l1_loss


class TestTimeTrainingCallback(L.Callback):
    """
    Lightweight Test-Time Training with MIND-based self-supervised early stopping.
    No W&B logging to avoid DDP synchronization issues.
    """

    def __init__(
        self,
        lr: float = 1e-6,
        inner_steps: int = 7,
        log_every_n_steps: int = 5,  # 减少日志频率

        # ---- early stop mode ----
        early_stop_mode: str = "first_drop",  # "first_drop" | "robust"

        # thresholds used in BOTH modes
        min_delta_abs: float = 0.0,      # absolute minimal change to consider worsening
        min_delta_rel: float = 0.0,      # relative minimal change (e.g., 0.001 = 0.1%)

        # extras for "robust" mode
        patience: int = 1,               # require >= patience consecutive worsens before stop
        use_ema: bool = True,            # smooth metric with EMA before comparing
        ema_beta: float = 0.7,           # EMA decay (0.6~0.8 works well)

        # ---- MIND settings ----
        mind_patch: int = 7,             # odd
        mind_nonlocal: int = 9,          # odd, defines non-local window
        mind_sigma: float = 2.0,         # Gaussian sigma
        mind_stride: int = 2,            # subsampling of non-local offsets

        # ---- crop region for ranking ----
        crop_frac_w: float = 1.0/3.0,
        crop_frac_h: float = 1.0/2.0,

        # ---- optimization hygiene ----
        weight_decay: float = 0.0,
        freeze_bn_stats: bool = False,

        # ---- tie-breaker ----
        use_tie_breaker_kspace: bool = True,

        # ---- AMP (auto-detect from trainer.precision) ----
        use_amp_autodetect: bool = True,
    ):
        super().__init__()
        assert early_stop_mode in ("first_drop", "robust")

        self.lr = float(lr)
        self.inner_steps = int(inner_steps)
        self.log_every_n_steps = int(log_every_n_steps)

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
        self.use_tie_breaker_kspace = bool(use_tie_breaker_kspace)
        self.use_amp_autodetect = bool(use_amp_autodetect)

        self.normalized_l1 = normalized_l1_loss()

        # runtime buffers
        self._gauss_kernel: Optional[torch.Tensor] = None
        self._offsets: Optional[List[Tuple[int, int]]] = None

    # -------------------- utils --------------------
    @staticmethod
    def _get(batch: Any, key: str, default=None):
        if hasattr(batch, key):
            return getattr(batch, key)
        if isinstance(batch, dict):
            return batch.get(key, default)
        return default

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
        if full_mask is None:
            raise RuntimeError("batch.mask is required for TTT(DC).")
        if full_mask.dim() < 5:
            return full_mask
        chunks = torch.chunk(full_mask, chunks=full_mask.shape[1], dim=1)
        return chunks[cidx]

    @staticmethod
    def _center_crop_fraction(img: torch.Tensor, fw: float, fh: float) -> torch.Tensor:
        # supports (B,1,H,W) or (B,H,W)
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
        assert x.size(-1) == 2, "expected complex-last tensor (..., 2)"
        return torch.view_as_complex(x.contiguous())

    @torch.no_grad()
    def _lf_reference_from_kspace(self, masked_kspace: torch.Tensor, num_low_freq: int) -> torch.Tensor:
        if masked_kspace is None:
            raise RuntimeError("batch.masked_kspace is required to build LF reference.")
        B, C, H, W, _ = masked_kspace.shape
        if num_low_freq <= 0:
            num_low_freq = 16
        k_lf = torch.zeros_like(masked_kspace)
        half = int(num_low_freq // 2)
        y0 = max(0, int(H // 2 - half))
        y1 = min(H, int(y0 + num_low_freq))
        k_lf[:, :, y0:y1, :, :] = masked_kspace[:, :, y0:y1, :, :]
        img_c = torch.fft.ifft2(self._to_complex(k_lf), dim=(-2, -1), norm="ortho")
        mag = torch.abs(img_c)
        rss = torch.sqrt(torch.clamp((mag * mag).sum(dim=1, keepdim=True), min=1e-12))
        return rss  # (B,1,H,W)

    # -------------------- MIND --------------------
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
        """
        Input: (B,1,H,W), float
        Output: (B,K,H,W) where K=len(offsets)
        """
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

    # -------------------- Lightning hooks --------------------
    def _autocast_context(self, trainer, device) -> contextlib.AbstractContextManager:
        if not self.use_amp_autodetect:
            return contextlib.nullcontext()
        prec = str(getattr(trainer, "precision", "")).lower()
        if device.type != "cuda":
            return contextlib.nullcontext()
        if "bf16" in prec:
            return torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if "16" in prec:
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return contextlib.nullcontext()

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx: int = 0):
        device = next(pl_module.parameters()).device
        device_name = f"cuda:{device.index}" if device.type == 'cuda' else str(device)

        fname = self._get(batch, 'fname', ['unknown'])
        fname = fname[0] if isinstance(fname, (list, tuple)) else str(fname)
        slice_num = self._get(batch, 'slice_num', [0])
        slice_num = int(slice_num[0] if isinstance(slice_num, (list, tuple)) else slice_num)

        masked_kspace = self._get(batch, "masked_kspace")
        mask = self._get(batch, "mask")
        num_low_frequencies = int(self._get(batch, "num_low_frequencies") or 16)
        mask_type = self._get(batch, "mask_type")
        compute_sens_per_coil = getattr(pl_module, "compute_sens_per_coil", False)

        if masked_kspace is None or mask is None:
            print(f"[{device_name}] ERROR: masked_kspace/mask not found. Skip TTT for {fname}/slice{slice_num}.")
            return

        # Save original weights to restore later
        if not hasattr(pl_module, "_orig_state"):
            pl_module._orig_state = copy.deepcopy(pl_module.state_dict())

        # train mode + grads on
        pl_module.train()
        if self.freeze_bn_stats:
            self._set_bn_eval(pl_module)
        for p in pl_module.parameters():
            p.requires_grad_(True)

        opt = torch.optim.Adam(
            (p for p in pl_module.parameters() if p.requires_grad),
            lr=self.lr, weight_decay=self.weight_decay
        )

        # center-slice mask for DC
        cidx = self._center_slice_index(pl_module)
        center_mask = self._center_mask(mask, cidx)

        # LF pseudo-reference
        with torch.no_grad():
            ref_lf = self._lf_reference_from_kspace(masked_kspace, num_low_frequencies)  # (B,1,H,W)

        # baseline forward (no grad)
        with torch.no_grad():
            out0 = pl_module(
                masked_kspace, mask,
                num_low_frequencies, mask_type,
                compute_sens_per_coil=compute_sens_per_coil,
            )
            img0 = out0["img_pred"]
            if img0.dim() == 3:
                img0 = img0.unsqueeze(1)
            img0c = self._center_crop_fraction(img0, self.crop_frac_w, self.crop_frac_h)
            refc  = self._center_crop_fraction(ref_lf, self.crop_frac_w, self.crop_frac_h)
            F_img0 = self._mind(img0c)
            F_ref  = self._mind(refc)
            best_metric_raw = float(torch.mean(torch.abs(F_img0 - F_ref)).item())
            best_metric_judged = best_metric_raw  # for comparison logic
            best_kspace_tie = float("inf")        # tie-breaker
            best_state = copy.deepcopy(pl_module.state_dict())  # save baseline state as initial best
            # NaN guard
            if torch.isnan(torch.tensor(best_metric_raw)):
                print(f"[{device_name}] WARNING: MIND baseline is NaN for {fname}/slice{slice_num}, skipping TTT")
                return

            print(f"[{device_name}] TTT(MIND) start {fname}/slice{slice_num} (batch {batch_idx}); baseline={best_metric_raw:.6f}")

        # robust-mode state
        bad_count = 0
        ema_metric = best_metric_raw if self.use_ema else None
        it_when_best = -1

        # ---- inner loop (enable grads, optional AMP) ----
        autocast_ctx = self._autocast_context(trainer, device)
        for it in range(self.inner_steps):
            opt.zero_grad(set_to_none=True)

            # Enable gradients for training
            with torch.enable_grad():
                with autocast_ctx:
                    out = pl_module(
                        masked_kspace, mask,
                        num_low_frequencies, mask_type,
                        compute_sens_per_coil=compute_sens_per_coil,
                    )
                    pred_k = out["pred_kspace"]
                    original_k = out["original_kspace"]

                    if not (isinstance(pred_k, torch.Tensor) and pred_k.requires_grad):
                        raise RuntimeError(
                            "pred_kspace.requires_grad=False. Ensure pl_module.train() and no detach in forward."
                        )

                    # DC loss on acquired
                    if self._is_complex_last(pred_k):
                        cm = self._broadcast_like(center_mask, pred_k).float()
                    else:
                        cm = center_mask
                        if cm.dim() == 5:
                            cm = cm.squeeze(-1)
                        cm = self._broadcast_like(cm, pred_k).float()

                    loss_dc = self.normalized_l1(pred_k * cm, original_k * cm)

                if torch.isnan(loss_dc):
                    print(f"[{device_name}] WARNING: iter {it+1} NaN train loss, skip step")
                    continue

                # backward (AMP-safe)
                if isinstance(autocast_ctx, torch.cuda.amp.autocast):
                    scaler = getattr(trainer, "scaler", None)
                    if scaler is None:
                        loss_dc.backward()
                        opt.step()
                    else:
                        scaler.scale(loss_dc).backward()
                        scaler.step(opt)
                        scaler.update()
                else:
                    loss_dc.backward()
                    opt.step()

            # compute MIND metric (no grad)
            with torch.no_grad():
                img = out["img_pred"]
                if img.dim() == 3:
                    img = img.unsqueeze(1)
                imgc = self._center_crop_fraction(img, self.crop_frac_w, self.crop_frac_h)
                F_img = self._mind(imgc)
                metric_cur_raw = float(torch.mean(torch.abs(F_img - F_ref)).item())
                if torch.isnan(torch.tensor(metric_cur_raw)):
                    print(f"[{device_name}] WARNING: iter {it+1} NaN val metric, skip judge")
                    continue

            # EMA (robust mode)
            metric_for_judge = metric_cur_raw
            if self.early_stop_mode == "robust" and self.use_ema:
                ema_metric = metric_cur_raw if ema_metric is None else self.ema_beta * ema_metric + (1.0 - self.ema_beta) * metric_cur_raw
                metric_for_judge = float(ema_metric)

            # judge against best
            threshold = best_metric_judged * (1.0 + self.min_delta_rel) + self.min_delta_abs
            worse = metric_for_judge > threshold

            # console logging (reduced frequency)
            if (it % self.log_every_n_steps == 0) or (it == self.inner_steps - 1):
                print(f"[{device_name}] TTT(MIND) iter {it+1}/{self.inner_steps}: "
                      f"train_dc={loss_dc.item():.6f} | mind_raw={metric_cur_raw:.6f} "
                      f"| judged={metric_for_judge:.6f} | best={best_metric_judged:.6f} (thr={threshold:.6f})")

            if not worse:
                best_metric_judged = metric_for_judge
                best_metric_raw = metric_cur_raw
                best_kspace_tie = float(loss_dc.item())
                it_when_best = it
                best_state = copy.deepcopy(pl_module.state_dict())  # save current best state
                bad_count = 0
                print(f"[{device_name}]     ✓ New best: judged={best_metric_judged:.6f} (raw={best_metric_raw:.6f})")
            else:
                if self.early_stop_mode == "first_drop":
                    print(f"[{device_name}]     ✗ Early stop at iter {it+1} (judged worse).")
                    break
                else:
                    bad_count += 1
                    print(f"[{device_name}]     worse #{bad_count}: judged={metric_for_judge:.6f} > thr={threshold:.6f}")
                    if bad_count >= self.patience:
                        print(f"[{device_name}]     ✗ Early stop (patience reached).")
                        break

        # restore best state and copy back to pl_module
        pl_module.load_state_dict(best_state)
        pl_module.eval()
        
        print(f"[{device_name}] TTT(MIND) done {fname}/slice{slice_num}: best_judged={best_metric_judged:.6f} "
              f"(raw={best_metric_raw:.6f}) at iter={it_when_best}")

    def on_predict_batch_end(self, trainer, pl_module, prediction, batch, dataloader_idx: int = 0, **kwargs):
        # Restore original weights so the next sample starts fresh
        if hasattr(pl_module, "_orig_state"):
            pl_module.load_state_dict(pl_module._orig_state)
            pl_module.eval()
            del pl_module._orig_state

