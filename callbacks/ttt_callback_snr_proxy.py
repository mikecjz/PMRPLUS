# callbacks/ttt_callback_snr_proxy.py
import copy
import contextlib
from typing import Optional, Callable, Tuple, Any, Dict, List

import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only


# ---------------------- small utilities ----------------------
def _complex_as_real(x: torch.Tensor) -> torch.Tensor:
    return torch.view_as_real(x) if x.is_complex() else x


def _real_as_complex(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_complex() else torch.view_as_complex(x.contiguous())


def _to_2d_mask(acq_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert arbitrary-shaped acquired mask to [H, W] boolean.
    Accepts shapes like [H,W], [1,H,W], [H,W,1], [B,H,W], [coils,H,W], [B,coils,H,W,1], etc.
    Strategy: squeeze last singleton if present, then OR-reduce over all leading dims except (H,W).
    """
    m = acq_mask.bool()
    if m.dim() == 2:
        return m
    if m.dim() >= 3 and m.size(-1) == 1:
        m = m.squeeze(-1)
    if m.dim() > 2:
        reduce_dims = tuple(range(0, m.dim() - 2))
        m = m.any(dim=reduce_dims)
    assert m.dim() == 2, f"Mask must be [H,W] after conversion, got {tuple(m.shape)}"
    return m


def split_ssdu_masks(acq_mask_2d: torch.Tensor, val_ratio: float = 0.1, keep_acs: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split acquired 2D mask M into R (train) and V (val), both subsets of M and disjoint.
    - acq_mask_2d: [H, W] bool
    - keep_acs: keep 2*keep_acs low-freq ky lines in R to stabilize
    """
    acq_mask_2d = acq_mask_2d.bool()
    H, W = acq_mask_2d.shape
    R = torch.zeros_like(acq_mask_2d, dtype=torch.bool)
    V = torch.zeros_like(acq_mask_2d, dtype=torch.bool)

    if keep_acs > 0:
        cy = H // 2
        sl = slice(max(0, cy - keep_acs), min(H, cy + keep_acs))
        R[sl, :] = acq_mask_2d[sl, :]

    free_mask = acq_mask_2d & (~R)
    free_idx = torch.nonzero(free_mask, as_tuple=False)
    n_free = free_idx.size(0)
    n_val = int(max(1, round(n_free * val_ratio))) if n_free > 0 else 0

    if n_val > 0:
        perm = torch.randperm(n_free)
        V_sel = free_idx[perm[:n_val]]
        R_sel = free_idx[perm[n_val:]]
        V[V_sel[:, 0], V_sel[:, 1]] = True
        R[R_sel[:, 0], R_sel[:, 1]] = True
    else:
        R |= free_mask

    return R, V


# ---------------------- callback ----------------------
class SNRProxyTTTCallback(L.Callback):
    """
    Unsupervised SSDU-style Test-Time Training with selectable early-stopping metric.

    Train loss (on R):   k-space consistency (L1) using model's pred_kspace if available (fallback: FFT)
    Val metric (on V):   val_mse (MSE)  OR  proxy_snr (maximize)
    Proxy-SNR:           10*log10( mean(|img_pred|^2) / mean(|(Khat-K)*V|^2) )

    Expected batch:
      - masked_kspace : [coils?, H, W, 2] or complex
      - mask          : arbitrary shape → will be converted to [H,W]
      - (optional) num_low_frequencies, mask_type, fname, slice_num
    Model forward (recommended):
      out = pl_module(masked_kspace, mask, num_low_frequencies, mask_type, compute_sens_per_coil=...)
      out should contain "img_pred" and preferably "pred_kspace" (and "original_kspace" if you want to log DC).
    """

    def __init__(
        self,
        lr: float = 1e-5,
        max_steps: int = 50,
        val_ratio: float = 0.1,
        keep_acs: int = 8,
        patience: int = 3,
        # early-stop selector
        early_stop_metric: str = "val_mse",  # "val_mse" | "proxy_snr"
        # thresholds
        min_delta_val: float = 0.0,     # absolute decrease required for val_mse
        min_delta_proxy: float = 0.05,  # dB increase required for proxy_snr
        # smoothing proxy_snr (to reduce jitter in decision)
        proxy_smooth_window: int = 1,   # 1 = no smoothing; 3 is a good start
        # logging & optimization hygiene
        log_every_n_steps: int = 5,
        freeze_bn_stats: bool = True,
        freeze_bn_params: bool = False,
        weight_decay: float = 0.0,
        # AMP auto-detect
        use_amp_autodetect: bool = True,
        # optional param filter (e.g., only last cascades / prompts)
        param_filter_fn: Optional[Callable[[torch.nn.Module], List[torch.nn.Parameter]]] = None,
    ):
        super().__init__()
        assert early_stop_metric in ("val_mse", "proxy_snr")
        assert proxy_smooth_window >= 1

        self.lr = lr
        self.max_steps = max_steps
        self.val_ratio = val_ratio
        self.keep_acs = keep_acs
        self.patience = patience

        self.early_stop_metric = early_stop_metric
        self.min_delta_val = min_delta_val
        self.min_delta_proxy = min_delta_proxy
        self.proxy_smooth_window = proxy_smooth_window

        self.log_every_n_steps = log_every_n_steps
        self.freeze_bn_stats = freeze_bn_stats
        self.freeze_bn_params = freeze_bn_params
        self.weight_decay = weight_decay

        self.use_amp_autodetect = use_amp_autodetect
        self.param_filter_fn = param_filter_fn

        self._wandb: Optional[WandbLogger] = None

    # ---------- batch helpers ----------
    @staticmethod
    def _get(batch: Any, key: str, default=None):
        if hasattr(batch, key):
            return getattr(batch, key)
        if isinstance(batch, dict):
            return batch.get(key, default)
        return default

    # ---------- AMP ctx ----------
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

    # ---------- logger detection ----------
    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        loggers = trainer.logger if isinstance(trainer.logger, list) else [trainer.logger]
        self._wandb = next((lg for lg in loggers if isinstance(lg, WandbLogger)), None)

    def _log_wandb(self, trainer, payload: Dict[str, Any]):
        if trainer.is_global_zero and self._wandb is not None:
            self._wandb.experiment.log(payload)

    # ---------- forward wrappers ----------
    def _forward_and_cache(self, pl_module, masked_kspace: torch.Tensor, batch: dict):
        """Call model forward with your project's signature; return (img_pred, out_dict)."""
        mask = self._get(batch, "mask")
        nfl = int(self._get(batch, "num_low_frequencies") or 16)
        mtype = self._get(batch, "mask_type")
        cspc = getattr(pl_module, "compute_sens_per_coil", False)

        out = pl_module(masked_kspace, mask, nfl, mtype, compute_sens_per_coil=cspc)
        img = out["img_pred"]
        # ensure shape [..., 2] or [B,1,H,W] magnitude—proxy SNR uses |img|^2，所以两者都可
        if img.dim() == 3:
            # [H,W] → [1,H,W] for power computation
            img = img.unsqueeze(0)
        return img, out

    def _image_to_kspace(self, pl_module, img: torch.Tensor, batch: dict, out_cache: Optional[dict]) -> torch.Tensor:
        """Prefer model's pred_kspace if present; else fallback to FFT."""
        if out_cache is not None and isinstance(out_cache.get("pred_kspace", None), torch.Tensor):
            return _complex_as_real(out_cache["pred_kspace"])
        # fallback FFT (single-coil proxy)
        img_c = _real_as_complex(img) if not img.dtype.is_complex el_
