from typing import Sequence, List, Tuple
import math
import numpy as np
import torch
import logging
import os
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler
from math import ceil, floor
from torch import Tensor
import math
import cv2

def get_logger(save_path=None, level='INFO'):
    logger = logging.getLogger('log')
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('[%(asctime)s][%(filename)15s][line:%(lineno)4d][%(levelname)8s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dinov3_weight_name(encoder_name):
    NAMES ={
        'dinov3_vitb16': 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        'dinov3_vitl16': 'dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth',
        'dinov3_vits16': 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
        'dinov3_vits16plus': 'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    }
    return NAMES[encoder_name]


def multi_scale_feat_fusion(feats, feat_dim):
    if len(feats) != 1:
        feats = torch.mean(torch.stack(feats, dim=1), dim=1)
    else:
        feats = feats[0]

    B, L, C = feats.shape
    feats = F.adaptive_avg_pool1d(feats.reshape(B*L, 1, C), feat_dim)
    feats = feats.reshape((B, L, feat_dim))
    return feats


def anomaly_map_normalization(anomaly_maps, eps = 1e-4):
    min_value = (np.percentile(anomaly_maps, 1) + np.percentile(anomaly_maps, 0.5)) * 0.5
    max_value = (np.percentile(anomaly_maps, 99.5) + np.percentile(anomaly_maps, 99.9)) * 0.5
    anomaly_maps = np.clip((anomaly_maps - min_value) / (max_value - min_value + eps), a_min=0, a_max=1)
    return anomaly_maps


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]


def torch_quantile(
    tensor: Tensor,
    q: float | Tensor,
    dim: int | None = None,
    *,
    keepdim: bool = False,
    interpolation: str = "linear",
    out: Tensor | None = None,
) -> Tensor:
    r"""Improved ``torch.quantile`` for one scalar quantile.

    Arguments
    ---------
    tensor: ``Tensor``
        See ``torch.quantile``.
    q: ``float``
        See ``torch.quantile``. Supports only scalar values currently.
    dim: ``int``, optional
        See ``torch.quantile``.
    keepdim: ``bool``
        See ``torch.quantile``. Supports only ``False`` currently.
        Defaults to ``False``.
    interpolation: ``{"linear", "lower", "higher", "midpoint", "nearest"}``
        See ``torch.quantile``. Defaults to ``"linear"``.
    out: ``Tensor``, optional
        See ``torch.quantile``. Currently not supported.

    Notes
    -----
    Uses ``torch.kthvalue``. Better than ``torch.quantile`` since:

    #. it has no :math:`2^{24}` tensor `size limit <https://github.com/pytorch/pytorch/issues/64947#issuecomment-2304371451>`_;
    #. it is much faster, at least on big tensor sizes.

    """
    # Sanitization of: q
    q_float = float(q)  # May raise an (unpredictible) error
    if not 0 <= q_float <= 1:
        msg = f"Only values 0<=q<=1 are supported (got {q_float!r})"
        raise ValueError(msg)

    # Sanitization of: dim
    # Because one cannot pass  `dim=None` to `squeeze()` or `kthvalue()`
    if dim_was_none := dim is None:
        dim = 0
        tensor = tensor.reshape((-1, *(1,) * (tensor.ndim - 1)))

    # Sanitization of: inteporlation
    idx_float = q_float * (tensor.shape[dim] - 1)
    if interpolation == "nearest":
        idxs = [round(idx_float)]
    elif interpolation == "lower":
        idxs = [floor(idx_float)]
    elif interpolation == "higher":
        idxs = [ceil(idx_float)]
    elif interpolation in {"linear", "midpoint"}:
        low = floor(idx_float)
        idxs = [low] if idx_float == low else [low, low + 1]
        weight = idx_float - low if interpolation == "linear" else 0.5
    else:
        msg = (
            "Currently supported interpolations are {'linear', 'lower', 'higher', "
            f"'midpoint', 'nearest'}} (got {interpolation!r})"
        )
        raise ValueError(msg)

    # Sanitization of: out
    if out is not None:
        msg = f"Only None value is currently supported for out (got {out!r})"
        raise ValueError(msg)

    # Logic
    outs = [torch.kthvalue(tensor, idx + 1, dim, keepdim=True)[0] for idx in idxs]
    out = outs[0] if len(outs) == 1 else outs[0].lerp(outs[1], weight)

    # Rectification of: keepdim
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)
