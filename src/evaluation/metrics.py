from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def basic_image_metrics(ref: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    ref_np = ref.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    pred_np = pred.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    psnr = float(peak_signal_noise_ratio(ref_np, pred_np, data_range=1.0))
    ssim = float(structural_similarity(ref_np, pred_np, channel_axis=-1, data_range=1.0))
    return {"psnr": psnr, "ssim": ssim}
