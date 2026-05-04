from __future__ import annotations

from functools import lru_cache
from typing import Dict

import torch
import lpips as lpips_lib
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@lru_cache(maxsize=4)
def _lpips_model(device_name: str) -> lpips_lib.LPIPS:
    model = lpips_lib.LPIPS(net="alex")
    return model.to(device_name).eval()


def lpips_distance(ref: torch.Tensor, pred: torch.Tensor) -> float:
    device_name = str(ref.device)
    model = _lpips_model(device_name)
    ref_t = ref.detach().to(ref.device).clamp(0, 1) * 2 - 1
    pred_t = pred.detach().to(ref.device).clamp(0, 1) * 2 - 1
    with torch.inference_mode():
        score = model(ref_t, pred_t).mean()
    return float(score.detach().cpu())


def basic_image_metrics(ref: torch.Tensor, pred: torch.Tensor) -> Dict[str, float]:
    ref_np = ref.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    pred_np = pred.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    psnr = float(peak_signal_noise_ratio(ref_np, pred_np, data_range=1.0))
    ssim = float(structural_similarity(ref_np, pred_np, channel_axis=-1, data_range=1.0))
    lpips = lpips_distance(ref, pred)
    return {"psnr": psnr, "ssim": ssim, "lpips": lpips}
