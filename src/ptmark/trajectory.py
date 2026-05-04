from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from src.pipelines.base_diffusion import DiffusionCore
from src.watermark.tree_ring import TreeRingWatermark, WatermarkPayload


@dataclass
class TrajectoryBundle:
    z_star: List[torch.Tensor]
    z_hat: List[torch.Tensor]
    payload: WatermarkPayload


def build_watermarked_trajectory(
    core: DiffusionCore,
    prompt: str,
    z_star: List[torch.Tensor],
    watermark: TreeRingWatermark,
) -> TrajectoryBundle:
    z_star_t = z_star[0]
    payload = watermark.build_payload(z_star_t)
    z_hat_t = watermark.embed(z_star_t, payload)
    z_hat = core.sample_trajectory(prompt=prompt, z_t=z_hat_t, guidance_scale=core.cfg.guidance_scale)
    return TrajectoryBundle(z_star=z_star, z_hat=z_hat, payload=payload)
