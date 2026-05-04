from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from src.config import WatermarkConfig


@dataclass
class WatermarkPayload:
    mask: torch.Tensor
    message: torch.Tensor


class TreeRingWatermark:
    def __init__(self, cfg: WatermarkConfig):
        self.cfg = cfg

    def _build_ring_mask(self, h: int, w: int, device: torch.device) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij",
        )
        cy, cx = h // 2, w // 2
        dist = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
        r = self.cfg.ring_radius
        rw = self.cfg.ring_width
        mask = ((dist >= r - rw) & (dist <= r + rw)).float()
        return mask

    def build_payload(self, latent: torch.Tensor) -> WatermarkPayload:
        _, _, h, w = latent.shape
        device = latent.device
        mask = self._build_ring_mask(h, w, device)
        gen = torch.Generator(device=device).manual_seed(self.cfg.watermark_seed)
        msg = torch.randn((h, w), generator=gen, device=device, dtype=latent.dtype)
        return WatermarkPayload(mask=mask, message=msg)

    def embed(self, z_t: torch.Tensor, payload: WatermarkPayload) -> torch.Tensor:
        z_new = z_t.clone()
        c = self.cfg.channel_index
        freq = torch.fft.fft2(z_new[:, c, :, :])
        msg_complex = payload.message.to(freq.dtype)
        mask = payload.mask.to(freq.dtype)
        freq = (1 - mask) * freq + mask * msg_complex
        z_new[:, c, :, :] = torch.fft.ifft2(freq).real
        return z_new

    def extract_score(self, z_t: torch.Tensor, payload: WatermarkPayload) -> torch.Tensor:
        c = self.cfg.channel_index
        freq = torch.fft.fft2(z_t[:, c, :, :])
        diff = payload.mask * (freq.real - payload.message)
        return (diff.pow(2).sum(dim=(-1, -2)) / payload.mask.sum()).sqrt()
