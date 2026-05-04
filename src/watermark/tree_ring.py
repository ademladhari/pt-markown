from __future__ import annotations

from dataclasses import dataclass

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
        mask = (dist >= r - rw) & (dist <= r + rw)
        return mask

    def build_payload(self, latent: torch.Tensor) -> WatermarkPayload:
        _, _, h, w = latent.shape
        device = latent.device
        mask = self._build_ring_mask(h, w, device)
        gen = torch.Generator(device=device).manual_seed(self.cfg.watermark_seed)
        real = torch.randn((h, w), generator=gen, device=device, dtype=torch.float32)
        imag = torch.randn((h, w), generator=gen, device=device, dtype=torch.float32)
        message = torch.complex(real, imag) * mask.to(dtype=torch.float32)
        return WatermarkPayload(mask=mask, message=message)

    def embed(self, z_t: torch.Tensor, payload: WatermarkPayload) -> torch.Tensor:
        z_new = torch.nan_to_num(z_t.float()).clone()
        c = self.cfg.channel_index
        freq = torch.fft.fftshift(torch.fft.fft2(z_new[:, c, :, :]), dim=(-2, -1))
        mask = payload.mask.to(device=freq.device)
        message = payload.message.to(device=freq.device, dtype=freq.dtype)
        freq = freq * (~mask).to(dtype=freq.dtype) + message
        z_new[:, c, :, :] = torch.fft.ifft2(torch.fft.ifftshift(freq, dim=(-2, -1))).real
        return torch.nan_to_num(z_new.to(dtype=z_t.dtype))

    def extract_score(self, z_t: torch.Tensor, payload: WatermarkPayload) -> torch.Tensor:
        c = self.cfg.channel_index
        freq = torch.fft.fftshift(torch.fft.fft2(torch.nan_to_num(z_t.float())[:, c, :, :]), dim=(-2, -1))
        mask = payload.mask.to(device=freq.device)
        message = payload.message.to(device=freq.device, dtype=freq.dtype)
        diff = mask.to(dtype=freq.dtype) * (freq - message)
        denom = mask.float().sum().clamp_min(1.0)
        return (diff.abs().pow(2).sum(dim=(-1, -2)) / denom).sqrt()
