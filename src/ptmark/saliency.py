from __future__ import annotations

import torch


def latent_difference_mask(
    z_watermarked_prev: torch.Tensor,
    z_original_prev: torch.Tensor,
    threshold_quantile: float = 0.8,
) -> torch.Tensor:
    """
    Simple saliency prior from latent differences.
    """
    diff = (z_watermarked_prev - z_original_prev).abs().mean(dim=1, keepdim=True)
    flat = diff.flatten(start_dim=1).float()
    q = torch.quantile(flat, threshold_quantile, dim=1, keepdim=True)
    q = q.view(-1, 1, 1, 1)
    mask = (diff >= q).float()
    return mask
