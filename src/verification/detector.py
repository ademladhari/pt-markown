from __future__ import annotations

from typing import Dict

import torch

from src.watermark.tree_ring import TreeRingWatermark, WatermarkPayload


class WatermarkDetector:
    def __init__(self, watermark: TreeRingWatermark):
        self.watermark = watermark

    def score(self, z_t: torch.Tensor, payload: WatermarkPayload) -> torch.Tensor:
        return self.watermark.extract_score(z_t, payload)

    def classify(self, z_t: torch.Tensor, payload: WatermarkPayload, threshold: float) -> Dict[str, float]:
        score = float(self.score(z_t, payload).mean().detach().cpu())
        return {
            "score": score,
            "is_watermarked": float(score <= threshold),
        }
