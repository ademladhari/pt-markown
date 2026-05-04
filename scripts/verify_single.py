from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, WatermarkConfig
from src.inversion.ddim import DDIMInverter
from src.pipelines.base_diffusion import DiffusionCore
from src.verification.detector import WatermarkDetector
from src.watermark.tree_ring import TreeRingWatermark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify watermark score for one image.")
    p.add_argument("--prompt", required=True, help="Prompt used during generation")
    p.add_argument("--image", required=True, help="Path to candidate image")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    core = DiffusionCore(ModelConfig())
    inverter = DDIMInverter(core)
    watermark = TreeRingWatermark(WatermarkConfig())
    detector = WatermarkDetector(watermark)

    image = Image.open(args.image).convert("RGB")
    tensor = ToTensor()(image).unsqueeze(0).to(core.cfg.device, dtype=core.pipe.unet.dtype)
    z_star = inverter.invert(args.prompt, tensor)
    payload = watermark.build_payload(z_star[0])
    score = detector.score(z_star[0], payload)
    print(f"Watermark score: {float(score.mean().cpu()):.6f}")


if __name__ == "__main__":
    main()
