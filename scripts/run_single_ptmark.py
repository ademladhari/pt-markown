from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, PTMarkConfig, RunConfig, WatermarkConfig
from src.inversion.ddim import DDIMInverter
from src.pipelines.base_diffusion import DiffusionCore
from src.ptmark.trajectory import build_watermarked_trajectory
from src.ptmark.tuning import SemanticAwarePivotalTuning
from src.watermark.tree_ring import TreeRingWatermark


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PT-Mark on a single prompt.")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--outdir", type=str, default="runs/single", help="Output directory")
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=None,
        help="Override diffusion inference steps for lower memory or faster runs.",
    )
    p.add_argument(
        "--null-opt-steps",
        type=int,
        default=None,
        help="Override null-text optimization steps for lower memory or faster runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_cfg = ModelConfig()
    run_cfg = RunConfig(seed=args.seed, output_dir=args.outdir)
    wm_cfg = WatermarkConfig()
    pt_cfg = PTMarkConfig()
    if args.num_inference_steps is not None:
        model_cfg.num_inference_steps = args.num_inference_steps
    if args.null_opt_steps is not None:
        pt_cfg.null_opt_steps = args.null_opt_steps

    outdir = Path(run_cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(run_cfg.seed)

    core = DiffusionCore(model_cfg)
    inverter = DDIMInverter(core)
    watermark = TreeRingWatermark(wm_cfg)
    tuner = SemanticAwarePivotalTuning(core, pt_cfg)

    z_t = core.sample_initial_latent(seed=run_cfg.seed)
    clean_traj = core.sample_trajectory(args.prompt, z_t=z_t)
    clean_image = core.decode_latent(clean_traj[-1])
    save_image(clean_image, outdir / "clean.png")

    z_star = inverter.invert(args.prompt, clean_image)
    bundle = build_watermarked_trajectory(core, args.prompt, z_star, watermark)
    wm_image = core.decode_latent(bundle.z_hat[-1])
    save_image(wm_image, outdir / "tree_ring_baseline.png")

    pt_result = tuner.run(args.prompt, bundle.z_star, bundle.z_hat)
    pt_image = core.decode_latent(pt_result.trajectory[-1])
    save_image(pt_image, outdir / "pt_mark.png")

    with open(outdir / "losses.txt", "w", encoding="utf-8") as f:
        for i, (ls, lw, lt) in enumerate(
            zip(pt_result.losses["semantic"], pt_result.losses["watermark"], pt_result.losses["total"])
        ):
            f.write(f"step={i}, semantic={ls:.6f}, watermark={lw:.6f}, total={lt:.6f}\n")

    print(f"Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
