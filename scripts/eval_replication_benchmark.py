#!/usr/bin/env python3
"""Multi-prompt PT-Mark benchmark: fidelity vs clean + detector scores + pooled ROC/AUC.

Use this for a stronger "replication check" than a single prompt × few seeds:

  python scripts/eval_replication_benchmark.py \\
      --prompts-file data/replication_prompts.txt \\
      --out-json runs/replication_benchmark.json

One full pipeline load; one run per prompt (seed = seed-base + line index unless --same-seed).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, PTMarkConfig, WatermarkConfig
from src.evaluation.metrics import basic_image_metrics
from src.evaluation.roc_metrics import format_roc_report, roc_watermark_vs_clean
from src.inversion.ddim import DDIMInverter
from src.pipelines.base_diffusion import DiffusionCore
from src.ptmark.run_once import detector_score_after_invert
from src.ptmark.trajectory import build_watermarked_trajectory
from src.ptmark.tuning import SemanticAwarePivotalTuning
from src.verification.detector import WatermarkDetector
from src.watermark.tree_ring import TreeRingWatermark


def load_prompts(path: Path, limit: int | None = None) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
        if limit is not None and len(lines) >= limit:
            break
    if not lines:
        raise ValueError(f"No prompts found in {path}")
    return lines


def _agg(name: str, xs: list[float]) -> dict:
    out = {"mean": mean(xs)}
    if len(xs) > 1:
        out["std"] = stdev(xs)
    else:
        out["std"] = 0.0
    out["min"] = min(xs)
    out["max"] = max(xs)
    out["n"] = len(xs)
    return {name: out}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-prompt PT-Mark replication benchmark.")
    p.add_argument(
        "--prompts-file",
        type=str,
        default=str(ROOT / "data" / "replication_prompts.txt"),
        help="Text file: one prompt per line, # for comments.",
    )
    p.add_argument("--limit", type=int, default=None, help="Use only first N prompts (debug).")
    p.add_argument("--seed-base", type=int, default=1000, help="seed[i] = seed-base + i unless --same-seed.")
    p.add_argument(
        "--same-seed",
        action="store_true",
        help="Use seed-base for every prompt (controls for seed; less diversity).",
    )
    p.add_argument("--out-json", type=str, default="runs/replication_benchmark.json")
    p.add_argument("--null-opt-steps", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    return p.parse_args()


def run_one_with_images(
    core: DiffusionCore,
    prompt: str,
    seed: int,
    wm_cfg: WatermarkConfig,
    pt_cfg: PTMarkConfig,
) -> dict:
    """Same pipeline as run_ptmark_once_scores but returns images + per-image metrics."""
    torch.manual_seed(seed)
    inverter = DDIMInverter(core)
    watermark = TreeRingWatermark(wm_cfg)
    detector = WatermarkDetector(watermark)
    tuner = SemanticAwarePivotalTuning(core, pt_cfg)

    z_t = core.sample_initial_latent(seed=seed)
    clean_traj = core.sample_trajectory(prompt, z_t=z_t)
    clean_image = core.decode_latent(clean_traj[-1]).clamp(0, 1)

    z_star = inverter.invert(prompt, clean_image)
    bundle = build_watermarked_trajectory(core, prompt, z_star, watermark)
    wm_image = core.decode_latent(bundle.z_hat[-1]).clamp(0, 1)

    pt_result = tuner.run(prompt, bundle.z_star, bundle.z_hat)
    pt_image = core.decode_latent(pt_result.trajectory[-1]).clamp(0, 1)

    s_clean = detector_score_after_invert(core, inverter, detector, prompt, clean_image.float())
    s_base = detector_score_after_invert(core, inverter, detector, prompt, wm_image.float())
    s_pt = detector_score_after_invert(core, inverter, detector, prompt, pt_image.float())

    m_base = basic_image_metrics(clean_image, wm_image)
    m_pt = basic_image_metrics(clean_image, pt_image)

    return {
        "prompt": prompt,
        "seed": seed,
        "detector_clean": s_clean,
        "detector_tree_ring": s_base,
        "detector_pt_mark": s_pt,
        "psnr_clean_vs_tree_ring": m_base["psnr"],
        "ssim_clean_vs_tree_ring": m_base["ssim"],
        "lpips_clean_vs_tree_ring": m_base["lpips"],
        "psnr_clean_vs_pt_mark": m_pt["psnr"],
        "ssim_clean_vs_pt_mark": m_pt["ssim"],
        "lpips_clean_vs_pt_mark": m_pt["lpips"],
    }


def main() -> None:
    args = parse_args()
    prompts_path = Path(args.prompts_file)
    if not prompts_path.is_file():
        print(f"Missing prompts file: {prompts_path}", file=sys.stderr)
        sys.exit(1)

    prompts = load_prompts(prompts_path, limit=args.limit)
    mc = ModelConfig()
    wc = WatermarkConfig()
    pc = PTMarkConfig()
    if args.null_opt_steps is not None:
        pc.null_opt_steps = args.null_opt_steps
    if args.num_inference_steps is not None:
        mc.num_inference_steps = args.num_inference_steps

    print(f"Loading pipeline once … ({len(prompts)} prompts)", flush=True)
    core = DiffusionCore(mc)

    rows: list[dict] = []
    clean_d: list[float] = []
    base_d: list[float] = []
    pt_d: list[float] = []

    for i, prompt in enumerate(prompts):
        sid = args.seed_base if args.same_seed else args.seed_base + i
        print(f"[{i + 1}/{len(prompts)}] seed={sid} …", flush=True)
        row = run_one_with_images(core, prompt, sid, wc, pc)
        rows.append(row)
        clean_d.append(row["detector_clean"])
        base_d.append(row["detector_tree_ring"])
        pt_d.append(row["detector_pt_mark"])

    psnr_b = [r["psnr_clean_vs_tree_ring"] for r in rows]
    ssim_b = [r["ssim_clean_vs_tree_ring"] for r in rows]
    lpips_b = [r["lpips_clean_vs_tree_ring"] for r in rows]
    psnr_p = [r["psnr_clean_vs_pt_mark"] for r in rows]
    ssim_p = [r["ssim_clean_vs_pt_mark"] for r in rows]
    lpips_p = [r["lpips_clean_vs_pt_mark"] for r in rows]

    roc_b = roc_watermark_vs_clean(base_d, clean_d)
    roc_p = roc_watermark_vs_clean(pt_d, clean_d)

    aggregate = {}
    aggregate.update(_agg("psnr_clean_vs_tree_ring", psnr_b))
    aggregate.update(_agg("ssim_clean_vs_tree_ring", ssim_b))
    aggregate.update(_agg("lpips_clean_vs_tree_ring", lpips_b))
    aggregate.update(_agg("psnr_clean_vs_pt_mark", psnr_p))
    aggregate.update(_agg("ssim_clean_vs_pt_mark", ssim_p))
    aggregate.update(_agg("lpips_clean_vs_pt_mark", lpips_p))
    aggregate.update(_agg("detector_clean", clean_d))
    aggregate.update(_agg("detector_tree_ring", base_d))
    aggregate.update(_agg("detector_pt_mark", pt_d))

    paper_diffusion_db_ref = {
        "tree_ring_mean_psnr_approx": 15.18,
        "tree_ring_mean_ssim_approx": 0.56,
        "pt_mark_mean_psnr_approx": 28.18,
        "pt_mark_mean_ssim_approx": 0.94,
        "pt_mark_mean_lpips_approx": 0.03,
        "note": "Paper Table 1 (DiffusionDB). Your sweep is fewer samples; compare means qualitatively.",
    }

    summary = {
        "prompts_file": str(prompts_path.resolve()),
        "n_prompts": len(prompts),
        "seed_base": args.seed_base,
        "same_seed": args.same_seed,
        "samples": rows,
        "aggregate": aggregate,
        "roc_tree_ring_vs_clean": {
            "auc": roc_b["auc"],
            "report": format_roc_report(roc_b),
        },
        "roc_pt_mark_vs_clean": {
            "auc": roc_p["auc"],
            "report": format_roc_report(roc_p),
        },
        "paper_reference_approx_diffusion_db": paper_diffusion_db_ref,
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["aggregate"], indent=2))
    print("\n--- Tree-Ring vs clean (pooled ROC) ---\n", format_roc_report(roc_b))
    print("\n--- PT-Mark vs clean (pooled ROC) ---\n", format_roc_report(roc_p))
    print(f"\nWrote {out.resolve()}")
    print(
        "\nInterpretation for a thesis snippet: cite mean ± std PSNR/SSIM/LPIPS vs clean "
        "and pooled AUC; note n prompts and compare directionally to paper Table 1 (DiffusionDB)."
    )


if __name__ == "__main__":
    main()
