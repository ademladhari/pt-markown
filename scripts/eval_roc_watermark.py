#!/usr/bin/env python3
"""Estimate ROC AUC: PT-Mark and Tree-Ring baseline vs clean over multiple seeds.

Example:
  python scripts/eval_roc_watermark.py --prompt \"a fantasy innkeeper portrait\" --num-seeds 8 \\
      --out-json runs/roc_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ModelConfig, PTMarkConfig, WatermarkConfig
from src.evaluation.roc_metrics import format_roc_report, roc_watermark_vs_clean
from src.pipelines.base_diffusion import DiffusionCore
from src.ptmark.run_once import run_ptmark_once_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ROC watermark vs clean across random seeds.")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--num-seeds", type=int, default=8)
    p.add_argument("--seed-start", type=int, default=0, help="First seed offset; sample i uses seed_start + i.")
    p.add_argument("--out-json", type=str, default=None, help="Optional path to save JSON summary.")
    p.add_argument("--null-opt-steps", type=int, default=None)
    p.add_argument("--num-inference-steps", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    mc = ModelConfig()
    wc = WatermarkConfig()
    pc = PTMarkConfig()
    if args.null_opt_steps is not None:
        pc.null_opt_steps = args.null_opt_steps
    if args.num_inference_steps is not None:
        mc.num_inference_steps = args.num_inference_steps

    clean_scores: list[float] = []
    baseline_wm: list[float] = []
    pt_scores: list[float] = []

    summary: dict = {"prompt": args.prompt, "num_seeds": args.num_seeds, "methods": {}}

    # One load only: previously each seed re-called from_pretrained (~minutes per seed on Colab).
    print("Loading diffusion pipeline (once) ...", flush=True)
    core = DiffusionCore(mc)

    for i in range(args.num_seeds):
        sid = args.seed_start + i
        print(f"seed={sid} ...", flush=True)
        scores = run_ptmark_once_scores(args.prompt, sid, mc, wc, pc, core=core)
        clean_scores.append(scores.clean)
        baseline_wm.append(scores.tree_ring_baseline)
        pt_scores.append(scores.pt_mark)

    rb = roc_watermark_vs_clean(baseline_wm, clean_scores)
    summary["methods"]["tree_ring_baseline"] = {
        "auc": rb["auc"],
        "mean_watermarked": rb["mean_score_wm"],
        "mean_clean": rb["mean_score_clean"],
        "scores_watermarked": baseline_wm,
        "scores_clean": clean_scores,
    }
    print("\n--- Tree-Ring baseline vs clean ---")
    print(format_roc_report(rb))

    rpt = roc_watermark_vs_clean(pt_scores, clean_scores)
    summary["methods"]["pt_mark"] = {
        "auc": rpt["auc"],
        "mean_watermarked": rpt["mean_score_wm"],
        "mean_clean": rpt["mean_score_clean"],
        "scores_watermarked": pt_scores,
        "scores_clean": clean_scores,
    }
    print("\n--- PT-Mark vs clean ---")
    print(format_roc_report(rpt))

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSaved JSON summary to {out}")


if __name__ == "__main__":
    main()
