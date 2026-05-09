"""Single-sample PT-Mark pipeline + watermark detector scores (for batch ROC)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.config import ModelConfig, PTMarkConfig, WatermarkConfig
from src.inversion.ddim import DDIMInverter
from src.pipelines.base_diffusion import DiffusionCore
from src.ptmark.trajectory import build_watermarked_trajectory
from src.ptmark.tuning import SemanticAwarePivotalTuning
from src.verification.detector import WatermarkDetector
from src.watermark.tree_ring import TreeRingWatermark


@dataclass
class PTMarkDetectionScores:
    clean: float
    tree_ring_baseline: float
    pt_mark: float


def detector_score_after_invert(
    core: DiffusionCore,
    inverter: DDIMInverter,
    detector: WatermarkDetector,
    prompt: str,
    rgb_0_1: torch.Tensor,
) -> float:
    """rgb_0_1 shape (1,3,H,W). Same protocol as notebooks/verify_single."""
    inp = rgb_0_1.to(core.cfg.device, dtype=core.pipe.unet.dtype)
    z_star = inverter.invert(prompt, inp)
    payload = detector.watermark.build_payload(z_star[0])
    return float(detector.score(z_star[0], payload).mean().detach().cpu())


def run_ptmark_once_scores(
    prompt: str,
    seed: int,
    model_cfg: ModelConfig,
    wm_cfg: WatermarkConfig,
    pt_cfg: PTMarkConfig,
    core: DiffusionCore | None = None,
) -> PTMarkDetectionScores:
    """If ``core`` is omitted, loads a new pipeline (slow). For batch ROC, pass a single shared ``core``."""
    torch.manual_seed(seed)
    if core is None:
        core = DiffusionCore(model_cfg)
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

    return PTMarkDetectionScores(clean=s_clean, tree_ring_baseline=s_base, pt_mark=s_pt)
