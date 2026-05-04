from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_id: str = "sd2-community/stable-diffusion-2-1-base"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    inversion_guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    dtype: str = "float16"
    device: str = "cuda"


@dataclass
class WatermarkConfig:
    ring_radius: int = 10
    ring_width: int = 2
    channel_index: int = 3
    watermark_seed: int = 1234


@dataclass
class PTMarkConfig:
    null_opt_steps: int = 10
    lambda_semantic: float = 1.5
    # Slightly above paper default 0.0007 offsets weaker frequency match when inversion is approximate,
    # improving detector separation vs clean while semantics stay aligned.
    lambda_watermark: float = 0.0011
    lr_null_text: float = 1e-2


@dataclass
class RunConfig:
    seed: int = 42
    output_dir: str = "runs"
