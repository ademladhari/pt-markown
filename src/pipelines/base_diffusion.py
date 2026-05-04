from __future__ import annotations

from dataclasses import asdict
from typing import List, Tuple

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline

from src.config import ModelConfig


class DiffusionCore:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        self.pipe = StableDiffusionPipeline.from_pretrained(
            cfg.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(cfg.device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.unet.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.unet.eval()
        self.pipe.vae.eval()
        self.pipe.text_encoder.eval()
        self.pipe.enable_attention_slicing()

    def encode_prompt(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.pipe.tokenizer
        text_encoder = self.pipe.text_encoder
        max_len = tokenizer.model_max_length

        cond_inputs = tokenizer(
            [prompt],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.cfg.device)
        uncond_inputs = tokenizer(
            [""],
            padding="max_length",
            max_length=max_len,
            truncation=True,
            return_tensors="pt",
        ).to(self.cfg.device)
        with torch.inference_mode():
            cond = text_encoder(cond_inputs.input_ids)[0]
            uncond = text_encoder(uncond_inputs.input_ids)[0]
        return cond, uncond

    def sample_initial_latent(self, batch_size: int = 1, seed: int = 0) -> torch.Tensor:
        generator = torch.Generator(device=self.cfg.device).manual_seed(seed)
        h = self.cfg.height // 8
        w = self.cfg.width // 8
        return torch.randn(
            (batch_size, self.pipe.unet.config.in_channels, h, w),
            generator=generator,
            device=self.cfg.device,
            dtype=self.pipe.unet.dtype,
        )

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            latent = latent / self.pipe.vae.config.scaling_factor
            image = self.pipe.vae.decode(latent).sample
            image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def denoise_step(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        uncond_emb: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        latent_input = torch.cat([z_t, z_t], dim=0)
        embeds = torch.cat([uncond_emb, cond_emb], dim=0)
        noise = self.pipe.unet(latent_input, t, encoder_hidden_states=embeds).sample
        noise_uncond, noise_cond = noise.chunk(2)
        noise_guided = (noise_uncond + guidance_scale * (noise_cond - noise_uncond)).float()
        prev = self.pipe.scheduler.step(noise_guided, t, z_t.float()).prev_sample
        return prev.to(dtype=z_t.dtype)

    def sample_trajectory(
        self,
        prompt: str,
        z_t: torch.Tensor,
        guidance_scale: float | None = None,
    ) -> List[torch.Tensor]:
        g = guidance_scale if guidance_scale is not None else self.cfg.guidance_scale
        with torch.inference_mode():
            cond, uncond = self.encode_prompt(prompt)
            self.pipe.scheduler.set_timesteps(self.cfg.num_inference_steps, device=self.cfg.device)
            trajectory = [z_t.clone()]
            cur = z_t
            for t in self.pipe.scheduler.timesteps:
                cur = self.denoise_step(cur, t, cond, uncond, g)
                trajectory.append(cur.clone())
        return trajectory
