from __future__ import annotations

from typing import List

import torch
from diffusers import DDIMScheduler

from src.pipelines.base_diffusion import DiffusionCore


class DDIMInverter:
    """
    Lightweight inversion helper.
    This implementation is a practical approximation for thesis prototyping.
    """

    def __init__(self, core: DiffusionCore):
        self.core = core

    def image_to_latent(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            image_tensor = torch.nan_to_num(image_tensor.float() * 2 - 1).to(dtype=self.core.pipe.vae.dtype)
            posterior = self.core.pipe.vae.encode(image_tensor).latent_dist
            latent = posterior.sample() * self.core.pipe.vae.config.scaling_factor
            return torch.nan_to_num(latent).to(dtype=self.core.pipe.unet.dtype)

    def invert(self, prompt: str, image_tensor: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns an approximate original trajectory z*_0..z*_T.
        """
        with torch.inference_mode():
            z0 = self.image_to_latent(image_tensor).to(self.core.cfg.device, dtype=self.core.pipe.unet.dtype)
            cond, uncond = self.core.encode_prompt(prompt)
            scheduler = DDIMScheduler.from_config(self.core.pipe.scheduler.config)
            scheduler.set_timesteps(self.core.cfg.num_inference_steps, device=self.core.cfg.device)

            # Reverse DDIM-like pass: produce increasingly noisy states.
            trajectory = [z0.clone()]
            cur = z0
            for t in reversed(scheduler.timesteps):
                t_idx = int(t.item()) if hasattr(t, "item") else int(t)
                latent_input = torch.cat([torch.nan_to_num(cur), torch.nan_to_num(cur)], dim=0).to(dtype=self.core.pipe.unet.dtype)
                embeds = torch.cat([uncond, cond], dim=0).to(dtype=self.core.pipe.unet.dtype)
                noise = self.core.pipe.unet(latent_input, t, encoder_hidden_states=embeds).sample
                noise_uncond, noise_cond = noise.chunk(2)
                noise_guided = (noise_uncond + self.core.cfg.inversion_guidance_scale * (noise_cond - noise_uncond)).float()
                alpha_t = scheduler.alphas_cumprod[t_idx].float()
                first_idx = int(scheduler.timesteps[0].item()) if hasattr(scheduler.timesteps[0], "item") else int(scheduler.timesteps[0])
                next_idx = max(t_idx - 1, 0)
                next_alpha = alpha_t if t_idx == first_idx else scheduler.alphas_cumprod[next_idx].float()
                cur_fp32 = cur.float()
                cur_fp32 = (cur_fp32 - (1 - alpha_t).sqrt() * noise_guided) / alpha_t.sqrt()
                cur_fp32 = next_alpha.sqrt() * cur_fp32 + (1 - next_alpha).sqrt() * noise_guided
                cur = torch.nan_to_num(cur_fp32).to(dtype=z0.dtype)
                trajectory.append(cur.clone())
            trajectory.reverse()
            return trajectory
