from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import copy
import torch

from src.config import PTMarkConfig
from src.pipelines.base_diffusion import DiffusionCore
from src.ptmark.saliency import latent_difference_mask


@dataclass
class PTMarkResult:
    trajectory: List[torch.Tensor]
    null_embeddings: List[torch.Tensor]
    losses: Dict[str, List[float]]


class SemanticAwarePivotalTuning:
    def __init__(self, core: DiffusionCore, cfg: PTMarkConfig):
        self.core = core
        self.cfg = cfg

    def run(
        self,
        prompt: str,
        z_star: List[torch.Tensor],
        z_hat: List[torch.Tensor],
    ) -> PTMarkResult:
        cond, uncond_init = self.core.encode_prompt(prompt)
        cond = cond.clone()
        uncond_init = uncond_init.clone()
        scheduler = copy.deepcopy(self.core.pipe.scheduler)
        scheduler.set_timesteps(self.core.cfg.num_inference_steps, device=self.core.cfg.device)

        z_bar = [None] * len(z_hat)
        z_bar[0] = z_hat[0].clone()

        null_states: List[torch.Tensor] = [torch.zeros_like(uncond_init) for _ in range(len(z_hat))]
        null_states[0] = uncond_init.detach().clone()

        losses = {"semantic": [], "watermark": [], "total": []}

        for i, t in enumerate(scheduler.timesteps):
            z_cur = z_bar[i]
            z_star_prev = z_star[i + 1].detach().clone()
            z_hat_prev = z_hat[i + 1].detach().clone()
            null_emb = null_states[i].detach().clone().requires_grad_(True)
            optim = torch.optim.Adam([null_emb], lr=self.cfg.lr_null_text)

            for _ in range(self.cfg.null_opt_steps):
                z_pred_prev = self.core.denoise_step(
                    z_t=z_cur,
                    t=t,
                    cond_emb=cond,
                    uncond_emb=null_emb,
                    guidance_scale=self.core.cfg.guidance_scale,
                    scheduler=scheduler,
                )
                m = latent_difference_mask(z_hat_prev, z_star_prev)
                inv_m = 1.0 - m

                # Preserve semantics mostly outside watermark-salient regions.
                sem_num = (inv_m * (z_pred_prev - z_star_prev).pow(2)).sum()
                sem_den = inv_m.sum().clamp_min(1.0)
                l_sem = sem_num / sem_den

                # Preserve watermark inside salient regions with L1 objective.
                l_wm = (m * torch.abs(z_pred_prev - z_hat_prev)).sum(dim=(1, 2, 3)).mean()
                loss = self.cfg.lambda_semantic * l_sem + self.cfg.lambda_watermark * l_wm

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

            with torch.no_grad():
                z_next = self.core.denoise_step(
                    z_t=z_cur,
                    t=t,
                    cond_emb=cond,
                    uncond_emb=null_emb.detach(),
                    guidance_scale=self.core.cfg.guidance_scale,
                    scheduler=scheduler,
                )
            z_bar[i + 1] = z_next
            null_states[i + 1] = null_emb.detach().clone()
            losses["semantic"].append(float(l_sem.detach().cpu()))
            losses["watermark"].append(float(l_wm.detach().cpu()))
            losses["total"].append(float(loss.detach().cpu()))

        return PTMarkResult(
            trajectory=z_bar,
            null_embeddings=null_states,
            losses=losses,
        )
