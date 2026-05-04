# PT-Mark Thesis Implementation (Starter)

This repository now contains a runnable starter implementation of PT-Mark:

- Stable Diffusion v2.1 base pipeline wrapper
- DDIM-style inversion helper
- Tree-Ring watermark embedding in latent Fourier space
- Semantic-aware pivotal tuning loop with:
  - semantic loss (MSE to original trajectory)
  - watermark preservation loss (L1 in salient regions)
- Single-sample generation and verification scripts

## Install

```bash
pip install -r requirements.txt
```

## Run single prompt

```bash
python scripts/run_single_ptmark.py --prompt "A fantasy innkeeper portrait" --seed 42 --outdir runs/single
```

Outputs:
- `clean.png`
- `tree_ring_baseline.png`
- `pt_mark.png`
- `losses.txt`

## Verify one image

```bash
python scripts/verify_single.py --prompt "A fantasy innkeeper portrait" --image runs/single/pt_mark.png
```

## Notes

- This is an implementation scaffold oriented for thesis experimentation.
- DDIM inversion and saliency are intentionally lightweight first-pass implementations and should be refined for strict reproduction-quality results.
- Next recommended step is adding batch evaluation scripts for AUC/PSNR/SSIM and perturbation robustness (JPEG/crop/blur/noise/brightness/rotation).
