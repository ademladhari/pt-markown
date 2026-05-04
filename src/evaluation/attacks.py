from __future__ import annotations

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    arr = image_tensor.detach().cpu().clamp(0, 1)[0].permute(1, 2, 0).numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def pil_to_tensor(image: Image.Image, device: str) -> torch.Tensor:
    arr = np.asarray(image).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def jpeg_attack(image: Image.Image, quality: int = 25) -> Image.Image:
    import io

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def crop_attack(image: Image.Image, keep_ratio: float = 0.75) -> Image.Image:
    w, h = image.size
    nw, nh = int(w * keep_ratio), int(h * keep_ratio)
    left = (w - nw) // 2
    top = (h - nh) // 2
    cropped = image.crop((left, top, left + nw, top + nh))
    return cropped.resize((w, h), Image.BICUBIC)


def blur_attack(image: Image.Image, radius: int = 4) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def noise_attack(image: Image.Image, intensity: float = 0.1) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, intensity, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 1)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def bright_attack(image: Image.Image, factor: float = 1.6) -> Image.Image:
    return ImageEnhance.Brightness(image).enhance(factor)


def rotate_attack(image: Image.Image, degrees: float = 75) -> Image.Image:
    return image.rotate(degrees, resample=Image.BICUBIC)
