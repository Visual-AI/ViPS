from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tvf


def load_video_frames(video_path: Path, num_frames: int = 32) -> list[Image.Image]:
    """Uniformly sample the same 32 frames used by the original extractor."""

    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("torch")
    reader = VideoReader(str(video_path), ctx=cpu(0))
    indices = np.linspace(0, len(reader) - 1, num_frames, dtype=int)
    frames = reader.get_batch(indices.tolist()).numpy()
    return [Image.fromarray(frame) for frame in frames]


def preprocess_pil_frames(frames: list[Image.Image]) -> torch.Tensor:
    """VGGT crop preprocessing for in-memory VSIBench video frames."""

    images = []
    target_size = 518
    for image in frames:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
        width, height = image.size
        new_height = round(height * (target_size / width) / 14) * 14
        image = image.resize((target_size, new_height), Image.Resampling.BICUBIC)
        tensor = tvf.to_tensor(image)
        if new_height > target_size:
            start = (new_height - target_size) // 2
            tensor = tensor[:, start : start + target_size, :]
        images.append(tensor)
    return torch.stack(images)
