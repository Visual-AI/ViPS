# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in
# vips/feature_extraction/vggt_multihead/LICENSE.

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as tvf


def load_and_preprocess_images(image_paths: list[str | Path]) -> torch.Tensor:
    """Apply the VGGT crop preprocessing used by the feature extractor."""

    if not image_paths:
        raise ValueError("At least one image is required")

    images = []
    shapes = set()
    target_size = 518
    for image_path in image_paths:
        image = Image.open(image_path)
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
        shapes.add(tuple(tensor.shape[1:]))
        images.append(tensor)

    if len(shapes) > 1:
        max_height = max(height for height, _ in shapes)
        max_width = max(width for _, width in shapes)
        padded = []
        for image in images:
            height_padding = max_height - image.shape[1]
            width_padding = max_width - image.shape[2]
            top = height_padding // 2
            left = width_padding // 2
            padded.append(
                torch.nn.functional.pad(
                    image,
                    (
                        left,
                        width_padding - left,
                        top,
                        height_padding - top,
                    ),
                    mode="constant",
                    value=1.0,
                )
            )
        images = padded
    return torch.stack(images)
