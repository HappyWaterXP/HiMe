"""Image utility functions for server-side processing.

Currently includes:
- combine_two_images_horizontally: horizontally concatenate two image files.
- combine_two_pil_horizontally: horizontally concatenate two PIL images.
"""

from __future__ import annotations

import os
from typing import Optional, Union, List
from PIL import Image
from dataclasses import dataclass

@dataclass
class RobotImageInput:
    """
    Canonical input structure for a robot image observation.

    - waist_image: optional waist camera image(s) (PIL or List[PIL])
    - image: required main camera image(s) (PIL or List[PIL])

    If lists are provided, they represent a temporal sequence (buffer).
    """
    # 允许单张图片（兼容 create_task）或 图片列表（兼容 step buffer）
    waist_image: Union[Image.Image, List[Image.Image], None]
    image: Union[Image.Image, List[Image.Image]]


def combine_two_images_horizontally(
    img_path_left: str,
    img_path_right: str,
    save_path: str,
) -> None:
    """
    Concatenate two images horizontally and save:

    - left:  img_path_left
    - right: img_path_right
    - result saved to save_path
    """
    img_left = Image.open(img_path_left).convert("RGB")
    img_right = Image.open(img_path_right).convert("RGB")

    h_left = img_left.height
    h_right = img_right.height
    target_height = max(h_left, h_right)

    def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        return img.resize((new_w, target_h), Image.BILINEAR)

    img_left = resize_to_height(img_left, target_height)
    img_right = resize_to_height(img_right, target_height)

    total_width = img_left.width + img_right.width
    combined = Image.new("RGB", (total_width, target_height))

    combined.paste(img_left, (0, 0))
    combined.paste(img_right, (img_left.width, 0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    combined.save(save_path)
    print(f"Saved combined image: {save_path}")


def combine_two_pil_horizontally(
    img_left: Image.Image,
    img_right: Image.Image,
) -> Image.Image:
    """
    Concatenate two PIL images horizontally and return the combined PIL image.

    The left image will be placed on the left, and the right on the right.
    Heights are aligned by resizing while preserving aspect ratio.
    """
    img_left = img_left.convert("RGB")
    img_right = img_right.convert("RGB")

    h_left = img_left.height
    h_right = img_right.height
    target_height = max(h_left, h_right)

    def resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
        if img.height == target_h:
            return img
        ratio = target_h / img.height
        new_w = int(img.width * ratio)
        return img.resize((new_w, target_h), Image.BILINEAR)

    img_left = resize_to_height(img_left, target_height)
    img_right = resize_to_height(img_right, target_height)

    total_width = img_left.width + img_right.width
    combined = Image.new("RGB", (total_width, target_height))
    combined.paste(img_left, (0, 0))
    combined.paste(img_right, (img_left.width, 0))
    return combined