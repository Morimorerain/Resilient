"""Annotated four-panel visual-fault comparisons."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_name in ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def annotate_image(frame: Any, lines: list[str]) -> Image.Image:
    """Add a readable top-left label without changing panel dimensions."""
    panel = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(panel)
    font = _font(max(12, panel.height // 22))
    wrapped_lines = [
        segment
        for line in lines
        for segment in (textwrap.wrap(str(line), width=34) or [""])
    ]
    text = "\n".join(wrapped_lines)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=3)
    padding = 6
    draw.rectangle(
        (0, 0, box[2] - box[0] + 2 * padding, box[3] - box[1] + 2 * padding),
        fill=(0, 0, 0),
    )
    draw.multiline_text(
        (padding, padding), text, fill=(255, 255, 255), font=font, spacing=3
    )
    return panel


def compose_visual_fault_grid(
    nominal_images: Mapping[str, Any],
    faulted_images: Mapping[str, Any],
    *,
    fault_label: str,
    title: str = "Visual Fault comparison",
) -> np.ndarray:
    """Compose nominal/faulted agent and wrist views in an aligned 2 x 2 grid."""
    keys = ("image", "wrist_image")
    if any(key not in nominal_images or key not in faulted_images for key in keys):
        raise KeyError("Visual comparison requires image and wrist_image entries.")
    arrays = [np.asarray(nominal_images[key]) for key in keys]
    arrays.extend(np.asarray(faulted_images[key]) for key in keys)
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("All visual-fault panels must have identical shapes.")
    height, width = arrays[0].shape[:2]
    title_height = 44
    canvas = Image.new("RGB", (width * 2, height * 2 + title_height), color=(16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill=(255, 255, 255), font=_font(20))
    fault_lines = [part.strip() for part in fault_label.split("|")]
    panels = (
        annotate_image(arrays[0], ["NOMINAL", "third-person / agentview"]),
        annotate_image(arrays[1], ["NOMINAL", "wrist / eye-in-hand"]),
        annotate_image(arrays[2], ["FAULT", "third-person / agentview", *fault_lines]),
        annotate_image(arrays[3], ["FAULT", "wrist / eye-in-hand", *fault_lines]),
    )
    positions = (
        (0, title_height),
        (width, title_height),
        (0, title_height + height),
        (width, title_height + height),
    )
    for panel, position in zip(panels, positions, strict=True):
        canvas.paste(panel, position)
    return np.asarray(canvas)


def compose_multi_camera_fault_grid(
    nominal_images: Mapping[str, Any],
    faulted_images: Mapping[str, Any],
    *,
    camera_order: list[str] | tuple[str, ...],
    fault_label: str,
    title: str = "Multi-camera visual Fault comparison",
    panel_size: tuple[int, int] = (384, 288),
) -> np.ndarray:
    """Compose nominal and faulted rows for an ordered multi-camera simulator."""
    cameras = tuple(str(camera) for camera in camera_order)
    if not cameras or len(set(cameras)) != len(cameras):
        raise ValueError("camera_order must contain unique camera names.")
    missing = [
        camera
        for camera in cameras
        if camera not in nominal_images or camera not in faulted_images
    ]
    if missing:
        raise KeyError(f"Visual comparison is missing cameras: {missing}.")
    width, height = (int(panel_size[0]), int(panel_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("panel_size must contain positive width and height.")
    title_height = 52
    canvas = Image.new(
        "RGB",
        (width * len(cameras), height * 2 + title_height),
        color=(16, 16, 16),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255), font=_font(20))
    fault_lines = [part.strip() for part in fault_label.split("|")]
    for column, camera in enumerate(cameras):
        nominal = Image.fromarray(np.asarray(nominal_images[camera], dtype=np.uint8)).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        faulted = Image.fromarray(np.asarray(faulted_images[camera], dtype=np.uint8)).resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        nominal_panel = annotate_image(nominal, ["NOMINAL", camera])
        faulted_panel = annotate_image(faulted, ["FAULT", camera, *fault_lines])
        canvas.paste(nominal_panel, (column * width, title_height))
        canvas.paste(faulted_panel, (column * width, title_height + height))
    return np.asarray(canvas)
