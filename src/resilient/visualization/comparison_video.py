"""Time-aligned side-by-side frames for reusable fault demonstrations."""

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


def _as_rgb(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Comparison frame must have shape [H, W, 3], got {array.shape}.")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _telemetry_lines(values: Mapping[str, Any]) -> list[str]:
    lines = []
    for key, value in values.items():
        if isinstance(value, float):
            lines.append(f"{key}: {value:.3f}")
        else:
            lines.append(f"{key}: {value}")
    return lines


def compose_comparison_frame(
    clean_frame: Any,
    fault_frame: Any,
    *,
    title: str,
    clean_label: str,
    fault_label: str,
    clean_telemetry: Mapping[str, Any],
    fault_telemetry: Mapping[str, Any],
    timestamp_seconds: float,
    panel_scale: int = 2,
) -> np.ndarray:
    """Compose two equal-sized RGB frames with synchronized telemetry overlays."""
    clean = _as_rgb(clean_frame)
    fault = _as_rgb(fault_frame)
    if clean.shape != fault.shape:
        raise ValueError(f"Clean and fault frames differ: {clean.shape} versus {fault.shape}.")
    if panel_scale <= 0:
        raise ValueError("panel_scale must be positive.")

    height, width, _ = clean.shape
    if panel_scale != 1:
        width *= panel_scale
        height *= panel_scale
        clean = np.asarray(
            Image.fromarray(clean).resize((width, height), resample=Image.Resampling.BILINEAR)
        )
        fault = np.asarray(
            Image.fromarray(fault).resize((width, height), resample=Image.Resampling.BILINEAR)
        )
    header_height = 76
    canvas = Image.new("RGB", (width * 2, height + header_height), color=(18, 18, 18))
    canvas.paste(Image.fromarray(clean), (0, header_height))
    canvas.paste(Image.fromarray(fault), (width, header_height))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(18)
    body_font = _font(14)
    small_font = _font(12)

    draw.text((12, 8), title, fill=(255, 255, 255), font=title_font)
    timestamp = f"t = {timestamp_seconds:05.2f} s"
    timestamp_box = draw.textbbox((0, 0), timestamp, font=title_font)
    timestamp_width = timestamp_box[2] - timestamp_box[0]
    draw.text(
        (width * 2 - timestamp_width - 12, 8),
        timestamp,
        fill=(255, 220, 90),
        font=title_font,
    )
    draw.text((12, 38), clean_label, fill=(100, 235, 145), font=body_font)
    wrapped_fault_label = "\n".join(textwrap.wrap(fault_label, width=52))
    draw.multiline_text(
        (width + 12, 38),
        wrapped_fault_label,
        fill=(255, 120, 110),
        font=small_font,
        spacing=1,
    )
    draw.line((width, 32, width, height + header_height), fill=(255, 255, 255), width=2)

    def add_overlay(x_offset: int, values: Mapping[str, Any]) -> None:
        lines = _telemetry_lines(values)
        box_top = header_height + 8
        box_height = 8 + len(lines) * 18
        draw.rectangle(
            (x_offset + 8, box_top, x_offset + min(width - 8, 250), box_top + box_height),
            fill=(0, 0, 0),
        )
        for line_index, line in enumerate(lines):
            draw.text(
                (x_offset + 14, box_top + 4 + line_index * 18),
                line,
                fill=(255, 255, 255),
                font=small_font,
            )

    add_overlay(0, clean_telemetry)
    add_overlay(width, fault_telemetry)
    return np.asarray(canvas)
