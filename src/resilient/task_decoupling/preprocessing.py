"""LIBERO camera preprocessing shared by baseline and decoupling paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from experiments.libero.libero_utils import get_libero_image


def center_crop_resize(image: np.ndarray, *, width: int, height: int) -> np.ndarray:
    """Resize the shortest side and center crop with the baseline interpolation."""
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize(
        (round(src_w * scale), round(src_h * scale)),
        resample=Image.BILINEAR,
    )
    resized_w, resized_h = resized.size
    left = max((resized_w - width) // 2, 0)
    top = max((resized_h - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.ascontiguousarray(np.asarray(cropped, dtype=np.uint8))


def preprocess_libero_cameras(obs: Mapping, processor) -> dict[str, np.ndarray]:
    """Return separately resized uint8 camera images in Fast-WAM camera order."""
    images = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    num_cameras = int(processor.num_output_cameras)
    if len(image_meta) < num_cameras or num_cameras not in {1, 2}:
        raise ValueError("LIBERO task decoupling supports one or two configured cameras.")

    def target_size(index: int) -> tuple[int, int]:
        shape = image_meta[index]["shape"]
        if len(shape) != 3:
            raise ValueError(f"Camera shape must be [C,H,W], got {shape}.")
        return int(shape[2]), int(shape[1])

    primary_width, primary_height = target_size(0)
    result = {
        "image": center_crop_resize(
            images["image"], width=primary_width, height=primary_height
        )
    }
    if num_cameras == 2:
        wrist_width, wrist_height = target_size(1)
        result["wrist_image"] = center_crop_resize(
            images["wrist_image"], width=wrist_width, height=wrist_height
        )
    return result


def concatenate_cameras(
    cameras: Mapping[str, np.ndarray],
    *,
    camera_order: Sequence[str],
    mode: str,
) -> np.ndarray:
    """Concatenate preprocessed cameras using the released Fast-WAM layout."""
    arrays = [np.asarray(cameras[name]) for name in camera_order]
    if not arrays:
        raise ValueError("At least one camera is required.")
    if any(array.ndim != 3 or array.shape[-1] != 3 for array in arrays):
        raise ValueError("Every camera image must have shape [H,W,3].")
    if len(arrays) == 1:
        return np.ascontiguousarray(arrays[0])
    if mode == "horizontal":
        return np.ascontiguousarray(np.concatenate(arrays, axis=1))
    if mode == "vertical":
        return np.ascontiguousarray(np.concatenate(arrays, axis=0))
    raise ValueError(f"Invalid concat_multi_camera mode: {mode}")


def rgb_to_model_tensor(
    rgb: np.ndarray,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Normalize uint8 HWC RGB to one Fast-WAM image tensor."""
    tensor = torch.as_tensor(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device=device, dtype=dtype)
    return tensor * (2.0 / 255.0) - 1.0


def stack_camera_videos(
    observations: Sequence[Mapping],
    processor,
) -> dict[str, np.ndarray]:
    """Build camera-separated videos shaped [T,H,W,3]."""
    videos: dict[str, list[np.ndarray]] = {}
    for observation in observations:
        for name, image in preprocess_libero_cameras(observation, processor).items():
            videos.setdefault(name, []).append(image)
    if not videos or any(len(frames) != len(observations) for frames in videos.values()):
        raise RuntimeError("Camera observations are incomplete or inconsistent.")
    return {
        name: np.stack(frames, axis=0).astype(np.uint8, copy=False)
        for name, frames in videos.items()
    }


def camera_videos_to_model_video(
    videos: Mapping[str, np.ndarray],
    *,
    camera_order: Sequence[str],
    mode: str,
) -> torch.Tensor:
    """Return a normalized video [1,C,T,H,W] after per-frame camera concat."""
    lengths = {int(np.asarray(video).shape[0]) for video in videos.values()}
    if len(lengths) != 1:
        raise ValueError("Camera videos must contain the same number of frames.")
    frames = [
        concatenate_cameras(
            {name: videos[name][index] for name in camera_order},
            camera_order=camera_order,
            mode=mode,
        )
        for index in range(next(iter(lengths)))
    ]
    array = np.stack(frames, axis=0)
    tensor = torch.from_numpy(array).permute(3, 0, 1, 2).unsqueeze(0).float()
    return tensor * (2.0 / 255.0) - 1.0
