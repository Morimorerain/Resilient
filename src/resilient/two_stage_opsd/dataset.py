"""Trajectory-backed Fast-WAM dataset without duplicated overlapping windows."""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


def _text_cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.t5_len{context_len}.wan22ti2v5b.pt"


class FaultRolloutWindowDataset(Dataset):
    """Read 32-action/9-frame windows from compact on-policy trajectory shards."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        text_embedding_cache_dir: str | Path,
        context_len: int = 128,
        cache_episodes: int = 8,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Unsupported Fault rollout dataset manifest schema.")
        self.frame_steps = tuple(int(value) for value in payload["frame_steps"])
        if self.frame_steps != tuple(range(0, 33, 4)):
            raise ValueError("Stage-I data must use frames 0,4,...,32.")
        self.windows = list(payload["windows"])
        if not self.windows:
            raise ValueError("Fault rollout manifest contains no training windows.")
        self.task_description = str(payload["task_description"])
        self.prompt = DEFAULT_PROMPT.format(task=self.task_description)
        if payload.get("text_context") is not None:
            text_path = self.root / str(payload["text_context"])
        else:
            text_path = _text_cache_path(
                Path(text_embedding_cache_dir).resolve(), self.prompt, int(context_len)
            )
        cached = torch.load(text_path, map_location="cpu", weights_only=False)
        self.context = cached["context"].to(dtype=torch.float32)
        cached_mask = cached["mask"].bool()
        if self.context.ndim != 2 or cached_mask.ndim != 1:
            raise ValueError("Cached Fast-WAM text tensors have invalid dimensions.")
        if self.context.shape[0] != int(context_len) or cached_mask.shape[0] != int(
            context_len
        ):
            raise ValueError("Cached Fast-WAM text length does not match context_len.")
        self.context[~cached_mask] = 0.0
        # Match RobotVideoDataset and released Fast-WAM behavior exactly.
        self.context_mask = torch.ones_like(cached_mask)
        self.cache_episodes = max(int(cache_episodes), 1)
        self._episode_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.windows)

    def _episode(self, relative_path: str) -> dict[str, Any]:
        cached = self._episode_cache.pop(relative_path, None)
        if cached is None:
            cached = torch.load(
                self.root / relative_path, map_location="cpu", weights_only=False
            )
        self._episode_cache[relative_path] = cached
        while len(self._episode_cache) > self.cache_episodes:
            self._episode_cache.popitem(last=False)
        return cached

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        window = self.windows[index]
        episode = self._episode(str(window["episode"]))
        start = int(window["start"])
        action = episode["action"][start : start + 32].float()
        proprio = episode["proprio"][start : start + 32].float()
        frame_indices = torch.tensor(
            [start + value for value in self.frame_steps], dtype=torch.long
        )
        video_uint8 = episode["image"].index_select(0, frame_indices)
        if action.shape != (32, 7) or proprio.shape != (32, 8):
            raise RuntimeError("Stored Fault trajectory has an invalid action/proprio window.")
        if video_uint8.shape[0] != 9:
            raise RuntimeError("Stored Fault trajectory has an invalid video window.")
        video = video_uint8.permute(1, 0, 2, 3).float().mul_(2.0 / 255.0).sub_(1.0)
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": self.prompt,
            "context": self.context,
            "context_mask": self.context_mask,
            "image_is_pad": torch.zeros(9, dtype=torch.bool),
            "action_is_pad": torch.zeros(32, dtype=torch.bool),
            "proprio_is_pad": torch.zeros(32, dtype=torch.bool),
        }
