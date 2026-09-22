"""Dependency-free filesystem RPC client for an isolated SAM 3.1 service."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np

from ..types import EntitySpec, MaskBatch

SCHEMA_VERSION = 1


def _content_hash(
    videos: dict[str, np.ndarray],
    entities: tuple[EntitySpec, ...],
    model_identity: str,
    output_probability_threshold: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(model_identity.encode())
    digest.update(repr(output_probability_threshold).encode())
    for name, video in videos.items():
        digest.update(name.encode())
        array = np.ascontiguousarray(video)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    for entity in entities:
        digest.update(entity.entity_id.encode())
        digest.update(entity.prompt.encode())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


class FileServiceMaskProvider:
    """Exchange validated arrays with SAM without sharing Python dependencies."""

    def __init__(
        self,
        queue_dir: str | Path,
        *,
        model_identity: str,
        output_probability_threshold: float = 0.5,
        timeout_seconds: float = 300.0,
        poll_seconds: float = 0.1,
        cache_enabled: bool = True,
    ) -> None:
        self.queue_dir = Path(queue_dir)
        self.model_identity = str(model_identity)
        self.output_probability_threshold = float(output_probability_threshold)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.cache_enabled = bool(cache_enabled)
        if self.timeout_seconds <= 0 or self.poll_seconds <= 0:
            raise ValueError("Service timeout and polling interval must be positive.")
        if not 0.0 < self.output_probability_threshold < 1.0:
            raise ValueError("SAM output probability threshold must be in (0,1).")
        for child in ("requests", "responses", "cache"):
            (self.queue_dir / child).mkdir(parents=True, exist_ok=True)

    def segment(
        self,
        videos: dict[str, np.ndarray],
        entities: tuple[EntitySpec, ...],
    ) -> MaskBatch:
        camera_names = tuple(videos)
        if not camera_names or not entities:
            raise ValueError("SAM requests require cameras and entities.")
        shapes = {np.asarray(video).shape for video in videos.values()}
        if len(shapes) != 1:
            raise ValueError("All camera videos must have the same shape.")
        video_shape = next(iter(shapes))
        if len(video_shape) != 4 or video_shape[-1] != 3:
            raise ValueError("Camera videos must have shape [T,H,W,3].")

        request_id = _content_hash(
            videos,
            entities,
            self.model_identity,
            self.output_probability_threshold,
        )
        cache_npz = self.queue_dir / "cache" / f"{request_id}.npz"
        cache_json = self.queue_dir / "cache" / f"{request_id}.json"
        if self.cache_enabled and cache_npz.is_file() and cache_json.is_file():
            return self._read_response(cache_npz, cache_json, camera_names, entities)

        token = f"{request_id}-{uuid.uuid4().hex}"
        request_npz = self.queue_dir / "requests" / f"{token}.npz"
        request_json = self.queue_dir / "requests" / f"{token}.json"
        response_npz = self.queue_dir / "responses" / f"{token}.npz"
        response_json = self.queue_dir / "responses" / f"{token}.json"
        _atomic_npz(
            request_npz,
            **{f"camera_{index}": videos[name] for index, name in enumerate(camera_names)},
        )
        _atomic_json(
            request_json,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "payload_file": request_npz.name,
                "camera_names": list(camera_names),
                "entities": [
                    {"id": entity.entity_id, "prompt": entity.prompt} for entity in entities
                ],
                "model_identity": self.model_identity,
                "output_probability_threshold": self.output_probability_threshold,
            },
        )

        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if response_npz.is_file() and response_json.is_file():
                result = self._read_response(
                    response_npz, response_json, camera_names, entities
                )
                if self.cache_enabled:
                    if not cache_npz.exists():
                        os.replace(response_npz, cache_npz)
                        os.replace(response_json, cache_json)
                    else:
                        response_npz.unlink(missing_ok=True)
                        response_json.unlink(missing_ok=True)
                request_npz.unlink(missing_ok=True)
                request_json.unlink(missing_ok=True)
                return result
            time.sleep(self.poll_seconds)
        raise TimeoutError(f"SAM service timed out for request {request_id}.")

    def _read_response(
        self,
        npz_path: Path,
        json_path: Path,
        camera_names: tuple[str, ...],
        entities: tuple[EntitySpec, ...],
    ) -> MaskBatch:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("SAM response schema version mismatch.")
        if metadata.get("status") != "ok":
            raise RuntimeError(f"SAM service failed: {metadata.get('error', 'unknown error')}")
        if metadata.get("model_identity") != self.model_identity:
            raise ValueError("SAM response model identity mismatch.")
        if not np.isclose(
            float(metadata.get("output_probability_threshold", -1.0)),
            self.output_probability_threshold,
        ):
            raise ValueError("SAM response probability threshold mismatch.")
        with np.load(npz_path, allow_pickle=False) as data:
            packed = data["packed_masks"]
            shape = tuple(int(value) for value in data["mask_shape"])
            scores = data["scores"] if "scores" in data.files else None
        masks = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
        expected = (len(camera_names), len(entities))
        if masks.shape[:2] != expected:
            raise ValueError("SAM response camera/entity dimensions do not match the request.")
        return MaskBatch(
            camera_names=camera_names,
            entity_ids=tuple(entity.entity_id for entity in entities),
            masks=masks,
            scores=scores,
            metadata=metadata,
        )
