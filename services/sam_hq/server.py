#!/usr/bin/env python3
"""Persistent HQ-SAM filesystem service for audit-only box-prompt masks."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

PACKAGE_VERSION = "0.3"
CHECKPOINT_SHA256 = "a7ac14a085326d9fa6199c8c698c4f0e7280afdbb974d2c4660ec60877b45e35"
MODEL_IDENTITY = f"samhq:{PACKAGE_VERSION}:{CHECKPOINT_SHA256}"
PROMPT_MODE = "box_per_frame"
SCHEMA_VERSION = 1


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SamHqWorker:
    """Load the frozen official HQ-SAM ViT-H predictor once."""

    def __init__(self, checkpoint: Path, *, device: str) -> None:
        if _sha256(checkpoint) != CHECKPOINT_SHA256:
            raise ValueError("HQ-SAM checkpoint SHA-256 does not match the pinned asset.")
        installed = importlib.metadata.version("segment-anything-hq")
        if installed != PACKAGE_VERSION:
            raise ValueError(
                f"Expected segment-anything-hq {PACKAGE_VERSION}, found {installed}."
            )
        from segment_anything_hq import SamPredictor, sam_model_registry

        model = sam_model_registry["vit_h"](checkpoint=str(checkpoint))
        model.to(device=device)
        model.eval()
        self.predictor = SamPredictor(model)

    def segment(
        self,
        videos: list[np.ndarray],
        boxes_xyxy: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Segment every valid camera/entity/frame box without temporal tracking."""
        shape = videos[0].shape
        camera_count, entity_count, frame_count = boxes_xyxy.shape[:3]
        masks = np.zeros(
            (camera_count, entity_count, frame_count, shape[1], shape[2]), dtype=bool
        )
        scores = np.full((camera_count, entity_count, frame_count), np.nan, dtype=np.float32)
        for camera_index, video in enumerate(videos):
            for frame_index, image in enumerate(video):
                self.predictor.set_image(image.astype(np.uint8, copy=False))
                for entity_index in range(entity_count):
                    if not valid[camera_index, entity_index, frame_index]:
                        continue
                    box = boxes_xyxy[camera_index, entity_index, frame_index]
                    predicted, quality, _ = self.predictor.predict(
                        box=box,
                        multimask_output=False,
                        hq_token_only=False,
                    )
                    masks[camera_index, entity_index, frame_index] = predicted[0].astype(bool)
                    scores[camera_index, entity_index, frame_index] = float(quality[0])
        return masks, scores


def _process_request(worker: SamHqWorker, request_json: Path, queue_dir: Path) -> None:
    token = request_json.stem
    response_npz = queue_dir / "responses" / f"{token}.npz"
    response_json = queue_dir / "responses" / f"{token}.json"
    started = time.monotonic()
    try:
        request = json.loads(request_json.read_text(encoding="utf-8"))
        if int(request.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Request schema version mismatch.")
        if request.get("model_identity") != MODEL_IDENTITY:
            raise ValueError("Request model identity does not match the running service.")
        if request.get("prompt_mode") != PROMPT_MODE:
            raise ValueError("HQ-SAM requires per-frame box prompts.")
        if not np.isclose(float(request["output_probability_threshold"]), 0.5):
            raise ValueError("HQ-SAM binary mask threshold is fixed at probability 0.5.")
        payload_path = queue_dir / "requests" / str(request["payload_file"])
        camera_names = tuple(str(value) for value in request["camera_names"])
        entities = tuple(request["entities"])
        with np.load(payload_path, allow_pickle=False) as payload:
            videos = [payload[f"camera_{index}"] for index in range(len(camera_names))]
            boxes_xyxy = payload["prompt_boxes_xyxy"].astype(np.float32, copy=False)
            valid = payload["prompt_valid"].astype(bool, copy=False)
        shape = videos[0].shape
        if any(video.shape != shape for video in videos):
            raise ValueError("Camera video shapes differ.")
        expected = (len(videos), len(entities), shape[0])
        if boxes_xyxy.shape != (*expected, 4) or valid.shape != expected:
            raise ValueError("Prompt boxes are not aligned with videos and entities.")
        masks, scores = worker.segment(videos, boxes_xyxy, valid)
        _atomic_npz(
            response_npz,
            packed_masks=np.packbits(masks.reshape(-1)),
            mask_shape=np.asarray(masks.shape, dtype=np.int64),
            scores=scores,
        )
        _atomic_json(
            response_json,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "request_id": request["request_id"],
                "model_identity": MODEL_IDENTITY,
                "output_probability_threshold": 0.5,
                "prompt_mode": PROMPT_MODE,
                "prompt_source": request.get("prompt_source"),
                "duration_seconds": time.monotonic() - started,
                "camera_names": list(camera_names),
                "entity_ids": [str(item["id"]) for item in entities],
            },
        )
    except Exception as error:
        _atomic_npz(
            response_npz,
            packed_masks=np.zeros(0, dtype=np.uint8),
            mask_shape=np.zeros(5, dtype=np.int64),
        )
        _atomic_json(
            response_json,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "model_identity": MODEL_IDENTITY,
                "output_probability_threshold": 0.5,
                "prompt_mode": PROMPT_MODE,
            },
        )
    finally:
        request_json.unlink(missing_ok=True)
        (queue_dir / "requests" / f"{token}.npz").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for child in ("requests", "responses"):
        (args.queue_dir / child).mkdir(parents=True, exist_ok=True)
    worker = SamHqWorker(args.checkpoint, device=args.device)
    while True:
        requests = sorted((args.queue_dir / "requests").glob("*.json"))
        if requests:
            _process_request(worker, requests[0], args.queue_dir)
            if args.once:
                return
        elif args.once:
            return
        else:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
