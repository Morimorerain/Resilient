#!/usr/bin/env python3
"""Persistent SAM 3.1 filesystem service for the Python 3.12 environment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

SAM_SOURCE_COMMIT = "2345a4ad109ac29c569da749c91d84f10dc08c40"
SAM_CHECKPOINT_SHA256 = "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6"
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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_union(outputs: dict, height: int, width: int) -> np.ndarray:
    raw = outputs.get("out_binary_masks")
    if raw is None:
        return np.zeros((height, width), dtype=bool)
    array = np.asarray(raw)
    while array.ndim > 3 and array.shape[1] == 1:
        array = array[:, 0]
    if array.size == 0:
        return np.zeros((height, width), dtype=bool)
    if array.ndim == 2:
        array = array[None]
    if array.ndim != 3 or array.shape[-2:] != (height, width):
        raise ValueError(f"Unexpected SAM output mask shape: {array.shape}")
    return array.astype(bool).any(axis=0)


class Sam3Worker:
    """Load the upstream predictor once and serve deterministic short videos."""

    def __init__(
        self,
        *,
        checkpoint: Path,
        output_probability_threshold: float,
        compile_model: bool,
    ) -> None:
        if _sha256(checkpoint) != SAM_CHECKPOINT_SHA256:
            raise ValueError("SAM 3.1 checkpoint SHA-256 does not match the pinned asset.")
        from sam3.model_builder import build_sam3_predictor

        self.predictor = build_sam3_predictor(
            checkpoint_path=str(checkpoint),
            version="sam3.1",
            compile=compile_model,
            warm_up=compile_model,
            max_num_objects=16,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=True,
            async_loading_frames=False,
        )
        self.output_probability_threshold = float(output_probability_threshold)

    def segment_video(self, video: np.ndarray, prompt: str, frame_dir: Path) -> np.ndarray:
        """Run one semantic prompt in one isolated video session."""
        if video.ndim != 4 or video.shape[-1] != 3 or video.dtype != np.uint8:
            raise ValueError("SAM video input must be uint8 [T,H,W,3].")
        frame_dir.mkdir(parents=True)
        for index, frame in enumerate(video):
            Image.fromarray(frame).save(frame_dir / f"{index:05d}.jpg", quality=100)
        response = self.predictor.handle_request(
            {"type": "start_session", "resource_path": str(frame_dir)}
        )
        session_id = response["session_id"]
        masks = np.zeros(video.shape[:3], dtype=bool)
        try:
            initial = self.predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": prompt,
                    "output_prob_thresh": self.output_probability_threshold,
                }
            )
            masks[0] = _output_union(
                initial["outputs"], int(video.shape[1]), int(video.shape[2])
            )
            for item in self.predictor.handle_stream_request(
                {
                    "type": "propagate_in_video",
                    "session_id": session_id,
                    "propagation_direction": "forward",
                    "start_frame_index": 0,
                    "max_frame_num_to_track": int(video.shape[0]),
                    "output_prob_thresh": self.output_probability_threshold,
                    "evict_cached_frame_outputs": True,
                }
            ):
                frame_index = int(item["frame_index"])
                if 0 <= frame_index < video.shape[0]:
                    masks[frame_index] = _output_union(
                        item["outputs"], int(video.shape[1]), int(video.shape[2])
                    )
        finally:
            self.predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )
        return masks


def _process_request(worker: Sam3Worker, request_json: Path, queue_dir: Path) -> None:
    token = request_json.stem
    response_npz = queue_dir / "responses" / f"{token}.npz"
    response_json = queue_dir / "responses" / f"{token}.json"
    work_dir = queue_dir / "work" / token
    started = time.monotonic()
    try:
        request = json.loads(request_json.read_text(encoding="utf-8"))
        if int(request.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError("Request schema version mismatch.")
        expected_identity = f"sam3.1:{SAM_SOURCE_COMMIT}:{SAM_CHECKPOINT_SHA256}"
        if request.get("model_identity") != expected_identity:
            raise ValueError("Request model identity does not match the running service.")
        requested_threshold = float(request["output_probability_threshold"])
        if not np.isclose(requested_threshold, worker.output_probability_threshold):
            raise ValueError("Request threshold does not match the running service.")
        payload_path = queue_dir / "requests" / str(request["payload_file"])
        camera_names = tuple(str(value) for value in request["camera_names"])
        entities = tuple(request["entities"])
        with np.load(payload_path, allow_pickle=False) as payload:
            videos = [payload[f"camera_{index}"] for index in range(len(camera_names))]
        shape = videos[0].shape
        if any(video.shape != shape for video in videos):
            raise ValueError("Camera video shapes differ.")
        masks = np.zeros(
            (len(videos), len(entities), shape[0], shape[1], shape[2]),
            dtype=bool,
        )
        for camera_index, video in enumerate(videos):
            for entity_index, entity in enumerate(entities):
                frame_dir = work_dir / f"camera_{camera_index}" / f"entity_{entity_index}"
                masks[camera_index, entity_index] = worker.segment_video(
                    video.astype(np.uint8, copy=False), str(entity["prompt"]), frame_dir
                )
        flat = np.packbits(masks.reshape(-1))
        _atomic_npz(
            response_npz,
            packed_masks=flat,
            mask_shape=np.asarray(masks.shape, dtype=np.int64),
        )
        _atomic_json(
            response_json,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "ok",
                "request_id": request["request_id"],
                "model_identity": expected_identity,
                "output_probability_threshold": worker.output_probability_threshold,
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
            },
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        request_json.unlink(missing_ok=True)
        (queue_dir / "requests" / f"{token}.npz").unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-probability-threshold", type=float, default=0.5)
    parser.add_argument("--poll-seconds", type=float, default=0.1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.output_probability_threshold < 1.0:
        parser.error("--output-probability-threshold must be in (0,1).")
    for child in ("requests", "responses", "work"):
        (args.queue_dir / child).mkdir(parents=True, exist_ok=True)
    worker = Sam3Worker(
        checkpoint=args.checkpoint,
        output_probability_threshold=args.output_probability_threshold,
        compile_model=args.compile,
    )
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
