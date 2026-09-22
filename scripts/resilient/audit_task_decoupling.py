#!/usr/bin/env python3
"""Collect and analyze Gate-1 through Gate-3 task-decoupling evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party" / "LIBERO"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from libero.libero import benchmark, get_libero_path  # noqa: E402

from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _obs_to_model_input,
)
from experiments.libero.libero_utils import (  # noqa: E402
    LIBERO_ENV_RESOLUTION,
    capture_libero_observation,
    get_libero_dummy_action,
    get_libero_env,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from resilient.faults import build_fault_pipeline  # noqa: E402
from resilient.outcome_fpo.runtime import _denormalize_candidate  # noqa: E402
from resilient.task_decoupling import (  # noqa: E402
    TaskRegionDisentangler,
    load_entity_specification,
    validate_task_decoupling_config,
)
from resilient.task_decoupling.audit import (  # noqa: E402
    binary_mask_metrics,
    effective_rank,
    residual_cosine,
    ridge_probe,
    robot_contamination,
    temporal_mask_iou,
)
from resilient.task_decoupling.libero_oracle import (  # noqa: E402
    install_robosuite_numpy2_segmentation_compatibility,
    stack_preprocessed_oracle_masks,
)
from resilient.task_decoupling.postprocess import postprocess_masks  # noqa: E402
from resilient.task_decoupling.preprocessing import stack_camera_videos  # noqa: E402
from resilient.task_decoupling.prompts import boxes_from_binary_masks  # noqa: E402
from resilient.task_decoupling.providers.file_service import (  # noqa: E402
    FileServiceMaskProvider,
)
from resilient.task_decoupling.providers.oracle import ArrayOracleMaskProvider  # noqa: E402
from resilient.task_decoupling.types import MaskBatch  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _git_sha() -> str:
    """Return the exact source revision recorded with an ignored audit run."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _resolved_config_sha256(cfg: DictConfig) -> str:
    payload = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_processor(cfg: DictConfig) -> FastWAMProcessor:
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    statistics = load_dataset_stats_from_json(
        str(_project_path(cfg.two_stage_opsd.dataset_stats_path))
    )
    processor.set_normalizer_from_stats(statistics)
    return processor


def _state_episode(dataset_dir: Path, state_index: int) -> tuple[dict[str, Any], Path]:
    state_record = json.loads(
        (dataset_dir / "states" / f"state_{state_index:02d}.json").read_text(encoding="utf-8")
    )
    episode_record = state_record["episodes"][0]
    return episode_record, dataset_dir / episode_record["path"]


def _collect(cfg: DictConfig, output_dir: Path) -> None:
    install_robosuite_numpy2_segmentation_compatibility()
    processor = _load_processor(cfg)
    section = cfg.task_decoupling.audit
    entities = load_entity_specification(
        _project_path(cfg.task_decoupling.entities.config),
        expected_suite=str(cfg.two_stage_opsd.task.suite),
        expected_task_id=int(cfg.two_stage_opsd.task.task_id),
    )
    dataset_dir = _project_path(cfg.two_stage_opsd.stage1.dataset_dir)
    suite = benchmark.get_benchmark_dict()[str(cfg.two_stage_opsd.task.suite)]()
    task_id = int(cfg.two_stage_opsd.task.task_id)
    task = suite.get_task(task_id)
    state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(state_path, map_location="cpu", weights_only=False)
    anchors = [int(value) for value in section.anchor_steps]
    frame_steps = [int(value) for value in section.future_frame_steps]
    requested_steps = sorted({anchor + offset for anchor in anchors for offset in frame_steps})
    maximum_step = max(requested_steps)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    fault_config = OmegaConf.to_container(cfg.fault, resolve=True)
    for state_index_value in section.state_indices:
        state_index = int(state_index_value)
        episode_record, episode_path = _state_episode(dataset_dir, state_index)
        episode = torch.load(episode_path, map_location="cpu", weights_only=False)
        if episode["action"].shape[0] < maximum_step:
            raise ValueError(f"State {state_index} does not contain {maximum_step} actions.")
        raw_actions = _denormalize_candidate(
            episode["action"][:maximum_step], processor, binarize_gripper=True
        )
        env, _ = get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            int(episode_record["environment_seed"]),
            fault_pipeline=build_fault_pipeline(fault_config),
            fault_episode_index=int(episode_record["episode_index"]),
            segmentation=True,
        )
        try:
            env.reset()
            observation = env.set_init_state(initial_states[state_index])
            for _ in range(int(cfg.two_stage_opsd.stage1.collection.num_steps_wait)):
                observation, _, done, _ = env.step(get_libero_dummy_action())
                if done:
                    raise RuntimeError("Task terminated during audit warm-up.")
            reference, _, _ = _obs_to_model_input(
                observation,
                cfg,
                processor,
                448,
                224,
                "cpu",
                torch.float32,
            )
            reference_uint8 = reference[0].add(1).mul(127.5).round().to(torch.uint8)
            replay_max_difference = int(
                (reference_uint8.to(torch.int16) - episode["image"][0].to(torch.int16))
                .abs()
                .max()
            )
            observations = {0: observation}
            for step, action in enumerate(raw_actions, start=1):
                observation, _, _, _ = env.step(action)
                observation = capture_libero_observation(env)
                if step in requested_steps:
                    observations[step] = observation
            for anchor in anchors:
                selected = tuple(observations[anchor + offset] for offset in frame_steps)
                videos = stack_camera_videos(selected, processor)
                task_masks, robot_masks = stack_preprocessed_oracle_masks(
                    env, selected, entities
                )
                initial = selected[0]
                final = selected[-1]
                start_distance = np.mean(
                    [
                        np.linalg.norm(initial[f"{name}_pos"] - initial["basket_1_pos"])
                        for name in ("alphabet_soup_1", "cream_cheese_1")
                    ]
                )
                end_distance = np.mean(
                    [
                        np.linalg.norm(final[f"{name}_pos"] - final["basket_1_pos"])
                        for name in ("alphabet_soup_1", "cream_cheese_1")
                    ]
                )
                joint_motion = abs(
                    float(final["robot0_joint_pos"][0])
                    - float(initial["robot0_joint_pos"][0])
                )
                sample_id = f"state_{state_index:02d}_anchor_{anchor:03d}"
                sample_path = samples_dir / f"{sample_id}.npz"
                np.savez_compressed(
                    sample_path,
                    image=videos["image"],
                    wrist_image=videos["wrist_image"],
                    oracle_task_masks=task_masks,
                    oracle_robot_masks=robot_masks,
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "state_index": state_index,
                        "anchor_step": anchor,
                        "path": str(sample_path.relative_to(output_dir)),
                        "task_progress": float(start_distance - end_distance),
                        "joint1_motion": float(joint_motion),
                        "replay_max_pixel_difference": replay_max_difference,
                    }
                )
        finally:
            env.close()
    manifest = {
        "schema_version": 1,
        "suite": str(cfg.two_stage_opsd.task.suite),
        "task_id": task_id,
        "entities": [entity.__dict__ for entity in entities],
        "frame_steps": frame_steps,
        "anchors": anchors,
        "source_git_sha": _git_sha(),
        "resolved_config_sha256": _resolved_config_sha256(cfg),
        "records": records,
    }
    (output_dir / "collection_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Collected {len(records)} audit clips in {output_dir}")


def _shifted_masks(masks: np.ndarray, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    shifted = np.empty_like(masks)
    for camera in range(masks.shape[0]):
        for entity in range(masks.shape[1]):
            for frame in range(masks.shape[2]):
                dy = int(generator.integers(32, max(33, masks.shape[-2] - 31)))
                dx = int(generator.integers(32, max(33, masks.shape[-1] - 31)))
                shifted[camera, entity, frame] = np.roll(
                    masks[camera, entity, frame], shift=(dy, dx), axis=(0, 1)
                )
    return shifted


def _pooled_features(latent: torch.Tensor) -> np.ndarray:
    future = latent.float()[0]
    mean = future.mean(dim=(-2, -1))
    standard_deviation = future.std(dim=(-2, -1))
    return torch.cat([mean.flatten(), standard_deviation.flatten()]).numpy()


def _overlay(image: np.ndarray, mask: np.ndarray, *, label: str) -> Image.Image:
    """Render one deterministic mask overlay for manual Gate-1 inspection."""
    rgb = np.asarray(image, dtype=np.uint8).copy()
    selected = np.asarray(mask, dtype=bool)
    tint = np.asarray([255, 64, 32], dtype=np.float32)
    rgb[selected] = np.rint(0.55 * rgb[selected] + 0.45 * tint).astype(np.uint8)
    rendered = Image.fromarray(rgb)
    draw = ImageDraw.Draw(rendered)
    box = draw.textbbox((6, 6), label)
    draw.rectangle((3, 3, box[2] + 3, box[3] + 3), fill=(0, 0, 0))
    draw.text((6, 6), label, fill=(255, 255, 255))
    return rendered


def _save_mask_overlay(
    path: Path,
    videos: dict[str, np.ndarray],
    sam_union: np.ndarray,
    oracle_union: np.ndarray,
) -> None:
    """Save raw/SAM/oracle views for first, middle, and last frames of both cameras."""
    camera_names = ("image", "wrist_image")
    frame_indices = (0, sam_union.shape[1] // 2, sam_union.shape[1] - 1)
    width = int(videos[camera_names[0]].shape[2])
    height = int(videos[camera_names[0]].shape[1])
    canvas = Image.new("RGB", (3 * width, len(camera_names) * 3 * height))
    row = 0
    for camera_index, camera_name in enumerate(camera_names):
        for frame_index in frame_indices:
            image = videos[camera_name][frame_index]
            panels = (
                _overlay(
                    image,
                    np.zeros_like(sam_union[camera_index, frame_index]),
                    label=f"{camera_name} t={frame_index} raw",
                ),
                _overlay(
                    image,
                    sam_union[camera_index, frame_index],
                    label=f"{camera_name} t={frame_index} SAM",
                ),
                _overlay(
                    image,
                    oracle_union[camera_index, frame_index],
                    label=f"{camera_name} t={frame_index} oracle",
                ),
            )
            for column, panel in enumerate(panels):
                canvas.paste(panel, (column * width, row * height))
            row += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _pipeline(cfg: DictConfig, provider, entities) -> TaskRegionDisentangler:
    return TaskRegionDisentangler(
        mask_provider=provider,
        entities=entities,
        camera_order=("image", "wrist_image"),
        concat_mode=str(cfg.data.train.concat_multi_camera),
        minimum_component_area_px=int(
            cfg.task_decoupling.segmentation.minimum_component_area_px
        ),
        fill_hole_area_px=int(cfg.task_decoupling.segmentation.fill_hole_area_px),
        dilation_radius_px=int(cfg.task_decoupling.segmentation.dilation_radius_px),
        sigma_short_edge_ratio=float(
            cfg.task_decoupling.counterfactual.sigma_short_edge_ratio
        ),
    )


def _analyze(cfg: DictConfig, output_dir: Path) -> None:
    collection_dir = _project_path(
        cfg.task_decoupling.artifacts.get(
            "collection_dir", cfg.task_decoupling.artifacts.output_dir
        )
    )
    manifest = json.loads(
        (collection_dir / "collection_manifest.json").read_text(encoding="utf-8")
    )
    entities = load_entity_specification(
        _project_path(cfg.task_decoupling.entities.config),
        expected_suite=str(cfg.two_stage_opsd.task.suite),
        expected_task_id=int(cfg.two_stage_opsd.task.task_id),
    )
    provider = FileServiceMaskProvider(
        _project_path(cfg.task_decoupling.mask_provider.queue_dir),
        model_identity=str(cfg.task_decoupling.mask_provider.model_identity),
        output_probability_threshold=float(
            cfg.task_decoupling.segmentation.output_probability_threshold
        ),
        timeout_seconds=float(cfg.task_decoupling.mask_provider.timeout_seconds),
        poll_seconds=float(cfg.task_decoupling.mask_provider.poll_seconds),
        cache_enabled=bool(cfg.task_decoupling.mask_provider.cache_enabled),
        prompt_mode=str(cfg.task_decoupling.mask_provider.prompt_mode),
    )
    cfg.model.load_text_encoder = False
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
    _load_model_checkpoint(model, str(_project_path(cfg.ckpt)))
    model.eval().requires_grad_(False)
    rows: list[dict[str, Any]] = []
    sam_features: list[np.ndarray] = []
    full_features: list[np.ndarray] = []
    progress: list[float] = []
    joint_motion: list[float] = []
    state_indices: list[int] = []
    random_seed = int(cfg.task_decoupling.audit.random_seed)
    for sample_index, record in enumerate(manifest["records"]):
        with np.load(collection_dir / record["path"], allow_pickle=False) as sample:
            videos = {"image": sample["image"], "wrist_image": sample["wrist_image"]}
            oracle_masks = sample["oracle_task_masks"].astype(bool)
            robot_masks = sample["oracle_robot_masks"].astype(bool)
        prompt_hints = None
        if str(cfg.task_decoupling.mask_provider.type) == "samhq_file_service":
            prompt_hints = boxes_from_binary_masks(
                oracle_masks,
                camera_names=("image", "wrist_image"),
                entity_ids=tuple(entity.entity_id for entity in entities),
                source=str(cfg.task_decoupling.mask_provider.prompt_source),
                padding_px=int(cfg.task_decoupling.mask_provider.box_padding_px),
            )
        sam_result = _pipeline(cfg, provider, entities).extract(
            videos, model, prompt_hints=prompt_hints
        )
        oracle_provider = ArrayOracleMaskProvider(
            oracle_masks, ("image", "wrist_image")
        )
        oracle_result = _pipeline(cfg, oracle_provider, entities).extract(videos, model)
        random_provider = ArrayOracleMaskProvider(
            _shifted_masks(oracle_masks, random_seed + sample_index),
            ("image", "wrist_image"),
        )
        random_result = _pipeline(cfg, random_provider, entities).extract(videos, model)
        cleaned_oracle = postprocess_masks(
            MaskBatch(
                ("image", "wrist_image"),
                tuple(entity.entity_id for entity in entities),
                oracle_masks,
            ),
            minimum_component_area_px=int(
                cfg.task_decoupling.segmentation.minimum_component_area_px
            ),
            fill_hole_area_px=int(cfg.task_decoupling.segmentation.fill_hole_area_px),
            dilation_radius_px=int(cfg.task_decoupling.segmentation.dilation_radius_px),
        )
        overlay_frequency = int(cfg.task_decoupling.audit.overlay_every_n_samples)
        if bool(cfg.task_decoupling.audit.save_overlays) and (
            sample_index % overlay_frequency == 0
        ):
            _save_mask_overlay(
                output_dir / "overlays" / f"{record['sample_id']}.png",
                videos,
                sam_result.mask_batch.union,
                cleaned_oracle.union,
            )
        segmentation = binary_mask_metrics(sam_result.mask_batch.union, cleaned_oracle.union)
        entity_segmentation: dict[str, float] = {}
        for entity_index, entity in enumerate(entities):
            metrics = binary_mask_metrics(
                sam_result.mask_batch.masks[:, entity_index],
                cleaned_oracle.masks[:, entity_index],
            )
            entity_segmentation.update(
                {f"entity_{entity.entity_id}_{key}": value for key, value in metrics.items()}
            )
        temporal_values = [
            temporal_mask_iou(sam_result.mask_batch.union[camera])
            for camera in range(2)
        ]
        row = {
            **record,
            **{f"mask_{key}": value for key, value in segmentation.items()},
            **entity_segmentation,
            "temporal_iou": float(np.mean(temporal_values)),
            "robot_contamination": robot_contamination(
                sam_result.mask_batch.union, robot_masks
            ),
            "sam_oracle_residual_cosine": residual_cosine(
                sam_result.future_delta, oracle_result.future_delta
            ),
            "random_oracle_residual_cosine": residual_cosine(
                random_result.future_delta, oracle_result.future_delta
            ),
            **sam_result.mask_statistics,
        }
        rows.append(row)
        sam_features.append(_pooled_features(sam_result.future_delta))
        full_features.append(_pooled_features(sam_result.full_latent[:, :, 1:]))
        progress.append(float(record["task_progress"]))
        joint_motion.append(float(record["joint1_motion"]))
        state_indices.append(int(record["state_index"]))
        print(f"Analyzed {sample_index + 1}/{len(manifest['records'])}: {record['sample_id']}")

    sam_array = np.stack(sam_features)
    full_array = np.stack(full_features)
    progress_array = np.asarray(progress)
    joint_array = np.asarray(joint_motion)
    train_states = {int(value) for value in cfg.task_decoupling.audit.train_state_indices}
    test_states = {int(value) for value in cfg.task_decoupling.audit.test_state_indices}
    train_mask = np.asarray([value in train_states for value in state_indices])
    test_mask = np.asarray([value in test_states for value in state_indices])
    regularization = float(cfg.task_decoupling.audit.ridge_regularization)
    probes = {}
    for name, features in (("task_residual", sam_array), ("full_latent", full_array)):
        probes[f"{name}_task_progress_r2"] = ridge_probe(
            features[train_mask],
            progress_array[train_mask],
            features[test_mask],
            progress_array[test_mask],
            regularization=regularization,
        ).r2
        probes[f"{name}_joint1_motion_r2"] = ridge_probe(
            features[train_mask],
            joint_array[train_mask],
            features[test_mask],
            joint_array[test_mask],
            regularization=regularization,
        ).r2
    numeric_keys = [
        key for key, value in rows[0].items() if isinstance(value, int | float)
    ]
    aggregate = {key: float(np.mean([float(row[key]) for row in rows])) for key in numeric_keys}
    aggregate.update(probes)
    aggregate["task_residual_effective_rank"] = effective_rank(
        torch.from_numpy(sam_array)
    )
    aggregate["sample_count"] = len(rows)
    aggregate["minimum_entity_recall"] = min(
        aggregate[f"entity_{entity.entity_id}_recall"] for entity in entities
    )
    gates = cfg.task_decoupling.audit.gates
    aggregate["gate0_pass"] = bool(aggregate["replay_max_pixel_difference"] == 0.0)
    mask_quality_pass = bool(
        aggregate["mask_recall"] >= float(gates.minimum_union_recall)
        and aggregate["mask_iou"] >= float(gates.minimum_union_iou)
        and aggregate["minimum_entity_recall"] >= float(gates.minimum_entity_recall)
        and aggregate["robot_contamination"]
        <= float(gates.maximum_robot_contamination)
        and aggregate["temporal_iou"] >= float(gates.minimum_temporal_iou)
    )
    provider_audit_only = bool(cfg.task_decoupling.mask_provider.audit_only)
    aggregate["gate1_mask_quality_pass"] = mask_quality_pass
    aggregate["gate1_pass"] = None if provider_audit_only else mask_quality_pass
    # Gate 2 violations raise during construction, before any metric can be written.
    aggregate["gate2_pass"] = True
    residual_specificity_pass = bool(
        aggregate["sam_oracle_residual_cosine"]
        >= aggregate["random_oracle_residual_cosine"]
        + float(gates.residual_cosine_margin_over_shifted_control)
    )
    aggregate["gate3_oracle_prompted_upper_bound_pass"] = residual_specificity_pass
    aggregate["gate3_residual_specificity_pass"] = (
        None if provider_audit_only else residual_specificity_pass
    )
    aggregate["mask_provider_type"] = str(cfg.task_decoupling.mask_provider.type)
    aggregate["mask_provider_audit_only"] = provider_audit_only
    aggregate["prompt_mode"] = str(cfg.task_decoupling.mask_provider.prompt_mode)
    aggregate["prompt_source"] = cfg.task_decoupling.mask_provider.get(
        "prompt_source", "entity_text_then_video_tracking"
    )
    aggregate["analysis_git_sha"] = _git_sha()
    aggregate["collection_source_git_sha"] = manifest["source_git_sha"]
    aggregate["analysis_config_sha256"] = _resolved_config_sha256(cfg)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "per_clip_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(aggregate, indent=2))


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="two_stage_opsd/fastwam_libero10_task7_joint1_half",
)
def main(cfg: DictConfig) -> None:
    """Run the configured collection or analysis phase."""
    validate_task_decoupling_config(cfg.task_decoupling)
    output_dir = _project_path(cfg.task_decoupling.artifacts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    phase = str(cfg.task_decoupling.audit.phase)
    if phase == "collect":
        collection_dir = _project_path(
            cfg.task_decoupling.artifacts.get(
                "collection_dir", cfg.task_decoupling.artifacts.output_dir
            )
        )
        collection_dir.mkdir(parents=True, exist_ok=True)
        _collect(cfg, collection_dir)
    elif phase == "analyze":
        _analyze(cfg, output_dir)
    else:
        raise ValueError("task_decoupling.audit.phase must be collect or analyze.")


if __name__ == "__main__":
    main()
