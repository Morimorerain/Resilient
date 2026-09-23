#!/usr/bin/env python3
"""Run the held-out Gate-4 task-decoupled outcome-reward audit."""

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
from accelerate import Accelerator
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for import_root in (PROJECT_ROOT, PROJECT_ROOT / "src", PROJECT_ROOT / "third_party/LIBERO"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from libero.libero import benchmark, get_libero_path  # noqa: E402

from experiments.libero.eval_libero_single import (  # noqa: E402
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import (  # noqa: E402
    FastWAMProcessor,
)
from fastwam.datasets.lerobot.utils.normalizer import (  # noqa: E402
    load_dataset_stats_from_json,
)
from resilient.faults import build_fault_pipeline  # noqa: E402
from resilient.opsd.adapters import (  # noqa: E402
    FastWAMLoraConfig,
    inject_fastwam_aligned_lora,
    load_adapter_state_dict,
)
from resilient.opsd.runtime import RolloutDescriptor, _derive_seed  # noqa: E402
from resilient.outcome_fpo.collector import (  # noqa: E402
    collect_action_chunk_future,
    restore_shadow_state,
)
from resilient.outcome_fpo.outcome import FrozenNominalOutcomeTeacher  # noqa: E402
from resilient.outcome_fpo.policy import sample_action_chunk  # noqa: E402
from resilient.outcome_fpo.runtime import (  # noqa: E402
    _advance_to_anchor,
    _denormalize_candidate,
    _execute_warmup,
    _make_conditioning,
    _training_state_records,
)
from resilient.state_banks import state_fingerprint  # noqa: E402
from resilient.task_decoupling import (  # noqa: E402
    TaskRegionDisentangler,
    load_entity_specification,
)
from resilient.task_decoupling.gate4 import (  # noqa: E402
    evaluate_gate4,
    summarize_gate4_rows,
)
from resilient.task_decoupling.libero_oracle import (  # noqa: E402
    install_robosuite_numpy2_segmentation_compatibility,
    stack_preprocessed_oracle_masks,
)
from resilient.task_decoupling.preprocessing import stack_camera_videos  # noqa: E402
from resilient.task_decoupling.prompts import boxes_from_binary_masks  # noqa: E402
from resilient.task_decoupling.providers.file_service import (  # noqa: E402
    FileServiceMaskProvider,
)
from resilient.task_decoupling.reward import task_outcome_rewards  # noqa: E402


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sha256_config(cfg: DictConfig) -> str:
    return hashlib.sha256(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _provider(cfg: DictConfig) -> FileServiceMaskProvider:
    section = cfg.task_decoupling
    if str(section.mask_provider.type) != "samhq_file_service":
        raise ValueError("Gate 4 currently requires the audit-only HQ-SAM box backend.")
    return FileServiceMaskProvider(
        _project_path(section.mask_provider.queue_dir),
        model_identity=str(section.mask_provider.model_identity),
        output_probability_threshold=float(section.segmentation.output_probability_threshold),
        timeout_seconds=float(section.mask_provider.timeout_seconds),
        poll_seconds=float(section.mask_provider.poll_seconds),
        cache_enabled=bool(section.mask_provider.cache_enabled),
        prompt_mode=str(section.mask_provider.prompt_mode),
    )


def _disentangler(cfg: DictConfig, provider, entities) -> TaskRegionDisentangler:
    section = cfg.task_decoupling
    return TaskRegionDisentangler(
        mask_provider=provider,
        entities=entities,
        camera_order=("image", "wrist_image"),
        concat_mode=str(cfg.data.train.concat_multi_camera),
        minimum_component_area_px=int(section.segmentation.minimum_component_area_px),
        fill_hole_area_px=int(section.segmentation.fill_hole_area_px),
        dilation_radius_px=int(section.segmentation.dilation_radius_px),
        sigma_short_edge_ratio=float(section.counterfactual.sigma_short_edge_ratio),
    )


def _task_progress(initial: dict[str, Any], final: dict[str, Any], cfg: DictConfig) -> float:
    goal = str(cfg.task_reward_audit.labels.goal_position_key)
    objects = [str(value) for value in cfg.task_reward_audit.labels.manipulated_position_keys]
    start = np.mean([np.linalg.norm(initial[key] - initial[goal]) for key in objects])
    end = np.mean([np.linalg.norm(final[key] - final[goal]) for key in objects])
    return float(start - end)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_summary(cfg: DictConfig, output_dir: Path, rows: list[dict[str, Any]]) -> None:
    section = cfg.task_reward_audit
    dev = summarize_gate4_rows(
        rows,
        state_indices=section.development_state_indices,
        minimum_reward_std=float(section.gates.minimum_reward_std),
        progress_tolerance=float(section.gates.progress_tolerance),
    )
    heldout = summarize_gate4_rows(
        rows,
        state_indices=section.heldout_state_indices,
        minimum_reward_std=float(section.gates.minimum_reward_std),
        progress_tolerance=float(section.gates.progress_tolerance),
    )
    decision = evaluate_gate4(
        heldout, OmegaConf.to_container(section.gates, resolve=True)
    )
    summary = {
        "schema_version": 1,
        "method": "one_sided_task_region_outcome_reward_audit",
        "formal_training_reward_changed": False,
        "teacher_task_region_decomposed": False,
        "boundary": (
            "The realized candidate is task-decomposed; the nominal Teacher target "
            "remains full-frame because temporary HQ-SAM requires oracle box prompts."
        ),
        "source_git_sha": _git_sha(),
        "resolved_config_sha256": _sha256_config(cfg),
        "development": dev,
        "heldout": heldout,
        "gate4": decision,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_finite_json(summary), indent=2) + "\n", encoding="utf-8"
    )
    columns = list(rows[0]) if rows else []
    with (output_dir / "candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _run(cfg: DictConfig) -> None:
    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    section = cfg.task_reward_audit
    requested_processes = int(section.num_processes)
    if accelerator.num_processes != requested_processes:
        raise ValueError("Accelerate world size does not match task_reward_audit.num_processes.")
    install_robosuite_numpy2_segmentation_compatibility()
    output_dir = _project_path(section.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    accelerator.wait_for_everyone()

    dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=dtype, device=str(accelerator.device))
    _load_model_checkpoint(model, str(_project_path(cfg.ckpt)))
    adapter_cfg = FastWAMLoraConfig.from_config(OmegaConf.to_container(cfg.adapter, resolve=True))
    adapter_audit = inject_fastwam_aligned_lora(model, adapter_cfg)
    adapter_path = _project_path(cfg.two_stage_opsd.stage2.initial_adapter)
    load_adapter_state_dict(
        model, torch.load(adapter_path, map_location="cpu", weights_only=True)
    )
    model.requires_grad_(False)
    model.eval()

    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    statistics = load_dataset_stats_from_json(
        str(_project_path(cfg.outcome_fpo.dataset_stats_path))
    )
    processor.set_normalizer_from_stats(statistics)
    video_height, video_width = (int(value) for value in cfg.data.train.video_size)
    entities = load_entity_specification(
        _project_path(cfg.task_decoupling.entities.config),
        expected_suite=str(cfg.two_stage_opsd.task.suite),
        expected_task_id=int(cfg.two_stage_opsd.task.task_id),
    )
    provider = _provider(cfg)
    disentangler = _disentangler(cfg, provider, entities)
    teacher = FrozenNominalOutcomeTeacher(
        model,
        layer_index=int(cfg.outcome_fpo.outcome.layer_index),
        reward_timestep=float(cfg.outcome_fpo.outcome.reward_timestep),
    )

    suite_name = str(cfg.two_stage_opsd.task.suite)
    task_id = int(cfg.two_stage_opsd.task.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    state_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    initial_states = torch.load(state_path, map_location="cpu", weights_only=False)
    training_records = _training_state_records(
        _project_path(cfg.outcome_fpo.split.training_reference_manifest)
    )
    states = [int(value) for value in section.state_indices]
    anchors = [int(value) for value in section.anchor_steps]
    descriptors: list[RolloutDescriptor] = []
    state_anchor_pairs = [
        (state_index, anchor) for state_index in states for anchor in anchors
    ]
    for sample_id, (state_index, anchor_step) in enumerate(state_anchor_pairs):
        identity = ("gate4", state_index, anchor_step)
        descriptors.append(
            RolloutDescriptor(
                suite=suite_name,
                task_id=task_id,
                rollout_id=state_index,
                initial_state_index=state_index,
                environment_seed=_derive_seed(int(section.environment_seed), *identity),
                inference_seed=_derive_seed(int(section.inference_seed), *identity),
                sample_id=sample_id,
                anchor_index=anchors.index(anchor_step),
                anchor_step=anchor_step,
            )
        )

    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    rows: list[dict[str, Any]] = []
    local_descriptors = descriptors[accelerator.process_index :: accelerator.num_processes]
    for local_index, descriptor in enumerate(local_descriptors):
        state = initial_states[descriptor.initial_state_index]
        record_key = (suite_name, task_id, descriptor.initial_state_index)
        expected = training_records.get(record_key)
        if expected is None or state_fingerprint(state) != expected["state_sha256"]:
            raise ValueError(f"Training state manifest mismatch: {record_key}")
        episode_index = 900000 + descriptor.sample_id
        environments = [
            get_libero_env(
                task,
                LIBERO_ENV_RESOLUTION,
                descriptor.environment_seed,
                fault_pipeline=build_fault_pipeline(fault_cfg),
                fault_episode_index=episode_index,
                segmentation=True,
            )[0]
            for _ in range(2)
        ]
        main_env, shadow_env = environments
        try:
            main_env.reset()
            observation = main_env.set_init_state(state)
            warmed = _execute_warmup(main_env, int(cfg.outcome_fpo.rollout.num_steps_wait))
            if warmed is not None:
                observation = warmed
            observation, reached_step, terminal = _advance_to_anchor(
                env=main_env,
                initial_state=state,
                initial_observation=observation,
                descriptor=descriptor,
                task_description=task.language,
                cfg=cfg,
                processor=processor,
                model=model,
                video_height=video_height,
                video_width=video_width,
            )
            conditioning = _make_conditioning(
                observation,
                task_description=task.language,
                cfg=cfg,
                processor=processor,
                model=model,
                video_height=video_height,
                video_width=video_width,
            )
            simulator_state = main_env.get_sim_state()
            fault_state = main_env.fault_runtime_state_dict()
            target = teacher.build_target(
                conditioning,
                num_video_frames=int(cfg.outcome_fpo.outcome.num_video_frames),
                num_inference_steps=int(cfg.outcome_fpo.outcome.num_inference_steps),
                sigma_shift=None,
                prediction_seed=_derive_seed(descriptor.inference_seed, "gate4-teacher"),
                reward_noise_seed=_derive_seed(descriptor.inference_seed, "gate4-noise"),
                rand_device=str(cfg.outcome_fpo.rollout.rand_device),
            )
            group_id = (
                f"state_{descriptor.initial_state_index:02d}_"
                f"anchor_{descriptor.anchor_step:03d}"
            )
            for candidate_index in range(int(section.group_size)):
                candidate_seed = _derive_seed(
                    descriptor.inference_seed, candidate_index, "gate4-candidate"
                )
                normalized_action = sample_action_chunk(
                    model,
                    conditioning,
                    action_horizon=int(cfg.outcome_fpo.rollout.action_horizon),
                    num_inference_steps=int(cfg.outcome_fpo.rollout.num_inference_steps),
                    sigma_shift=None,
                    seed=candidate_seed,
                    rand_device=str(cfg.outcome_fpo.rollout.rand_device),
                    compile_action_infer=bool(cfg.outcome_fpo.rollout.compile_action_infer),
                )
                action = _denormalize_candidate(
                    normalized_action,
                    processor,
                    binarize_gripper=bool(cfg.outcome_fpo.rollout.binarize_gripper),
                )
                shadow_observation = restore_shadow_state(
                    shadow_env,
                    simulator_state=simulator_state,
                    fault_runtime_state=fault_state,
                )
                future = collect_action_chunk_future(
                    shadow_env,
                    shadow_observation,
                    action,
                    frame_steps=cfg.outcome_fpo.rollout.future_frame_steps,
                )
                videos = stack_camera_videos(future.observations, processor)
                oracle_masks, _ = stack_preprocessed_oracle_masks(
                    shadow_env, future.observations, entities
                )
                prompts = boxes_from_binary_masks(
                    oracle_masks,
                    camera_names=("image", "wrist_image"),
                    entity_ids=tuple(entity.entity_id for entity in entities),
                    source=str(cfg.task_decoupling.mask_provider.prompt_source),
                    padding_px=int(cfg.task_decoupling.mask_provider.box_padding_px),
                )
                residual = disentangler.extract(videos, model, prompt_hints=prompts)
                full_hidden = teacher.encode_realized_latent(
                    residual.full_latent, conditioning, target
                )
                counterfactual_hidden = teacher.encode_realized_latent(
                    residual.counterfactual_latent, conditioning, target
                )
                rewards = task_outcome_rewards(full_hidden, counterfactual_hidden, target)
                initial_observation, final_observation = (
                    future.observations[0],
                    future.observations[-1],
                )
                joint_key = str(section.labels.joint_position_key)
                joint_index = int(section.labels.joint_index)
                rows.append(
                    {
                        "group_id": group_id,
                        "group_size": int(section.group_size),
                        "state_index": descriptor.initial_state_index,
                        "anchor_step": descriptor.anchor_step,
                        "reached_anchor_step": reached_step,
                        "terminal_before_anchor": terminal,
                        "candidate_index": candidate_index,
                        "candidate_seed": candidate_seed,
                        "environment_seed": descriptor.environment_seed,
                        "mask_valid": residual.valid,
                        "mask_union_area_ratio": residual.mask_statistics["union_area_ratio"],
                        "task_progress": _task_progress(
                            initial_observation, final_observation, cfg
                        ),
                        "joint1_motion": abs(
                            float(final_observation[joint_key][joint_index])
                            - float(initial_observation[joint_key][joint_index])
                        ),
                        "success": bool(future.success),
                        "reward_full": float(rewards.full),
                        "reward_counterfactual": float(rewards.counterfactual),
                        "reward_margin": float(rewards.margin),
                        "reward_task_direction": float(rewards.task_direction),
                    }
                )
            print(
                f"rank={accelerator.process_index} group={local_index + 1}/"
                f"{len(local_descriptors)} id={group_id}",
                flush=True,
            )
        finally:
            main_env.close()
            shadow_env.close()

    rank_path = output_dir / f"candidates_rank_{accelerator.process_index:02d}.jsonl"
    rank_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    if accelerator.is_main_process:
        (output_dir / "adapter_audit.json").write_text(
            json.dumps(adapter_audit, indent=2) + "\n", encoding="utf-8"
        )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        all_rows: list[dict[str, Any]] = []
        for rank in range(accelerator.num_processes):
            rank_file = output_dir / f"candidates_rank_{rank:02d}.jsonl"
            with rank_file.open(encoding="utf-8") as handle:
                all_rows.extend(json.loads(line) for line in handle if line.strip())
        all_rows.sort(
            key=lambda row: (
                row["state_index"],
                row["anchor_step"],
                row["candidate_index"],
            )
        )
        _write_summary(cfg, output_dir, all_rows)
        print(f"Gate-4 audit complete: {output_dir / 'summary.json'}", flush=True)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="task_reward_audit/samhq_libero10_task7_joint1_half",
)
def main(cfg: DictConfig) -> None:
    _run(cfg)


if __name__ == "__main__":
    main()
