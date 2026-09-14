"""Collect compact base-policy trajectories under a simulator-level Fault."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from hydra.utils import instantiate
from libero.libero import benchmark, get_libero_path
from omegaconf import DictConfig, OmegaConf

from experiments.libero.eval_libero_single import (
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _obs_to_model_input,
)
from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    capture_libero_observation,
    get_libero_dummy_action,
    get_libero_env,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from resilient.faults import build_fault_pipeline
from resilient.opsd.runtime import _derive_seed
from resilient.outcome_fpo.policy import sample_action_chunk
from resilient.outcome_fpo.runtime import _denormalize_candidate
from resilient.outcome_fpo.types import ActionConditioning
from resilient.state_banks import state_fingerprint

from .common import (
    resolve_project_path,
    resolved_config_hash,
    sha256_file,
    validate_fault_and_split,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _warmup(env: Any, steps: int) -> Any:
    observation = None
    for _ in range(int(steps)):
        observation, _, done, _ = env.step(get_libero_dummy_action())
        if done:
            raise RuntimeError("LIBERO task terminated during Stage-I warm-up.")
    return observation


def _encoded_observation(
    observation: dict[str, Any],
    *,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    video_height: int,
    video_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image, proprio, _ = _obs_to_model_input(
        obs=observation,
        cfg=cfg,
        processor=processor,
        width=video_width,
        height=video_height,
        device="cpu",
        dtype=torch.float32,
    )
    image_uint8 = (
        image[0].add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8)
    )
    return image_uint8.contiguous(), proprio[0].float().contiguous()


def _normalize_applied_action(
    action: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    """Invert LIBERO's gripper convention and normalize the action actually applied."""
    if action.shape != (7,):
        raise ValueError(f"Expected one LIBERO action with shape (7,), got {action.shape}.")
    model_action = torch.as_tensor(action.copy(), dtype=torch.float32)
    model_action[-1] = ((-model_action[-1]) + 1.0) * 0.5
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Stage-I collection expects one merged action field.")
    normalizer = processor.normalizer.normalizers["action"][action_meta[0]["key"]]
    return normalizer.forward(model_action).float().contiguous()


def _collect_episode(
    *,
    cfg: DictConfig,
    model,
    processor: FastWAMProcessor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    task: Any,
    state: torch.Tensor,
    state_index: int,
    attempt: int,
    video_height: int,
    video_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    section = cfg.two_stage_opsd.stage1.collection
    suite_name = str(cfg.two_stage_opsd.task.suite)
    task_id = int(cfg.two_stage_opsd.task.task_id)
    identity = (suite_name, task_id, state_index, attempt)
    environment_seed = _derive_seed(int(section.environment_seed), *identity, "environment")
    episode_index = state_index * int(section.max_attempts_per_state) + attempt
    fault_cfg = OmegaConf.to_container(cfg.fault, resolve=True)
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        environment_seed,
        fault_pipeline=build_fault_pipeline(fault_cfg),
        fault_episode_index=episode_index,
    )
    try:
        env.reset()
        observation = env.set_init_state(state)
        warmed = _warmup(env, int(section.num_steps_wait))
        if warmed is not None:
            observation = warmed
        first_image, first_proprio = _encoded_observation(
            observation,
            cfg=cfg,
            processor=processor,
            video_height=video_height,
            video_width=video_width,
        )
        images = [first_image]
        proprio = [first_proprio]
        actions: list[torch.Tensor] = []
        success = False
        replan_index = 0
        while len(actions) < int(section.max_actions_per_episode) and not success:
            image, state_tensor, _ = _obs_to_model_input(
                obs=observation,
                cfg=cfg,
                processor=processor,
                width=video_width,
                height=video_height,
                device=str(model.device),
                dtype=model.torch_dtype,
            )
            with torch.no_grad():
                normalized_chunk = sample_action_chunk(
                    model,
                    conditioning=ActionConditioning(
                        image, context, context_mask, state_tensor
                    ),
                    action_horizon=int(section.action_horizon),
                    num_inference_steps=int(section.num_inference_steps),
                    sigma_shift=None,
                    seed=_derive_seed(
                        int(section.inference_seed), *identity, replan_index, "action"
                    ),
                    rand_device=str(section.rand_device),
                    compile_action_infer=bool(section.compile_action_infer),
                )
            raw_chunk = _denormalize_candidate(
                normalized_chunk,
                processor,
                binarize_gripper=bool(section.binarize_gripper),
            )
            remaining = int(section.max_actions_per_episode) - len(actions)
            execute = min(int(section.replan_steps), remaining, len(raw_chunk))
            for raw_action in raw_chunk[:execute]:
                observation, _, done, _ = env.step(raw_action)
                # Structural Faults may update qpos after the wrapped step returns.
                observation = capture_libero_observation(env)
                actions.append(_normalize_applied_action(np.asarray(raw_action), processor))
                image_uint8, proprio_tensor = _encoded_observation(
                    observation,
                    cfg=cfg,
                    processor=processor,
                    video_height=video_height,
                    video_width=video_width,
                )
                images.append(image_uint8)
                proprio.append(proprio_tensor)
                success = success or bool(done)
                if success:
                    break
            replan_index += 1
        episode = {
            "schema_version": 1,
            "image": torch.stack(images),
            "action": torch.stack(actions) if actions else torch.empty((0, 7)),
            "proprio": torch.stack(proprio),
        }
        metadata = {
            "suite": suite_name,
            "task_id": task_id,
            "task_description": task_description,
            "state_index": state_index,
            "attempt": attempt,
            "environment_seed": environment_seed,
            "episode_index": episode_index,
            "actions": len(actions),
            "success": success,
            "faults": env.fault_metadata["faults"],
        }
        return episode, metadata
    finally:
        env.close()


def collect_stage1_dataset(cfg: DictConfig) -> Path:
    """Collect exactly the configured number of valid windows for every train state."""
    accelerator = Accelerator(mixed_precision=str(cfg.mixed_precision))
    if accelerator.num_processes != int(cfg.two_stage_opsd.distributed.num_processes):
        raise ValueError("Accelerate world size does not match two_stage_opsd.distributed.")
    validated = validate_fault_and_split(cfg)
    section = cfg.two_stage_opsd.stage1.collection
    if int(section.windows_per_state) <= 0:
        raise ValueError("windows_per_state must be positive.")
    if int(section.action_horizon) != 32 or int(section.replan_steps) != 10:
        raise ValueError("Stage-I collection keeps Fast-WAM's 32/10 action semantics.")
    if int(section.max_actions_per_episode) < int(section.action_horizon):
        raise ValueError("max_actions_per_episode must permit one complete training window.")

    dataset_dir = resolve_project_path(cfg.two_stage_opsd.stage1.dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    config_hash = resolved_config_hash(cfg)
    if accelerator.is_main_process:
        _write_json(
            dataset_dir / "collection_provenance.json",
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "method": "fault_rollout_native_joint_adaptation",
                "policy": "frozen_base_fastwam",
                "future_semantics": "realized_fault_trajectory",
                "post_success_frames_included": False,
                "fault": validated["fault"],
            },
        )
    accelerator.wait_for_everyone()

    set_global_seed(int(cfg.seed) + accelerator.process_index, get_worker_init_fn=False)
    dtype = _mixed_precision_to_model_dtype(str(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=dtype, device=str(accelerator.device))
    checkpoint_path = resolve_project_path(cfg.ckpt)
    _load_model_checkpoint(model, str(checkpoint_path))
    model.requires_grad_(False)
    model.eval()
    stats = load_dataset_stats_from_json(
        str(resolve_project_path(cfg.two_stage_opsd.dataset_stats_path))
    )
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    video_height, video_width = (int(value) for value in cfg.data.train.video_size)

    suite_name = str(cfg.two_stage_opsd.task.suite)
    task_id = int(cfg.two_stage_opsd.task.task_id)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    with torch.no_grad():
        context, context_mask = model.encode_prompt(
            DEFAULT_PROMPT.format(task=task.language)
        )
    states_path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    official_states = torch.load(states_path, map_location="cpu", weights_only=False)
    records = sorted(validated["training_records"], key=lambda item: int(item["state_index"]))
    selected_indices = section.get("state_indices")
    if selected_indices is not None:
        selected = {int(value) for value in selected_indices}
        available = {int(item["state_index"]) for item in records}
        if not selected or not selected.issubset(available):
            raise ValueError("collection.state_indices must select available training states.")
        records = [item for item in records if int(item["state_index"]) in selected]
    local_records = records[accelerator.process_index :: accelerator.num_processes]
    for record in local_records:
        state_index = int(record["state_index"])
        state = official_states[state_index]
        if state_fingerprint(state) != record["state_sha256"]:
            raise ValueError(f"Training state fingerprint mismatch at index {state_index}.")
        state_dir = dataset_dir / "episodes" / f"state_{state_index:02d}"
        summary_path = dataset_dir / "states" / f"state_{state_index:02d}.json"
        if summary_path.is_file():
            prior = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                prior.get("config_sha256") == config_hash
                and len(prior.get("windows", [])) == int(section.windows_per_state)
                and all((dataset_dir / item["episode"]).is_file() for item in prior["windows"])
            ):
                continue
        state_dir.mkdir(parents=True, exist_ok=True)
        windows: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        for attempt in range(int(section.max_attempts_per_state)):
            episode, metadata = _collect_episode(
                cfg=cfg,
                model=model,
                processor=processor,
                context=context,
                context_mask=context_mask,
                task=task,
                state=state,
                state_index=state_index,
                attempt=attempt,
                video_height=video_height,
                video_width=video_width,
            )
            valid_count = max(int(episode["action"].shape[0]) - 31, 0)
            if valid_count == 0:
                continue
            episode_path = state_dir / f"attempt_{attempt:02d}.pt"
            torch.save(episode, episode_path)
            relative = episode_path.relative_to(dataset_dir).as_posix()
            take = min(valid_count, int(section.windows_per_state) - len(windows))
            windows.extend({"episode": relative, "start": start} for start in range(take))
            episodes.append(
                {
                    **metadata,
                    "path": relative,
                    "sha256": sha256_file(episode_path),
                    "valid_windows": valid_count,
                    "selected_windows": take,
                }
            )
            if len(windows) == int(section.windows_per_state):
                break
        if len(windows) != int(section.windows_per_state):
            raise RuntimeError(
                f"State {state_index} produced only {len(windows)}/"
                f"{int(section.windows_per_state)} valid windows."
            )
        _write_json(
            summary_path,
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "state_index": state_index,
                "state_sha256": record["state_sha256"],
                "episodes": episodes,
                "windows": windows,
            },
        )
    accelerator.wait_for_everyone()

    manifest_path = dataset_dir / "dataset_manifest.json"
    if accelerator.is_main_process:
        text_context_path = dataset_dir / "text_context.pt"
        torch.save(
            {
                "context": context[0].detach().cpu().float(),
                "mask": context_mask[0].detach().cpu().bool(),
            },
            text_context_path,
        )
        windows: list[dict[str, Any]] = []
        episodes: list[dict[str, Any]] = []
        for record in records:
            state_index = int(record["state_index"])
            state_payload = json.loads(
                (dataset_dir / "states" / f"state_{state_index:02d}.json").read_text(
                    encoding="utf-8"
                )
            )
            if state_payload["config_sha256"] != config_hash:
                raise ValueError("A collected state shard belongs to a different configuration.")
            windows.extend(
                {**window, "state_index": state_index}
                for window in state_payload["windows"]
            )
            episodes.extend(state_payload["episodes"])
        _write_json(
            manifest_path,
            {
                "schema_version": 1,
                "config_sha256": config_hash,
                "suite": suite_name,
                "task_id": task_id,
                "task_description": task.language,
                "text_context": text_context_path.relative_to(dataset_dir).as_posix(),
                "fault": validated["fault"],
                "policy_checkpoint": {
                    "path": str(cfg.ckpt),
                    "sha256": sha256_file(checkpoint_path),
                },
                "state_count": len(records),
                "windows_per_state": int(section.windows_per_state),
                "window_count": len(windows),
                "action_horizon": int(section.action_horizon),
                "replan_steps": int(section.replan_steps),
                "frame_steps": [0, 4, 8, 12, 16, 20, 24, 28, 32],
                "episodes": episodes,
                "windows": windows,
            },
        )
    accelerator.wait_for_everyone()
    return manifest_path
