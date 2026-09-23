"""Strict validation for the first task-decoupling implementation."""

from __future__ import annotations

from omegaconf import DictConfig


def validate_task_decoupling_config(section: DictConfig) -> None:
    """Reject declared options that the Gate-3 implementation does not support."""
    mode = str(section.mode)
    if mode not in {"audit", "training_gate4_override"}:
        raise ValueError("Unsupported task-decoupling mode.")
    provider_type = str(section.mask_provider.type)
    if provider_type not in {"sam3_file_service", "samhq_file_service"}:
        raise ValueError("Unsupported task-decoupling mask provider.")
    prompt_mode = str(section.mask_provider.prompt_mode)
    expected_prompt_mode = {
        "sam3_file_service": "text_video",
        "samhq_file_service": "box_per_frame",
    }[provider_type]
    if prompt_mode != expected_prompt_mode:
        raise ValueError(
            f"{provider_type} requires prompt_mode={expected_prompt_mode}."
        )
    if provider_type == "samhq_file_service" and not bool(
        section.mask_provider.audit_only
    ):
        raise ValueError("The oracle-box HQ-SAM backend must remain audit-only.")
    if mode == "training_gate4_override":
        reward = section.get("reward")
        if reward is None or str(reward.get("type")) != "task_direction":
            raise ValueError("The Gate-4 override requires reward.type=task_direction.")
        if not bool(reward.get("gate4_override", False)):
            raise ValueError("Gate-4 override training requires explicit acknowledgement.")
        if str(reward.get("invalid_mask_policy")) != "error":
            raise ValueError("Gate-4 override training must stop on an invalid task mask.")
    threshold = float(section.segmentation.output_probability_threshold)
    if not 0.0 < threshold < 1.0:
        raise ValueError("SAM output probability threshold must be in (0,1).")
    if provider_type == "samhq_file_service" and threshold != 0.5:
        raise ValueError("HQ-SAM binary masks require probability threshold 0.5.")
    if str(section.counterfactual.type) != "local_gaussian_blur":
        raise ValueError("Only local_gaussian_blur counterfactuals are implemented.")
    if int(section.counterfactual.edge_feather_px) != 0:
        raise ValueError("Mask edge feathering is not implemented in the Gate-3 audit.")
    if str(section.segmentation.empty_mask_policy) != "invalid":
        raise ValueError("The Gate-3 audit requires empty_mask_policy=invalid.")
    if str(section.latent.encoder) != "frozen_fastwam_vae":
        raise ValueError("The Gate-3 audit requires the frozen Fast-WAM VAE.")
    if not bool(section.latent.batch_full_and_counterfactual):
        raise ValueError("Full and counterfactual videos must share one VAE batch.")
    if not bool(section.latent.exclude_initial_latent):
        raise ValueError("The initial latent slice must be excluded.")
    if int(section.audit.overlay_every_n_samples) <= 0:
        raise ValueError("Overlay frequency must be positive.")
    probability_like = (
        "minimum_union_recall",
        "minimum_union_iou",
        "minimum_entity_recall",
        "maximum_robot_contamination",
        "minimum_temporal_iou",
        "residual_cosine_margin_over_shifted_control",
    )
    for name in probability_like:
        value = float(section.audit.gates[name])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Audit gate {name} must be in [0,1].")
