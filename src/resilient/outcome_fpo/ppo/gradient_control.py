"""Independent actor and critic gradient clipping for PPO-style FPO."""

from __future__ import annotations

import math
from typing import Any

import torch
from accelerate.utils import DistributedType


def clip_actor_critic_gradients(
    *,
    accelerator: Any,
    optimizer: Any,
    actor_max_norm: float,
    critic_max_norm: float,
) -> dict[str, float]:
    """Clip two optimizer groups without coupling the critic to actor gradients.

    Under ZeRO-2, Accelerate's ``clip_grad_norm_`` only reports DeepSpeed's
    cached global norm. Gradients are already partitioned at this point, so
    clip each DeepSpeed optimizer group's reduced partitions before ``step``.
    """
    limits = (float(actor_max_norm), float(critic_max_norm))
    if any(not math.isfinite(limit) or limit <= 0 for limit in limits):
        raise ValueError("Actor and critic gradient limits must be finite and positive.")

    if getattr(accelerator, "distributed_type", None) == DistributedType.DEEPSPEED:
        wrapped = accelerator.deepspeed_engine_wrapped
        if wrapped is None:
            raise RuntimeError("DeepSpeed gradient control requires a prepared engine.")
        engine = wrapped.engine
        zero = engine.optimizer
        if engine.gradient_clipping() != 0 or not zero.partition_gradients or zero.cpu_offload:
            raise RuntimeError(
                "Separate PPO clipping requires ZeRO-2 without built-in clipping/offload."
            )
        if len(zero.bit16_groups) != 2 or len(zero.optimizer.param_groups) != 2:
            raise RuntimeError("Expected distinct actor and critic DeepSpeed optimizer groups.")
        norms = []
        for index, limit in enumerate(limits):
            gradients = zero.averaged_gradients[index]
            parameters = zero.params_in_partition[index]
            if gradients is None or len(gradients) != len(parameters):
                raise RuntimeError("ZeRO-2 reduced gradients do not match the PPO parameter group.")
            scaled_norm = zero.get_grad_norm_direct(gradients, parameters)
            norm = float(scaled_norm) / float(zero.loss_scale)
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite PPO optimizer-group gradient norm.")
            coefficient = min(1.0, limit / (norm + 1e-6))
            for gradient in gradients:
                gradient.mul_(coefficient)
            norms.append(norm)
    else:
        if len(optimizer.param_groups) != 2:
            raise RuntimeError("Expected distinct actor and critic optimizer groups.")
        if hasattr(accelerator, "unscale_gradients"):
            accelerator.unscale_gradients()
        norms = []
        for index, limit in enumerate(limits):
            norm = float(
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[index]["params"], limit)
            )
            if not math.isfinite(norm):
                raise FloatingPointError("Non-finite PPO optimizer-group gradient norm.")
            norms.append(norm)

    return {
        "actor_grad_norm": norms[0],
        "critic_grad_norm": norms[1],
        "actor_grad_clipped": float(norms[0] > limits[0]),
        "critic_grad_clipped": float(norms[1] > limits[1]),
    }
