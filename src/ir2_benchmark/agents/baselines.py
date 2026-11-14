"""Untrained frontier and original pretrained IR2 navigation with fixed communication."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict, freeze

from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.codecs.packet import Codec
from ir2_benchmark.config import ConfigError, HostConfig
from ir2_benchmark.evaluation.errors import ModelExecutionError
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.scenarios import Method
from ir2_benchmark.models.ir2_policy import GraphPolicy, convert_torch_policy_weights
from ir2_benchmark.types import Action, LocalObservation, make_action

if TYPE_CHECKING:
    from jax import Array

    from ir2_benchmark.env.rollout import RolloutResult
    from ir2_benchmark.evaluation.batching import EvaluationKernel, ModelParameters
    from ir2_benchmark.types import MapInput

Motion = Literal["nearest_frontier", "ir2_adapted"]
Communication = Literal["off", "plan", "map", "mixed"]
_POLICY_TENSORS: Final = 90


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Static navigation and communication choices, independent of policy weights."""

    motion: Motion
    communication: Communication

    def __post_init__(self) -> None:
        """Reject unsupported manifest values at the adapter boundary."""
        if self.motion not in ("nearest_frontier", "ir2_adapted"):
            raise ConfigError(
                field="baseline.motion", detail="expected nearest_frontier or ir2_adapted"
            )
        if self.communication not in ("off", "plan", "map", "mixed"):
            raise ConfigError(
                field="baseline.communication", detail="expected off, plan, map or mixed"
            )


def make_baseline_method(
    motion: Motion, communication: Communication, checkpoint: str | Path | None = None
) -> Method:
    """Bind one of eight canonical method identities to an actual optional checkpoint."""
    spec = BaselineSpec(motion, communication)
    match spec.motion:
        case "nearest_frontier":
            if checkpoint:
                raise ConfigError(
                    field="baseline.checkpoint", detail="nearest_frontier does not use weights"
                )
            prefix, path, digest = "nf", "", ""
        case "ir2_adapted":
            if not checkpoint:
                raise ConfigError(
                    field="baseline.checkpoint", detail="ir2_adapted requires original weights"
                )
            path = str(checkpoint)
            prefix, digest = "ir2", file_hash(Path(path))
        case unreachable:
            assert_never(unreachable)
    return Method(
        f"{prefix}_{communication}",
        "baseline",
        communication,
        "none",
        -1,
        path,
        digest,
        0,
        decision_rule=motion,
    )


def spec_from_method(method: Method) -> BaselineSpec:
    """Parse a fixed baseline without accepting learner metadata as a baseline identity."""
    if method.route != "baseline":
        raise ConfigError(field="baseline.route", detail="expected baseline")
    spec = BaselineSpec(method.decision_rule, method.communication)
    prefix = "nf" if spec.motion == "nearest_frontier" else "ir2"
    if (
        method.method_id != f"{prefix}_{spec.communication}"
        or method.estimator != "none"
        or method.training_seed != -1
        or method.real_training_events != 0
    ):
        raise ConfigError(field="baseline.method", detail="noncanonical fixed baseline identity")
    match spec.motion:
        case "nearest_frontier":
            if method.checkpoint or method.checkpoint_hash:
                raise ConfigError(
                    field="baseline.checkpoint", detail="nearest_frontier requires empty identity"
                )
        case "ir2_adapted":
            if not method.checkpoint or not method.checkpoint_hash:
                raise ConfigError(
                    field="baseline.checkpoint", detail="ir2_adapted requires checkpoint and hash"
                )
        case unreachable:
            assert_never(unreachable)
    return spec


def load_policy_parameters(path: str | Path) -> FrozenDict:
    """Convert only original PolicyNet tensors; never initialize replacement weights."""
    checkpoint = Path(path)
    match checkpoint.suffix.lower():
        case ".npz":
            with np.load(checkpoint, allow_pickle=False) as archive:
                tensors = {name: archive[name] for name in archive.files}
        case ".pth" | ".pt":
            import torch  # noqa: PLC0415 - Optional original-checkpoint compatibility.

            original = torch.load(checkpoint, map_location="cpu", weights_only=True)["policy_model"]
            tensors = {name: value.detach().cpu().numpy() for name, value in original.items()}
        case _:
            raise ConfigError(
                field="baseline.checkpoint", detail="expected original .npz or .pth policy tensors"
            )
    if len(tensors) != _POLICY_TENSORS or any(
        value.dtype != np.float32 for value in tensors.values()
    ):
        raise ModelExecutionError(detail="original_policy_requires_90_float32_tensors")
    return freeze(convert_torch_policy_weights(tensors))


def baseline_communication(round_index: Array, communication: Communication) -> tuple[Array, Array]:
    """Send every fourth own action, preserving the declared codec even for empty maps."""
    match communication:
        case "off":
            return jnp.zeros((), bool), jnp.asarray(Codec.PLAN, jnp.int32)
        case "plan":
            codec = jnp.asarray(Codec.PLAN, jnp.int32)
        case "map":
            codec = jnp.asarray(Codec.MAP_DELTA, jnp.int32)
        case "mixed":
            cycle = jnp.array([Codec.PLAN, Codec.MAP_DELTA, Codec.MAP_DELTA, Codec.POSITION])
            codec = cycle[(round_index // 4) % 4]
        case unreachable:
            assert_never(unreachable)
    return round_index % 4 == 0, codec


def baseline_action(
    observation: LocalObservation,
    config: HostConfig,
    spec: BaselineSpec,
    parameters: ModelParameters = (),
) -> Action:
    """Choose a local legal move and independently apply the fixed communication schedule."""
    if observation.graph is None:
        raise ConfigError(
            field="baseline.navigation_mode", detail="requires ir2_graph observations"
        )
    match spec.motion:
        case "nearest_frontier":
            waypoint = nearest_frontier_action(observation, config).waypoint
        case "ir2_adapted":
            if not parameters:
                raise ModelExecutionError(detail="original_policy_parameters_missing")
            scores = GraphPolicy().apply(parameters, *observation.graph)
            allowed = observation.candidates.mask
            index = jnp.argmax(jnp.where(allowed, scores, -jnp.inf))
            waypoint = jnp.where(
                jnp.any(allowed), observation.candidates.positions[index], observation.position
            )
        case unreachable:
            assert_never(unreachable)
    send, codec = baseline_communication(observation.round, spec.communication)
    return make_action(waypoint, send, codec)


@cache
def get_baseline_kernel(config: HostConfig, spec: BaselineSpec) -> EvaluationKernel:
    """Cache a six-input batched rollout, dynamically supplied with the paired scenario seeds."""
    from ir2_benchmark.env.reset import reset  # noqa: PLC0415
    from ir2_benchmark.env.rollout import rollout  # noqa: PLC0415

    if config.navigation_mode != "ir2_graph":
        raise ConfigError(field="baseline.navigation_mode", detail="requires ir2_graph")

    def single(  # noqa: PLR0913, PLR0917 - Shared six-array evaluation kernel contract.
        parameters: ModelParameters,
        map_data: MapInput,
        key: Array,
        spawns: Array,
        fault_key: Array,
        isolated_mask: Array,
    ) -> tuple[RolloutResult, Array]:
        initial = reset(map_data, config, key, spawns)._replace(
            key=fault_key, isolated_mask=isolated_mask
        )
        return (
            rollout(
                initial,
                config,
                policy=lambda observation, settings: baseline_action(
                    observation, settings, spec, parameters
                ),
            ),
            initial.physical_seen,
        )

    return jax.jit(jax.vmap(single))
