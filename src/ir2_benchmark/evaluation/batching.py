"""Bounded metadata grouping and hash-bound working-window model parameters."""

from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, assert_never

from ir2_benchmark.config import ConfigError, HostConfig
from ir2_benchmark.evaluation.errors import ModelExecutionError
from ir2_benchmark.evaluation.records import file_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from flax.core import FrozenDict
    from jax import Array

    from ir2_benchmark.agents.baselines import BaselineSpec
    from ir2_benchmark.env.rollout import RolloutResult
    from ir2_benchmark.evaluation.scenarios import EvaluationTask
    from ir2_benchmark.training.b_state import BTrainingConfig
    from ir2_benchmark.training.collect_a import Parameters as AParameters
    from ir2_benchmark.training.evaluate_b import Parameters as BParameters
    from ir2_benchmark.training.state_a import TrainAConfig
    from ir2_benchmark.types import MapInput

ModelParameters: TypeAlias = "AParameters | BParameters | FrozenDict | tuple[()]"
EvaluationKernel: TypeAlias = (
    "Callable[[ModelParameters, MapInput, Array, Array, Array, Array], tuple[RolloutResult, Array]]"
)


@dataclass(frozen=True, slots=True)
class AModel:
    """Route-A static inference settings, with weights kept out of the task queue."""

    settings: TrainAConfig
    metadata_path: Path
    metadata_hash: str


@dataclass(frozen=True, slots=True)
class BModel:
    """Route-B static inference settings, with weights kept out of the task queue."""

    settings: BTrainingConfig
    metadata_path: Path
    metadata_hash: str


@dataclass(frozen=True, slots=True)
class BaselineModel:
    """A fixed baseline needs only its static adapter spec, with no learner metadata file."""

    spec: BaselineSpec


@dataclass(frozen=True, slots=True)
class GroupKey:
    """Only native geometry, environment semantics and model inference structure group lanes."""

    config: HostConfig
    bucket: int
    route: str
    profile: tuple[str | int | bool, ...]


@dataclass(frozen=True, slots=True)
class PreparedTask:
    """Small immutable metadata; no device map or checkpoint parameters are retained here."""

    task: EvaluationTask
    group: GroupKey
    model: AModel | BModel | BaselineModel | None


def verify_checkpoint(task: EvaluationTask) -> None:
    """Never let a parameter-cache hit hide a modified checkpoint at the same path."""
    if task.method.route == "heuristic":
        return
    if task.method.route == "baseline":
        from ir2_benchmark.agents.baselines import spec_from_method  # noqa: PLC0415

        if spec_from_method(task.method).motion == "nearest_frontier":
            return
    if file_hash(Path(task.method.checkpoint)) != task.method.checkpoint_hash:
        detail = "checkpoint_content_hash_mismatch"
        raise ModelExecutionError(detail)


def prepare_task(task: EvaluationTask) -> PreparedTask:
    """Read only inference metadata until a bounded group is ready for device execution."""
    verify_checkpoint(task)
    model = None
    profile = ()
    match task.method.route:
        case "A":
            metadata_path = Path(task.method.checkpoint).parent / "configuration.json"
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            from ir2_benchmark.training.state_a import TrainAConfig  # noqa: PLC0415

            raw["training"].setdefault("candidate_attention", False)
            settings = TrainAConfig(**raw["training"])
            model = AModel(settings, metadata_path, file_hash(metadata_path))
            profile = (
                settings.hidden,
                settings.communication_mode,
                settings.neighbor_estimator,
                settings.candidate_attention,
                settings.fixed_plan,
            )
        case "B":
            metadata_path = Path(task.method.checkpoint).parent / "config.json"
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            from ir2_benchmark.training.b_state import BTrainingConfig  # noqa: PLC0415

            raw["training"].setdefault("candidate_attention", False)
            settings = BTrainingConfig(**raw["training"])
            model = BModel(settings, metadata_path, file_hash(metadata_path))
            profile = (
                settings.hidden,
                settings.categories,
                settings.classes,
                settings.learned_communication,
                settings.predicted_plans,
                settings.use_neighbor_plan,
                settings.candidate_attention,
            )
        case "heuristic":
            pass
        case "baseline":
            from ir2_benchmark.agents.baselines import spec_from_method  # noqa: PLC0415

            if task.scenario.config.navigation_mode != "ir2_graph":
                raise ConfigError(field="baseline.navigation_mode", detail="requires ir2_graph")
            spec = spec_from_method(task.method)
            model = BaselineModel(spec)
            profile = (spec.motion, spec.communication)
        case unreachable:
            assert_never(unreachable)
    key = GroupKey(task.scenario.config, task.scenario.bucket, task.method.route, profile)
    return PreparedTask(task, key, model)


def _model_metadata_hash(model: AModel | BModel | BaselineModel, detail: str) -> str | None:
    """Verify learner metadata, or identify asset-free and original-policy baseline bundles."""
    match model:
        case AModel() | BModel():
            if file_hash(model.metadata_path) != model.metadata_hash:
                raise ModelExecutionError(detail)
            return model.metadata_hash
        case BaselineModel(spec=spec):
            return None if spec.motion == "nearest_frontier" else ""
        case unreachable:
            assert_never(unreachable)


class ParameterCache:
    """A mutable LRU keeps at most one execution window of hash-verified model arrays."""

    __slots__ = ("_entries", "capacity")

    def __init__(self, capacity: int) -> None:
        """Bound resident checkpoint bundles independently of total manifest/model counts."""
        self.capacity = capacity
        self._entries: OrderedDict[tuple[str, str, str], ModelParameters] = OrderedDict()

    @property
    def size(self) -> int:
        """Return the number of resident parameter bundles for boundedness checks."""
        return len(self._entries)

    def load(self, prepared: PreparedTask) -> ModelParameters:
        """Verify every hit and bind loaded arrays to stable checkpoint and metadata hashes."""
        if prepared.model is None:
            return ()
        verify_checkpoint(prepared.task)
        metadata_hash = _model_metadata_hash(
            prepared.model, "checkpoint_metadata_changed_while_queued"
        )
        if metadata_hash is None:
            return ()
        method = prepared.task.method
        key = method.route, method.checkpoint_hash, metadata_hash
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        if len(self._entries) >= self.capacity:
            self._entries.popitem(last=False)
        match prepared.model:
            case AModel():
                from ir2_benchmark.training.evaluate_a import load_parameters  # noqa: PLC0415

                parameters, _ = load_parameters(method.checkpoint)
            case BModel():
                from ir2_benchmark.training.evaluate_b import load_parameters  # noqa: PLC0415

                parameters, _ = load_parameters(method.checkpoint)
            case BaselineModel():
                from ir2_benchmark.agents.baselines import load_policy_parameters  # noqa: PLC0415

                parameters = load_policy_parameters(method.checkpoint)
            case None:
                return ()
            case unreachable:
                assert_never(unreachable)
        verify_checkpoint(prepared.task)
        _model_metadata_hash(prepared.model, "checkpoint_metadata_changed_while_loading")
        self._entries[key] = parameters
        return parameters


class BoundedTaskQueue:
    """Mutable metadata queues emit complete groups or oldest tails under backpressure."""

    __slots__ = ("_groups", "_pending", "batch_size", "max_pending")

    def __init__(self, batch_size: int, max_pending: int) -> None:
        """Keep pending metadata bounded without scanning or materializing a full matrix."""
        if batch_size < 1 or max_pending < batch_size:
            raise ConfigError(field="batch_size", detail="require 1 <= batch_size <= max_pending")
        self.batch_size = batch_size
        self.max_pending = max_pending
        self._pending = 0
        self._groups: OrderedDict[GroupKey, list[PreparedTask]] = OrderedDict()

    @property
    def pending_count(self) -> int:
        """Return retained metadata count, excluding the batch already handed to execution."""
        return self._pending

    def append(self, prepared: PreparedTask) -> tuple[tuple[PreparedTask, ...], ...]:
        """Emit same-group lanes together; evict only whole actual tails if the bound is reached."""
        group = self._groups.setdefault(prepared.group, [])
        group.append(prepared)
        self._pending += 1
        ready = []
        if len(group) == self.batch_size:
            ready.append(tuple(self._groups.pop(prepared.group)))
            self._pending -= len(group)
        while self._pending > self.max_pending:
            _, oldest = self._groups.popitem(last=False)
            self._pending -= len(oldest)
            ready.append(tuple(oldest))
        return tuple(ready)

    def finish(self) -> tuple[tuple[PreparedTask, ...], ...]:
        """Flush real tail sizes; no padded synthetic task is scored or counted."""
        remaining = tuple(tuple(group) for group in self._groups.values())
        self._groups.clear()
        self._pending = 0
        return remaining


@cache
def get_heuristic_kernel(config: HostConfig) -> EvaluationKernel:
    """Persist the same dynamically seeded/map-fed batched program for the heuristic."""
    import jax  # noqa: PLC0415

    from ir2_benchmark.env.reset import reset  # noqa: PLC0415
    from ir2_benchmark.env.rollout import rollout  # noqa: PLC0415

    def single(
        _parameters: ModelParameters,
        map_data: MapInput,
        key: Array,
        spawns: Array,
        fault_key: Array,
        isolated_mask: Array,
    ) -> tuple[RolloutResult, Array]:
        initial = reset(map_data, config, key, spawns)._replace(
            key=fault_key, isolated_mask=isolated_mask
        )
        return rollout(initial, config), initial.physical_seen

    return jax.jit(jax.vmap(single))
