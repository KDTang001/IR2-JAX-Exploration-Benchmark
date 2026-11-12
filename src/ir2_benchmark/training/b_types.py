"""Typed checkpoint and report boundaries shared by route-B execution and evaluation."""

from collections.abc import Mapping
from typing import TypeAlias, TypedDict

from flax.training.train_state import TrainState
from jax import Array

from ir2_benchmark.models.world_model import RSSMState
from ir2_benchmark.training.imagination import PolicyCarry
from ir2_benchmark.types import EnvState

CheckpointValue: TypeAlias = (
    Array
    | TrainState
    | PolicyCarry
    | RSSMState
    | EnvState
    | str
    | int
    | float
    | Mapping[str, "CheckpointValue"]
    | None
)


class EvaluationSummary(TypedDict):
    """Real episode metrics plus the declared policy decision rule."""

    outcome: str
    success: bool
    timeout: bool
    auc: float | None
    final_coverage: float | None
    success_round: int | None
    success_event_offset: int | None
    observed_rounds: int
    decision_rule: str


class ModelDiagnostic(TypedDict):
    """Paired-action open-loop errors at one declared team-round horizon."""

    rounds: int
    position_error: float
    local_map_disagreement: float
    reward_mae: float
    delivery_error: float


class ReplaySummary(TypedDict):
    """Actual retained and evicted real-event replay counts."""

    size: int
    capacity: int
    evicted_events: int
    evicted_episodes: int


class TrainingSummary(TypedDict):
    """Measured training counters, independent real evaluation, and selected artifact locations."""

    route: str
    real_events: int
    total_training_events: int
    imagined_events: int
    model_updates: int
    actor_updates: int
    predictor_updates: int
    config_hash: str
    physical_microbatch: int
    physical_env_count: int
    session_real_events: int
    session_seconds: float
    session_events_per_second: float
    stage_seconds: dict[str, float]
    development: bool
    model_loss: float | None
    seconds: float
    actor_parameter_delta: float
    evaluation: EvaluationSummary | None
    checkpoint: str
    formal_complete: bool
    schedule_cursor: int
    group_events: dict[str, int]
    best_validation_auc: float | None
    last_validation_event: int
    best_checkpoint: str | None
    source_event_counts: dict[str, int]
    family_event_counts: dict[str, int]
    source_episode_counts: dict[str, int]
    family_episode_counts: dict[str, int]
    replay: dict[str, ReplaySummary]
    evaluation_status: str
