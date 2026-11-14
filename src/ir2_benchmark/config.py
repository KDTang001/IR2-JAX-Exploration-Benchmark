"""Validated host configuration; immutable values are captured statically by JAX."""

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Final, Literal

from ir2_benchmark.presets import preset_names

MAX_AGENTS: Final = 6
MAX_HORIZON: Final = 57
MAX_CANDIDATES: Final = 64
FAULT_BURST: Final = 2


@dataclass(frozen=True, slots=True)
class ConfigError(ValueError):
    """A host configuration violates a benchmark contract."""

    field: str
    detail: str

    def __str__(self) -> str:
        """Render the offending field and its concrete constraint."""
        return f"{self.field}: {self.detail}"


@dataclass(frozen=True, slots=True)
class HostConfig:
    """One static robot/map compilation bucket and its public scenario parameters."""

    n_agents: int = 2
    agent_bucket: int | None = None
    sensor_radius: float = 80
    max_distance: int = 20
    plan_horizon: int = 5
    max_rounds: int = 512
    initial_budget: int = 8192
    max_candidates: int = 29
    navigation_mode: str = "ir2_one_hop"
    rendezvous_mode: Literal["strict_ir2", "reachable_only"] = "strict_ir2"
    global_graph_fusion: bool = False
    graph_nodes_per_axis: int = 50
    graph_local_radius: int = 80
    graph_global_radius: int = 160
    graph_capacity: int = 4096
    graph_padding_size: int = 400
    graph_coordinate_scale: float = 640.0
    graph_utility_radius: int = 70
    graph_peer_radius: int = 20
    graph_merge_enabled: bool = True
    graph_merge_every: int = 10
    graph_merge_threshold: int = 200
    graph_unique_radius: int = 30
    graph_offshoot_unique_radius: int = 40
    graph_offshoot_max_radius: int = 80
    graph_offshoot_count: int = 6
    graph_frontier_protect_radius: int = 120
    coverage_target: float = 0.99
    fault_mode: int = 0
    dropout_probability: float = 0
    burst_start: int = 128
    burst_duration: int = 64
    spawn_radius: int = 10
    signal_threshold: float = -70
    reward_model: str = "coverage_delta"
    reward_coverage_weight: float = 100.0
    reward_distance_weight: float = 1 / 512
    reward_event_cost: float | None = None
    reward_success_bonus: float = 40.0
    reward_graph_statistics: bool = False
    signal_model: str = "attenuation"
    communication_range: float = 100.0
    transmit_power: float = -20.0
    reference_loss: float = 31.0
    gamma_free: float = 2.0
    gamma_obstacle: float = 4.0
    reference_distance: float = 35.0
    signal_noise_free_min: float = 0.0
    signal_noise_free_max: float = 0.0
    signal_noise_obstacle_min: float = 0.0
    signal_noise_obstacle_max: float = 0.0
    exchange_mode: str = "broadcast"
    charge_mode: str = "attempt"
    send_policy: str = "action"
    send_period: int = 1
    message_codec: int = 0
    custom_parameters: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:  # noqa: C901
        """Validate capacities and ranges once at the host boundary."""
        parameters = tuple(tuple(pair) for pair in self.custom_parameters)
        if any(
            len(pair) != 2 or not isinstance(pair[0], str) or not math.isfinite(pair[1])
            for pair in parameters
        ) or len({pair[0] for pair in parameters}) != len(parameters):
            raise ConfigError("custom_parameters", "expected unique name/finite-number pairs")
        object.__setattr__(self, "custom_parameters", parameters)
        for kind, name in (
            ("reward", self.reward_model),
            ("signal", self.signal_model),
            ("sending", self.send_policy),
        ):
            if name not in preset_names(kind):
                raise ConfigError(kind, f"unknown preset {name!r}; register custom functions first")
        if (
            self.reward_model == "ir2_individual" or self.reward_graph_statistics
        ) and self.navigation_mode != "ir2_graph":
            raise ConfigError("reward_model", "ir2_individual requires ir2_graph observations")
        if self.exchange_mode not in ("broadcast", "addressed"):
            raise ConfigError("exchange_mode", "expected broadcast or addressed")
        if self.charge_mode not in ("attempt", "success"):
            raise ConfigError("charge_mode", "expected attempt or success")
        if not isinstance(self.send_period, int) or self.send_period < 1:
            raise ConfigError("send_period", "expected a positive integer number of robot turns")
        if self.message_codec not in (0, 1, 2, 3, 4):
            raise ConfigError(
                "message_codec", "0 preserves method action; 1..4 select standard codecs"
            )
        nonnegative = (
            "reward_coverage_weight",
            "reward_distance_weight",
            "reward_success_bonus",
            "communication_range",
            "gamma_free",
            "gamma_obstacle",
        )
        finite = nonnegative + (
            "transmit_power",
            "reference_loss",
            "reference_distance",
            "signal_threshold",
            "signal_noise_free_min",
            "signal_noise_free_max",
            "signal_noise_obstacle_min",
            "signal_noise_obstacle_max",
        )
        if any(not math.isfinite(getattr(self, name)) for name in finite):
            raise ConfigError("presets", "numeric preset parameters must be finite")
        if any(getattr(self, name) < 0 for name in nonnegative) or self.reference_distance <= 0:
            raise ConfigError(
                "presets", "costs/ranges/exponents nonnegative; reference distance positive"
            )
        if self.reward_event_cost is not None and (
            not math.isfinite(self.reward_event_cost) or self.reward_event_cost < 0
        ):
            raise ConfigError("reward_event_cost", "expected a finite nonnegative cost")
        if (
            self.signal_noise_free_min > self.signal_noise_free_max
            or self.signal_noise_obstacle_min > self.signal_noise_obstacle_max
        ):
            raise ConfigError("signal_noise", "minimum must not exceed maximum")
        if self.signal_model != "ir2" and any(
            (
                self.signal_noise_free_min,
                self.signal_noise_free_max,
                self.signal_noise_obstacle_min,
                self.signal_noise_obstacle_max,
            )
        ):
            raise ConfigError(
                "signal_noise", "episode noise currently belongs to the ir2 signal preset"
            )
        if self.navigation_mode not in ("ir2_graph", "ir2_one_hop", "grid_path"):
            field = "navigation_mode"
            raise ConfigError(field, "expected ir2_graph, ir2_one_hop or legacy grid_path")
        if self.rendezvous_mode not in ("strict_ir2", "reachable_only"):
            field = "rendezvous_mode"
            raise ConfigError(field, "expected strict_ir2 or reachable_only")
        if (
            self.graph_nodes_per_axis < 2  # noqa: PLR2004
            or self.graph_capacity < 1
            or self.graph_padding_size < 2  # noqa: PLR2004
            or self.graph_coordinate_scale <= 0
            or min(
                self.graph_utility_radius,
                self.graph_peer_radius,
                self.graph_merge_every,
                self.graph_merge_threshold,
                self.graph_unique_radius,
            )
            < 1
            or not 0 < self.graph_local_radius <= self.graph_global_radius
        ):
            field = "graph"
            raise ConfigError(
                field, "positive capacity/radii and at least two grid nodes per axis required"
            )
        if self.global_graph_fusion and self.navigation_mode == "grid_path":
            field = "global_graph_fusion"
            raise ConfigError(field, "fusion requires one-hop graph navigation")
        if not 1 <= self.n_agents <= self.n_slots <= MAX_AGENTS:
            field = "n_agents"
            raise ConfigError(field, "expected 1 <= active <= bucket <= 6")
        if self.sensor_radius < 0 or self.max_distance < 1 or self.max_rounds < 1:
            field = "geometry"
            raise ConfigError(field, "radius must be nonnegative; distance and rounds positive")
        if (
            not 1 <= self.plan_horizon <= MAX_HORIZON
            or not 0 <= self.max_candidates <= MAX_CANDIDATES
        ):
            field = "capacity"
            raise ConfigError(field, "PLAN must fit 240 bytes and at most 64 waypoints")
        if self.initial_budget < 0 or not 0 < self.coverage_target <= 1:
            field = "budget/coverage"
            raise ConfigError(field, "budget nonnegative; target in (0,1]")
        if self.fault_mode not in (0, 1, 2) or not 0 <= self.dropout_probability <= 1:
            field = "fault"
            raise ConfigError(field, "unknown condition or dropout outside [0,1]")
        if self.fault_mode != 1 and self.dropout_probability != 0:
            field = "dropout_probability"
            raise ConfigError(field, "Dropout is a separate condition from Normal and Burst")
        if min(self.burst_start, self.burst_duration, self.spawn_radius) < 0:
            field = "schedule"
            raise ConfigError(field, "burst and spawn parameters must be nonnegative")

    @property
    def n_slots(self) -> int:
        """Return the static robot capacity, distinct from the active population."""
        return self.n_agents if self.agent_bucket is None else self.agent_bucket


def load_config(path: str | Path) -> HostConfig:
    """Read a JSON object, rejecting unknown keys instead of silently ignoring them."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(HostConfig)}
    if not isinstance(raw, dict) or not set(raw) <= allowed:
        field = "json"
        raise ConfigError(field, "expected an object containing only HostConfig fields")
    return HostConfig(**raw)
