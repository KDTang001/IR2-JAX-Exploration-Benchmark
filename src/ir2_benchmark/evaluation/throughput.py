"""Synchronized whole-environment and learned-kernel timing with explicit valid work counts."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from functools import partial
from shutil import which
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ir2_benchmark.evaluation.errors import is_oom
from ir2_benchmark.evaluation.timing import Timing, measure

if TYPE_CHECKING:
    from pathlib import Path

    from jax import Array

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.types import EnvState, MapInput


@dataclass(frozen=True, slots=True)
class ThroughputCase:
    """One native-map, robot-count, batch-size measurement or an explicit OOM."""

    shape: tuple[int, int]
    n_agents: int
    batch_size: int
    device: str
    platform: str
    precision: str
    status: str
    reset: Timing | None
    scripted_events: Timing | None
    policy_rollout: Timing | None
    team_rounds_per_second: float | None
    peak_device_bytes: int | None
    memory_measurement: str
    error: str | None
    policy_rollout_events: int = 32


class ScriptResult(NamedTuple):
    """Carry the full real final state without retaining a time-by-map state tensor."""

    final_state: EnvState
    active_events: Array
    distance: Array
    spent: Array


def scripted_events(initial: EnvState, config: HostConfig, steps: int) -> ScriptResult:
    """Alternate known-free endpoints through observation/navigation/sensing/traffic."""
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax import lax  # noqa: PLC0415

    from ir2_benchmark.env.observe import observe  # noqa: PLC0415
    from ir2_benchmark.env.step import step_agent  # noqa: PLC0415
    from ir2_benchmark.types import make_action  # noqa: PLC0415

    origins = initial.positions
    observations = jax.vmap(lambda agent: observe(initial, agent, config))(
        jnp.arange(config.n_slots)
    )
    ends = (
        jnp.where(
            observations.candidates.mask[:, 1:2], observations.candidates.positions[:, 1], origins
        )
        if config.max_candidates
        else origins
    )

    def event(state: EnvState, _unused: None) -> tuple[EnvState, tuple[Array, Array, Array]]:
        agent = state.current_agent
        observation = observe(state, agent, config)
        goal = jnp.where(state.round % 2 == 0, ends[agent], origins[agent])
        if config.navigation_mode != "grid_path":
            candidates = observation.candidates
            squared = jnp.sum(jnp.square(candidates.positions - goal), axis=-1)
            index = jnp.argmin(jnp.where(candidates.mask, squared, jnp.iinfo(jnp.int32).max))
            goal = candidates.positions[index]
        result = step_agent(state, make_action(goal, jnp.bool_(1)), config, observation=observation)
        return result.state, (result.info.active, result.info.distance, result.info.spent)

    final, (active, distance, spent) = lax.scan(event, initial, None, length=steps)
    return ScriptResult(final, jnp.sum(active), jnp.sum(distance), jnp.sum(spent))


def benchmark_environment(  # noqa: PLR0913
    map_data: MapInput,
    config: HostConfig,
    *,
    batches: tuple[int, ...] = (1, 4, 16),
    steps: int = 32,
    warmups: int = 3,
    repeats: int = 10,
) -> tuple[ThroughputCase, ...]:
    """Stop increasing a native geometry/N batch after OOM, keeping that failure in the results."""
    import jax  # noqa: PLC0415

    from ir2_benchmark.env.reset import reset  # noqa: PLC0415
    from ir2_benchmark.env.rollout import rollout  # noqa: PLC0415

    device = jax.devices()[0]
    measurements = []
    for batch in batches:
        keys = jax.random.split(jax.random.PRNGKey(71), batch)
        reset_fn = jax.jit(jax.vmap(lambda key: reset(map_data, config, key)))
        try:
            reset_timing = measure(
                partial(reset_fn, keys),
                lambda _, batch=batch: batch,
                warmups=warmups,
                repeats=repeats,
            )
            initial = reset_fn(keys)
            scripted = jax.jit(jax.vmap(lambda state: scripted_events(state, config, steps)))
            events = measure(
                partial(scripted, initial),
                lambda output: int(np.asarray(output.active_events).sum()),
                warmups=warmups,
                repeats=repeats,
            )
            policy = jax.jit(jax.vmap(lambda state: rollout(state, config, steps)))
            policy_timing = measure(
                partial(policy, initial),
                lambda output: int(np.any(np.asarray(output.trace.active), axis=-1).sum()),
                warmups=warmups,
                repeats=repeats,
            )
            stats = device.memory_stats() or {}
            measurements.append(
                ThroughputCase(
                    map_data.occupancy.shape,
                    config.n_agents,
                    batch,
                    str(device),
                    device.platform,
                    "float32/int32",
                    "measured",
                    reset_timing,
                    events,
                    policy_timing,
                    events.units_per_second / config.n_agents
                    if events.units_per_second is not None
                    else None,
                    stats.get("peak_bytes_in_use"),
                    "jax.Device.memory_stats peak_bytes_in_use (process cumulative)",
                    None,
                    policy_rollout_events=steps,
                )
            )
        except (MemoryError, RuntimeError) as error:
            if not is_oom(error):
                raise
            measurements.append(
                ThroughputCase(
                    map_data.occupancy.shape,
                    config.n_agents,
                    batch,
                    str(device),
                    device.platform,
                    "float32/int32",
                    "oom",
                    None,
                    None,
                    None,
                    None,
                    None,
                    "allocation failed; no invented peak",
                    str(error),
                    policy_rollout_events=steps,
                )
            )
            break
    return tuple(measurements)


def write_throughput(
    path: Path, cases: tuple[ThroughputCase, ...], *, scope: str, model: dict | None = None
) -> Path:
    """Save actual cases and mark model kernels unmeasured unless real hooks ran."""
    import jax  # noqa: PLC0415
    import jaxlib  # noqa: PLC0415

    monitor = which("nvidia-smi")
    driver = (
        subprocess.run(  # noqa: S603
            [monitor, "--query-gpu=driver_version,name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        if monitor
        else None
    )
    expected = {
        (side, agents, batch)
        for side in (512, 768, 1024)
        for agents in (2, 4, 6)
        for batch in (1, 4, 16)
    }
    observed = {
        (case.shape[0], case.n_agents, case.batch_size)
        for case in cases
        if case.shape[0] == case.shape[1]
    }
    stopped = {
        key
        for key in expected - observed
        if any(
            case.status == "oom"
            and case.shape[0] == key[0]
            and case.n_agents == key[1]
            and case.batch_size < key[2]
            for case in cases
        )
    }
    remaining = sorted(expected - observed - stopped) if scope == "formal" else []

    payload = {
        "scope": scope,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "python_version": platform.python_version(),
        "driver_device_memory_query": driver,
        "warmups": sorted({case.reset.warmup_count for case in cases if case.reset is not None}),
        "synchronized_repetitions": sorted(
            {len(case.reset.samples_seconds) for case in cases if case.reset is not None}
        ),
        "cases": [asdict(case) for case in cases],
        "world_model": model or {"status": "unmeasured", "reason": "no B checkpoint supplied"},
        "formal_matrix_complete": scope == "formal"
        and not remaining
        and model is not None
        and model.get("status") == "measured",
        "remaining_native_cases": remaining,
        "skipped_after_oom": sorted(stopped),
        "throughput_denominator": "actual unfrozen agent events; team rounds = events/N",
        "policy_throughput_unit": "active rollout segments; event length is recorded per case",
        "policy": "nearest_frontier_with_fixed_communication",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = (
    "ScriptResult",
    "ThroughputCase",
    "Timing",
    "benchmark_environment",
    "measure",
    "scripted_events",
    "write_throughput",
)
