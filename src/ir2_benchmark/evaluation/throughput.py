"""Synchronized whole-environment and learned-kernel timing with explicit valid work counts."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, replace
from functools import partial
from shutil import which
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ir2_benchmark.evaluation.errors import is_oom
from ir2_benchmark.evaluation.timing import Timing, measure

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from jax import Array

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.env.rollout import RolloutResult
    from ir2_benchmark.types import EnvState, MapInput


@dataclass(frozen=True, slots=True)
class ThroughputCase:
    """One native-map, robot-count, batch-size measurement or an explicit OOM."""

    shape: tuple[int, int]
    n_agents: int
    batch_size: int
    device: str
    platform: str
    precision: str = "float32/int32"
    status: str = "measured"
    reset: Timing | None = None
    scripted_events: Timing | None = None
    policy_rollout: Timing | None = None
    team_rounds_per_second: float | None = None
    peak_device_bytes: int | None = None
    memory_measurement: str = (
        "whole-process peak across setup, compilation, reset, scripted, nearest-frontier and "
        "optional IR2 phases; JAX peak_bytes_in_use and Linux peak RSS; no per-workload attribution"
    )
    error: str | None = None
    policy_rollout_events: int = 32
    ir2_policy_inference: Timing | None = None
    ir2_policy_rollout: Timing | None = None
    peak_host_rss_bytes: int | None = None
    policy_rollout_segment_length: int = 32


def active_rollout_events(result: RolloutResult) -> int:
    """Count every live event across lanes, excluding frozen scan padding."""
    return int(np.asarray(result.trace.active).sum())


def peak_host_rss_bytes() -> int | None:
    """Return Linux process high-water RSS, including compilation and every timed phase."""
    if platform.system() != "Linux":
        return None
    import resource  # noqa: PLC0415 - Server measurements use Linux ru_maxrss in KiB.

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


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
    policy_measurement: Callable[[EnvState], tuple[Timing, Timing]] | None = None,
    seed: int = 71,
) -> tuple[ThroughputCase, ...]:
    """Stop increasing a native geometry/N batch after OOM, keeping that failure in the results."""
    import jax  # noqa: PLC0415

    from ir2_benchmark.env.reset import reset  # noqa: PLC0415
    from ir2_benchmark.env.rollout import rollout  # noqa: PLC0415

    device = jax.devices()[0]
    measurements = []
    for batch in batches:
        keys = jax.random.split(jax.random.PRNGKey(seed), batch)
        reset_fn = jax.jit(jax.vmap(lambda key: reset(map_data, config, key)))
        case = ThroughputCase(
            map_data.occupancy.shape,
            config.n_agents,
            batch,
            str(device),
            device.platform,
            policy_rollout_events=steps,
            policy_rollout_segment_length=steps,
        )
        try:
            reset_timing = measure(
                partial(reset_fn, keys),
                lambda _, batch=batch: batch,
                warmups=warmups,
                repeats=repeats,
            )
            initial = reset_fn(keys)
            jax.block_until_ready(initial)
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
                active_rollout_events,
                warmups=warmups,
                repeats=repeats,
            )
            inference_timing, ir2_timing = (
                policy_measurement(initial) if policy_measurement is not None else (None, None)
            )
            stats = device.memory_stats() or {}
            measurements.append(
                replace(
                    case,
                    reset=reset_timing,
                    scripted_events=events,
                    policy_rollout=policy_timing,
                    team_rounds_per_second=events.units_per_second / config.n_agents
                    if events.units_per_second is not None
                    else None,
                    peak_device_bytes=stats.get("peak_bytes_in_use"),
                    ir2_policy_inference=inference_timing,
                    ir2_policy_rollout=ir2_timing,
                    peak_host_rss_bytes=peak_host_rss_bytes(),
                )
            )
        except (MemoryError, RuntimeError) as error:
            if not is_oom(error):
                raise
            measurements.append(
                replace(
                    case,
                    status="oom",
                    memory_measurement="allocation failed; no invented device peak",
                    error=str(error),
                    peak_host_rss_bytes=peak_host_rss_bytes(),
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
        "policy_throughput_unit": "actual active agent events across all batch lanes",
        "reset_throughput_unit": "environment resets",
        "ir2_inference_throughput_unit": "single-current-agent graph forward passes",
        "ir2_inference_scope": "real reset-state graphs; excludes checkpoint IO and observation "
        "construction",
        "ir2_policy": "original frozen weights with mixed communication every fourth own action",
        "policy_rollout_events_semantics": "legacy schedule-length alias; actual counts are stored "
        "in Timing.actual_units and segment length in policy_rollout_segment_length",
        "timing_scope": "first_call_seconds includes tracing/compilation/first synchronized "
        "execution; "
        "samples_seconds are synchronized warm steady repetitions on the same initial states",
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
