"""Fresh real-episode evaluation of a trained route-A checkpoint."""

import json
from collections.abc import Callable
from dataclasses import asdict, replace
from functools import cache
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from flax.core import freeze
from jax import Array
from PIL import Image

from ir2_benchmark.agents.actor import Actor, initial_memory
from ir2_benchmark.agents.critic import Critic
from ir2_benchmark.communication.faults import KeyPurpose, fold_key, sample_isolated
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import _sample_spawns, reset
from ir2_benchmark.env.rollout import EventTrace, RolloutResult
from ir2_benchmark.evaluation.metrics import EpisodeOutcome, summarize_episode
from ir2_benchmark.models.plan_predictor import PlanPredictor, initial_plan_memory
from ir2_benchmark.presets import verify_preset_manifest
from ir2_benchmark.training.checkpoint import load_checkpoint
from ir2_benchmark.training.collect_a import (
    Collection,
    Collector,
    Parameters,
    SamplerState,
    make_collector,
)
from ir2_benchmark.training.diagnostics_a import plan_diagnostics
from ir2_benchmark.training.train_a import TrainAConfig, TrainingError
from ir2_benchmark.types import EnvState, MapInput


class InferenceSettings(NamedTuple):
    """Only network width and policy choices affect inference compilation."""

    hidden: int
    communication_mode: Literal["fixed", "learned"]
    neighbor_estimator: Literal["zoh", "predict", "none"]
    candidate_attention: bool
    fixed_plan: bool


EvaluationKernel = Callable[
    [Parameters, MapInput, Array, Array, Array, Array], tuple[RolloutResult, Array]
]


def inference_settings(settings: TrainAConfig) -> InferenceSettings:
    """Exclude training quotas, optimizers and replay settings from the cache key."""
    return InferenceSettings(
        settings.hidden,
        settings.communication_mode,
        settings.neighbor_estimator,
        settings.candidate_attention,
        settings.fixed_plan,
    )


@cache
def _collector(config: HostConfig, settings: InferenceSettings) -> Collector:
    """Reuse one compiled real-event collector for each inference structure."""
    return make_collector(
        config,
        Actor(settings.hidden, candidate_attention=settings.candidate_attention),
        Critic(settings.hidden),
        PlanPredictor(config.plan_horizon, settings.hidden),
        length=config.max_rounds * config.n_agents,
        learned_communication=settings.communication_mode == "learned",
        predict_neighbors=settings.neighbor_estimator == "predict",
        use_neighbor_plan=settings.neighbor_estimator != "none",
        fixed_plan=settings.fixed_plan,
        compute_values=False,
    )


def _sampler(environment: EnvState, hidden: int, policy_key: Array) -> SamplerState:
    """Start fresh local memories; policy_key has the legacy pre-POLICY-fold semantics."""
    slots = environment.positions.shape[0]
    return SamplerState(
        jax.tree.map(lambda value: value[None], environment),
        initial_memory((1, slots), hidden),
        initial_plan_memory((1, slots, slots), hidden),
        fold_key(policy_key, KeyPurpose.POLICY)[None],
        jnp.zeros(1, jnp.int32),
    )


def _rollout_result(result: Collection) -> RolloutResult:
    """Remove the collector's inner singleton axis without dropping diagnostics."""
    trace = jax.tree.map(lambda value: value[0], result.trace)
    events = EventTrace(
        trace.event,
        trace.agent_id,
        trace.valid,
        trace.positions,
        trace.goals,
        trace.balances,
        trace.spent,
        trace.delivered,
        trace.coverage,
        trace.reward,
        trace.invalid_action,
        trace.terminated,
        trace.truncated,
    )
    return RolloutResult(
        jax.tree.map(lambda value: value[0], result.sampler.environment),
        events,
        jax.tree.map(lambda value: value[0], result.plan_diagnostics),
    )


@cache
def _evaluation_kernel(config: HostConfig, settings: InferenceSettings) -> EvaluationKernel:
    """Keep parameters, maps, scenario keys and masks as dynamic batch inputs."""
    collector = _collector(config, settings)

    def single(  # noqa: PLR0913, PLR0917
        params: Parameters,
        map_data: MapInput,
        policy_key: Array,
        spawns: Array,
        fault_key: Array,
        isolated_mask: Array,
    ) -> tuple[RolloutResult, Array]:
        environment = reset(map_data, config, policy_key, spawns=spawns)._replace(
            key=fault_key, isolated_mask=isolated_mask
        )
        result = collector(
            _sampler(environment, settings.hidden, policy_key),
            params,
            jnp.asarray([config.max_rounds * config.n_agents]),
            jnp.int32(0),
        )
        return _rollout_result(result), environment.physical_seen

    return jax.jit(jax.vmap(single))


def get_evaluation_kernel(config: HostConfig, settings: TrainAConfig) -> EvaluationKernel:
    """Return a cached six-input batched kernel and batched (rollout, birth footprint)."""
    return _evaluation_kernel(config, inference_settings(settings))


def load_parameters(checkpoint: str | Path) -> tuple[Parameters, TrainAConfig]:
    """Verify the original training identity; test budgets and faults may differ."""
    path = Path(checkpoint)
    configuration = json.loads((path.parent / "configuration.json").read_text(encoding="utf-8"))
    verify_preset_manifest(
        HostConfig(**configuration["environment"]), configuration.get("implementations")
    )
    configuration["training"].setdefault("candidate_attention", False)
    settings = TrainAConfig(**configuration["training"])
    raw = serialization.msgpack_restore(load_checkpoint(path, configuration["configuration_hash"]))
    if "state" in raw:
        raw = raw["state"]["models"]
    return Parameters(
        freeze(jax.tree.map(jnp.asarray, raw["actor"]["params"])),
        freeze(jax.tree.map(jnp.asarray, raw["critic"]["params"])),
        freeze(jax.tree.map(jnp.asarray, raw["predictor"]["params"])),
    ), settings


def run_evaluation(  # noqa: PLR0913
    checkpoint: str | Path,
    map_data: MapInput,
    config: HostConfig,
    settings: TrainAConfig,
    key: Array,
    *,
    spawns: Array | None = None,
    fault_key: Array | None = None,
    isolated_mask: Array | None = None,
) -> RolloutResult:
    """Expose real final footprints and event traces to the paired evaluation runner."""
    parameters, trained = load_parameters(checkpoint)
    settings = replace(settings, candidate_attention=trained.candidate_attention)
    if settings.hidden != trained.hidden:
        raise TrainingError(field="hidden", detail="evaluation width differs from checkpoint")
    points = (
        _sample_spawns(map_data, config, key, jnp.int32(0))[: config.n_agents]
        if spawns is None
        else spawns
    )
    physical_key = key if fault_key is None else fault_key
    isolated = (
        sample_isolated(physical_key, jnp.int32(0), jnp.arange(config.n_slots) < config.n_agents)
        if isolated_mask is None
        else isolated_mask
    )
    result, _ = get_evaluation_kernel(config, settings)(
        jax.tree.map(lambda value: value[None], parameters),
        jax.tree.map(lambda value: value[None], map_data),
        key[None],
        points[None],
        physical_key[None],
        isolated[None],
    )
    return jax.tree.map(lambda value: value[0], result)


def evaluate_a(  # noqa: PLR0913, PLR0917
    checkpoint: str | Path,
    map_data: MapInput,
    config: HostConfig,
    settings: TrainAConfig,
    key: Array,
    output: str | Path,
) -> Path:
    """Restore trained parameters and write an independent real-episode evaluation."""
    parameters, trained = load_parameters(checkpoint)
    if settings.hidden != trained.hidden:
        raise TrainingError(field="hidden", detail="evaluation width differs from checkpoint")
    environment = reset(map_data, config, key)
    result = _collector(config, inference_settings(settings))(
        _sampler(environment, settings.hidden, key),
        parameters,
        jnp.asarray([config.max_rounds * config.n_agents]),
        jnp.int32(0),
    )
    trace = jax.tree.map(lambda value: np.asarray(value[0]), result.trace)
    actual = trace.valid.astype(bool)
    coverage = trace.coverage[actual]
    rounds = list(coverage[config.n_agents - 1 :: config.n_agents])
    if len(coverage) % config.n_agents:
        rounds.append(coverage[-1])
    final = jax.tree.map(lambda value: value[0], result.sampler.environment)
    outcome = EpisodeOutcome.SUCCESS if bool(final.terminated) else EpisodeOutcome.TIMEOUT
    metrics = summarize_episode(
        rounds,
        horizon=config.max_rounds,
        n_agents=config.n_agents,
        outcome=outcome,
        event_coverage=coverage.tolist(),
        threshold=config.coverage_target,
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    summary = (
        asdict(metrics)
        | plan_diagnostics(result)
        | {
            "real_events": int(actual.sum()),
            "spent_bytes": np.asarray(final.spent_bytes).tolist(),
            "total_spent_bytes": int(jnp.sum(final.spent_bytes)),
            "travel_distance": np.asarray(final.travel_distance).tolist(),
            "seed_key": np.asarray(key).tolist(),
            "checkpoint": str(checkpoint),
        }
    )
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (destination / "events.jsonl").open("w", encoding="utf-8") as stream:
        for index in np.flatnonzero(actual):
            stream.write(
                json.dumps(
                    {
                        name: np.asarray(value[index]).tolist()
                        for name, value in zip(trace._fields, trace, strict=True)
                    }
                )
                + "\n"
            )
    np.savez_compressed(
        destination / "trajectory.npz",
        positions=trace.positions[actual],
        balances=trace.balances[actual],
        coverage=coverage,
        local_maps=np.asarray(final.local_maps),
    )
    np.savez_compressed(
        destination / "plan-diagnostics.npz",
        **{
            name: np.asarray(value[0])
            for name, value in zip(
                result.plan_diagnostics._fields, result.plan_diagnostics, strict=True
            )
        },
    )
    palette = np.asarray([[255, 255, 255], [25, 25, 25], [140, 140, 140]], np.uint8)
    panels = [palette[np.asarray(final.truth)]]
    panels.extend(palette[np.asarray(final.local_maps[i])] for i in range(config.n_agents))
    Image.fromarray(np.concatenate(panels, axis=1)).save(destination / "maps.png")
    return destination / "summary.json"
