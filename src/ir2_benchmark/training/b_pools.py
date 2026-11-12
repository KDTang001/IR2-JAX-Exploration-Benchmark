"""Actual native-map assignment and independent physical lanes with shared bounded host replay."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from ir2_benchmark.agents.actor import initial_memory
from ir2_benchmark.communication.faults import sample_isolated
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.maps.sampler import TrainingMapSampler, TrainingSample
from ir2_benchmark.models.plan_predictor import initial_plan_memory
from ir2_benchmark.models.world_model import RSSMState, initial_rssm
from ir2_benchmark.training.b_state import BTrainingConfig, TrainingPool
from ir2_benchmark.training.imagination import PolicyCarry
from ir2_benchmark.training.replay import ReplayBuffer
from ir2_benchmark.types import EnvState


def initial_carry(config: HostConfig, options: BTrainingConfig) -> tuple[PolicyCarry, RSSMState]:
    """Reset one lane's recurrent state independently of the shared model parameters."""
    return (
        PolicyCarry(
            initial_memory((config.n_slots,), options.hidden),
            initial_plan_memory(
                (config.n_slots, config.n_slots),
                options.hidden,
                options.categories,
                options.classes,
            ),
        ),
        initial_rssm(
            deterministic=options.hidden, categories=options.categories, classes=options.classes
        ),
    )


def _assign(
    pool: TrainingPool,
    lane: int,
    state: EnvState,
    sample_index: int,
    options: BTrainingConfig,
) -> TrainingPool:
    """Replace one free lane while preserving every other lane, replay, and schedule identity."""
    carry, latent = initial_carry(pool.config, options)
    state, carry, latent = jax.tree.map(
        lambda full, new: full.at[lane].set(new),
        (pool.state, pool.carry, pool.latent),
        (state, carry, latent),
    )
    return replace(
        pool,
        state=state,
        carry=carry,
        latent=latent,
        sample_indices=(
            *pool.sample_indices[:lane],
            sample_index,
            *pool.sample_indices[lane + 1 :],
        ),
        lane_episodes=(*pool.lane_episodes[:lane], pool.episode, *pool.lane_episodes[lane + 1 :]),
        episode=pool.episode + 1,
    )


def new_pool(
    sample: TrainingSample,
    sampler: TrainingMapSampler,
    base: HostConfig,
    options: BTrainingConfig,
    seed: int,
) -> TrainingPool:
    """Load exactly one actual episode; unassigned shape-template lanes cannot be collected."""
    config = replace(base, n_agents=sample.n_agents, agent_bucket=6)
    state = reset(sampler.load(sample), config, jax.random.PRNGKey(sample.spawn_seed))
    fault_key = jax.random.PRNGKey(sample.fault_seed)
    state = state._replace(
        key=fault_key, isolated_mask=sample_isolated(fault_key, 0, state.active_mask)
    )
    carry, latent = initial_carry(config, options)
    state, carry, latent = jax.tree.map(
        lambda value: jnp.repeat(value[None], options.physical_env_count, axis=0),
        (state, carry, latent),
    )
    return TrainingPool(
        config,
        state,
        carry,
        latent,
        ReplayBuffer(
            seed + sample.bucket + sample.n_agents,
            options.snapshot_interval,
            options.replay_capacity,
        ),
        1,
        (sample.index, *(-1 for _ in range(options.physical_env_count - 1))),
        (0, *(-1 for _ in range(options.physical_env_count - 1))),
    )


def activate_pool(  # noqa: PLR0913, PLR0917
    pools: dict[str, TrainingPool],
    cursor: int,
    sampler: TrainingMapSampler,
    base: HostConfig,
    options: BTrainingConfig,
    seed: int,
) -> tuple[str, int, tuple[TrainingSample, ...]]:
    """Mutate actual pool assignments in schedule order; return only a completely assigned group.

    Already-reset partial groups wait without taking real events. A full group
    blocking the next sample runs before the cursor advances. No reservation or
    synthetic replacement is counted as an assigned episode.
    """
    assigned = []
    while True:
        sample = sampler.schedule(cursor)
        name = f"{sample.bucket}x{sample.n_agents}"
        if name not in pools:
            pool = new_pool(sample, sampler, base, options, seed)
        else:
            pool = pools[name]
            free = (
                (jnp.asarray(pool.sample_indices) < 0)
                | pool.state.terminated
                | pool.state.truncated
            )
            if not bool(jnp.any(free)):
                return name, cursor, tuple(assigned)
            lane = int(jnp.argmax(free))
            state = reset(
                sampler.load(sample),
                pool.config,
                jax.random.PRNGKey(sample.spawn_seed),
                episode=pool.episode,
            )
            key = jax.random.PRNGKey(sample.fault_seed)
            state = state._replace(
                key=key, isolated_mask=sample_isolated(key, pool.episode, state.active_mask)
            )
            pool = _assign(pool, lane, state, sample.index, options)
        pools[name] = pool
        cursor += 1
        assigned.append(sample)
        live = ~(pool.state.terminated | pool.state.truncated)
        if all(index >= 0 for index in pool.sample_indices) and bool(jnp.all(live)):
            return name, cursor, tuple(assigned)


def micro_pool(
    config: HostConfig, options: BTrainingConfig, seed: int, path: Path | None
) -> TrainingPool:
    """Create independent real lanes; lane zero retains the original serial reset/key stream."""
    states = [
        reset(
            load_scenario(config, seed + lane, path),
            config,
            jax.random.PRNGKey(seed)
            if lane == 0
            else jax.random.fold_in(jax.random.PRNGKey(seed), lane),
            episode=lane,
        )
        for lane in range(options.physical_env_count)
    ]
    carry, latent = initial_carry(config, options)
    return TrainingPool(
        config,
        jax.tree.map(lambda *values: jnp.stack(values), *states),
        jax.tree.map(
            lambda value: jnp.repeat(value[None], options.physical_env_count, axis=0), carry
        ),
        jax.tree.map(
            lambda value: jnp.repeat(value[None], options.physical_env_count, axis=0), latent
        ),
        ReplayBuffer(seed, options.snapshot_interval, options.replay_capacity),
        options.physical_env_count,
        tuple(range(options.physical_env_count)),
        tuple(range(options.physical_env_count)),
    )


def refresh_micro(
    pool: TrainingPool, options: BTrainingConfig, seed: int, path: Path | None
) -> TrainingPool:
    """Reset completed lanes only, retaining independent counters and contiguous replay episodes."""
    finished = pool.state.terminated | pool.state.truncated
    for lane in range(options.physical_env_count):
        if bool(finished[lane]):
            key = jax.random.PRNGKey(seed)
            if lane:
                key = jax.random.fold_in(key, lane)
            state = reset(
                load_scenario(pool.config, seed + pool.episode, path),
                pool.config,
                key,
                episode=pool.episode,
            )
            pool = _assign(pool, lane, state, pool.episode, options)
    return pool
