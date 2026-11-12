"""Assign the global map schedule to actual independent native-bucket lanes."""

import json
from dataclasses import asdict, replace
from typing import TextIO

import jax
import jax.numpy as jnp

from ir2_benchmark.config import HostConfig
from ir2_benchmark.maps.sampler import TrainingMapSampler
from ir2_benchmark.training.collect_a import Collection, SamplerState
from ir2_benchmark.training.pools_a import FAMILIES, SOURCES, FormalState, PoolState, _pool
from ir2_benchmark.training.state_a import TrainAConfig


def _assign_lane(
    pool: PoolState,
    lane: int,
    sample_index: int,
    new: SamplerState,
) -> PoolState:
    """Reset only the chosen lane; retain every other physical/recurrent/RNG leaf."""
    sampler = jax.tree.map(lambda full, value: full.at[lane].set(value[0]), pool.sampler, new)
    indices = (*pool.sample_indices[:lane], sample_index, *pool.sample_indices[lane + 1 :])
    return pool._replace(sample_indices=indices, sampler=sampler)


def activate_pool(
    state: FormalState,
    maps: TrainingMapSampler,
    config: HostConfig,
    settings: TrainAConfig,
    episodes: TextIO,
) -> tuple[FormalState, int, tuple[int, int]]:
    """Return a full live vmap group without skipping any scheduled map assignment.

    Cursor advances only after a sample is actually reset into a free lane.
    Unfilled groups hold their already-assigned episodes, including all RNG and
    physical state, until their native bucket/N has a complete physical batch.
    A full group blocking the next schedule index runs before any new assignment.
    With physical_env_count=1 this is the original serial scheduling order.
    """
    while True:
        sample = maps.schedule(state.cursor)
        key = (sample.bucket, sample.n_agents)
        keys = [(pool.bucket, pool.n_agents) for pool in state.pools]
        index = keys.index(key) if key in keys else len(keys)
        absent = index == len(keys)
        if absent:
            selected = _pool(sample, maps, config, settings, state.root_key)
        else:
            pool = state.pools[index]
            free = jnp.asarray(pool.sample_indices) < 0
            finished = pool.sampler.environment.terminated | pool.sampler.environment.truncated
            available = free | finished
            if not bool(jnp.any(available)):
                return state, index, key
            lane = int(jnp.argmax(available))
            new = _pool(
                sample, maps, config, replace(settings, physical_env_count=1), state.root_key
            )
            selected = _assign_lane(pool, lane, sample.index, new.sampler)
        pools = (
            (*state.pools, selected)
            if absent
            else (*state.pools[:index], selected, *state.pools[index + 1 :])
        )
        state = state._replace(
            pools=pools,
            cursor=state.cursor + 1,
            source_episodes=state.source_episodes.at[SOURCES.index(sample.source)].add(1),
            family_episodes=state.family_episodes.at[FAMILIES.index(sample.family)].add(1),
        )
        episodes.write(json.dumps(asdict(sample)) + "\n")
        episodes.flush()
        live = ~(selected.sampler.environment.terminated | selected.sampler.environment.truncated)
        if all(index >= 0 for index in selected.sample_indices) and bool(jnp.all(live)):
            return state, index, key


def account_events(
    state: FormalState,
    index: int,
    result: Collection,
    maps: TrainingMapSampler,
    valid_count: int,
) -> FormalState:
    """Attribute only actual accepted lane events to their original map-source identity."""
    pool = state.pools[index]
    actual = [maps.schedule(sample) for sample in pool.sample_indices]
    per_lane = jnp.sum(result.rollout.valid, axis=1, dtype=jnp.int32)
    source_counts = jnp.bincount(
        jnp.asarray([SOURCES.index(sample.source) for sample in actual]),
        weights=per_lane,
        length=len(SOURCES),
    ).astype(jnp.int32)
    family_counts = jnp.bincount(
        jnp.asarray([FAMILIES.index(sample.family) for sample in actual]),
        weights=per_lane,
        length=len(FAMILIES),
    ).astype(jnp.int32)
    return state._replace(
        pools=(
            *state.pools[:index],
            pool._replace(sampler=result.sampler),
            *state.pools[index + 1 :],
        ),
        real_events=state.real_events + valid_count,
        source_events=state.source_events + source_counts,
        family_events=state.family_events + family_counts,
    )
