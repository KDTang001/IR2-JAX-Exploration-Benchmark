"""A bounded compact ring for recent/history intent supervision, without maps."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from ir2_benchmark.training.collect_a import PlanTrainingBatch


class PredictorReplay(NamedTuple):
    """Every leaf is an array; data, cursor and sampling key restore exactly."""

    data: PlanTrainingBatch
    cursor: Array
    size: Array
    key: Array


def mix_live_predictor_events(
    replay: PredictorReplay, plans: PlanTrainingBatch, valid: Array, recent_fraction: float
) -> tuple[PredictorReplay, PlanTrainingBatch]:
    """Pack real [environment,event] rows between collection and supervised optimization."""
    indices = jnp.nonzero(valid.reshape(-1), size=int(jnp.sum(valid)))[0]
    recent = jax.tree.map(lambda value: value.reshape((-1, *value.shape[2:]))[indices], plans)
    return sample_and_store(replay, recent, recent_fraction)


def empty_predictor_replay(
    prototype: PlanTrainingBatch, capacity: int, key: Array
) -> PredictorReplay:
    """Allocate static event slots, each containing only model input/memory/targets."""
    data = jax.tree.map(lambda value: jnp.zeros((capacity, *value.shape), value.dtype), prototype)
    return PredictorReplay(data, jnp.int32(0), jnp.int32(0), key)


@partial(jax.jit, static_argnames=("recent_fraction",))
def sample_and_store(
    replay: PredictorReplay, recent: PlanTrainingBatch, recent_fraction: float = 0.5
) -> tuple[PredictorReplay, PlanTrainingBatch]:
    """Sample current/history, then insert current events so history is genuinely older.

    The validated configuration guarantees capacity >= one logical rollout batch.
    An empty initial ring supplies the complete current batch, with no fabricated
    targets. History samples use replacement; current samples use a permutation.
    """
    rows = recent.key.shape[0]
    history_count = int(rows * (1 - recent_fraction))
    recent_count = rows - history_count
    next_key, current_key, history_key = jax.random.split(replay.key, 3)
    permutation = jax.random.permutation(current_key, rows)
    history_indices = jax.random.randint(
        history_key, (history_count,), 0, jnp.maximum(replay.size, 1)
    )
    mixed = jax.tree.map(
        lambda current, old: jnp.concatenate(
            (current[permutation[:recent_count]], old[history_indices]), axis=0
        ),
        recent,
        replay.data,
    )
    batch = jax.tree.map(
        lambda current, blend: jnp.where(replay.size > 0, blend, current[permutation]),
        recent,
        mixed,
    )
    capacity = replay.data.key.shape[0]
    indices = (replay.cursor + jnp.arange(rows)) % capacity
    stored = jax.tree.map(lambda old, new: old.at[indices].set(new), replay.data, recent)
    return PredictorReplay(
        stored,
        (replay.cursor + rows) % capacity,
        jnp.minimum(capacity, replay.size + rows),
        next_key,
    ), batch
