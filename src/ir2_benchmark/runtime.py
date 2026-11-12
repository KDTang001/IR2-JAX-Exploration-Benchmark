"""Run small, real computations to verify the training runtime."""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from time import perf_counter
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


@dataclass(frozen=True, slots=True)
class DeviceReport:
    """Describe one device exposed to JAX."""

    platform: str
    kind: str
    identifier: int


@dataclass(frozen=True, slots=True)
class RuntimeReport:
    """Return measured runtime evidence, not environment throughput claims."""

    jax_version: str
    flax_version: str
    optax_version: str
    devices: tuple[DeviceReport, ...]
    batch_size: int
    horizon: int
    scan_correct: bool
    initial_loss: float
    final_loss: float
    compile_and_run_seconds: float
    warm_run_seconds: float
    purpose: str = "runtime_smoke_only"


class ScanResult(NamedTuple):
    """Carry and complete trajectory of a deterministic accumulator."""

    final: jax.Array
    history: jax.Array


class Regressor(nn.Module):
    """Exercise Flax parameters and an Optax update on the selected device."""

    @nn.compact
    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Predict one scalar for each input row."""
        hidden = nn.tanh(nn.Dense(16)(inputs))
        return nn.Dense(1)(hidden)


def batched_scan(initial: jax.Array, increments: jax.Array) -> ScanResult:
    """Accumulate independent sequences, using vmap outside the temporal scan."""

    def one(start: jax.Array, changes: jax.Array) -> ScanResult:
        def step(carry: jax.Array, change: jax.Array) -> tuple[jax.Array, jax.Array]:
            updated = carry + change
            return updated, updated

        final, history = jax.lax.scan(step, start, changes)
        return ScanResult(final, history)

    return jax.vmap(one)(initial, increments)


def run_doctor() -> RuntimeReport:
    """Verify compiled batching, temporal scan and a real gradient-based update."""
    batch_size, horizon = 32, 16
    initial = jnp.arange(batch_size, dtype=jnp.int32)
    increments = jnp.ones((batch_size, horizon), dtype=jnp.int32)
    compiled = jax.jit(batched_scan)
    start = perf_counter()
    first = jax.block_until_ready(compiled(initial, increments))
    compile_seconds = perf_counter() - start
    start = perf_counter()
    warm = jax.block_until_ready(compiled(initial, increments))
    warm_seconds = perf_counter() - start
    expected = initial[:, None] + jnp.arange(1, horizon + 1)[None, :]
    correct = bool(jnp.array_equal(first.history, expected)) and bool(
        jnp.array_equal(warm.final, initial + horizon),
    )

    model = Regressor()
    inputs = jnp.linspace(-1.0, 1.0, 32).reshape(-1, 1)
    targets = 2.0 * inputs + 0.25
    params = model.init(jax.random.key(7), inputs)["params"]
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.03))

    @jax.jit
    def update(current: TrainState) -> tuple[TrainState, jax.Array]:
        def loss_fn(parameters: Mapping[str, Mapping[str, jax.Array]]) -> jax.Array:
            predicted = current.apply_fn({"params": parameters}, inputs)
            return jnp.mean(jnp.square(predicted - targets))

        loss, gradient = jax.value_and_grad(loss_fn)(current.params)
        return current.apply_gradients(grads=gradient), loss

    initial_loss = float(jnp.mean(jnp.square(model.apply({"params": params}, inputs) - targets)))
    for _ in range(40):
        state, _loss = update(state)
    final_loss = float(
        jnp.mean(jnp.square(state.apply_fn({"params": state.params}, inputs) - targets)),
    )
    return RuntimeReport(
        jax_version=version("jax"),
        flax_version=version("flax"),
        optax_version=version("optax"),
        devices=tuple(
            DeviceReport(device.platform, device.device_kind, device.id)
            for device in jax.devices()
        ),
        batch_size=batch_size,
        horizon=horizon,
        scan_correct=correct,
        initial_loss=initial_loss,
        final_loss=final_loss,
        compile_and_run_seconds=compile_seconds,
        warm_run_seconds=warm_seconds,
    )
