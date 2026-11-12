"""Synchronize complete result PyTrees and count actual work in stable repeated measurements."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, TypeVar

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable
Result = TypeVar("Result")


@dataclass(frozen=True, slots=True)
class Timing:
    """Compilation/first execution is separate from at least ten synchronized samples."""

    first_call_seconds: float
    samples_seconds: tuple[float, ...]
    mean_seconds: float
    actual_units: int
    units_per_second: float | None
    warmup_count: int = 3


def measure(
    function: Callable[[], Result],
    units: Callable[[Result], int],
    *,
    warmups: int = 3,
    repeats: int = 10,
) -> Timing:
    """Block every output PyTree, including the first call, warmups and timed repetitions."""
    import jax  # noqa: PLC0415

    start = perf_counter()
    first = function()
    jax.block_until_ready(first)
    first_seconds = perf_counter() - start
    for _ in range(warmups):
        jax.block_until_ready(function())
    samples, counts = [], []
    for _ in range(repeats):
        start = perf_counter()
        result = function()
        jax.block_until_ready(result)
        samples.append(perf_counter() - start)
        counts.append(units(result))
    total_units = sum(counts)
    return Timing(
        first_seconds,
        tuple(samples),
        float(np.mean(samples)),
        counts[-1],
        total_units / sum(samples) if total_units else None,
        warmups,
    )
