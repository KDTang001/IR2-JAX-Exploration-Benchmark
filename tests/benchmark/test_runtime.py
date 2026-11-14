"""Independent expected values validate the actual compiled execution."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark import runtime
from ir2_benchmark.cli import main
from ir2_benchmark.runtime import batched_scan, run_doctor

if TYPE_CHECKING:
    from pathlib import Path


def test_compiled_scan_keeps_each_environment_independent() -> None:
    # Given: different initial values and signed, nonuniform actions.
    initial = jnp.array([10, -3], dtype=jnp.int32)
    changes = jnp.array([[1, -2, 4], [0, 5, -1]], dtype=jnp.int32)
    # When: the same compiled temporal program runs across the batch.
    result = jax.jit(batched_scan)(initial, changes)
    # Then: neither environment borrows the other environment's state.
    np.testing.assert_array_equal(result.history, [[11, 9, 13], [-3, 2, 1]])
    np.testing.assert_array_equal(result.final, [13, 1])


def test_training_runtime_reduces_regression_error() -> None:
    # Given: the fixed dataset and seed built into the smoke diagnostic.
    # When: Flax parameters are trained through real JAX/Optax updates.
    report = run_doctor()
    # Then: the optimized model improves substantially on its initialization.
    assert report.scan_correct
    assert report.initial_loss is not None
    assert report.final_loss is not None
    assert report.final_loss < report.initial_loss * 0.25


def test_runtime_only_cli_runs_inference_without_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Given: optimizer construction is forbidden for the inference-only diagnostic.
    def forbid_optimizer(_learning_rate: float) -> None:
        pytest.fail("runtime-only doctor attempted to create an optimizer")

    output = tmp_path / "doctor.json"
    monkeypatch.setattr(runtime.optax, "adam", forbid_optimizer)
    monkeypatch.setattr(
        sys,
        "argv",
        ["ir2-benchmark", "doctor", "--runtime-only", "--output", str(output)],
    )
    # When: the public CLI executes actual compiled scan and Flax inference.
    status = main()
    # Then: no optimizer is constructed and the successful JSON has no invented loss.
    report = json.loads(output.read_text(encoding="utf-8"))
    assert status == 0
    assert report["scan_correct"]
    assert report["inference_correct"]
    assert report["optimizer_status"] == "skipped_runtime_only"
    assert report["optimizer_updates"] == 0
    assert report["initial_loss"] is report["final_loss"] is None
