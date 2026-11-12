"""Independent expected values validate the actual compiled execution."""

import jax
import jax.numpy as jnp
import numpy as np

from ir2_benchmark.runtime import batched_scan, run_doctor


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
    assert report.final_loss < report.initial_loss * 0.25
