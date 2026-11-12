"""The four A controls spend equal real events on paired genuine scenarios."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from ir2_benchmark.config import HostConfig
from ir2_benchmark.training.comparison_a import run_a_comparison
from ir2_benchmark.training.train_a import TrainAConfig
from ir2_benchmark.types import MapInput


def test_comparison_when_only_communication_and_estimator_vary(tmp_path: Path) -> None:
    # Given: every control receives the same actual map, event quota, seed and optimizer settings.
    grid = jnp.ones((10, 10), jnp.uint8).at[1:9, 1:9].set(0)
    scenario = MapInput(grid, grid == 0, jnp.asarray([10, 10]), jnp.asarray([4, 4]))
    config = HostConfig(
        n_agents=2,
        sensor_radius=2,
        max_distance=1,
        max_rounds=6,
        max_candidates=3,
        initial_budget=90,
        spawn_radius=1,
    )
    settings = TrainAConfig(
        events=8,
        warmup_events=2,
        rollout_length=2,
        ppo_epochs=1,
        batch_size=1,
        hidden=8,
        suite="micro",
    )
    # When: each control performs actual environment sampling, optimization and fresh evaluation.
    result = run_a_comparison(scenario, config, settings, jax.random.PRNGKey(5), tmp_path)
    comparison = json.loads(result.read_text(encoding="utf-8"))
    # Then: equal interaction budgets produce independently recorded outcomes, including losses.
    rows = comparison["methods"]
    assert {row["method"] for row in rows} == {
        "fixed_zoh",
        "fixed_predict",
        "learned_zoh",
        "learned_predict",
    }
    assert {row["training"]["real_events"] for row in rows} == {8}
    assert all(
        "auc" in row["evaluation"] and "total_spent_bytes" in row["evaluation"] for row in rows
    )
