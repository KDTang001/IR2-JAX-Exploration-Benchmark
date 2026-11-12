"""Pairing and stored workload identities are independent of method ordering."""

from dataclasses import replace
from pathlib import Path

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    make_scenario,
    read_task_manifest,
    tasks,
    write_task_manifest,
)


def test_method_order_does_not_change_scenarios_or_result_ids(tmp_path: Path) -> None:
    # Given: one map and two separately trained methods sharing a paired scenario.
    map_data = EvaluationMap(
        "test.png", "map-content", "test", "ir2", "corridor", 512, "fixture-anchor"
    )
    scenario = make_scenario(map_data, HostConfig(), 3)
    methods = (
        Method("A_fixed_zoh", "A", "fixed", "zoh", 1, "a", "weights-a", 2000000),
        Method("B_fixed_zoh", "B", "fixed", "zoh", 1, "b", "weights-b", 2000000),
    )
    # When: methods are scheduled in opposite orders and the workload is restored.
    forward, backward = tuple(tasks((scenario,), methods)), tuple(tasks((scenario,), methods[::-1]))
    manifest = tmp_path / "tasks.jsonl"
    write_task_manifest(manifest, forward)
    restored = tuple(read_task_manifest(manifest))
    # Then: paired RNG counters and checkpoint-bound IDs match exactly.
    assert {task.result_id for task in forward} == {task.result_id for task in backward}
    assert restored == forward


def test_changing_budget_or_fault_condition_produces_distinct_task_identity() -> None:
    # Given: the same map/seed with three requested independent conditions.
    map_data = EvaluationMap(
        "test.png", "map-content", "test", "ir2", "complex", 1024, "fixture-anchor"
    )
    base = HostConfig()
    # When: changing budget or enabling Burst without Dropout.
    normal = make_scenario(map_data, base, 9)
    budget = make_scenario(map_data, replace(base, initial_budget=2048), 9)
    burst = make_scenario(map_data, replace(base, fault_mode=2, n_agents=6), 9)
    # Then: scenarios differ, while spawn/fault streams remain paired across conditions.
    assert len({normal.scenario_id, budget.scenario_id, burst.scenario_id}) == 3  # noqa: PLR2004
    assert normal.spawn_seed == budget.spawn_seed == burst.spawn_seed
    assert normal.fault_seed == budget.fault_seed == burst.fault_seed
    assert len(burst.isolation_ids) == 3  # noqa: PLR2004
    assert burst.drop_probability == 0


def test_changed_spawn_anchor_changes_the_frozen_scenario_identity() -> None:
    # Given: identical oriented occupancy with two different anchor hashes.
    first = EvaluationMap("map.png", "same-occupancy", "test", "ir2", "rooms", 512, "anchor-one")
    second = replace(first, scenario_hash="anchor-two")
    # When: the same scenario counter is applied to each distinct anchor.
    a, b = make_scenario(first, HostConfig(), 1), make_scenario(second, HostConfig(), 1)
    # Then: cached results from the old birth region cannot be reused for the new region.
    assert a.scenario_id != b.scenario_id
    assert a.map_scenario_hash != b.map_scenario_hash
