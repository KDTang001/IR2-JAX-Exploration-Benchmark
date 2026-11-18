"""Capacity acceptance preserves workload identity and rejects semantic changes."""

from __future__ import annotations

import gzip
import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest

from ir2_benchmark.config import ConfigError, load_config
from ir2_benchmark.evaluation.scenarios import EvaluationTask, read_task_manifest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "capacity_check", ROOT / "scripts/freeze_capacity_check.py"
)
assert SPEC is not None
assert SPEC.loader is not None
PLANNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLANNER)


@pytest.fixture
def parent(tmp_path: Path) -> tuple[EvaluationTask, ...]:
    """Use the actual frozen C400 manifest without loading maps or policy weights."""
    manifest = tmp_path / "parent.jsonl"
    archive = ROOT / "research/evidence/batch-cost-v1-tasks.jsonl.gz"
    manifest.write_bytes(gzip.decompress(archive.read_bytes()))
    return tuple(read_task_manifest(manifest))


def test_capacity_changes_identity_but_preserves_paired_seeds_and_inputs(
    parent: tuple[EvaluationTask, ...],
) -> None:
    extended = load_config(ROOT / "configs/benchmark_capacity1024.json")
    planned = PLANNER.capacity_tasks(parent, extended)
    assert len({task.result_id for task in planned}) == 10  # noqa: PLR2004
    assert not {task.result_id for task in planned} & {task.result_id for task in parent}
    for old, new in zip(parent, planned, strict=True):
        assert old.method == new.method
        assert old.scenario.map_hash == new.scenario.map_hash
        assert old.scenario.map_scenario_hash == new.scenario.map_scenario_hash
        assert old.scenario.spawn_seed == new.scenario.spawn_seed
        assert old.scenario.fault_seed == new.scenario.fault_seed
        assert old.scenario.config_hash != new.scenario.config_hash


@pytest.mark.parametrize("mutation", ["neighbors", "merge", "split", "identity", "missing"])
def test_capacity_planner_rejects_undeclared_scope_or_semantic_changes(
    parent: tuple[EvaluationTask, ...], mutation: str
) -> None:
    extended = load_config(ROOT / "configs/benchmark_capacity1024.json")
    if mutation == "neighbors":
        extended = replace(extended, max_candidates=31)
    elif mutation == "merge":
        extended = replace(extended, graph_merge_threshold=800)
    elif mutation == "split":
        changed = replace(parent[0], scenario=replace(parent[0].scenario, split="test"))
        parent = (changed, *parent[1:])
    elif mutation == "identity":
        parent = (replace(parent[0], result_id="changed"), *parent[1:])
    else:
        parent = parent[:-1]
    with pytest.raises(ConfigError):
        PLANNER.capacity_tasks(parent, extended)
