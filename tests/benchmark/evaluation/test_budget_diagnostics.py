"""Budget allocation oracles and legacy artifacts keep diagnostic scope explicit."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.commands.evaluate import add_evaluate_arguments, evaluate_handler
from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation.budgets import allocate_budget
from ir2_benchmark.evaluation.records import completed_result, file_hash
from ir2_benchmark.evaluation.report import load_report_rows
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    identity,
    make_scenario,
    read_task_manifest,
    tasks,
)
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash


@pytest.mark.parametrize(("total", "balance"), [(8192, 1365), (32768, 5461), (131072, 21845)])
def test_team_six_allocations_match_the_independent_integer_oracle(
    total: int, balance: int
) -> None:
    # Given: each approved team budget divided among six robots.
    # When: allocating equal integer-byte balances.
    allocation = allocate_budget("team_total", total, 6, 512)
    # Then: the two leftover bytes are reported, never silently assigned to one robot.
    assert (
        allocation.per_robot,
        allocation.requested_team_bytes,
        allocation.unallocated_bytes,
    ) == (balance, total, 2)
    assert allocation.per_robot * 6 + allocation.unallocated_bytes == total


def test_unlimited_bound_is_exactly_one_maximum_wire_packet_for_each_own_action() -> None:
    # Given: a finite episode with at most one wire packet per robot event.
    # When: constructing its nonbinding budget.
    allocation = allocate_budget("unlimited", 0, 6, 512)
    # Then: even 512 maximal 256-byte packets fit, independent of receivers or faults.
    assert allocation.per_robot == 131072  # noqa: PLR2004
    assert allocation.requested_team_bytes is None
    assert allocation.unallocated_bytes == 0


@pytest.mark.parametrize("agents", [2, 4])
def test_team_two_and_four_leave_no_unallocated_bytes(agents: int) -> None:
    # Given: approved totals with team sizes that divide every amount exactly.
    for total in (8192, 32768, 131072):
        # When: allocating the requested team total.
        allocation = allocate_budget("team_total", total, agents, 512)
        # Then: equal integer balances use every requested byte.
        assert allocation.unallocated_bytes == 0
        assert allocation.per_robot * agents == total


def test_tagged_budget_modes_have_distinct_ids_but_keep_paired_seed_streams() -> None:
    # Given: equal actual balances represented as core or explicitly unlimited diagnostics.
    data = EvaluationMap("map.png", "map", "test", "ir2", "rooms", 64, "anchor")
    config = HostConfig(max_rounds=2, initial_budget=512)
    core = make_scenario(data, config, 0)
    unlimited = make_scenario(data, config, 0, budget_mode="unlimited")
    team = make_scenario(
        data,
        replace(config, initial_budget=1365),
        0,
        budget_mode="team_total",
        requested_team_bytes=8192,
        unallocated_bytes=2,
    )
    # When: recording budget provenance independently of its ledger balance.
    # Then: diagnostic results cannot alias core IDs, while all methods retain paired RNG roots.
    assert len({core.scenario_id, unlimited.scenario_id, team.scenario_id}) == 3  # noqa: PLR2004
    assert core.spawn_seed == unlimited.spawn_seed == team.spawn_seed
    assert core.fault_seed == unlimited.fault_seed == team.fault_seed


def test_unlimited_budget_pays_every_maximum_packet_in_the_real_ledger() -> None:
    # Given: an actual ledger and a short finite episode of maximum-size attempted sends.
    pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    from ir2_benchmark.codecs.packet import MAX_PACKET_SIZE  # noqa: PLC0415
    from ir2_benchmark.communication.ledger import charge_attempt  # noqa: PLC0415

    balance = jnp.int32(allocate_budget("unlimited", 0, 2, 4).per_robot)
    # When: all four own actions attempt the protocol's maximum wire size.
    for _ in range(4):
        attempt = charge_attempt(
            balance, jnp.int32(MAX_PACKET_SIZE), requested=jnp.bool_(1), packet_valid=jnp.bool_(1)
        )
        assert bool(attempt.eligible)
        balance = attempt.balance
    # Then: no action is rejected by the finite representation of unlimited budget.
    assert int(balance) == 0


def test_core_ids_and_old_manifest_and_result_checksums_remain_compatible(tmp_path: Path) -> None:
    # Given: the pre-diagnostic scenario/result formats, without new default metadata fields.
    config = HostConfig()
    map_data = EvaluationMap("map.png", "occupancy", "test", "ir2", "rooms", 512, "anchor")
    scene = make_scenario(map_data, config, 0)
    legacy_id = identity(
        {
            "map_hash": "occupancy",
            "map_scenario_hash": "anchor",
            "split": "test",
            "spawn_seed": scene.spawn_seed,
            "fault_seed": scene.fault_seed,
            "config": asdict(config),
        }
    )
    assert scene.scenario_id == identity(
        {
            "map_hash": "occupancy",
            "map_scenario_hash": "anchor",
            "split": "test",
            "spawn_seed": scene.spawn_seed,
            "fault_seed": scene.fault_seed,
            "config": asdict(config),
            "implementations": scene.implementations,
        }
    )
    scene = replace(scene, scenario_id=legacy_id, implementations=None)
    method = Method(
        "A_learned_predict", "A", "learned", "predict", 0, "checkpoint", "weights", 2000000
    )
    task = next(tasks((scene,), (method,)))
    assert task.result_id == identity(
        [legacy_id, {name: value for name, value in asdict(method).items() if name != "checkpoint"}]
    )
    raw_task = asdict(task)
    for field in ("budget_mode", "requested_team_bytes", "unallocated_bytes", "implementations"):
        raw_task["scenario"].pop(field)
    manifest = tmp_path / "old-tasks.jsonl"
    manifest.write_text(json.dumps(raw_task) + "\n", encoding="utf-8")
    trace = tmp_path / "old-trace.npz"
    np.savez(trace, coverage=np.array([0.25]))
    raw = {
        "result_id": task.result_id,
        "method_id": method.method_id,
        "training_seed": 0,
        "source": "ir2",
        "family": "rooms",
        "map_hash": "occupancy",
        "outcome": "timeout",
        "checkpoint_hash": "weights",
        "configuration_hash": scene.config_hash,
        "trace_file": str(trace),
        "trace_hash": file_hash(trace),
        "metrics": {"auc": 0.25},
    }
    envelope = {"result": raw, "checksum": identity(raw)}
    path = tmp_path / "results" / f"{task.result_id}.json"
    path.parent.mkdir()
    path.write_text(json.dumps(envelope), encoding="utf-8")
    # When: restoring old tasks and loading old result envelopes before applying defaults.
    assert next(read_task_manifest(manifest)) == task
    assert completed_result(path, task)
    rows, corrupt = load_report_rows(tmp_path)
    # Then: default provenance does not change the original checksum or identity.
    assert corrupt == ()
    assert rows[0].budget_mode == "per_robot"
    assert json.loads(path.read_text()) == envelope


@pytest.mark.parametrize(
    ("mode", "budgets", "expected"),
    [
        ("team_total", [8192], (1365, 8192, 2)),
        ("unlimited", [], (768, None, 0)),
    ],
)
def test_real_evaluate_planning_cli_records_budget_mode_and_allocation(
    tmp_path: Path,
    mode: str,
    budgets: list[int],
    expected: tuple[int, int | None, int],
) -> None:
    # Given: a real retained-map registry and an explicit diagnostic declaration.
    pixels = np.full((7, 7), 195, np.uint8)
    pixels[3, 3] = 208
    Image.fromarray(pixels).save(tmp_path / "map.png")
    occupancy = np.zeros((7, 7), np.uint8)
    entry = {
        "path": "map.png",
        "split": "test",
        "family": "rooms",
        "bucket": 64,
        "occupancy_hash": obstacle_hash(occupancy),
        "scenario_hash": scenario_hash(occupancy, (3, 3)),
    }
    (tmp_path / "maps.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    sources = tmp_path / "sources.json"
    sources.write_text(
        json.dumps([{"source": "ir2", "root": ".", "manifest": "maps.jsonl"}]), encoding="utf-8"
    )
    declaration = tmp_path / "suite.json"
    declaration.write_text(
        json.dumps(
            {
                "budget_mode": mode,
                "n_agents": [6],
                "budgets": budgets,
                "conditions": [[0, 0, 64]],
                "scenario_seeds": [0],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(json.dumps(asdict(HostConfig(max_rounds=3))), encoding="utf-8")
    parser = argparse.ArgumentParser()
    add_evaluate_arguments(parser)
    args = parser.parse_args(
        [
            "--route",
            "heuristic",
            "--suite",
            "formal",
            "--config",
            str(config),
            "--map-sources",
            str(sources),
            "--suite-config",
            str(declaration),
            "--plan-only",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    # When: the production CLI constructs the immutable execution workload.
    assert evaluate_handler(args) == 0
    scene = next(read_task_manifest(tmp_path / "out/tasks.jsonl")).scenario
    # Then: the ledger balance and its policy provenance are independently visible.
    assert scene.budget_mode == mode
    assert (scene.budget, scene.requested_team_bytes, scene.unallocated_bytes) == expected
