"""Real transmissions separate physical wire bytes from the channel's paid fee."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from ir2_benchmark.codecs.packet import Codec  # noqa: E402
from ir2_benchmark.config import HostConfig  # noqa: E402
from ir2_benchmark.env.reset import reset  # noqa: E402
from ir2_benchmark.env.rollout import EventTrace, rollout  # noqa: E402
from ir2_benchmark.env.step import step_agent  # noqa: E402
from ir2_benchmark.evaluation.episode import (  # noqa: E402
    EpisodeExecution,
    summarize_execution,
)
from ir2_benchmark.evaluation.runner import run_manifest  # noqa: E402
from ir2_benchmark.evaluation.scenarios import (  # noqa: E402
    EvaluationMap,
    EvaluationTask,
    Method,
    make_scenario,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash  # noqa: E402
from ir2_benchmark.types import Action, LocalObservation, MapInput, make_action  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path


def _config() -> HostConfig:
    """Keep the physical corridor live while guaranteeing its nonself radio links."""
    return HostConfig(
        sensor_radius=1,
        max_distance=1,
        plan_horizon=2,
        max_rounds=1,
        max_candidates=3,
        graph_capacity=16,
        initial_budget=128,
        coverage_target=1,
        signal_threshold=-1e6,
        fault_mode=1,
        dropout_probability=1,
        charge_mode="success",
    )


def _wait_and_send(observation: LocalObservation, _config: HostConfig) -> Action:
    """Exercise actual POSITION serialization without changing the known footprint."""
    return make_action(observation.position, send=True, codec=Codec.POSITION)


def _execution(config: HostConfig) -> tuple[EvaluationTask, EpisodeExecution]:
    """Run a real two-robot corridor through live events and a frozen tail."""
    truth = jnp.ones((5, 12), jnp.uint8).at[2, 1:11].set(0)
    data = MapInput(truth, truth == 0, jnp.array([5, 12]), jnp.array([1, 2]))
    initial = reset(data, config, jax.random.PRNGKey(7), jnp.array([[1, 2], [2, 2]]))
    scene = make_scenario(
        EvaluationMap(
            "corridor.png",
            obstacle_hash(np.asarray(truth)),
            "test",
            "ir2",
            "corridor",
            12,
            scenario_hash(np.asarray(truth), (1, 2)),
        ),
        config,
        0,
    )
    method = Method("emission-fixture", "heuristic", "fixed", "none", -1, "", "", 0)
    result = rollout(initial, config, steps=2 * config.n_agents, policy=_wait_and_send)
    return next(tasks((scene,), (method,))), EpisodeExecution(
        result, np.asarray(initial.physical_seen), int(np.asarray(initial.component).sum())
    )


def _legacy(execution: EpisodeExecution) -> EpisodeExecution:
    """Construct the original thirteen-field trace to prove optional-field compatibility."""
    trace = EventTrace(*execution.rollout.trace[:13])
    return replace(execution, rollout=execution.rollout._replace(trace=trace))


def test_success_charge_lost_send_records_wire_without_fee(tmp_path: Path) -> None:
    # Given: real, valid sends with sufficient budget and every receiver externally dropped.
    task, execution = _execution(_config())
    path = tmp_path / "events.npz"
    # When: summarizing the real rollout and saving its raw packets and channel masks.
    result = summarize_execution(task, execution, path)
    # Then: serialized wire lengths are positive even though no fee or delivery occurred.
    with np.load(path, allow_pickle=False) as saved:
        active = saved["active"].astype(bool)
        wire_lengths = np.array(
            [
                struct.calcsize("<BBBBIIHBB") + struct.unpack_from("<H", raw, 12)[0]
                for raw in saved["packet_buffer"][active]
            ]
        )
        assert np.all(wire_lengths > 0)
        np.testing.assert_array_equal(saved["emitted_length"][active], wire_lengths)
        np.testing.assert_array_equal(saved["packet_buffer"][active, 2], saved["agent_id"][active])
        assert np.any(saved["physical_links"][active])
        assert not np.any(saved["external_links"][active])
        assert not np.any(saved["delivered"])
        assert not np.any(saved["spent"])
        assert result.emitted_bytes == int(wire_lengths.sum())
        assert all(saved[name].dtype.kind != "O" for name in saved.files)
    assert result.spent_bytes == 0
    assert result.delivered_payload_bytes == 0
    np.testing.assert_array_equal(execution.rollout.final_state.balances, task.scenario.budget)


def test_attempt_charge_paid_fee_agrees_with_actual_lost_wire(tmp_path: Path) -> None:
    # Given: the identical all-lost channel charges each eligible attempt.
    task, execution = _execution(replace(_config(), charge_mode="attempt"))
    # When: summarizing actual channel and budget transitions.
    result = summarize_execution(task, execution, tmp_path / "attempt.npz")
    # Then: wire traffic equals the actual balance decrease and paid fee.
    decrease = task.scenario.n_agents * task.scenario.budget - int(
        np.asarray(execution.rollout.final_state.balances).sum()
    )
    assert result.emitted_bytes > 0
    assert result.spent_bytes == result.emitted_bytes == decrease
    assert not np.any(execution.rollout.trace.delivered)


@pytest.mark.parametrize(("initial_budget", "send_policy"), [(1, "action"), (128, "none")])
def test_ineligible_send_emits_no_bytes(
    initial_budget: int, send_policy: str, tmp_path: Path
) -> None:
    # Given: a requested valid packet is unaffordable, or sending is disabled by its preset.
    config = replace(_config(), initial_budget=initial_budget, send_policy=send_policy)
    task, execution = _execution(config)
    # When: actual packet locking and channel eligibility run through the evaluator.
    result = summarize_execution(task, execution, tmp_path / "suppressed.npz")
    # Then: a constructed packet is not mistaken for a physical emission.
    assert np.any(execution.rollout.trace.packet_buffer)
    assert not np.any(execution.rollout.trace.emitted_length)
    assert result.emitted_bytes == result.spent_bytes == 0
    assert result.delivered_payload_bytes == 0


def test_legacy_attempt_trace_uses_paid_lengths_without_object_arrays(tmp_path: Path) -> None:
    # Given: an actual attempt-charge rollout restored through the old trace constructor.
    task, execution = _execution(replace(_config(), charge_mode="attempt"))
    execution = _legacy(execution)
    path = tmp_path / "legacy.npz"
    # When: evaluating its still-identifiable physical traffic.
    result = summarize_execution(task, execution, path)
    # Then: old attempt traces retain correct wire totals and omit unavailable telemetry.
    assert result.emitted_bytes == result.spent_bytes > 0
    with np.load(path, allow_pickle=False) as saved:
        assert "emitted_length" not in saved.files
        assert "packet_buffer" not in saved.files
        assert "physical_links" not in saved.files
        assert "external_links" not in saved.files
        assert all(saved[name].dtype.kind != "O" for name in saved.files)


def test_legacy_success_trace_rejects_unidentifiable_emissions(tmp_path: Path) -> None:
    # Given: real all-lost success-charge sends whose old trace contains only zero fees.
    task, execution = _execution(_config())
    execution = _legacy(execution)
    path = tmp_path / "rejected.npz"
    # When: evaluation is asked to recover physical traffic from that incomplete trace.
    with pytest.raises(RuntimeError, match="requires actual emitted_length telemetry"):
        summarize_execution(task, execution, path)
    # Then: no misleading zero-traffic artifact is written.
    assert not path.exists()


def test_frozen_state_and_rollout_tail_emit_no_bytes() -> None:
    # Given: a real corridor rollout has reached its time limit.
    config = _config()
    _, execution = _execution(config)
    state = execution.rollout.final_state
    # When: another sending action is requested after the episode froze.
    result = step_agent(state, make_action(state.positions[0], send=True), config)
    # Then: neither the frozen event nor the earlier scanned frozen tail emits traffic.
    assert not bool(result.info.active)
    assert int(result.info.emitted_length) == 0
    trace = execution.rollout.trace
    assert np.any(~np.asarray(trace.active))
    np.testing.assert_array_equal(np.asarray(trace.emitted_length)[~np.asarray(trace.active)], 0)
    for before, after in zip(jax.tree.leaves(state), jax.tree.leaves(result.state), strict=True):
        np.testing.assert_array_equal(before, after)


def test_real_runner_persists_lost_success_charge_wire_and_fee(tmp_path: Path) -> None:
    # Given: a real PNG and manifest with always-send success charging and complete dropout.
    pixels = np.full((5, 12), 127, np.uint8)
    pixels[2, 1:11], pixels[2, 1] = 195, 208
    path = tmp_path / "corridor.png"
    Image.fromarray(pixels).save(path)
    occupancy = (pixels <= 150).astype(np.uint8)  # noqa: PLR2004
    scene = make_scenario(
        EvaluationMap(
            str(path),
            obstacle_hash(occupancy),
            "test",
            "ir2",
            "corridor",
            12,
            scenario_hash(occupancy, (1, 2)),
        ),
        replace(_config(), send_policy="always", spawn_radius=1),
        0,
    )
    method = Method("emission-fixture", "heuristic", "fixed", "none", -1, "", "", 0)
    manifest, output = tmp_path / "tasks.jsonl", tmp_path / "evaluation"
    write_task_manifest(manifest, tasks((scene,), (method,)))
    # When: the public runner loads, batches, executes, summarizes and saves the task.
    summary = run_manifest(manifest, output)
    # Then: persisted result and actual trace agree on positive wire and zero paid fees.
    assert (summary.executed, summary.failures) == (1, 0)
    result_path = next((output / "results").glob("*.json"))
    result = json.loads(result_path.read_text(encoding="utf-8"))["result"]
    assert result["emitted_bytes"] > 0
    assert result["spent_bytes"] == 0
    with np.load(result["trace_file"], allow_pickle=False) as saved:
        assert result["emitted_bytes"] == int(saved["emitted_length"].sum())
        assert not np.any(saved["spent"])
        assert not np.any(saved["delivered"])
