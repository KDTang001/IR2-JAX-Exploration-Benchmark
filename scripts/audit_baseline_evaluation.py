"""Independently replay fixed-baseline traces with original IR2 sensing and host accounting."""

# Literal codec/header rules and exact geometry are deliberate independent oracles.
# ruff: noqa: PLR2004, S101

from __future__ import annotations

import argparse
import importlib
import json
import struct
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image
from scipy.ndimage import label


class VerifiedEpisode(TypedDict):
    """Only independently recomputed values are returned to the acceptance gate."""

    result_id: str
    events: int
    final_coverage: float
    auc: float
    emitted_bytes: int
    spent_bytes: int
    delivered_payload_bytes: int
    overlap: float
    allocation_concentration: float


def digest(value: dict) -> str:
    """Encode the documented canonical JSON identity without production helpers."""
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_episode(job: tuple[str, str]) -> VerifiedEpisode:  # noqa: C901, PLR0912, PLR0915
    """Reconstruct birth scans, every physical scan, standard wire cost and end metrics."""
    serialized, directory = job
    task = json.loads(serialized)
    scene, method = task["scenario"], task["method"]
    config, n = scene["config"], scene["n_agents"]
    assert method["route"] == "baseline"
    assert config["signal_model"] == "ir2"
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sensor_work = importlib.import_module("sensor").sensor_work
    model = importlib.import_module("ss_realistic_model").SS_realistic_model(
        P_T=config["transmit_power"],
        threshold_ss=config["signal_threshold"],
        gamma=config["gamma_free"],
        gamma_obst=config["gamma_obstacle"],
        dist_o=config["reference_distance"],
        PL_o=config["reference_loss"],
    )
    assert all(
        config[key] == 0
        for key in (
            "signal_noise_free_min",
            "signal_noise_free_max",
            "signal_noise_obstacle_min",
            "signal_noise_obstacle_max",
        )
    )
    with Image.open(scene["map_path"]) as image:
        pixels = np.asarray(image.convert("L"))
    truth = np.where(pixels <= 150, 1, 255).astype(np.int32)
    marker = np.argwhere(pixels == 208)
    y, x = marker[np.argmin(np.sum((marker - marker.mean(axis=0)) ** 2, axis=1))]
    labels, _ = label(truth == 255)
    component = labels == labels[y, x]
    area = int(component.sum())
    cells = np.argwhere(component)[:, ::-1]
    squared = np.sum((cells - [x, y]) ** 2, axis=1)
    radius = max(config["spawn_radius"], int(np.ceil(np.sqrt(np.sort(squared)[n - 1]))))
    eligible = cells[squared <= radius**2]
    positions = eligible[
        np.random.default_rng(scene["spawn_seed"]).choice(len(eligible), n, replace=False)
    ]

    @lru_cache(maxsize=64)
    def scan(x: int, y: int) -> tuple[np.ndarray, np.ndarray]:
        known = sensor_work(
            np.array([x, y]), config["sensor_radius"], np.full(truth.shape, 127), truth
        )
        return np.flatnonzero(known != 127), np.flatnonzero((known == 255) & component)

    footprints = np.zeros((n, truth.size), bool)
    knowledge = np.zeros((n, truth.size), bool)
    union = np.zeros(truth.size, bool)
    allocation = np.zeros(n, np.int64)
    for agent in range(n):
        observed, free = scan(*map(int, positions[agent]))
        knowledge[agent, observed] = True
        footprints[agent, free] = True
        allocation[agent] += np.count_nonzero(footprints[agent] & ~union)
        union |= footprints[agent]
    initial = union.sum() / area
    record = Path(directory) / "results" / f"{task['result_id']}.json"
    envelope = json.loads(record.read_text())
    row = envelope["result"]
    assert envelope["checksum"] == digest(row)
    assert row["checkpoint_hash"] == method["checkpoint_hash"]
    assert row["configuration_hash"] == scene["config_hash"]
    trace_file = Path(row["trace_file"])
    assert sha256(trace_file.read_bytes()).hexdigest() == row["trace_hash"]
    assert row["outcome"] in ("success", "timeout"), row.get("error")
    balances = np.full(n, scene["budget"], np.int64)
    sequences = np.zeros(n, np.int64)
    pending: dict[tuple[int, int], list[int]] = {}
    seen = np.zeros(n, bool)
    cold = emitted = paid = payload_delivered = 0
    coverage = []
    with np.load(trace_file, allow_pickle=False) as trace:
        active = np.flatnonzero(trace["active"])
        for event in active:
            agent = int(trace["agent_id"][event])
            assert event == int(trace["event"][event])
            assert agent == event % n
            for sender in range(n):
                knowledge[agent, pending.pop((agent, sender), [])] = True
            packet = trace["packet_buffer"][event].tobytes()
            version, codec, sender, flags, sequence, generated, size, recipients, reserved = (
                struct.unpack("<BBBBIIHBB", packet[:16])
            )
            assert (version, sender, flags, sequence, generated, recipients, reserved) == (
                1,
                agent,
                0,
                sequences[agent],
                event,
                ((1 << n) - 1) ^ (1 << agent),
                0,
            )
            own_round = event // n
            comm = method["communication"]
            send = comm != "off" and own_round % 4 == 0
            expected_codec = {
                "off": 3,
                "plan": 3,
                "map": 4,
                "mixed": (3, 4, 4, 1)[(own_round // 4) % 4],
            }[comm]
            assert codec == expected_codec
            delta = []
            if codec == 3:
                assert size == 9 + 4 * config["plan_horizon"]
                assert struct.unpack("<HH", packet[16:20]) == tuple(positions[agent])
                assert struct.unpack("<HH", packet[20:24]) == tuple(trace["goals"][event, agent])
            elif codec == 4:
                count = packet[16]
                assert count <= 47
                assert size == 1 + 5 * count
                for index in range(count):
                    px, py, state = struct.unpack("<HHB", packet[17 + 5 * index : 22 + 5 * index])
                    assert px < truth.shape[1]
                    assert py < truth.shape[0]
                    flat = py * truth.shape[1] + px
                    assert state == int(truth[py, px] != 255)
                    assert knowledge[agent, flat]
                    delta.append(flat)
            else:
                assert codec == 1
                assert size == 4
                assert struct.unpack("<HH", packet[16:20]) == tuple(positions[agent])
            length = 16 + size if send and balances[agent] >= 16 + size else 0
            assert int(trace["emitted_length"][event]) == length
            after = trace["positions"][event, :n]
            physical = np.array(
                [
                    receiver != agent
                    and model.is_within_signal_strength(truth, after[agent], after[receiver])
                    for receiver in range(n)
                ]
            )
            np.testing.assert_array_equal(trace["physical_links"][event, :n], physical)
            external = trace["external_links"][event, :n]
            if scene["fault_mode"] == 0:
                expected_external = np.arange(n) != agent
                np.testing.assert_array_equal(external, expected_external)
            if scene["fault_mode"] == 2:
                isolated = set(scene["isolation_ids"])
                in_burst = (
                    scene["burst_start"]
                    <= own_round
                    < scene["burst_start"] + scene["burst_duration"]
                )
                expected_external = np.array(
                    [
                        r != agent and not (in_burst and (agent in isolated or r in isolated))
                        for r in range(n)
                    ]
                )
                np.testing.assert_array_equal(external, expected_external)
            deliveries = physical & external & bool(length)
            np.testing.assert_array_equal(trace["delivered"][event, :n], deliveries)
            cost = length if config["charge_mode"] == "attempt" or deliveries.any() else 0
            assert int(trace["spent"][event]) == cost
            balances[agent] -= cost
            np.testing.assert_array_equal(trace["balances"][event, :n], balances)
            sequences[agent] += bool(length)
            emitted += length
            paid += cost
            payload_delivered += max(length - 16, 0) * int(deliveries.sum())
            for receiver in np.flatnonzero(deliveries):
                pending[int(receiver), agent] = delta.copy()
            observed, free = scan(*map(int, after[agent]))
            allocation[agent] += np.count_nonzero(~union[free])
            footprints[agent, free] = True
            union[free] = True
            knowledge[agent, observed] = True
            fraction = union.sum() / area
            assert abs(float(trace["coverage"][event]) - fraction) <= 6e-8
            coverage.append(fraction)
            cold += int(not seen[agent])
            seen |= deliveries
            positions = after.copy()
        assert not np.any(trace["emitted_length"][~trace["active"]])
    rounds = coverage[n - 1 :: n]
    if len(coverage) % n:
        rounds.append(coverage[-1])
    final_coverage = coverage[-1] if coverage else initial
    success = final_coverage >= config["coverage_target"]
    assert (row["outcome"] == "success") == success
    if not success:
        assert len(coverage) == config["max_rounds"] * n
    auc = (sum(rounds) + (config["max_rounds"] - len(rounds)) * final_coverage) / config[
        "max_rounds"
    ]
    overlap = 1 - union.sum() / footprints.sum()
    concentration = float(np.sum((allocation / allocation.sum()) ** 2))
    for field, value in [("final_coverage", final_coverage), ("auc", auc)]:
        assert abs(row["metrics"][field] - value) <= 1e-7, field
    for field, value in [
        ("emitted_bytes", emitted),
        ("spent_bytes", paid),
        ("delivered_payload_bytes", payload_delivered),
        ("cold_start_events", cold),
        ("active_events", len(coverage)),
    ]:
        assert row[field] == value, field
    assert abs(row["overlap"] - overlap) <= 1e-12
    assert abs(row["allocation_concentration"] - concentration) <= 1e-12
    return VerifiedEpisode(
        result_id=task["result_id"],
        events=len(coverage),
        final_coverage=float(final_coverage),
        auc=float(auc),
        emitted_bytes=emitted,
        spent_bytes=paid,
        delivered_payload_bytes=payload_delivered,
        overlap=float(overlap),
        allocation_concentration=concentration,
    )


def main() -> int:
    """Audit all supplied actual tasks; any discrepancy exits without a passing artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    jobs = [(line, str(args.results)) for line in args.manifest.read_text().splitlines()]
    verified = []
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(verify_episode, jobs, chunksize=2):
            verified.append(result)
            if len(verified) % 32 == 0:
                print(f"Independently verified {len(verified)}/{len(jobs)} episodes", flush=True)  # noqa: T201
    args.output.write_text(
        json.dumps({"status": "passed", "episodes": len(verified), "results": verified}, indent=2)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
