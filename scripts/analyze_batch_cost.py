#!/usr/bin/env python3
"""Audit full-episode B1/B4 artifacts without importing JAX or initializing a GPU."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np

REWARD_ATOL: Final = 3e-5
AUC_ATOL: Final = 1e-12
EXPECTED_CASES: Final = 4
EXPECTED_PAIRS: Final = 10
BUDGET: Final = 8192


def require(condition: bool, detail: str) -> None:  # noqa: FBT001 - Assertion predicate, not a flag.
    """Keep artifact validation enabled even when Python assertions are disabled."""
    if not condition:
        raise RuntimeError(detail)


def digest(path: Path) -> str:
    """Return the byte identity of a compact input or result artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(root: Path) -> None:  # noqa: C901, PLR0912, PLR0915
    """Independently validate saved results, accounting, event metrics and paired traces."""
    declaration = json.loads((root / "mini-declaration.json").read_text())
    request = json.loads((root / "profile-request.json").read_text())
    for path, expected in request["input_hashes"].items():
        require(digest(Path(path)) == expected, f"frozen input changed: {path}")
    out = root / "resource-cost-mini"
    cases = []
    pairs = defaultdict(dict)
    for case in declaration["cases"]:
        name = case["case_id"]
        record_path = out / (name + ".record.json")
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text())
        require(record["exit_code"] == 0 and not record["timeout"], f"case failed: {name}")
        summary = json.loads((out / name / "runner_summary.json").read_text())
        require(
            summary == {"requested": 5, "executed": 5, "resumed": 0, "failures": 0},
            f"incomplete runner: {name}",
        )
        memory = json.loads((out / (name + ".memory.json")).read_text())
        timings = json.loads((out / (name + ".memory.timing.json")).read_text())["boundaries"]
        areas = {
            row["result_id"]: row["component_area"]
            for row in timings
            if row["phase"] == "load_lane"
        }
        phases = defaultdict(float)
        for row in timings:
            phases[row["phase"]] += row["seconds"]
        batches = [row for row in timings if row["phase"] == "execute_batch"]
        expected_sizes = [1] * 5 if case["batch_size"] == 1 else [4, 1]
        require(
            [row["batch_size"] for row in batches] == expected_sizes,
            f"incorrect actual batching: {name}",
        )
        tasks = [json.loads(line) for line in Path(case["task_manifest"]).read_text().splitlines()]
        require(
            [key for row in batches for key in row["result_ids"]]
            == [task["result_id"] for task in tasks],
            f"batch task order changed: {name}",
        )
        episodes = []
        for task in tasks:
            result_id = task["result_id"]
            path = out / name / "results" / (result_id + ".json")
            envelope = json.loads(path.read_text())
            result = envelope["result"]
            canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
            require(
                hashlib.sha256(canonical).hexdigest() == envelope["checksum"],
                f"result checksum failed: {path}",
            )
            require(result["result_id"] == result_id, f"wrong result identity: {path}")
            scene, method = task["scenario"], task["method"]
            require(
                result["configuration_hash"] == scene["config_hash"]
                and result["checkpoint_hash"] == method["checkpoint_hash"]
                and result["method_id"] == method["method_id"]
                and result["map_hash"] == scene["map_hash"]
                and result["map_scenario_hash"] == scene["map_scenario_hash"],
                f"changed configuration or input identity: {path}",
            )
            trace_path = Path(result["trace_file"])
            require(digest(trace_path) == result["trace_hash"], f"trace checksum failed: {path}")
            with np.load(trace_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            active = arrays["active"].astype(bool)
            events = int(active.sum())
            require(events == result["active_events"], f"event count failed: {path}")
            require(not arrays["invalid_action"][active].any(), f"illegal action: {path}")
            coverage = arrays["coverage"][active].astype(np.float64)
            counts = np.rint(coverage * areas[result_id]).astype(np.int64)
            require(not (np.diff(counts) < 0).any(), f"decreasing coverage: {path}")
            emitted = arrays["emitted_length"][active].astype(np.int64)
            spent = arrays["spent"][active].astype(np.int64)
            delivered = arrays["delivered"][active].sum(axis=-1, dtype=np.int64)
            require(np.array_equal(emitted, spent), f"attempt fee mismatch: {path}")
            require(int(emitted.sum()) == result["emitted_bytes"], f"emission mismatch: {path}")
            require(int(spent.sum()) == result["spent_bytes"], f"spent mismatch: {path}")
            payload_copies = int((np.maximum(emitted - 16, 0) * delivered).sum())
            require(
                payload_copies == result["delivered_payload_bytes"],
                f"delivered payload mismatch: {path}",
            )
            require(
                int(arrays["balances"][-1].sum()) + int(spent.sum()) == 4 * 8192,
                f"balance conservation failed: {path}",
            )
            for agent in range(4):
                paid = spent[arrays["agent_id"][active] == agent].sum()
                require(
                    int(arrays["balances"][-1][agent] + paid) == BUDGET,
                    f"individual balance conservation failed: {path}:{agent}",
                )
            metrics = result["metrics"]
            round_coverage = list(coverage[3::4])
            if len(coverage) % 4:
                round_coverage.append(coverage[-1])
            require(bool(round_coverage), f"unexpected initially complete map: {path}")
            auc = (sum(round_coverage) + (512 - len(round_coverage)) * round_coverage[-1]) / 512
            require(abs(auc - metrics["auc"]) < AUC_ATOL, f"AUC mismatch: {path}")
            require(coverage[-1] == metrics["final_coverage"], f"final coverage mismatch: {path}")
            require(result["outcome"] in ("success", "timeout"), f"execution failure: {path}")
            require(
                bool(arrays["terminated"][-1]) == metrics["success"],
                f"termination mismatch: {path}",
            )
            require(
                bool(arrays["truncated"][-1]) == metrics["timeout"], f"truncation mismatch: {path}"
            )
            item = {
                "result_id": result_id,
                "family": result["family"],
                "active_events": events,
                "outcome": result["outcome"],
                "final_coverage": metrics["final_coverage"],
                "auc": metrics["auc"],
                "success_round": metrics["success_round"],
                "component_area": areas[result_id],
                "trace_sha256": result["trace_hash"],
                "trace_bytes": trace_path.stat().st_size,
                "emitted_bytes": int(emitted.sum()),
                "spent_bytes": int(spent.sum()),
                "final_balances": arrays["balances"][-1].tolist(),
            }
            episodes.append(item)
            pairs[result_id][case["batch_size"]] = (arrays, item)
        own_pids = {memory["pid"], record["supervised_pid"]}
        foreign = set()
        peak_mib = 0
        telemetry_failures = 0
        for line in record["resource_before"]["processes"].splitlines():
            if line.split(",")[0].strip().isdigit():
                foreign.add(int(line.split(",")[0]))
        for sample in record["gpu_samples"]:
            telemetry_failures += sample["processes"].startswith("query_failed:")
            for line in sample["processes"].splitlines():
                fields = line.split(",")
                if fields[0].strip().isdigit():
                    pid = int(fields[0])
                    if pid not in own_pids:
                        foreign.add(pid)
                    elif fields[1].strip().isdigit():
                        peak_mib = max(peak_mib, int(fields[1]))
        cases.append(
            {
                "case_id": name,
                "supervised_wall_seconds": record["wall_seconds"],
                "phases_seconds": dict(phases),
                "batches": batches,
                "episodes": episodes,
                "device_active_events_per_second": sum(e["active_events"] for e in episodes)
                / phases["device_execute"],
                "sampled_own_nvml_peak_mib": peak_mib,
                "foreign_gpu_pids": sorted(foreign),
                "telemetry_failures": telemetry_failures,
                "clean_timing": not foreign and not telemetry_failures,
                "memory": memory,
            }
        )
    paired = []
    for result_id, variants in pairs.items():
        if set(variants) != {1, 4}:
            continue
        (left, left_item), (right, right_item) = variants[1], variants[4]
        require(
            left_item["component_area"] == right_item["component_area"],
            f"different component area: {result_id}",
        )
        require(left.keys() == right.keys(), f"trace schema changed: {result_id}")
        differences = []
        for key in left:
            x, y = left[key], right[key]
            require(
                x.shape == y.shape and x.dtype == y.dtype,
                f"changed trace shape or dtype: {result_id}:{key}",
            )
            if np.array_equal(x, y):
                continue
            match key:
                case "coverage" | "initial_coverage":
                    area = left_item["component_area"]
                    same = np.array_equal(np.rint(x * area), np.rint(y * area))
                    close = np.all(np.abs(x - y) <= 2 * np.spacing(np.maximum(abs(x), abs(y))))
                    require(bool(same and close), f"different covered cells: {result_id}:{key}")
                case "reward":
                    require(
                        bool(np.allclose(x, y, rtol=0, atol=REWARD_ATOL)),
                        f"different rewards: {result_id}",
                    )
                case _:
                    detail = f"changed trajectory or communication: {result_id}:{key}"
                    raise RuntimeError(detail)
            differences.append({"field": key, "max_abs_difference": float(np.max(np.abs(x - y)))})
        paired.append({"result_id": result_id, "passed": True, "float_differences": differences})
    result = {
        "complete": len(cases) == EXPECTED_CASES and len(paired) == EXPECTED_PAIRS,
        "timing_comparable": len(cases) == EXPECTED_CASES
        and all(case["clean_timing"] for case in cases),
        "case_count": len(cases),
        "paired_task_count": len(paired),
        "cases": cases,
        "pairs": paired,
        "analyzer_sha256": digest(Path(__file__)),
    }
    (root / "batch-cost-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        json.dumps({key: result[key] for key in ("complete", "case_count", "paired_task_count")})
        + "\n"
    )


if __name__ == "__main__":
    root_path = Path(sys.argv[1]).resolve()
    if sys.argv[2:] == ["--wait"]:
        limit = json.loads((root_path / "gpu-window.json").read_text())["deadline_epoch"] + 30
        while not (root_path / "gpu-window-closed.json").exists() and time.time() < limit:
            time.sleep(15)
    main(root_path)
