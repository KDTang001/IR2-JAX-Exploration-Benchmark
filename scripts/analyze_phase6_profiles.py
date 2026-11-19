#!/usr/bin/env python3
"""Summarize captured GPU trace operators and their optimized HLO source metadata.

Usage: python analyze_phase6_profiles.py PHASE6_ROOT
Trace sums are device operation durations, not additive end-to-end wall time.
"""

from __future__ import annotations

import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path


def component(metadata: str) -> str:
    """Assign one category, retaining unmapped/memory-copy work explicitly."""
    groups = (
        ("sensing", ("sensing", "ir2_sensor.py")),
        ("communication", ("communication_", "message_", "/communication/", "/codecs/")),
        (
            "graph_maintenance",
            (
                "graph_maintenance",
                "graph_initialization",
                "graph_edges",
                "frontier_detection",
                "ir2_graph.py",
            ),
        ),
        (
            "candidate_generation",
            (
                "candidate_generation",
                "graph_features",
                "rendezvous",
                "ir2_observation.py",
                "observe.py",
            ),
        ),
        ("policy_inference", ("ir2_policy.py", "Dense_", "LayerNorm_", "baseline_action")),
        ("state_update", ("step.py", "rewards.py", "reset.py")),
    )
    return next(
        (label for label, markers in groups if any(m in metadata for m in markers)), "unattributed"
    )


# Keep the frozen trace aggregation flow intact during style-only cleanup.
def main() -> None:  # noqa: C901
    """Aggregate the captured trace files and hash their supporting raw artifacts."""
    root = Path(sys.argv[1]).resolve()
    records = []
    artifacts = {}
    for case in sorted((root / "profiles-baseline").glob("gpu-*")):
        if not case.is_dir():
            continue
        progress = json.loads((case / "progress.json").read_text())
        for row in progress["completed"]:
            phase = row["phase"]
            paths = [
                p
                for p in (case / phase).rglob("*.trace.json.gz")
                if p.name != "perfetto_trace.json.gz"
            ]
            if not paths:
                continue
            path = paths[0]
            events = json.loads(gzip.decompress(path.read_bytes()))["traceEvents"]
            gpu_pids = {
                e["pid"]
                for e in events
                if e.get("name") == "process_name" and "GPU" in e.get("args", {}).get("name", "")
            }
            gpu = [e for e in events if e.get("ph") == "X" and e.get("pid") in gpu_pids]
            hlo_path = case / f"{phase}.optimized-hlo.txt.gz"
            definitions = {}
            if hlo_path.exists():
                for line in gzip.decompress(hlo_path.read_bytes()).decode().splitlines():
                    match = re.match(r"\s*(?:ROOT\s+)?%?([\w.-]+)\s*=", line)
                    if match:
                        definitions[re.sub(r"\W", "_", match[1])] = line
            sums = collections.Counter()
            groups = collections.Counter()
            counts = collections.Counter()
            for event in gpu:
                name = event["name"]
                sums[name] += event.get("dur", 0)
                counts[name] += 1
                label = (
                    "copy_or_memset"
                    if name.startswith(("Memcpy", "Memset", "memcpy"))
                    else component(definitions.get(name, ""))
                )
                groups[label] += event.get("dur", 0)
            records.append(
                {
                    "case_id": case.name,
                    **row,
                    "trace_gpu_events": len(gpu),
                    "trace_gpu_duration_us": sum(sums.values()),
                    "trace_component_duration_us": dict(groups),
                    "trace_duration_scope": (
                        "sum of device operation durations, not wall time; one profiled invocation"
                    ),
                    "top_gpu_operators": [
                        {
                            "name": name,
                            "duration_us": total,
                            "count": counts[name],
                            "hlo_definition": definitions.get(name),
                        }
                        for name, total in sums.most_common(15)
                    ],
                    "trace_file": str(path.relative_to(root)),
                }
            )
        for path in case.rglob("*"):
            if path.is_file():
                artifacts[str(path.relative_to(root))] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    payload = {
        "protocol": "phase6-profiler-v1",
        "records": records,
        "raw_artifacts": artifacts,
        "warning": (
            "HLO metadata categories can contain fused operations; "
            "isolated stage percentages are not claimed."
        ),
    }
    (root / "profile-analysis.json").write_text(json.dumps(payload, indent=2) + "\n")
    # This JSON summary is the command-line completion output.
    print(json.dumps({"phase_records": len(records), "raw_files": len(artifacts)}))  # noqa: T201


if __name__ == "__main__":
    main()
