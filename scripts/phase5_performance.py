#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = []
# ///
# How to run:
#   Use the prepared CPU/GPU project environment (no dependency changes):
#   uv run python scripts/phase5_performance.py --config CONFIG --map MAP --bucket 512 \
#     --agents 2 --batch-size 1 --device gpu --policy-checkpoint WEIGHTS --output OUTPUT
"""Measure exactly one explicit map/N/B/device in a new Python child process."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Isolate JAX initialization and both device/host high-water marks for each invocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--bucket", type=int, required=True)
    parser.add_argument("--agents", type=int, choices=(2, 4, 6), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4, 16), required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        environment = os.environ.copy()
        environment.update(
            JAX_PLATFORMS=args.device,
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            JAX_ENABLE_COMPILATION_CACHE="false",
        )
        return subprocess.run(  # noqa: S603 - Current interpreter, literal local script and argv.
            [sys.executable, str(Path(__file__).resolve()), "--worker", *sys.argv[1:]],
            env=environment,
            check=False,
        ).returncode

    from ir2_benchmark.commands.performance import (  # noqa: PLC0415
        add_throughput_arguments,
        throughput_handler,
    )

    command = argparse.ArgumentParser()
    add_throughput_arguments(command)
    declared = command.parse_args(
        [
            "--config",
            str(args.config),
            "--map",
            str(args.map),
            "--bucket",
            str(args.bucket),
            "--agents",
            str(args.agents),
            "--batches",
            str(args.batch_size),
            "--policy-checkpoint",
            str(args.policy_checkpoint),
            "--output",
            str(args.output),
            "--seed",
            str(args.seed),
            "--steps",
            str(args.steps),
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
        ]
    )
    result = throughput_handler(declared)
    source = args.output / "throughput.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    process = {
        "mode": "one fresh child process for this map/N/B/device case",
        "pid": os.getpid(),
        "requested_device": args.device,
        "jax_platforms": os.environ["JAX_PLATFORMS"],
        "device_preallocation": False,
        "persistent_compilation_cache": False,
        "memory_scope": "whole child process includes all workload phases and compilation",
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "XLA_FLAGS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }
    payload["process_isolation"] = process
    source.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    requested = args.output / "requested_cases.json"
    declaration = json.loads(requested.read_text(encoding="utf-8"))
    declaration["process_isolation"] = process
    requested.write_text(json.dumps(declaration, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
