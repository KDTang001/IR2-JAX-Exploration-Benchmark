"""Command line entry points; argparse follows the approved project plan."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import jax

from ir2_benchmark.commands.evaluate import add_evaluate_arguments, evaluate_handler
from ir2_benchmark.commands.options import add_train_arguments
from ir2_benchmark.commands.performance import (
    add_report_arguments,
    add_throughput_arguments,
    add_validation_arguments,
    report_handler,
    throughput_handler,
    validation_manifest_handler,
)
from ir2_benchmark.commands.train import train_handler
from ir2_benchmark.config import load_config
from ir2_benchmark.env.artifacts import write_rollout
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.rollout import rollout
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.maps.generate import generate_pool
from ir2_benchmark.maps.manifest import audit_maps
from ir2_benchmark.runtime import run_doctor


def main() -> int:  # noqa: PLR0915
    """Run a runtime check and return a nonzero status for failed requirements."""
    parser = argparse.ArgumentParser(prog="ir2-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check JAX batching and Flax/Optax training")
    doctor.add_argument("--require-gpu", action="store_true")
    doctor.add_argument("--output", type=Path)
    run = commands.add_parser("rollout", help="Run the ordered-event exploration smoke scenario")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--map", type=Path)
    maps = commands.add_parser("maps", help="Audit IR2 sources or generate new maps")
    map_commands = maps.add_subparsers(dest="map_command", required=True)
    audit = map_commands.add_parser("audit")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--n-agents", type=int, default=6)
    generate = map_commands.add_parser("generate")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--count-per-template", type=int, default=1)
    generate.add_argument(
        "--split", choices=("train", "validation", "test", "ood"), default="train"
    )
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--shift", choices=("none", "narrow", "large_rooms"), default="none")
    for name, route in (("train-a", "A"), ("train-b", "B")):
        training = commands.add_parser(name, help=f"Run route {route} on real exploration events")
        add_train_arguments(training)
        training.set_defaults(handler=train_handler, route=route)
    evaluate = commands.add_parser(
        "evaluate", help="Run paired checkpoint scenarios in the real environment"
    )
    add_evaluate_arguments(evaluate)
    evaluate.set_defaults(handler=evaluate_handler)
    report = commands.add_parser(
        "report", help="Report actual results and missing matrix conditions"
    )
    add_report_arguments(report)
    report.set_defaults(handler=report_handler)
    throughput = commands.add_parser(
        "throughput", help="Measure synchronized whole-environment throughput"
    )
    add_throughput_arguments(throughput)
    throughput.set_defaults(handler=throughput_handler)
    validation = commands.add_parser(
        "validation-manifest", help="Freeze a validation-only model-selection workload"
    )
    add_validation_arguments(validation)
    validation.set_defaults(handler=validation_manifest_handler)
    args = parser.parse_args()
    if hasattr(args, "handler"):
        return args.handler(args)
    if args.command == "rollout":
        return rollout_handler(args)
    if args.command == "maps":
        if args.map_command == "audit":
            print(json.dumps(asdict(audit_maps(args.root, args.output, args.n_agents)), indent=2))  # noqa: T201
        else:
            print(  # noqa: T201
                generate_pool(
                    args.output,
                    args.count_per_template,
                    args.split,
                    args.seed,
                    shift=args.shift,
                )
            )
        return 0
    report = run_doctor()
    rendered = json.dumps(asdict(report), indent=2)
    print(rendered)  # noqa: T201
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    gpu_ok = not args.require_gpu or any(device.platform == "gpu" for device in report.devices)
    return 0 if report.scan_correct and report.final_loss < report.initial_loss and gpu_ok else 1


def rollout_handler(args: argparse.Namespace) -> int:
    """Compile, warm up and synchronously measure the same real micro scenario."""
    config = load_config(args.config)
    initial = reset(
        load_scenario(config, args.seed, args.map), config, jax.random.PRNGKey(args.seed)
    )
    compiled = jax.jit(lambda state: rollout(state, config))
    start = perf_counter()
    first = compiled(initial)
    jax.block_until_ready(first)
    compile_seconds = perf_counter() - start
    jax.block_until_ready(compiled(initial))
    start = perf_counter()
    result = compiled(initial)
    jax.block_until_ready(result)
    elapsed = perf_counter() - start
    write_rollout(result, config, args.seed, args.output, elapsed, compile_seconds)
    print((args.output / "summary.json").read_text(encoding="utf-8"))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
