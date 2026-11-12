"""Real report and synchronized performance command orchestration."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import jax

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import ConfigError, load_config
from ir2_benchmark.env.reset import from_prepared_map
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.evaluation.release import write_release_manifest
from ir2_benchmark.evaluation.report import write_report
from ir2_benchmark.evaluation.scenarios import evaluation_maps
from ir2_benchmark.evaluation.throughput import benchmark_environment, measure, write_throughput
from ir2_benchmark.evaluation.validation import build_validation_manifest
from ir2_benchmark.maps.prepare import load_ir2_map, prepare_map


def add_report_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare artifact paths and the intended matrix for completeness checks."""
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--core", type=Path, default=Path("configs/evaluation/core.json"))
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--map-sources", type=Path)


def report_handler(args: argparse.Namespace) -> int:
    """Preserve failed and missing tasks while exporting actual score/cost/quality data."""
    manifest = args.task_manifest or args.results / "tasks.jsonl"
    result = write_report(
        args.results,
        args.output,
        manifest=manifest if manifest.exists() else None,
        core=args.core,
        repetitions=args.bootstrap_repetitions,
        map_sources=args.map_sources,
    )
    write_release_manifest(args.results, args.output, manifest if manifest.exists() else None)
    print(result.read_text(encoding="utf-8"))  # noqa: T201
    return 0


def add_validation_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare one fixed stratified validation subset shared by both routes."""
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--map-sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maps-per-family", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)


def validation_manifest_handler(args: argparse.Namespace) -> int:
    """Freeze validation-only maps by hash before any checkpoint selection occurs."""
    path = build_validation_manifest(
        manifest_sources(args.map_sources),
        load_config(args.config),
        args.output,
        maps_per_family=args.maps_per_family,
        seed=args.seed,
    )
    print(str(path))  # noqa: T201
    return 0


def add_throughput_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare measured shapes, logical populations, batches and optional learned kernels."""
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("micro", "formal"), default="micro")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--map-sources", type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--limit-cases", type=int)


def throughput_handler(args: argparse.Namespace) -> int:
    """Time real reset, full events, policy segments and optional learned B kernels."""
    config = load_config(args.config)
    prepared_cases = []
    if args.suite == "formal":
        if args.map_sources is None:
            raise ConfigError(
                field="map_sources", detail="formal timing requires actual retained native buckets"
            )
        maps = evaluation_maps(manifest_sources(args.map_sources))
        for bucket in (512, 768, 1024):
            candidates = tuple(row for row in maps if row.bucket == bucket)
            if not candidates:
                raise ConfigError(field="bucket", detail=f"missing actual map for bucket {bucket}")
            prepared = prepare_map(load_ir2_map(candidates[0].path), buckets=(bucket,))
            map_data = from_prepared_map(prepared)
            prepared_cases.extend(
                (map_data, replace(config, n_agents=agents, agent_bucket=6)) for agents in (2, 4, 6)
            )
    else:
        prepared_cases.append((load_scenario(config, args.seed, args.map), config))
    selected = prepared_cases if args.limit_cases is None else prepared_cases[: args.limit_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for map_data, environment in selected:
        cases.extend(
            benchmark_environment(
                map_data, environment, batches=tuple(args.batches), steps=args.steps
            )
        )
        write_throughput(args.output / "throughput.json", tuple(cases), scope=args.suite)
        jax.clear_caches()
    model = None
    if args.model_checkpoint is not None:
        from ir2_benchmark.training.train_b import BTrainingConfig, benchmark_functions  # noqa: PLC0415

        metadata = json.loads(
            (args.model_checkpoint.parent / "config.json").read_text(encoding="utf-8")
        )
        environment = type(config)(**metadata["environment"])
        options = BTrainingConfig(**metadata["training"])
        functions = benchmark_functions(
            args.model_checkpoint,
            load_scenario(environment, args.seed, args.map),
            environment,
            options,
            jax.random.PRNGKey(args.seed),
        )
        model = {
            "status": "measured",
            "world_updates": asdict(
                measure(functions.world_update, lambda value: int(value.updates))
            ),
            "imagined_events": asdict(
                measure(functions.imagination, lambda value: int(value.valid_events))
            ),
            "checkpoint": str(args.model_checkpoint),
            "replay": "restored actual checkpoint replay",
            "measurement": asdict(functions.metadata),
        }
    path = write_throughput(
        args.output / "throughput.json", tuple(cases), scope=args.suite, model=model
    )
    requested = [
        (list(map_data.occupancy.shape), environment.n_agents, batch)
        for map_data, environment in prepared_cases
        for batch in args.batches
    ]
    declaration = {
        "requested_cases": requested,
        "limited": args.limit_cases is not None,
        "scope": args.suite,
        "steps_per_rollout": args.steps,
    }
    (args.output / "requested_cases.json").write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8"
    )
    print(path.read_text(encoding="utf-8"))  # noqa: T201
    return 0
