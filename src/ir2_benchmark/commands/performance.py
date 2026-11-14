"""Real report and synchronized performance command orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import ConfigError, load_config
from ir2_benchmark.env.reset import from_prepared_map
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.release import write_release_manifest
from ir2_benchmark.evaluation.report import write_report
from ir2_benchmark.evaluation.scenarios import evaluation_maps
from ir2_benchmark.evaluation.throughput import (
    Timing,
    active_rollout_events,
    benchmark_environment,
    measure,
    write_throughput,
)
from ir2_benchmark.evaluation.validation import build_validation_manifest
from ir2_benchmark.maps.prepare import load_ir2_map, prepare_map

if TYPE_CHECKING:
    import argparse

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.types import EnvState


def _benchmark_ir2(  # noqa: PLR0913, PLR0917
    initial: EnvState,
    config: HostConfig,
    checkpoint: Path,
    steps: int,
    warmups: int,
    repeats: int,
) -> tuple[Timing, Timing]:
    """Time original frozen weights on real reset graphs and real adapted-policy events."""
    if config.navigation_mode != "ir2_graph":
        raise ConfigError(field="policy_checkpoint", detail="IR2 timing requires ir2_graph")
    from ir2_benchmark.agents.baselines import (  # noqa: PLC0415
        BaselineSpec,
        baseline_action,
        load_policy_parameters,
    )
    from ir2_benchmark.env.observe import observe  # noqa: PLC0415
    from ir2_benchmark.env.rollout import rollout  # noqa: PLC0415
    from ir2_benchmark.models.ir2_policy import GraphPolicy  # noqa: PLC0415

    parameters = load_policy_parameters(checkpoint)
    graphs = jax.jit(jax.vmap(lambda state: observe(state, state.current_agent, config).graph))(
        initial
    )
    jax.block_until_ready(graphs)
    inference = jax.jit(
        jax.vmap(lambda weights, graph: GraphPolicy().apply(weights, *graph), in_axes=(None, 0))
    )
    inferred = measure(
        partial(inference, parameters, graphs),
        lambda output: int(output.shape[0]),
        warmups=warmups,
        repeats=repeats,
    )
    spec = BaselineSpec("ir2_adapted", "mixed")
    policy = jax.jit(
        jax.vmap(
            lambda weights, state: rollout(
                state,
                config,
                steps,
                policy=lambda observation, settings: baseline_action(
                    observation, settings, spec, weights
                ),
            ),
            in_axes=(None, 0),
        )
    )
    adapted = measure(
        partial(policy, parameters, initial),
        active_rollout_events,
        warmups=warmups,
        repeats=repeats,
    )
    return inferred, adapted


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
    parser.add_argument("--bucket", type=int)
    parser.add_argument("--agents", type=int, choices=(2, 4, 6))
    parser.add_argument("--map-sources", type=Path)
    parser.add_argument("--batches", nargs="+", type=int, default=(1, 4, 16))
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--limit-cases", type=int)


def throughput_handler(args: argparse.Namespace) -> int:
    """Time real reset, active policy events, optional frozen IR2 and learned B kernels."""
    config = load_config(args.config)
    config = replace(config, n_agents=args.agents or config.n_agents)
    if args.steps < 1 or args.warmups < 0 or args.repeats < 1 or min(args.batches) < 1:
        raise ConfigError(field="timing", detail="positive steps/batches/repeats and warmups >= 0")
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
        if args.bucket is not None:
            if args.map is None:
                raise ConfigError(field="map", detail="an explicit bucket requires an explicit map")
            prepared = prepare_map(load_ir2_map(args.map), buckets=(args.bucket,))
            map_data = from_prepared_map(prepared)
        else:
            map_data = load_scenario(config, args.seed, args.map)
        prepared_cases.append((map_data, config))
    selected = prepared_cases if args.limit_cases is None else prepared_cases[: args.limit_cases]
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for map_data, environment in selected:
        cases.extend(
            benchmark_environment(
                map_data,
                environment,
                batches=tuple(args.batches),
                steps=args.steps,
                warmups=args.warmups,
                repeats=args.repeats,
                policy_measurement=partial(
                    _benchmark_ir2,
                    config=environment,
                    checkpoint=args.policy_checkpoint,
                    steps=args.steps,
                    warmups=args.warmups,
                    repeats=args.repeats,
                )
                if args.policy_checkpoint is not None
                else None,
                seed=args.seed,
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
        "map": str(args.map) if args.map is not None else None,
        "bucket": args.bucket,
        "seed": args.seed,
        "configuration": asdict(config),
        "policy_checkpoint": str(args.policy_checkpoint) if args.policy_checkpoint else None,
        "policy_checkpoint_hash": file_hash(args.policy_checkpoint)
        if args.policy_checkpoint
        else None,
        "map_file_hash": file_hash(args.map) if args.map is not None else None,
    }
    (args.output / "requested_cases.json").write_text(
        json.dumps(declaration, indent=2) + "\n", encoding="utf-8"
    )
    print(path.read_text(encoding="utf-8"))  # noqa: T201
    return 0
