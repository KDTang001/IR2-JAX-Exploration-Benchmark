"""Checkpoint inference and immutable suite construction for real paired evaluation."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import ConfigError, HostConfig, load_config
from ir2_benchmark.evaluation.budgets import allocate_budget
from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.runner import run_manifest
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap,
    Method,
    evaluation_maps,
    make_scenario,
    scenario_matrix,
    tasks,
    write_task_manifest,
)
from ir2_benchmark.maps.generate import generate_map
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import load_ir2_map


def add_evaluate_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose a saved workload or explicit map/checkpoint suite, with visible limits."""
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--route", choices=("A", "B", "heuristic"))
    parser.add_argument("--suite", choices=("micro", "formal"), default="micro")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--scenario-seeds", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--method-manifest", type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--map-sources", type=Path)
    parser.add_argument("--suite-config", type=Path, default=Path("configs/evaluation/core.json"))
    parser.add_argument("--map", type=Path)
    parser.add_argument("--limit-scenarios", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-pending-tasks", type=int, default=128)
    parser.add_argument("--budget-mode", choices=("per_robot", "team_total", "unlimited"))


def checkpoint_method(
    checkpoint: Path | None,
    route: str | None = None,
    training_seed: int | None = None,
    *,
    real_events: int | None = None,
) -> tuple[Method, HostConfig]:
    """Derive variant/quota from actual saved metadata, rather than caller-supplied scores."""
    if route == "heuristic":
        return Method("heuristic", "heuristic", "fixed", "none", -1, "", "", 0), HostConfig()
    if checkpoint is None:
        raise ConfigError(
            field="checkpoint", detail="A/B evaluation requires an actual saved checkpoint"
        )
    root = checkpoint.parent
    detected = "B" if (root / "config.json").exists() else "A"
    if route is not None and route != detected:
        raise ConfigError(field="route", detail="does not match checkpoint metadata")
    metadata_path = root / ("config.json" if detected == "B" else "configuration.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    settings = metadata["training"]
    communication = (
        ("learned" if settings["learned_communication"] else "fixed")
        if detected == "B"
        else settings["communication_mode"]
    )
    estimator = (
        ("predict" if settings["predicted_plans"] else "zoh")
        if detected == "B"
        else settings["neighbor_estimator"]
    )
    if detected == "B" and not settings.get("use_neighbor_plan", True):
        estimator = "none"
    experiment_path = root / "experiment.json"
    experiment = (
        json.loads(experiment_path.read_text(encoding="utf-8")) if experiment_path.exists() else {}
    )
    seed = (
        training_seed
        if training_seed is not None
        else experiment.get(
            "training_seed", metadata.get("training_seed", metadata.get("seed", -1))
        )
    )
    if detected == "A" and settings.get("suite") == "formal" and seed < 0:
        raise ConfigError(
            field="training_seed",
            detail="formal A checkpoint requires its training seed or --training-seed",
        )
    if real_events is None:
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        events = summary.get("total_training_events", summary["real_events"])
        selected_at = summary.get("selected_at_events")
    else:
        events = real_events
        selected_at = real_events
    method_id = f"{detected}_{communication}_{estimator}"
    if detected == "A" and settings.get("fixed_plan", False):
        method_id = f"A_fixed_plan_{estimator}"
    if detected == "B" and settings.get("real_updates", False):
        method_id = f"B_real_updates_{communication}_{estimator}"
    return Method(
        method_id,
        detected,
        communication,
        estimator,
        seed,
        str(checkpoint.resolve()),
        file_hash(checkpoint),
        events,
        selected_at_events=selected_at,
    ), HostConfig(**metadata["environment"])


def _micro_map(args: argparse.Namespace, config: HostConfig) -> EvaluationMap:
    """Persist a new held-out development map so every paired task uses identical geometry."""
    if args.map is None:
        data = generate_map(
            "rooms",
            args.seed,
            size=64,
            n_agents=config.n_agents,
            width_range=(2, 4),
            room_scale_range=(12, 24),
        ).data
        pixels = np.where(data.occupancy == 0, 195, 127).astype(np.uint8)
        pixels[data.anchor[1], data.anchor[0]] = 208
        path = args.output / "micro-map.png"
        Image.fromarray(pixels).save(path)
    else:
        path, data = args.map, load_ir2_map(args.map)
    bucket = next(side for side in (64, 512, 768, 1024) if side >= max(data.valid_shape))
    return EvaluationMap(
        str(path.resolve()),
        obstacle_hash(data.occupancy),
        "development_test",
        "procedural",
        "micro_rooms",
        bucket,
        scenario_hash(data.occupancy, data.anchor),
    )


def _manifest_method(entry: dict, reference_checkpoint: Path | None) -> tuple[Method, HostConfig]:
    """Restore a named fixed baseline or the existing checkpoint-derived learning method."""
    if entry.get("route") == "baseline":
        from ir2_benchmark.agents.baselines import make_baseline_method  # noqa: PLC0415

        checkpoint = (
            (Path(entry["checkpoint"]) if entry.get("checkpoint") else reference_checkpoint)
            if entry["motion"] == "ir2_adapted"
            else None
        )
        return make_baseline_method(
            entry["motion"], entry["communication"], checkpoint
        ), HostConfig()
    return checkpoint_method(
        None if entry.get("route") == "heuristic" else Path(entry["checkpoint"]),
        entry.get("route"),
        entry.get("training_seed"),
    )


def _evaluation_methods(args: argparse.Namespace) -> tuple[tuple[Method, ...], HostConfig]:
    """Validate distinct fixed/checkpoint method identities before planning any task."""
    if args.method_manifest is None:
        method, config = checkpoint_method(args.checkpoint, args.route, args.training_seed)
        return (method,), config
    entries = json.loads(args.method_manifest.read_text(encoding="utf-8"))
    restored = tuple(
        _manifest_method(entry, getattr(args, "reference_checkpoint", None)) for entry in entries
    )
    if not restored:
        raise ConfigError(field="method_manifest", detail="requires at least one method")
    methods = tuple(item[0] for item in restored)
    if len({(m.method_id, m.training_seed) for m in methods}) != len(methods):
        raise ConfigError(field="method_manifest", detail="duplicate method/repetition identity")
    if any(m.route == "baseline" for m in methods) and args.config is None:
        raise ConfigError(
            field="config", detail="baseline evaluation requires an explicit graph configuration"
        )
    return methods, restored[0][1]


def evaluate_handler(args: argparse.Namespace) -> int:
    """Execute a persisted suite in the real simulator; limits remain in every result."""
    args.output.mkdir(parents=True, exist_ok=True)
    if args.task_manifest is not None:
        manifest = args.task_manifest
    else:
        methods, trained_config = _evaluation_methods(args)
        config = load_config(args.config) if args.config is not None else trained_config
        if args.suite == "micro":
            if args.route == "heuristic" and args.config is None:
                config = HostConfig(
                    sensor_radius=8, max_distance=2, max_rounds=128, initial_budget=2048
                )
            map_data = _micro_map(args, config)
            mode = args.budget_mode or "per_robot"
            allocation = allocate_budget(
                mode, config.initial_budget, config.n_agents, config.max_rounds
            )
            config = replace(config, initial_budget=allocation.per_robot)
            planned = (
                make_scenario(
                    map_data,
                    config,
                    seed,
                    limited=True,
                    budget_mode=mode,
                    requested_team_bytes=allocation.requested_team_bytes,
                    unallocated_bytes=allocation.unallocated_bytes,
                )
                for seed in (args.scenario_seeds or (0,))
            )
        else:
            if args.map_sources is None:
                raise ConfigError(
                    field="map_sources",
                    detail="formal evaluation requires retained test/OOD manifests",
                )
            declaration = json.loads(args.suite_config.read_text(encoding="utf-8"))
            if "method_ids" in declaration:
                methods = tuple(
                    method for method in methods if method.method_id in declaration["method_ids"]
                )
                if not methods:
                    raise ConfigError(
                        field="suite_config",
                        detail="no supplied method matches declared method_ids",
                    )
            maps = evaluation_maps(manifest_sources(args.map_sources))
            planned = scenario_matrix(
                maps,
                config,
                seeds=tuple(args.scenario_seeds or declaration["scenario_seeds"]),
                n_agents=tuple(declaration["n_agents"]),
                budgets=tuple(declaration["budgets"]),
                conditions=tuple(tuple(condition) for condition in declaration["conditions"]),
                budget_mode=args.budget_mode or declaration.get("budget_mode", "per_robot"),
            )
        if args.limit_scenarios is not None:
            from itertools import islice  # noqa: PLC0415

            planned = (
                replace(scenario, limited=True)
                for scenario in islice(planned, args.limit_scenarios)
            )
        manifest = args.output / "tasks.jsonl"
        write_task_manifest(manifest, tasks(planned, methods))
    if args.plan_only:
        print(json.dumps({"task_manifest": str(manifest), "executed": False}))  # noqa: T201
        return 0
    summary = run_manifest(
        manifest,
        args.output,
        retry_errors=args.retry_errors,
        batch_size=args.batch_size,
        max_pending_tasks=args.max_pending_tasks,
    )
    print(json.dumps(asdict(summary), indent=2))  # noqa: T201
    return 0 if summary.failures == 0 else 1
