"""Real A/B training dispatch with explicit micro reductions and formal source requirements."""

import argparse
import json
from dataclasses import asdict, replace

import jax

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.commands.options import resolve_training_options
from ir2_benchmark.config import ConfigError, load_config
from ir2_benchmark.env.scenarios import load_scenario
from ir2_benchmark.evaluation.validation import build_validation_callback
from ir2_benchmark.maps.sampler import TrainingMapSampler
from ir2_benchmark.visualization.recording import RecordingConfig


def train_handler(args: argparse.Namespace) -> int:
    """Record the actual experiment and call the appropriate real learning implementation."""
    config = load_config(args.config)
    recording = RecordingConfig(
        every_episodes=args.gif_every_episodes,
        frame_stride_rounds=args.gif_frame_stride_rounds,
        max_frames=args.gif_max_frames,
        duration_ms=getattr(args, "gif_duration_ms", 500),
    )
    if args.route == "A":
        from ir2_benchmark.training.train_a import TrainAConfig, train_a  # noqa: PLC0415

        resolved = resolve_training_options(args, asdict(TrainAConfig()))
        settings = TrainAConfig(**resolved.settings)
    else:
        from ir2_benchmark.training.train_b import BTrainingConfig, train_b  # noqa: PLC0415

        resolved = resolve_training_options(args, asdict(BTrainingConfig()))
        settings = BTrainingConfig(**resolved.settings)
    formal = resolved.suite == "formal"
    if config.global_graph_fusion and getattr(settings, "fixed_plan", False):
        raise ConfigError(
            field="global_graph_fusion",
            detail="fixed PLAN sends no maps; use a schedule containing MAP_DELTA for fusion",
        )
    if formal and args.map_sources is None:
        raise ConfigError(
            field="map_sources", detail="formal training requires complete train manifests"
        )
    if formal and args.validation_manifest is None:
        raise ConfigError(
            field="validation_manifest",
            detail="formal training requires fixed validation-only model selection",
        )
    sampler = (
        TrainingMapSampler(
            manifest_sources(args.map_sources),
            args.seed,
            easy_episodes=getattr(settings, "curriculum_easy_episodes", 0),
        )
        if formal
        else None
    )
    if formal:
        config = replace(config, agent_bucket=6)
    events = resolved.events
    warmup = resolved.settings["warmup_events"]
    args.output.mkdir(parents=True, exist_ok=True)
    experiment = {
        "route": args.route,
        "suite": resolved.suite,
        "training_seed": args.seed,
        "events_requested": events,
        "warmup_events": warmup,
        "manifest_hash": sampler.manifest_hash if sampler else None,
        "environment": asdict(config),
        "training": dict(resolved.settings),
        "recording": asdict(recording),
    }
    validation = (
        build_validation_callback(
            args.validation_manifest, args.output / "validation", training_seed=args.seed
        )
        if args.validation_manifest
        else None
    )
    experiment["validation_manifest_hash"] = validation.manifest_hash if validation else None
    from ir2_benchmark.evaluation.scenarios import identity  # noqa: PLC0415

    experiment["effective_configuration_hash"] = identity(experiment)
    if args.resume is not None:
        original = (
            args.resume.parent / "experiment.json"
            if args.resume.is_file()
            else args.resume / "experiment.json"
        )
        if (
            original.exists()
            and json.loads(original.read_text(encoding="utf-8")).get("training_seed") != args.seed
        ):
            raise ConfigError(
                field="seed", detail="resume must use the checkpoint's original training seed"
            )
    (args.output / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    if args.route == "A":
        if formal:
            from ir2_benchmark.training.driver_a import train_a_formal  # noqa: PLC0415

            checkpoint = train_a_formal(
                sampler,
                config,
                settings,
                jax.random.PRNGKey(args.seed),
                args.output,
                resume=args.resume,
                validation=validation,
                recording=recording,
            )
            print(json.dumps({"checkpoint": str(checkpoint), "suite": resolved.suite}))  # noqa: T201
        else:
            summary = train_a(
                load_scenario(config, args.seed, args.map),
                config,
                settings,
                jax.random.PRNGKey(args.seed),
                args.output,
                resume=args.resume,
                recording=recording,
            )
            print(json.dumps(asdict(summary), indent=2))  # noqa: T201
    else:
        summary = train_b(
            config,
            args.seed,
            events,
            args.output,
            options=settings,
            map_path=args.map,
            resume=args.resume,
            sampler=sampler,
            validation=validation,
            recording=recording,
        )
        print(json.dumps(summary, indent=2))  # noqa: T201
    return 0
