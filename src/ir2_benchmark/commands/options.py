"""Training precedence: explicit CLI values, then method file, then declared suite defaults."""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from ir2_benchmark.config import ConfigError

Setting: TypeAlias = str | int | float | bool


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    """The complete effective method and real-event quota, after a single boundary parse."""

    suite: str
    events: int
    settings: Mapping[str, Setting]


def resolve_training_options(  # noqa: C901, PLR0912
    args: argparse.Namespace, defaults: Mapping[str, Setting]
) -> TrainingOptions:
    """Merge only explicitly supplied flags and reject unknown method-file fields."""
    raw = json.loads(args.method_config.read_text(encoding="utf-8")) if args.method_config else {}
    allowed = set(defaults) | {"events", "suite"}
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(field="method_config", detail=f"unknown fields: {sorted(unknown)}")
    suite = args.suite or raw.get("suite", "micro")
    values = dict(defaults)
    if suite == "micro":
        values["warmup_events"] = 256
        values["events"] = 4096
        if args.route == "B":
            values.update(logical_batch=1, update_every=64, development=True)
    else:
        values["events"] = 2000000
        if "physical_env_count" in defaults:
            values["physical_env_count"] = 4
        if args.route == "B" and "replay_capacity" in defaults:
            values["replay_capacity"] = 16384
    values.update(raw)
    keys = {
        "events": "events",
        "warmup_events": "warmup_events",
        "hidden": "hidden",
        "physical_env_count": "physical_env_count",
    }
    keys |= (
        {
            "communication": "communication_mode",
            "estimator": "neighbor_estimator",
            "rollout_length": "rollout_length",
            "ppo_epochs": "ppo_epochs",
            "batch_size": "batch_size",
        }
        if args.route == "A"
        else {"replay_capacity": "replay_capacity"}
    )
    if args.route == "A" and args.replay_capacity is not None:
        raise ConfigError(field="replay_capacity", detail="this flag configures route B replay")
    for argument, field in keys.items():
        value = getattr(args, argument)
        if value is not None:
            values[field] = value
    if args.route == "B":
        if args.communication is not None:
            values["learned_communication"] = args.communication == "learned"
        if args.estimator is not None:
            values["predicted_plans"] = args.estimator == "predict"
            if "use_neighbor_plan" in defaults:
                values["use_neighbor_plan"] = args.estimator != "none"
    events = int(values.pop("events"))
    if "events" in defaults:
        values["events"] = events
    values.pop("suite", None)
    if "suite" in defaults:
        values["suite"] = suite
    if suite not in ("micro", "formal"):
        raise ConfigError(field="suite", detail="expected micro or formal")
    return TrainingOptions(suite, events, MappingProxyType(values))


def add_train_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare method inputs; preserve formal defaults unless micro is explicit."""
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite", choices=("micro", "formal"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--events", type=int)
    parser.add_argument("--warmup-events", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--map", type=Path)
    parser.add_argument("--map-sources", type=Path)
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--communication", choices=("fixed", "learned"))
    parser.add_argument("--estimator", choices=("zoh", "predict", "none"))
    parser.add_argument("--hidden", type=int)
    parser.add_argument("--rollout-length", type=int)
    parser.add_argument("--ppo-epochs", type=int)
    parser.add_argument(
        "--batch-size", type=int, help="A logical batch size; also A micro environment count"
    )
    parser.add_argument(
        "--physical-env-count", type=int, help="Parallel real environments: A formal and route B"
    )
    parser.add_argument(
        "--replay-capacity", type=int, help="B retained real events per native pool"
    )
    parser.add_argument(
        "--gif-every-episodes",
        type=int,
        default=201,
        help="Save first and every Nth training episode; 0 disables",
    )
    parser.add_argument("--gif-frame-stride-rounds", type=int, default=4)
    parser.add_argument("--gif-max-frames", type=int, default=129)
    parser.add_argument("--gif-duration-ms", type=int, default=500)
    parser.add_argument("--validation-manifest", type=Path)
