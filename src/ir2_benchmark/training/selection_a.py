"""Real validation-driven best selection without discarding the latest recovery state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from flax import serialization

from ir2_benchmark.presets import preset_manifest
from ir2_benchmark.training.checkpoint import save_checkpoint
from ir2_benchmark.training.pools_a import _save
from ir2_benchmark.training.state_a import TrainingError, configuration_hash

if TYPE_CHECKING:
    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.evaluation.validation import ValidationEvaluator
    from ir2_benchmark.maps.sampler import TrainingMapSampler
    from ir2_benchmark.training.pools_a import FormalState
    from ir2_benchmark.training.state_a import TrainAConfig


def validate_due(
    state: FormalState,
    checkpoint: Path,
    identity: str,
    validation: ValidationEvaluator | None,
    interval: int,
) -> FormalState:
    """Evaluate an exact real-event boundary and retain better inference parameters."""
    if validation is None or state.real_events < state.last_validation + interval:
        return state
    score = validation(checkpoint, state.real_events)
    if score.real_events != state.real_events or score.manifest_hash != validation.manifest_hash:
        raise TrainingError(
            field="validation", detail="score does not match fixed manifest/event boundary"
        )
    selected = score.mean_auc > state.best_auc
    record = asdict(score) | {
        "output": str(score.output),
        "selected": selected,
        "configurations_searched": 1,
    }
    with (checkpoint.parent / "validation.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")
    if selected:
        best = checkpoint.parent / "best"
        best.mkdir(parents=True, exist_ok=True)
        (best / "configuration.json").write_text(
            (checkpoint.parent / "configuration.json").read_text(encoding="utf-8"), encoding="utf-8"
        )
        models = state.models
        parameters = {
            name: {"params": serialization.to_state_dict(model.params)}
            for name, model in (
                ("actor", models.actor),
                ("critic", models.critic),
                ("predictor", models.predictor),
            )
        }
        save_checkpoint(best / "checkpoint", serialization.msgpack_serialize(parameters), identity)
        (best / "selection.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        (best / "summary.json").write_text(
            json.dumps(
                {
                    "real_events": state.real_events,
                    "total_training_events": state.real_events,
                    "selected_at_events": state.real_events,
                    "best_validation_auc": score.mean_auc,
                    "validation_manifest_hash": score.manifest_hash,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (best / "experiment.json").write_text(
            json.dumps(
                {
                    "training_seed": validation.training_seed,
                    "selection_event": state.real_events,
                    "model_selection": "fixed_validation_mean_coverage_auc",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    updated = state._replace(
        best_auc=max(state.best_auc, score.mean_auc), last_validation=state.real_events
    )
    _save(checkpoint, updated, identity)
    return updated


def record_training_total(checkpoint: Path, real_events: int) -> None:
    """Keep selected-weight age distinct from the run's actual total interaction cost."""
    summary_path = checkpoint.parent / "best" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["real_events"] = real_events
        summary["total_training_events"] = real_events
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def prepare_output(  # noqa: PLR0913, PLR0917
    maps: TrainingMapSampler,
    config: HostConfig,
    settings: TrainAConfig,
    output: str | Path,
    validation: ValidationEvaluator | None,
    validation_interval: int,
) -> tuple[Path, str]:
    """Bind manifest, sampler seed, dynamics and optimization settings before training."""
    if validation_interval < 1 or (settings.events >= validation_interval and validation is None):
        raise TrainingError(
            field="validation",
            detail="periodic formal selection requires a fixed validation-only evaluator",
        )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        (
            configuration_hash(config, settings)
            + maps.manifest_hash
            + str(maps.seed)
            + (validation.manifest_hash if validation is not None else "")
            + str(validation_interval)
        ).encode()
    ).hexdigest()
    configuration = {
        "environment": asdict(config),
        "implementations": preset_manifest(config),
        "training": asdict(settings),
        "configuration_hash": identity,
        "driver": "bucket",
        "manifest_hash": maps.manifest_hash,
        "sampler_seed": maps.seed,
        "validation_manifest_hash": validation.manifest_hash if validation is not None else None,
        "validation_interval": validation_interval,
        "validation_configurations_searched": 1,
    }
    (destination / "configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    return destination, identity
