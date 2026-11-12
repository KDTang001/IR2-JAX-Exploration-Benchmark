"""Four route-A controls with equal real-interaction quotas and paired scenarios."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import jax
from jax import Array

from ir2_benchmark.config import HostConfig
from ir2_benchmark.training.evaluate_a import evaluate_a
from ir2_benchmark.training.train_a import TrainAConfig, train_a
from ir2_benchmark.types import MapInput


def run_a_comparison(
    map_data: MapInput, config: HostConfig, settings: TrainAConfig, key: Array, output: str | Path
) -> Path:
    """Train/evaluate all four controls; retain unfavorable results without retuning."""
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_key = jax.random.fold_in(key, 1)
    rows = []
    for communication in ("fixed", "learned"):
        for estimator in ("zoh", "predict"):
            variant = replace(
                settings, communication_mode=communication, neighbor_estimator=estimator
            )
            name = f"{communication}_{estimator}"
            training = train_a(map_data, config, variant, key, destination / name)
            evaluation = evaluate_a(
                training.checkpoint,
                map_data,
                config,
                variant,
                evaluation_key,
                destination / name / "evaluation",
            )
            rows.append(
                {
                    "method": name,
                    "training": asdict(training),
                    "evaluation": json.loads(evaluation.read_text(encoding="utf-8")),
                }
            )
    path = destination / "comparison.json"
    path.write_text(
        json.dumps(
            {
                "environment": asdict(config),
                "shared_training": asdict(settings),
                "seed_key": key.tolist(),
                "evaluation_key": evaluation_key.tolist(),
                "methods": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
