"""Fixed model selection manifests reject test leakage and later mutation before inference."""

from dataclasses import replace
from pathlib import Path

import pytest

from ir2_benchmark.config import ConfigError, HostConfig
from ir2_benchmark.evaluation.scenarios import EvaluationMap, make_scenario, write_scenario_manifest
from ir2_benchmark.evaluation.validation import build_validation_callback, validate_manifest


def test_validation_rejects_a_test_scene_before_loading_any_checkpoint(tmp_path: Path) -> None:
    # Given: a test row presented as a model-selection workload.
    scenario = make_scenario(
        EvaluationMap("not-opened.png", "map", "test", "ir2", "easy", 512, "fixture-anchor"),
        HostConfig(),
        0,
    )
    path = tmp_path / "validation.jsonl"
    write_scenario_manifest(path, (scenario,))
    # When: constructing a validation callback before training starts.
    # Then: test leakage is rejected before a model or map is read.
    with pytest.raises(ConfigError, match="validation-only"):
        build_validation_callback(path, tmp_path / "results")


def test_fixed_validation_manifest_cannot_change_during_training(tmp_path: Path) -> None:
    # Given: an accepted, frozen validation-only workload.
    scenario = make_scenario(
        EvaluationMap("not-opened.png", "map", "validation", "ir2", "easy", 512, "fixture-anchor"),
        HostConfig(),
        0,
    )
    path = tmp_path / "validation.jsonl"
    write_scenario_manifest(path, (scenario,))
    evaluator = build_validation_callback(path, tmp_path / "out")
    write_scenario_manifest(path, (replace(scenario, spawn_seed=scenario.spawn_seed + 1),))
    # When: a checkpoint selection is attempted after the workload was changed.
    # Then: a different search budget/scenario set cannot be used silently.
    with pytest.raises(ConfigError, match="changed during training"):
        evaluator(tmp_path / "checkpoint", 100000)


def test_validation_manifest_keeps_exact_config_identity(tmp_path: Path) -> None:
    # Given: a declared validation row whose full config hash remains intact.
    scenario = make_scenario(
        EvaluationMap(
            "not-opened.png", "map", "validation", "procedural", "rooms", 512, "fixture-anchor"
        ),
        HostConfig(),
        1,
    )
    path = tmp_path / "validation.jsonl"
    write_scenario_manifest(path, (scenario,))
    # When: the typed scenario workload is loaded before model selection.
    loaded = validate_manifest(path)
    # Then: map/spawn/fault/config metadata is preserved rather than regenerated.
    assert loaded == (scenario,)
