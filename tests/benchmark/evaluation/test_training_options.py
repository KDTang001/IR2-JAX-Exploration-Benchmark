"""Actual argparse values must preserve file-selected ablations unless explicitly overridden."""

import argparse
import json
from pathlib import Path

import pytest

from ir2_benchmark.commands.options import add_train_arguments, resolve_training_options
from ir2_benchmark.config import ConfigError


def test_a_method_file_survives_unset_cli_defaults(tmp_path: Path) -> None:
    # Given: an A fixed/ZOH method file and no explicit communication/estimator flags.
    method = tmp_path / "fixed_zoh.json"
    method.write_text(
        json.dumps({"communication_mode": "fixed", "neighbor_estimator": "zoh"}), encoding="utf-8"
    )
    parser = argparse.ArgumentParser()
    add_train_arguments(parser)
    parser.set_defaults(route="A")
    args = parser.parse_args(
        [
            "--config",
            "micro.json",
            "--output",
            "out",
            "--method-config",
            str(method),
            "--events",
            "4096",
        ]
    )
    defaults = {
        "communication_mode": "learned",
        "neighbor_estimator": "predict",
        "events": 2000000,
        "warmup_events": 100000,
        "suite": "formal",
    }
    # When: the real CLI namespace is merged with method and suite defaults.
    effective = resolve_training_options(args, defaults)
    # Then: file-selected ablations and the explicit real-event quota remain effective.
    assert effective.settings["communication_mode"] == "fixed"
    assert effective.settings["neighbor_estimator"] == "zoh"
    assert effective.events == effective.settings["events"] == 4096  # noqa: PLR2004


def test_explicit_cli_flag_overrides_b_method_file(tmp_path: Path) -> None:
    # Given: a B fixed/ZOH method file, with an explicit learned gate request.
    method = tmp_path / "b.json"
    method.write_text(
        json.dumps({"learned_communication": False, "predicted_plans": False}), encoding="utf-8"
    )
    parser = argparse.ArgumentParser()
    add_train_arguments(parser)
    parser.set_defaults(route="B")
    args = parser.parse_args(
        [
            "--config",
            "micro.json",
            "--output",
            "out",
            "--method-config",
            str(method),
            "--communication",
            "learned",
        ]
    )
    # When: resolving the parsed explicit flag over file and defaults.
    effective = resolve_training_options(
        args,
        {
            "learned_communication": True,
            "predicted_plans": True,
            "warmup_events": 100000,
            "logical_batch": 32,
            "update_every": 128,
            "development": False,
        },
    )
    # Then: only the explicitly changed gate overrides its file selection.
    assert effective.settings["learned_communication"]
    assert not effective.settings["predicted_plans"]


def test_unknown_method_field_is_rejected_at_the_cli_boundary(tmp_path: Path) -> None:
    # Given: a misspelled method parameter.
    method = tmp_path / "bad.json"
    method.write_text('{"communicaton_mode":"fixed"}', encoding="utf-8")
    parser = argparse.ArgumentParser()
    add_train_arguments(parser)
    parser.set_defaults(route="A")
    args = parser.parse_args(
        ["--config", "micro.json", "--output", "out", "--method-config", str(method)]
    )
    # When: method options are parsed before any real training is launched.
    # Then: the typo cannot silently select another ablation.
    with pytest.raises(ConfigError, match="unknown fields"):
        resolve_training_options(args, {"communication_mode": "learned", "warmup_events": 100000})


def test_cli_batch_options_reach_the_real_runner_validation(tmp_path: Path) -> None:
    # Given: the actual evaluation parser with explicit incompatible batching limits.
    from ir2_benchmark.commands.evaluate import add_evaluate_arguments, evaluate_handler  # noqa: PLC0415

    parser = argparse.ArgumentParser()
    add_evaluate_arguments(parser)
    manifest = tmp_path / "empty.jsonl"
    manifest.write_text("", encoding="utf-8")
    base = ["--task-manifest", str(manifest), "--output", str(tmp_path / "out")]
    assert parser.parse_args(base).batch_size == 1
    args = parser.parse_args([*base, "--batch-size", "3", "--max-pending-tasks", "2"])
    # When: the CLI reaches the production runner before reading any tasks.
    # Then: the explicit flags are enforced instead of silently using batch one.
    with pytest.raises(ConfigError, match="batch_size"):
        evaluate_handler(args)


def test_formal_parallel_defaults_and_explicit_capacity_preserve_logical_batch(
    tmp_path: Path,
) -> None:
    # Given: real argparse flags and a B method with a fixed logical learning batch.
    parser = argparse.ArgumentParser()
    add_train_arguments(parser)
    parser.set_defaults(route="B")
    base = ["--suite", "formal", "--config", "native.json", "--output", str(tmp_path)]
    defaults = {
        "warmup_events": 100000,
        "physical_env_count": 1,
        "replay_capacity": 4096,
        "logical_batch": 32,
    }
    # When: selecting the formal default and then explicit sixteen-environment capacity.
    normal = resolve_training_options(parser.parse_args(base), defaults)
    parallel = resolve_training_options(
        parser.parse_args([*base, "--physical-env-count", "16", "--replay-capacity", "65536"]),
        defaults,
    )
    # Then: physical concurrency changes explicitly while the learning quota stays fixed.
    assert (normal.settings["physical_env_count"], normal.settings["replay_capacity"]) == (4, 16384)
    assert (parallel.settings["physical_env_count"], parallel.settings["replay_capacity"]) == (
        16,
        65536,
    )
    assert normal.settings["logical_batch"] == parallel.settings["logical_batch"] == 32  # noqa: PLR2004
    assert normal.events == parallel.events == 2000000  # noqa: PLR2004


def test_gif_cli_settings_do_not_enter_learning_configuration(tmp_path: Path) -> None:
    # Given: capture is enabled by default and can be disabled independently of a model.
    parser = argparse.ArgumentParser()
    add_train_arguments(parser)
    parser.set_defaults(route="A")
    base = ["--config", "native.json", "--output", str(tmp_path)]
    defaults = {"events": 2000000, "warmup_events": 100000, "physical_env_count": 4}
    recorded = parser.parse_args(base)
    disabled = parser.parse_args([*base, "--gif-every-episodes", "0"])
    # When: resolving the actual training options on either side of this visualization choice.
    # Then: checkpoint learning identity inputs remain identical.
    assert recorded.gif_every_episodes == 201  # noqa: PLR2004
    assert disabled.gif_every_episodes == 0
    assert resolve_training_options(recorded, defaults) == resolve_training_options(
        disabled, defaults
    )
