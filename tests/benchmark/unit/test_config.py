"""Validate explicit rendezvous semantics at the host configuration boundary."""

from pathlib import Path

import pytest

from ir2_benchmark.config import ConfigError, HostConfig, load_config


def test_rendezvous_mode_defaults_to_original_abort_rule() -> None:
    assert HostConfig().rendezvous_mode == "strict_ir2"


def test_config_loads_reachable_only_rendezvous_mode(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text('{"rendezvous_mode": "reachable_only"}', encoding="utf-8")

    assert load_config(source).rendezvous_mode == "reachable_only"


def test_config_rejects_unknown_rendezvous_mode(tmp_path: Path) -> None:
    source = tmp_path / "benchmark.json"
    source.write_text('{"rendezvous_mode": "skip_all"}', encoding="utf-8")

    with pytest.raises(ConfigError, match="rendezvous_mode: expected strict_ir2 or reachable_only"):
        load_config(source)
