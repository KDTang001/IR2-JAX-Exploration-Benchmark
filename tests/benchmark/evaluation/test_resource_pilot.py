"""Validation-only resource planning preserves formal evaluation scope."""

# Literal assertions are the independent frozen planning contract.
# ruff: noqa: PLR2004

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.commands.common import manifest_sources
from ir2_benchmark.config import ConfigError
from ir2_benchmark.evaluation.scenarios import evaluation_maps, read_task_manifest
from ir2_benchmark.maps.manifest import canonical_obstacle_hash, obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import load_ir2_map

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "resource_pilot", ROOT / "scripts/freeze_resource_pilot.py"
)
assert SPEC is not None
assert SPEC.loader is not None
PILOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PILOT
SPEC.loader.exec_module(PILOT)


@pytest.fixture
def registry(tmp_path: Path) -> Path:
    """Create two differently hashed validation choices per stratum and excluded decoys."""
    entries = []
    for index, (source, family) in enumerate(PILOT.STRATA):
        root = tmp_path / f"maps{index}"
        rows = []
        for variant in range(2):
            relative = (
                f"train/{family}/{variant}.png"
                if source == "ir2"
                else f"validation_{family}_{variant}.png"
            )
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            pixels = np.full((12 + index, 12 + variant), 255, dtype=np.uint8)
            pixels[0, :] = 0
            pixels[2, 2] = 208
            Image.fromarray(pixels).save(path)
            data = load_ir2_map(path)
            rows.append(
                {
                    "path": relative,
                    "origin": "train",
                    "family": family,
                    "template": family,
                    "split": "validation",
                    "bucket": 512,
                    "valid_shape": list(pixels.shape),
                    "occupancy_hash": obstacle_hash(data.occupancy),
                    "canonical_hash": canonical_obstacle_hash(data.occupancy),
                    "scenario_hash": scenario_hash(data.occupancy, data.anchor),
                }
            )
        rows.extend(
            [
                {
                    **rows[0],
                    "split": split,
                    "occupancy_hash": "0" * 64,
                    "canonical_hash": f"decoy-{split}",
                }
                for split in ("test", "ood")
            ]
        )
        rows.append({**rows[0], "exclusion_reason": "rejected", "occupancy_hash": "0" * 64})
        manifest = tmp_path / f"manifest{index}.jsonl"
        manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        entries.append({"root": root.name, "manifest": manifest.name, "source": source})
    result = tmp_path / "sources.json"
    result.write_text(json.dumps(entries), encoding="utf-8")
    return result


def test_planner_has_exact_factors_and_deterministic_validation_identities(
    registry: Path,
    tmp_path: Path,
) -> None:
    # Given: native images, a registry with test/OOD decoys, and real checkpoint bytes.
    checkpoint = tmp_path / "policy.npz"
    checkpoint.write_bytes(b"identity-only-test-checkpoint")
    sources = manifest_sources(registry)
    formal_before = evaluation_maps(sources)
    # When: the planning API is repeated with identical sources and configuration.
    selected, planned = PILOT.plan_resource_pilot(
        registry, ROOT / "configs/benchmark.json", checkpoint
    )
    repeated = PILOT.plan_resource_pilot(registry, ROOT / "configs/benchmark.json", checkpoint)
    # Then: all five winners are deterministic and all 240 unique factor combinations exist.
    assert repeated == (selected, planned)
    assert len(selected) == 5
    assert len(planned) == len({task.result_id for task in planned}) == 240
    expected = {
        (source, family, budget, fault, probability, motion, comm)
        for source, family in PILOT.STRATA
        for budget in (2048, 8192)
        for fault, probability in ((0, 0.0), (1, 0.3), (2, 0.0))
        for motion in ("nearest_frontier", "ir2_adapted")
        for comm in ("off", "plan", "map", "mixed")
    }
    assert {
        (
            t.scenario.source,
            t.scenario.family,
            t.scenario.budget,
            t.scenario.fault_mode,
            t.scenario.drop_probability,
            t.method.decision_rule,
            t.method.communication,
        )
        for t in planned
    } == expected
    for item in selected:
        rows = [json.loads(line) for line in Path(item.manifest).read_text().splitlines()]
        assert item.map_data.map_hash == min(
            row["occupancy_hash"]
            for row in rows
            if row["split"] == "validation" and row.get("exclusion_reason") is None
        )
        assert item.png_hash
        assert item.canonical_hash
        assert item.valid_shape
    for task in planned:
        scene = task.scenario
        assert scene.split == "validation"
        assert scene.limited
        assert (scene.n_agents, scene.max_rounds, scene.burst_start, scene.burst_duration) == (
            4,
            512,
            128,
            64,
        )
        assert scene.config.coverage_target == 0.99
        assert scene.config.graph_capacity == 400
        assert scene.config.rendezvous_mode == "reachable_only"
    assert evaluation_maps(sources) == formal_before
    assert {row.split for row in formal_before} == {"test", "ood"}


def test_c1024_version_preserves_selection_seeds_and_changes_all_c400_ids(
    registry: Path, tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.npz"
    checkpoint.write_bytes(b"identity-only-test-checkpoint")
    original, v1 = PILOT.plan_resource_pilot(
        registry, ROOT / "configs/benchmark.json", checkpoint
    )
    selected, v2 = PILOT.plan_resource_pilot(
        registry, ROOT / "configs/benchmark_capacity1024.json", checkpoint,
        protocol="resource-pilot-v2-C1024",
    )
    assert selected == original
    assert len(v2) == len({task.result_id for task in v2}) == 240
    assert not {task.result_id for task in v1} & {task.result_id for task in v2}
    for old, new in zip(v1, v2, strict=True):
        assert old.method == new.method
        assert old.scenario.spawn_seed == new.scenario.spawn_seed
        assert old.scenario.fault_seed == new.scenario.fault_seed
        assert new.scenario.config.graph_capacity == 1024
        assert new.scenario.config.graph_padding_size == 1024
    probe = [task for task in v2
             if task.scenario.budget == 2048 and task.method.communication == "mixed"]
    assert len(probe) == 30
    assert {task.scenario.fault_mode for task in probe} == {0, 1, 2}


@pytest.mark.parametrize(("protocol", "config"), [
    ("resource-pilot-v1", "benchmark_capacity1024.json"),
    ("resource-pilot-v2-C1024", "benchmark.json"),
])
def test_protocol_rejects_capacity_mismatch_before_planning(
    registry: Path, tmp_path: Path, protocol: str, config: str,
) -> None:
    with pytest.raises(ConfigError, match="declared benchmark capacity"):
        PILOT.plan_resource_pilot(
            registry, ROOT / "configs" / config, tmp_path / "unused.npz", protocol=protocol
        )


@pytest.mark.parametrize(
    "mutation", ["test", "ood", "excluded", "family", "hash", "source", "geometry"]
)
def test_invalid_selection_never_falls_back(registry: Path, mutation: str) -> None:
    # Given: a valid registry whose entire first stratum is then made invalid.
    manifest = registry.parent / "manifest0.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()][:2]
    for row in rows:
        if mutation in ("test", "ood"):
            row["split"] = mutation
        elif mutation == "excluded":
            row["exclusion_reason"] = "rejected"
        elif mutation == "family":
            row["family"] = "difficult"
        elif mutation == "hash":
            row["scenario_hash"] = "tampered"
        elif mutation == "source":
            row["origin"] = "test"
        else:
            row["valid_shape"] = [100, 100]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    # When/Then: selection fails explicitly instead of using a different split/family/map.
    with pytest.raises(ConfigError):
        PILOT.select_validation_maps(manifest_sources(registry))


def test_cli_writes_only_plan_and_identity_sidecar(registry: Path, tmp_path: Path) -> None:
    # Given: a synthetic checkpoint and a fresh output location.
    checkpoint = tmp_path / "policy.npz"
    checkpoint.write_bytes(b"identity-only-test-checkpoint")
    output = tmp_path / "plan.jsonl"
    # When: the actual CLI plans without loading or executing the invalid checkpoint.
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(ROOT / "scripts/freeze_resource_pilot.py"),
            "--sources",
            str(registry),
            "--config",
            str(ROOT / "configs/benchmark.json"),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # Then: the CLI produces only a 240-task limited plan and its audit sidecar.
    assert result.returncode == 0, result.stderr
    assert len(tuple(read_task_manifest(output))) == 240
    evidence = json.loads(output.with_suffix(".selection.json").read_text())
    assert evidence["status"] == "planning_only"
    assert evidence["task_count"] == 240
    assert evidence["limited"] is True
    assert evidence["history_capacity"] == 513


def test_root_aliases_and_manifest_order_preserve_selected_identities(registry: Path) -> None:
    # Given: relative registry roots whose normalized aliases refer to the same images.
    original = PILOT.select_validation_maps(manifest_sources(registry))
    entries = json.loads(registry.read_text())
    for entry in entries:
        entry["root"] = f"{entry['root']}/../{entry['root']}"
        manifest = registry.parent / entry["manifest"]
        manifest.write_text(
            "\n".join(reversed(manifest.read_text().splitlines())), encoding="utf-8"
        )
    registry.write_text(json.dumps(list(reversed(entries))), encoding="utf-8")
    # When: resolving aliases and selecting from reordered input.
    selected = PILOT.select_validation_maps(manifest_sources(registry))
    # Then: absolute map paths and actual image identities remain identical.
    assert [(x.map_data, x.png_hash, x.canonical_hash, x.valid_shape) for x in selected] == [
        (x.map_data, x.png_hash, x.canonical_hash, x.valid_shape) for x in original
    ]


@pytest.mark.parametrize("field", ["occupancy_hash", "canonical_hash", "scenario_hash", "source"])
def test_metadata_tampering_is_rejected(registry: Path, field: str) -> None:
    # Given: one stratum with tampered content identity or source label.
    manifest = registry.parent / "manifest0.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()][:2]
    for row in rows:
        row[field] = "0" * 64 if field != "source" else "procedural"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    # When/Then: the planner rejects the mismatch before producing any task.
    with pytest.raises(ConfigError):
        PILOT.select_validation_maps(manifest_sources(registry))


def test_test_canonical_overlap_is_rejected(registry: Path) -> None:
    # Given: a winner also declared as a test canonical group in the same registry.
    manifest = registry.parent / "manifest0.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()][:2]
    winner = min(rows, key=lambda row: row["occupancy_hash"])
    rows.append({**winner, "split": "test"})
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    # When/Then: leakage is rejected even when the validation label itself is retained.
    with pytest.raises(ConfigError, match="overlaps test/OOD"):
        PILOT.select_validation_maps(manifest_sources(registry))
