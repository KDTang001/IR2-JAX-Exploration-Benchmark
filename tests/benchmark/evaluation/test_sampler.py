"""Formal schedule frequencies and train-only boundaries have independent count oracles."""

import json
from collections import Counter
from pathlib import Path

import pytest

from ir2_benchmark.maps.sampler import ManifestSource, SamplerError, TrainingMapSampler


def sources(root: Path) -> tuple[ManifestSource, ...]:
    """Create manifest-only strata; this test never claims images were generated or loaded."""
    root.mkdir(parents=True, exist_ok=True)
    ir2 = [
        {
            "family": family,
            "bucket": bucket,
            "path": f"{family}.png",
            "split": "train",
            "occupancy_hash": family,
            "canonical_hash": family,
        }
        for family, bucket in (("easy", 768), ("difficult", 512))
    ]
    procedural = [
        {
            "template": family,
            "size": bucket,
            "path": f"{family}_{bucket}.png",
            "split": "train",
            "occupancy_hash": f"{family}_{bucket}",
            "canonical_hash": f"{family}_{bucket}",
        }
        for family in ("rooms", "corridor", "hybrid")
        for bucket in (512, 768, 1024)
    ]
    forbidden = {
        "family": "easy",
        "bucket": 768,
        "path": "secret_test.png",
        "split": "test",
        "occupancy_hash": "test",
        "canonical_hash": "test",
    }
    first, second = root / "ir2.jsonl", root / "procedural.jsonl"
    first.write_text("\n".join(json.dumps(row) for row in [*ir2, forbidden]), encoding="utf-8")
    second.write_text("\n".join(json.dumps(row) for row in procedural), encoding="utf-8")
    return ManifestSource(root, first, "ir2"), ManifestSource(root, second, "procedural")


def test_complete_cycle_has_exact_joint_hierarchy_frequencies(tmp_path: Path) -> None:
    # Given: all formal train strata and an unrelated original test row.
    sampler = TrainingMapSampler(sources(tmp_path), seed=71)
    # When: a full108-entry balanced cycle is addressed out of order.
    samples = [sampler.schedule(index) for index in reversed(range(108))]
    # Then: source, robot count and all conditional families/buckets have exact weights.
    assert Counter(sample.source for sample in samples) == {"ir2": 54, "procedural": 54}
    assert Counter(sample.n_agents for sample in samples) == {2: 36, 4: 36, 6: 36}
    for n_agents in (2, 4, 6):
        ir2 = Counter(
            sample.family
            for sample in samples
            if sample.source == "ir2" and sample.n_agents == n_agents
        )
        procedural = Counter(
            (sample.family, sample.bucket)
            for sample in samples
            if sample.source == "procedural" and sample.n_agents == n_agents
        )
        assert ir2 == {"easy": 9, "difficult": 9}
        assert procedural == {
            (family, bucket): 2
            for family in ("rooms", "corridor", "hybrid")
            for bucket in (512, 768, 1024)
        }
    assert all(sample.split == "train" and "secret_test" not in sample.path for sample in samples)
    assert len({sample.sample_id for sample in samples}) == 108  # noqa: PLR2004


def test_easy_curriculum_is_reproducible_and_then_restores_balanced_schedule(
    tmp_path: Path,
) -> None:
    manifests = sources(tmp_path)
    sampler = TrainingMapSampler(manifests, seed=71, easy_episodes=8)
    baseline = TrainingMapSampler(manifests, seed=71)
    assert sampler.manifest_hash != baseline.manifest_hash
    for index in range(8):
        sample = sampler.schedule(index)
        assert (sample.source, sample.family, sample.n_agents) == ("ir2", "easy", 4)
        assert sample == TrainingMapSampler(manifests, seed=71, easy_episodes=8).schedule(index)
    later = [sampler.schedule(index) for index in range(8, 116)]
    assert Counter(sample.source for sample in later) == {"ir2": 54, "procedural": 54}
    assert Counter(sample.n_agents for sample in later) == {2: 36, 4: 36, 6: 36}


def test_counter_schedule_is_order_independent_and_relocation_keeps_identity(
    tmp_path: Path,
) -> None:
    # Given: identical relative manifests in two different workspace roots.
    first = TrainingMapSampler(sources(tmp_path / "first"), seed=3)
    second = TrainingMapSampler(sources(tmp_path / "second"), seed=3)
    # When: unrelated prior calls precede a fixed counter on the other instance.
    first.schedule(302)
    a, b = first.schedule(57), second.schedule(57)
    # Then: metadata and counter-derived spawn/fault streams remain reproducible.
    assert first.manifest_hash == second.manifest_hash
    assert (a.sample_id, a.map_hash, a.spawn_seed, a.fault_seed) == (
        b.sample_id,
        b.map_hash,
        b.spawn_seed,
        b.fault_seed,
    )


def test_missing_formal_stratum_is_rejected_instead_of_replaced(tmp_path: Path) -> None:
    # Given: the hybrid1024 training stratum is absent.
    inputs = sources(tmp_path)
    path = inputs[1].manifest
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
            if not (row["template"] == "hybrid" and row["size"] == 1024)  # noqa: PLR2004
        ),
        encoding="utf-8",
    )
    # When: a formal schedule is constructed.
    # Then: missing data is named, without selecting another family or test map.
    with pytest.raises(SamplerError, match="missing_training_stratum:procedural:hybrid:1024"):
        TrainingMapSampler(inputs, seed=1)
