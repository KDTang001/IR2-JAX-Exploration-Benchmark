"""Counter-addressed balanced formal training maps, never validation/test fallbacks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np

from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import MapError, load_ir2_map, prepare_map

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ir2_benchmark.types import MapInput

Source = Literal["ir2", "procedural"]
BUCKETS: Final = (512, 768, 1024)
FAMILIES: Final = ("rooms", "corridor", "hybrid")
CYCLE_LENGTH: Final = 108


@dataclass(frozen=True, slots=True)
class ManifestSource:
    """Root interpreting relative image paths in one audited/generated manifest."""

    root: Path
    manifest: Path
    source: Source


@dataclass(frozen=True, slots=True)
class TrainingMap:
    """One train-only immutable entry, preserving the native geometry bucket."""

    source: Source
    family: str
    bucket: int
    path: str
    map_hash: str
    canonical_hash: str
    scenario_hash: str = ""


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One independently reproducible dispatch to a (map bucket, active N) pool."""

    sample_id: str
    index: int
    source: Source
    family: str
    bucket: int
    n_agents: int
    path: str
    map_hash: str
    canonical_hash: str
    spawn_seed: int
    fault_seed: int
    split: str = "train"
    scenario_hash: str = ""


@dataclass(frozen=True, slots=True)
class SamplerError(ValueError):
    """Formal data cannot satisfy a named sampling stratum."""

    detail: str

    def __str__(self) -> str:
        """Render the missing data or changed-map evidence."""
        return self.detail


def _read_source(source: ManifestSource) -> tuple[TrainingMap, ...]:
    """Parse train records, excluding declared validation/test/error rows explicitly."""
    records = []
    for line in source.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") != "train" or row.get("exclusion_reason") is not None:
            continue
        family = row["family"] if source.source == "ir2" else row["template"]
        bucket = row.get("bucket", row.get("size"))
        records.append(
            TrainingMap(
                source.source,
                family,
                int(bucket),
                str((source.root / row["path"]).resolve()),
                row["occupancy_hash"],
                row["canonical_hash"],
                row.get("scenario_hash", ""),
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True, init=False)
class TrainingMapSampler:
    """Balance the complete hierarchy every108 samples, independently of call order.

    A driver must dispatch the returned (bucket,N) key to separate compiled pools.
    Conditioning this schedule on one bucket changes its global mixture and must
    never be reported as the complete formal training distribution.
    """

    records: tuple[TrainingMap, ...]
    seed: int
    manifest_hash: str
    easy_episodes: int

    def __init__(
        self, sources: tuple[ManifestSource, ...], seed: int, *, easy_episodes: int = 0
    ) -> None:
        """Require both IR2 families and every procedural template/bucket stratum."""
        if easy_episodes < 0:
            raise SamplerError(detail="negative_easy_episode_count")
        records = tuple(
            sorted(
                (row for source in sources for row in _read_source(source)),
                key=lambda row: (row.source, row.family, row.bucket, row.path),
            )
        )
        required = [("ir2", family, None) for family in ("easy", "difficult")]
        required += [("procedural", family, bucket) for family in FAMILIES for bucket in BUCKETS]
        for source, family, bucket in required:
            if not any(
                row.source == source
                and row.family == family
                and (bucket is None or row.bucket == bucket)
                for row in records
            ):
                detail = f"missing_training_stratum:{source}:{family}:{bucket}"
                raise SamplerError(detail)
        identity = json.dumps(
            [
                {key: value for key, value in asdict(row).items() if key != "path"}
                for row in records
            ],
            sort_keys=True,
        ).encode()
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "easy_episodes", easy_episodes)
        if easy_episodes:
            identity += f":easy_episodes={easy_episodes}:robots=4".encode()
        object.__setattr__(self, "manifest_hash", hashlib.sha256(identity).hexdigest())

    def schedule(self, index: int) -> TrainingSample:
        """Address one balanced stratum and map using semantic index-derived streams."""
        cycle, offset = divmod(max(index - self.easy_episodes, 0), CYCLE_LENGTH)
        permutation = np.random.default_rng(
            np.random.SeedSequence([self.seed, cycle, 0])
        ).permutation(CYCLE_LENGTH)
        slot = int(permutation[offset])
        opportunity = slot // 2
        n_agents = (2, 4, 6)[opportunity // 18]
        source: Source = "ir2" if slot % 2 == 0 else "procedural"
        family = (
            ("easy", "difficult")[opportunity % 2] if source == "ir2" else FAMILIES[opportunity % 3]
        )
        bucket = None if source == "ir2" else BUCKETS[(opportunity // 3) % 3]
        if index < self.easy_episodes:
            source, family, bucket, n_agents = "ir2", "easy", None, 4
        choices = tuple(
            row
            for row in self.records
            if row.source == source
            and row.family == family
            and (bucket is None or row.bucket == bucket)
        )
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, index, 1]))
        row = choices[int(rng.integers(len(choices)))]
        spawn_seed, fault_seed = (int(value) for value in rng.integers(0, 2**31, size=2))
        identity = (
            f"{self.manifest_hash}:{self.seed}:{index}:{row.map_hash}:"
            f"{n_agents}:{spawn_seed}:{fault_seed}"
        )
        return TrainingSample(
            hashlib.sha256(identity.encode()).hexdigest(),
            index,
            source,
            family,
            row.bucket,
            n_agents,
            row.path,
            row.map_hash,
            row.canonical_hash,
            spawn_seed,
            fault_seed,
            scenario_hash=row.scenario_hash,
        )

    def load(self, sample: TrainingSample) -> MapInput:
        """Verify the actual occupancy hash and retain its declared native map bucket."""
        data = load_ir2_map(sample.path)
        if obstacle_hash(data.occupancy) != sample.map_hash:
            raise MapError(sample.path, "training_manifest_occupancy_hash_mismatch")
        if (
            sample.scenario_hash
            and scenario_hash(data.occupancy, data.anchor) != sample.scenario_hash
        ):
            raise MapError(sample.path, "training_manifest_anchor_hash_mismatch")
        prepared = prepare_map(data, buckets=(sample.bucket,))
        if prepared.occupancy.shape != (sample.bucket, sample.bucket):
            raise MapError(sample.path, "training_manifest_bucket_mismatch")
        if int(prepared.component.sum()) < sample.n_agents:
            raise MapError(sample.path, "training_component_has_insufficient_spawns")
        from ir2_benchmark.env.reset import from_prepared_map  # noqa: PLC0415

        return from_prepared_map(prepared)

    def for_bucket(self, bucket: int, n_agents: int) -> Callable[[int], MapInput]:
        """Return a same-bucket adapter; global formal proportions require schedule()."""

        def selected(index: int) -> MapInput:
            cursor = index
            sample = self.schedule(cursor)
            while sample.bucket != bucket or sample.n_agents != n_agents:
                cursor += 1
                sample = self.schedule(cursor)
            return self.load(sample)

        return selected
