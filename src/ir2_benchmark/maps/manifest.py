"""Full-file audit, orientation-independent leakage grouping, and fixed split assignment."""

import json
import struct
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ir2_benchmark.maps.prepare import (
    ANCHOR_ALGORITHM,
    MapError,
    connected_component,
    decode_ir2_pixels,
    read_ir2_pixels,
)


def obstacle_hash(grid: NDArray[np.uint8]) -> str:
    """Hash the unpadded oriented occupancy together with (height,width)."""
    return sha256(struct.pack("<II", *grid.shape) + np.packbits(grid).tobytes()).hexdigest()


def canonical_obstacle_hash(grid: NDArray[np.uint8]) -> str:
    """Group all eight rotations/reflections independently of spawn markers."""
    return min(
        obstacle_hash(np.rot90(view, k)) for view in (grid, np.fliplr(grid)) for k in range(4)
    )


def scenario_hash(grid: NDArray[np.uint8], anchor: tuple[int, int]) -> str:
    """Identify oriented occupancy and anchor as a scenario, separate from split grouping."""
    return sha256(bytes.fromhex(obstacle_hash(grid)) + struct.pack("<II", *anchor)).hexdigest()


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One audit result per source file, including explicit exclusions."""

    path: str
    origin: str
    family: str
    valid_shape: tuple[int, int] | None = None
    bucket: int | None = None
    occupancy_hash: str | None = None
    canonical_hash: str | None = None
    scenario_hash: str | None = None
    anchor: tuple[int, int] | None = None
    anchor_algorithm: str = ANCHOR_ALGORITHM
    component_area: int | None = None
    total_free_area: int | None = None
    pixel_values: tuple[int, ...] = ()
    split: str = "excluded"
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """Counts include invalid and duplicate maps, never just retained maps."""

    total: int
    train: int
    validation: int
    test: int
    excluded: int
    errors: int
    extra_buckets: tuple[int, ...]
    output: str


def assign_splits(records: tuple[ManifestRecord, ...]) -> tuple[ManifestRecord, ...]:
    """Prefer original test groups, deduplicate origins, then reserve 10% per train family."""
    groups: dict[str, list[int]] = {}
    results = list(records)
    for index, record in enumerate(records):
        if record.exclusion_reason is None and record.canonical_hash is not None:
            groups.setdefault(record.canonical_hash, []).append(index)
    retained: dict[str, list[int]] = {}
    for indices in groups.values():
        ordered = sorted(
            indices, key=lambda index: (records[index].origin != "test", records[index].path)
        )
        winner = ordered[0]
        record = records[winner]
        results[winner] = replace(record, split=record.origin)
        if record.origin == "train":
            retained.setdefault(record.family, []).append(winner)
        for index in ordered[1:]:
            conflict = record.origin == "test" and records[index].origin == "train"
            reason = "test_priority" if conflict else "duplicate_occupancy"
            results[index] = replace(records[index], exclusion_reason=f"{reason}:{record.path}")
    for indices in retained.values():
        ordered = sorted(indices, key=lambda index: records[index].canonical_hash or "")
        for index in ordered[: len(ordered) // 10]:
            results[index] = replace(results[index], split="validation")
    return tuple(results)


def audit_maps(root: str | Path, output: str | Path, n_agents: int = 6) -> AuditSummary:
    """Audit every file under root without mutation, replacement, resizing, or hidden exclusions."""
    source_root = Path(root).resolve()
    output_path = Path(output).resolve()
    if not source_root.is_dir() or not 1 <= n_agents <= 6:  # noqa: PLR2004
        raise MapError(str(source_root), "invalid_audit_root_or_agent_count")
    paths = sorted(
        path for path in source_root.rglob("*") if path.is_file() and path.resolve() != output_path
    )
    if not paths:
        raise MapError(str(source_root), "no_source_files")
    records: list[ManifestRecord] = []
    for path in paths:
        relative = path.relative_to(source_root)
        parts = relative.parts
        origin = parts[0] if len(parts) >= 3 else "unknown"  # noqa: PLR2004
        family = parts[1] if len(parts) >= 3 else "unknown"  # noqa: PLR2004
        record = ManifestRecord(relative.as_posix(), origin, family)
        try:
            pixels = read_ir2_pixels(path)
            occupancy = (pixels <= 150).astype(np.uint8)  # noqa: PLR2004
            required = max(pixels.shape)
            bucket = next((side for side in (512, 768, 1024) if side >= required), required)
            record = replace(
                record,
                valid_shape=pixels.shape,
                bucket=bucket,
                occupancy_hash=obstacle_hash(occupancy),
                canonical_hash=canonical_obstacle_hash(occupancy),
                pixel_values=tuple(int(value) for value in np.unique(pixels)),
            )
            data = decode_ir2_pixels(pixels, str(path))
            component = connected_component(data.occupancy, data.anchor)
            record = replace(
                record,
                scenario_hash=scenario_hash(data.occupancy, data.anchor),
                anchor=data.anchor,
                component_area=int(component.sum()),
                total_free_area=int((data.occupancy == 0).sum()),
            )
            if int(component.sum()) < n_agents:
                raise MapError(  # noqa: TRY301
                    str(path), f"insufficient_component:{int(component.sum())}<{n_agents}"
                )
            if origin not in ("train", "test"):
                raise MapError(str(path), "unknown_origin_requires_train_or_test_directory")  # noqa: TRY301
        except MapError as error:
            record = replace(record, exclusion_reason=error.reason)
        records.append(record)
    assigned = assign_splits(tuple(records))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in assigned:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    return AuditSummary(
        len(assigned),
        sum(row.split == "train" for row in assigned),
        sum(row.split == "validation" for row in assigned),
        sum(row.split == "test" for row in assigned),
        sum(row.split == "excluded" for row in assigned),
        sum(
            row.exclusion_reason is not None
            and not row.exclusion_reason.startswith(("test_priority:", "duplicate_occupancy:"))
            for row in assigned
        ),
        tuple(
            sorted(
                {
                    row.bucket
                    for row in assigned
                    if row.bucket and row.bucket not in (512, 768, 1024)
                }
            )
        ),
        str(output_path),
    )


def backfill_rejected_metadata(root: str | Path, manifest: str | Path, output: str | Path) -> int:
    """Enrich rejected readable images without recomputing components, split assignment, or anchors."""  # noqa: E501
    source_root = Path(root)
    rows = [json.loads(line) for line in Path(manifest).read_text(encoding="utf-8").splitlines()]
    updated = 0
    for row in rows:
        if row.get("exclusion_reason") and row.get("valid_shape") is None:
            try:
                pixels = read_ir2_pixels(source_root / row["path"])
            except MapError:
                continue
            occupancy = (pixels <= 150).astype(np.uint8)  # noqa: PLR2004
            required = max(pixels.shape)
            row.update(
                valid_shape=list(pixels.shape),
                pixel_values=[int(value) for value in np.unique(pixels)],
                occupancy_hash=obstacle_hash(occupancy),
                canonical_hash=canonical_obstacle_hash(occupancy),
                bucket=next((side for side in (512, 768, 1024) if side >= required), required),
            )
            updated += 1
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return updated
