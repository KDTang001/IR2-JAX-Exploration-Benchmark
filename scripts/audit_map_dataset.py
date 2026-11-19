"""Recompute dataset geometry, split ownership and every procedural seed on the server."""

# The oracle deliberately spells out wire hashes and dataset constants independently.
# ruff: noqa: PLR2004, S101

from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import struct
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
from pathlib import Path
from typing import TypedDict

import numpy as np
from PIL import Image
from scipy.ndimage import label

from ir2_benchmark.maps.generate import generate_map


@dataclass(frozen=True, slots=True)
class MapJob:
    """One manifest row and the location of its source image."""

    root: str
    path: str
    row_json: str
    procedural: bool


@dataclass(frozen=True, slots=True)
class MapCheck:
    """Small per-file result; full pixels never cross worker process boundaries."""

    path: str
    file_hash: str
    git_blob: str
    canonical: str
    component: int
    free: int
    issues: tuple[str, ...]


def check_map(job: MapJob) -> MapCheck:  # noqa: C901
    """Use Pillow/SciPy and literal hash encoding, without production map preparation."""
    path = Path(job.root) / job.path
    raw = path.read_bytes()
    row = json.loads(job.row_json)
    issues: list[str] = []
    blob = sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L"))
    grid = (pixels <= 150).astype(np.uint8)

    def digest(view: np.ndarray) -> str:
        return sha256(struct.pack("<II", *view.shape) + np.packbits(view).tobytes()).hexdigest()

    oriented = digest(grid)
    canonical = min(digest(np.rot90(v, k)) for v in (grid, np.fliplr(grid)) for k in range(4))
    markers = np.argwhere(pixels == 208)
    anchor = None
    area = None
    scenario = None
    if len(markers):
        y, x = markers[np.argmin(np.sum((markers - np.mean(markers, axis=0)) ** 2, axis=1))]
        anchor = [int(x), int(y)]
        labels, _ = label(grid == 0, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        area = int(np.count_nonzero(labels == labels[y, x]))
        scenario = sha256(bytes.fromhex(oriented) + struct.pack("<II", *anchor)).hexdigest()
    free = int(np.count_nonzero(grid == 0))
    common = {
        "anchor": anchor,
        "component_area": area,
        "scenario_hash": scenario,
        "occupancy_hash": oriented,
        "canonical_hash": canonical,
    }
    if job.procedural:
        expected_shift = "narrow" if row["split"] == "ood" else "none"
        common.update(size=grid.shape[0], shift=expected_shift, version="ir2-procedural-v1")
        if area != free or free < 0.15 * grid.size or grid.shape[0] != grid.shape[1]:
            issues.append("procedural_geometry_contract")
        regenerated = generate_map(row["template"], row["requested_seed"], shift=row["shift"])
        reproduced = json.loads(json.dumps(asdict(regenerated.record)))
        for key, value in reproduced.items():
            if row[key] != value:
                issues.append(f"seed_reproduction:{key}")
    else:
        size = max(grid.shape)
        common.update(
            valid_shape=list(grid.shape),
            bucket=next((n for n in (512, 768, 1024) if n >= size), size),
            pixel_values=[int(v) for v in np.unique(pixels)],
            total_free_area=free if anchor is not None else None,
            anchor_algorithm="208-nearest-centroid-row-major-v1",
        )
        if anchor is None and (row["split"], row["exclusion_reason"]) != (
            "excluded",
            "missing_marker_208",
        ):
            issues.append("missing_marker_not_excluded")
    for key, value in common.items():
        if row[key] != value:
            issues.append(f"manifest_mismatch:{key}")
    if area is not None and area < 6:
        issues.append("insufficient_six_agent_component")
    return MapCheck(
        job.path,
        sha256(raw).hexdigest(),
        blob.hexdigest(),
        canonical,
        area or 0,
        free,
        tuple(issues),
    )


def split_issues(jobs: list[MapJob], checks: list[MapCheck]) -> list[str]:
    """Independently assign test-priority winners and hash-first validation members."""
    groups = defaultdict(list)
    records = {job.path: json.loads(job.row_json) for job in jobs if not job.procedural}
    for check in checks:
        if check.path in records and check.component:
            groups[check.canonical].append(check.path)
    expected = dict.fromkeys(records, ("excluded", "missing_marker_208"))
    training = defaultdict(list)
    for canonical, paths in groups.items():
        ordered = sorted(paths, key=lambda p: (records[p]["origin"] != "test", records[p]["path"]))
        winner = ordered[0]
        row = records[winner]
        expected[winner] = (row["origin"], None)
        if row["origin"] == "train":
            training[row["family"]].append((canonical, winner))
        for path in ordered[1:]:
            reason = (
                "test_priority"
                if row["origin"] == "test" and records[path]["origin"] == "train"
                else "duplicate_occupancy"
            )
            expected[path] = ("excluded", reason + ":" + row["path"])
    for members in training.values():
        for _, path in sorted(members)[: len(members) // 10]:
            expected[path] = ("validation", None)
    return [
        f"{path}:split_or_exclusion_mismatch"
        for path, row in records.items()
        if (row["split"], row["exclusion_reason"]) != expected[path]
    ]


class SamplingSummary(TypedDict):
    """Counts and identity from actual registry consumers."""

    scheduled: int
    train_records: int
    sampler_hash: str
    fixed_validation_scenarios: int


def sampling_checks(registry: Path, frozen: Path) -> SamplingSummary:
    """Exercise actual train-only scheduling and frozen validation loading, without training."""
    from ir2_benchmark.commands.common import manifest_sources  # noqa: PLC0415
    from ir2_benchmark.evaluation.validation import validate_manifest  # noqa: PLC0415
    from ir2_benchmark.maps.sampler import TrainingMapSampler  # noqa: PLC0415

    sources = manifest_sources(registry)
    sampler = TrainingMapSampler(sources, seed=71)
    samples = [sampler.schedule(i) for i in range(108)]
    assert all(sample.split == "train" for sample in samples)
    assert Counter(s.source for s in samples) == {"ir2": 54, "procedural": 54}
    assert Counter(s.n_agents for s in samples) == {2: 36, 4: 36, 6: 36}
    assert samples == [sampler.schedule(i) for i in range(108)]
    scenes = validate_manifest(frozen)
    validation = {}
    for source in sources:
        for line in source.manifest.read_text().splitlines():
            row = json.loads(line)
            if row["split"] == "validation":
                family = row.get("family", row.get("template"))
                validation[source.source, family, row["occupancy_hash"]] = row["scenario_hash"]
    for scene in scenes:
        assert validation[scene.source, scene.family, scene.map_hash] == scene.map_scenario_hash
    assert Counter((s.source, s.family) for s in scenes) == {
        ("ir2", "easy"): 5,
        ("ir2", "difficult"): 5,
        ("procedural", "rooms"): 5,
        ("procedural", "corridor"): 5,
        ("procedural", "hybrid"): 5,
    }
    return {
        "scheduled": len(samples),
        "train_records": len(sampler.records),
        "sampler_hash": sampler.manifest_hash,
        "fixed_validation_scenarios": len(scenes),
    }


def main() -> int:  # noqa: C901, PLR0915
    """Audit committed inputs in parallel and write evidence instead of another PNG pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=project)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--reference-git-inputs", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    expected = {}
    reference_commit = None
    if args.reference_git_inputs is not None:
        with gzip.open(args.reference_git_inputs, "rt") as stream:
            reference = json.load(stream)
        expected, reference_commit = reference["manifest_blobs"], reference["commit"]
    registry = args.data_root / "artifacts/maps/sources.json"
    jobs = []
    manifests = {}
    for source in json.loads(registry.read_text()):
        root = (registry.parent / source["root"]).resolve()
        manifest = (registry.parent / source["manifest"]).resolve()
        manifests[manifest.relative_to(args.data_root).as_posix()] = sha256(
            manifest.read_bytes()
        ).hexdigest()
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            path = (root / row["path"]).relative_to(args.data_root).as_posix()
            jobs.append(
                MapJob(
                    str(args.data_root),
                    path,
                    line,
                    source["source"] == "procedural",
                )
            )
    jobs.sort(key=lambda job: job.path)
    for path in [registry, *(args.data_root / p for p in manifests)]:
        relative = path.relative_to(args.data_root).as_posix()
        if expected:
            raw = path.read_bytes()
            blob = sha1(
                b"blob " + str(len(raw)).encode() + b"\0" + raw, usedforsecurity=False
            ).hexdigest()
            assert blob == expected[relative], f"registry_or_manifest_git_mismatch:{relative}"
    declared = {job.path for job in jobs}
    actual = {
        p.relative_to(args.data_root).as_posix()
        for root in (args.data_root / "DungeonMaps", args.data_root / "artifacts/maps/procedural")
        for p in root.rglob("*.png")
    }
    issues = [f"missing:{p}" for p in sorted(declared - actual)] + [
        f"undeclared:{p}" for p in sorted(actual - declared)
    ]
    checks = []
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for i, check in enumerate(pool.map(check_map, jobs, chunksize=8), start=1):
            checks.append(check)
            issues.extend(f"{check.path}:{error}" for error in check.issues)
            if i % 1000 == 0:
                print(f"Checked {i}/{len(jobs)} images and seeds", flush=True)  # noqa: T201
    issues.extend(split_issues(jobs, checks))
    png_tree = sha256("".join(f"{c.git_blob}  {c.path}\n" for c in checks).encode()).hexdigest()
    if expected and png_tree != reference["png_tree_sha256"]:
        issues.append("png_git_tree_mismatch")
    ownership = defaultdict(list)
    source_hashes = defaultdict(set)
    counts = Counter()
    exclusions = Counter()
    disconnected = Counter()
    for job, check in zip(jobs, checks, strict=True):
        row = json.loads(job.row_json)
        source = "procedural" if job.procedural else "ir2"
        source_hashes[source].add(check.canonical)
        family = row.get("family", row.get("template"))
        counts[f"{source}/{row['split']}/{family}"] += 1
        if row.get("exclusion_reason"):
            exclusions[row["exclusion_reason"].split(":")[0]] += 1
        else:
            ownership[check.canonical].append(job.path)
        if check.component and check.component != check.free:
            disconnected[f"{source}/{row['split']}/{family}"] += 1
    issues.extend(
        "retained_canonical_duplicate:" + ",".join(paths)
        for paths in ownership.values()
        if len(paths) > 1
    )
    overlap = source_hashes["ir2"] & source_hashes["procedural"]
    issues.extend("cross_source_canonical_overlap:" + h for h in sorted(overlap))
    checksum = args.output / "data-files.sha256"
    checksum.write_text("".join(f"{c.file_hash}  {c.path}\n" for c in checks))
    summary = {
        "status": "failed" if issues else "passed",
        "reference_commit": reference_commit,
        "png_git_tree_sha256": png_tree,
        "data_root": str(args.data_root),
        "workers": args.workers,
        "maps": len(checks),
        "procedural_seeds_reproduced": sum(j.procedural for j in jobs),
        "counts": dict(sorted(counts.items())),
        "exclusions": dict(exclusions),
        "anchor_component_smaller_than_all_free": dict(disconnected),
        "retained_unique_canonical": len(ownership),
        "cross_source_overlap": len(overlap),
        "manifest_sha256": manifests,
        "file_checksum_sha256": sha256(checksum.read_bytes()).hexdigest(),
        "sampling": sampling_checks(
            registry, project / "artifacts/maps/validation-scenarios.jsonl"
        ),
        "issues": issues,
        "elapsed_seconds": time.monotonic() - start,
    }
    (args.output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)  # noqa: T201
    return int(bool(issues))


if __name__ == "__main__":
    raise SystemExit(main())
