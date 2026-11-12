"""Behavioral map preparation and generation contracts."""

import json
import struct
from dataclasses import FrozenInstanceError
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.maps.generate import GenerationError, generate_map
from ir2_benchmark.maps.manifest import (
    audit_maps,
    backfill_rejected_metadata,
    canonical_obstacle_hash,
    scenario_hash,
)
from ir2_benchmark.maps.prepare import (
    MapData,
    MapError,
    connected_component,
    load_ir2_map,
    prepare_map,
    select_spawns,
)


def save_ir2(path: Path, grid: np.ndarray, anchor: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.where(grid == 0, 195, 127).astype(np.uint8)
    pixels[anchor[1], anchor[0]] = 208
    Image.fromarray(pixels).save(path)


def test_preserves_topology_and_padding_when_prepared(tmp_path: Path) -> None:
    # Given: asymmetric source values including an obstacle encoded as 1.
    pixels = np.array(
        [[127, 127, 127, 127], [127, 208, 195, 1], [127, 195, 195, 127]], dtype=np.uint8
    )
    path = tmp_path / "map.png"
    Image.fromarray(pixels).save(path)
    # When: decoding and padding to a fixed bucket.
    prepared = prepare_map(load_ir2_map(path), buckets=(8,))
    # Then: valid occupancy is exact, and all outside cells are occupied.
    np.testing.assert_array_equal(prepared.occupancy[:3, :4], pixels <= 150)  # noqa: PLR2004
    assert prepared.valid_shape == (3, 4)
    assert np.all(prepared.occupancy[~prepared.valid_mask] == 1)
    assert prepared.anchor == (1, 1)
    with pytest.raises(ValueError, match="read-only"):
        prepared.occupancy[1, 1] = 1
    with pytest.raises(ValueError, match="WRITEABLE"):
        prepared.occupancy.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        prepared.anchor = (2, 2)


@pytest.mark.parametrize("case", ["corrupt", "missing_marker", "insufficient_component"])
def test_reports_map_errors_without_replacement_when_invalid(tmp_path: Path, case: str) -> None:
    # Given: an invalid image beside a valid candidate replacement.
    path = tmp_path / "bad.png"
    save_ir2(tmp_path / "good.png", np.zeros((8, 8), dtype=np.uint8), (2, 2))
    match case:
        case "corrupt":
            path.write_bytes(b"this is not a png")
        case "missing_marker":
            Image.fromarray(np.full((8, 8), 195, dtype=np.uint8)).save(path)
        case "insufficient_component":
            save_ir2(path, np.ones((8, 8), dtype=np.uint8), (2, 2))
    # When / Then: the original failure is surfaced.
    with pytest.raises(MapError) as error:
        select_spawns(load_ir2_map(path), n_agents=6, seed=1)
    assert case in error.value.reason


def test_spawn_uses_connected_cells_and_expands_radius_when_needed() -> None:
    # Given: a one-cell-wide component and a disconnected neighboring room.
    grid = np.ones((30, 30), dtype=np.uint8)
    grid[1:29, 1] = 0
    grid[1:10, 3:10] = 0
    data = MapData(grid, (1, 1))
    # When: requiring six distinct spawns starting with a tiny radius.
    positions = select_spawns(data, 6, seed=91, initial_radius=1)
    # Then: IDs reproduce exactly and no spawn jumps components.
    np.testing.assert_array_equal(positions, select_spawns(data, 6, seed=91, initial_radius=1))
    assert len(set(map(tuple, positions))) == 6  # noqa: PLR2004
    assert np.all(positions[:, 0] == 1)
    assert np.all(positions[:, 1] <= 6)  # noqa: PLR2004


def test_component_matches_independent_bfs_when_geometry_has_random_holes() -> None:
    from collections import deque  # noqa: PLC0415

    # Given: random obstacles, including thin runs and disconnected diagonal contacts.
    rng = np.random.default_rng(72)
    for _ in range(30):
        grid = (rng.random((14, 17)) < 0.35).astype(np.uint8)  # noqa: PLR2004
        grid[2, 3] = 0
        expected = np.zeros(grid.shape, dtype=bool)
        pending = deque([(3, 2)])
        while pending:
            x, y = pending.popleft()
            if not (0 <= x < 17 and 0 <= y < 14) or expected[y, x] or grid[y, x]:  # noqa: PLR2004
                continue
            expected[y, x] = True
            pending.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
        # When: span-fill the anchor component.
        actual = connected_component(grid, (3, 2))
        # Then: the optimized traversal has exactly the same reachability as a cell BFS.
        np.testing.assert_array_equal(actual, expected)


def test_oversize_bucket_is_explicit_when_source_exceeds_default_buckets() -> None:
    # Given: a source wider than every configured bucket.
    data = MapData(np.zeros((3, 12), dtype=np.uint8), (1, 1))
    # When: preparing without a large enough default bucket.
    result = prepare_map(data, buckets=(4, 8))
    # Then: the source is preserved in an extra bucket, never cropped.
    assert result.occupancy.shape == (12, 12)
    np.testing.assert_array_equal(result.occupancy[:3, :12], data.occupancy)


def test_equivalent_occupancies_share_group_when_rotated_or_spawn_changed() -> None:
    # Given: asymmetric occupancy and alternate anchor metadata.
    grid = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 1, 1]], dtype=np.uint8)
    # When: grouping obstacles independently of scenario anchors.
    hashes = {canonical_obstacle_hash(np.rot90(np.fliplr(grid), k)) for k in range(4)}
    # Then: dimensions transform with occupancy, and scenario identity remains separate.
    assert hashes == {canonical_obstacle_hash(grid)}
    assert scenario_hash(grid, (1, 1)) != scenario_hash(grid, (2, 1))


def test_audit_prefers_test_and_reports_every_file_when_duplicates_exist(tmp_path: Path) -> None:
    # Given: test rotations, train collisions, independent train maps, and one corrupt file.
    root = tmp_path / "maps"
    grid = np.zeros((8, 10), dtype=np.uint8)
    grid[0, :3] = 1
    save_ir2(root / "test" / "complex" / "a.png", grid, (4, 4))
    save_ir2(root / "train" / "easy" / "collision.png", np.rot90(grid), (3, 3))
    for i in range(20):
        sample = np.zeros((10, 12), dtype=np.uint8)
        sample[0, : i % 10 + 1] = 1
        sample[2, i // 10 + 1] = 1
        save_ir2(root / "train" / "easy" / f"{i}.png", sample, (5, 5))
    corrupt = root / "train" / "easy" / "corrupt.png"
    corrupt.write_bytes(b"invalid")
    output = tmp_path / "manifest.jsonl"
    # When: auditing all sources.
    summary = audit_maps(root, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    # Then: test wins; valid training groups are split once and failures remain listed.
    assert summary.total == len(rows) == 23  # noqa: PLR2004
    assert next(row for row in rows if row["path"].endswith("collision.png"))[
        "exclusion_reason"
    ].startswith("test_priority")
    assert sum(row["split"] == "validation" for row in rows) == 2  # noqa: PLR2004
    assert sum(row["split"] == "test" for row in rows) == 1
    assert next(row for row in rows if row["path"].endswith("corrupt.png"))[
        "exclusion_reason"
    ].startswith("corrupt")
    original = output.read_bytes()
    audit_maps(root, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize("template", ["rooms", "corridor", "hybrid"])
def test_generator_reproduces_connected_formal_geometry_when_seed_fixed(template: str) -> None:
    # Given / When: independent runs at the smallest formal bucket.
    result = generate_map(template, 17, size=512)
    repeat = generate_map(template, 17, size=512)
    # Then: occupancy, selected seed, anchor, and all rejected seeds reproduce.
    np.testing.assert_array_equal(result.data.occupancy, repeat.data.occupancy)
    assert result.record == repeat.record
    assert result.record.component_area >= 0.15 * 512**2
    assert result.record.component_area == int(np.sum(result.data.occupancy == 0))
    assert 8 <= result.record.width <= 16  # noqa: PLR2004
    assert result.record.room_scale_range == (40, 120)


def test_generator_changes_only_approved_parameter_range_when_ood_requested() -> None:
    # Given / When: paired narrow-width and large-room variants of one template.
    narrow = generate_map("hybrid", 19, size=512, shift="narrow")
    large = generate_map("hybrid", 19, size=512, shift="large_rooms")
    # Then: each structural change is independently controlled and recorded.
    assert 4 <= narrow.record.width <= 7  # noqa: PLR2004
    assert narrow.record.room_scale_range == (40, 120)
    assert 8 <= large.record.width <= 16  # noqa: PLR2004
    assert large.record.room_scale_range == (121, 180)
    assert narrow.record.shift == "narrow"
    assert large.record.shift == "large_rooms"


def test_generator_has_distinct_templates_when_seed_shared() -> None:
    # Given / When: identical seed and size, different real structural templates.
    grids = [
        generate_map(template, 37, size=512).data.occupancy
        for template in ("rooms", "corridor", "hybrid")
    ]
    # Then: each has different topology, not just a relabeled common generator.
    assert len({canonical_obstacle_hash(grid) for grid in grids}) == 3  # noqa: PLR2004


def test_generator_reports_rejection_instead_of_fallback_when_geometry_impossible() -> None:
    # Given: a tiny map incompatible with formal room scales.
    # When / Then: exhaustion records the rejected seed and never returns a fallback.
    with pytest.raises(GenerationError) as error:
        generate_map("rooms", 1, size=16, max_attempts=2)
    assert len(error.value.rejections) == 2  # noqa: PLR2004
    assert error.value.rejections[0].seed == 1


def test_audit_keeps_image_metadata_when_source_has_no_marker(tmp_path: Path) -> None:
    # Given: readable occupancy with no allowed spawn anchor.
    root = tmp_path / "maps"
    source = root / "train" / "easy" / "missing.png"
    source.parent.mkdir(parents=True)
    Image.fromarray(np.full((3, 5), 195, np.uint8)).save(source)
    manifest = tmp_path / "manifest.jsonl"
    # When: image loading succeeds but anchor identification fails.
    audit_maps(root, manifest)
    row = json.loads(manifest.read_text())
    # Then: dimensions/obstacles remain inspectable; no anchor or split is invented.
    assert row["valid_shape"] == [3, 5]
    assert row["pixel_values"] == [195]
    assert row["occupancy_hash"] == sha256(struct.pack("<II", 3, 5) + bytes(2)).hexdigest()
    assert row["anchor"] is None
    assert row["split"] == "excluded"
    assert row["exclusion_reason"] == "missing_marker_208"


def test_metadata_backfill_does_not_reassign_split_when_existing_manifest_enriched(
    tmp_path: Path,
) -> None:
    # Given: an earlier manifest that lacks metadata for a readable rejected image.
    root = tmp_path / "maps"
    source = root / "train" / "easy" / "missing.png"
    source.parent.mkdir(parents=True)
    Image.fromarray(np.full((3, 5), 195, np.uint8)).save(source)
    old = tmp_path / "old.jsonl"
    old.write_text(
        json.dumps(
            {
                "path": "train/easy/missing.png",
                "split": "excluded",
                "exclusion_reason": "missing_marker_208",
                "anchor": None,
                "valid_shape": None,
                "component_area": None,
            }
        )
        + "\n"
    )
    new = tmp_path / "new.jsonl"
    # When: enriching only readable rejected image metadata.
    updated = backfill_rejected_metadata(root, old, new)
    row = json.loads(new.read_text())
    # Then: eligibility, reason, and unknown anchor remain exactly as audited.
    assert updated == 1
    assert row["split"] == "excluded"
    assert row["exclusion_reason"] == "missing_marker_208"
    assert row["anchor"] is None
    assert row["component_area"] is None
    assert row["valid_shape"] == [3, 5]
