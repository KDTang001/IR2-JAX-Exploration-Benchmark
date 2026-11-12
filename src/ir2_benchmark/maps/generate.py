"""Three actual structural generators with independently controlled OOD changes."""

import itertools
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final, Literal, assert_never

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ir2_benchmark.maps.manifest import canonical_obstacle_hash, obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import MapData, MapError, connected_component, select_spawns

GENERATOR_VERSION: Final = "ir2-procedural-v1"
Template = Literal["rooms", "corridor", "hybrid"]
Shift = Literal["none", "narrow", "large_rooms"]


@dataclass(frozen=True, slots=True)
class Rejection:
    """A seed that failed geometry or acceptance, with its concrete reason."""

    seed: int
    reason: str


@dataclass(frozen=True, slots=True)
class GenerationError(ValueError):
    """No acceptable map was generated within the stated retry bound."""

    template: str
    rejections: tuple[Rejection, ...]

    def __str__(self) -> str:
        """Describe the concrete typed failure."""
        return (
            f"{self.template}: exhausted {len(self.rejections)} seeds; {self.rejections[-1].reason}"
        )


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Complete reproducibility inputs and acceptance result for one generated map."""

    version: str
    template: Template
    shift: Shift
    requested_seed: int
    final_seed: int
    size: int
    width: int
    width_range: tuple[int, int]
    room_scale_range: tuple[int, int]
    corridor_spacing: int
    component_area: int
    anchor: tuple[int, int]
    rejections: tuple[Rejection, ...]
    occupancy_hash: str
    canonical_hash: str
    scenario_hash: str


@dataclass(frozen=True, slots=True)
class GeneratedMap:
    """Immutable accepted map and the seeds/parameters needed to reproduce it."""

    data: MapData
    record: GenerationRecord


def _partition(length: int, bounds: tuple[int, int], rng: np.random.Generator) -> list[int]:
    """Sample bounded room extents summing exactly to the usable map dimension."""
    lower, upper = bounds
    min_count = (length + upper - 1) // upper
    max_count = length // lower
    if min_count > max_count or min_count < 1:
        raise MapError("generator", f"room_scale_cannot_tile:{length}:{bounds}")  # noqa: EM101
    count = int(rng.integers(min_count, max_count + 1))
    remaining = length
    lengths: list[int] = []
    for index in range(count):
        rest = count - index - 1
        lo = max(lower, remaining - rest * upper)
        hi = min(upper, remaining - rest * lower)
        width = int(rng.integers(lo, hi + 1))
        lengths.append(width)
        remaining -= width
    return lengths


def _room_axes(
    size: int, scales: tuple[int, int], rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return independent room boundaries, keeping a two-cell exterior wall."""
    x = np.r_[2, 2 + np.cumsum(_partition(size - 4, scales, rng))]
    y = np.r_[2, 2 + np.cumsum(_partition(size - 4, scales, rng))]
    return x, y


def _rooms(
    size: int, width: int, scales: tuple[int, int], rng: np.random.Generator
) -> NDArray[np.uint8]:
    """Partition free interiors with two-cell walls; cut a door at every adjacency."""
    x, y = _room_axes(size, scales, rng)
    grid = np.ones((size, size), dtype=np.uint8)
    grid[2:-2, 2:-2] = 0
    for boundary in x[1:-1]:
        grid[2:-2, boundary : boundary + 2] = 1
    for boundary in y[1:-1]:
        grid[boundary : boundary + 2, 2:-2] = 1
    for boundary in x[1:-1]:
        for lo, hi in itertools.pairwise(y):
            start = int(rng.integers(int(lo) + 2, int(hi) - width - 1))
            grid[start : start + width, boundary : boundary + 2] = 0
    for boundary in y[1:-1]:
        for lo, hi in itertools.pairwise(x):
            start = int(rng.integers(int(lo) + 2, int(hi) - width - 1))
            grid[boundary : boundary + 2, start : start + width] = 0
    return grid


def _corridors(size: int, width: int, spacing: int, rng: np.random.Generator) -> NDArray[np.uint8]:
    """Carve a connected grid of narrow orthogonal corridors with no room interiors."""
    grid = np.ones((size, size), dtype=np.uint8)
    offset = int(rng.integers(0, max(1, spacing // 4)))
    centers = np.arange(spacing // 2 + offset, size - spacing // 2, spacing)
    if len(centers) < 2:  # noqa: PLR2004
        raise MapError("generator", "insufficient_corridor_intersections")  # noqa: EM101
    jitter = max(1, spacing // 10)
    xs = centers + rng.integers(-jitter, jitter + 1, size=len(centers))
    ys = centers + rng.integers(-jitter, jitter + 1, size=len(centers))
    lower = width // 2
    upper = width - lower
    for center in ys:
        grid[center - lower : center + upper, xs[0] - lower : xs[-1] + upper] = 0
    for center in xs:
        grid[ys[0] - lower : ys[-1] + upper, center - lower : center + upper] = 0
    return grid


def _hybrid(
    size: int, width: int, scales: tuple[int, int], rng: np.random.Generator
) -> NDArray[np.uint8]:
    """Carve separated rectangular rooms and connect their centers with narrow corridors."""
    gap = 24
    expanded = (scales[0] + gap, scales[1] + gap)
    x, y = _room_axes(size, expanded, rng)
    grid = np.ones((size, size), dtype=np.uint8)
    xs = (x[:-1] + x[1:]) // 2
    ys = (y[:-1] + y[1:]) // 2
    for x0, x1 in itertools.pairwise(x):
        for y0, y1 in itertools.pairwise(y):
            grid[y0 + gap // 2 : y1 - gap // 2, x0 + gap // 2 : x1 - gap // 2] = 0
    lower, upper = width // 2, width - width // 2
    for cy in ys:
        grid[cy - lower : cy + upper, xs[0] : xs[-1] + 1] = 0
    for cx in xs:
        grid[ys[0] : ys[-1] + 1, cx - lower : cx + upper] = 0
    return grid


def generate_map(  # noqa: C901, PLR0912, PLR0913, PLR0915
    template: Template,
    seed: int,
    *,
    size: int | None = None,
    shift: Shift = "none",
    n_agents: int = 6,
    max_attempts: int = 100,
    width_range: tuple[int, int] | None = None,
    room_scale_range: tuple[int, int] | None = None,
    corridor_spacing: int = 40,
) -> GeneratedMap:
    """Generate formal geometry; micro geometry requires explicit width/room/spacing overrides."""
    if template not in ("rooms", "corridor", "hybrid") or shift not in (
        "none",
        "narrow",
        "large_rooms",
    ):
        raise MapError("generator", "unknown_template_or_shift")  # noqa: EM101
    if max_attempts < 1 or seed < 0 or corridor_spacing < 2:  # noqa: PLR2004
        raise MapError("generator", "invalid_generation_parameters")  # noqa: EM101
    if template == "corridor" and shift == "large_rooms":
        raise MapError("generator", "large_rooms_shift_has_no_rooms_in_corridor_template")  # noqa: EM101
    match shift:
        case "none":
            default_width, default_scales = (8, 16), (40, 120)
        case "narrow":
            default_width, default_scales = (4, 7), (40, 120)
        case "large_rooms":
            default_width, default_scales = (8, 16), (121, 180)
        case unreachable:
            assert_never(unreachable)
    widths = width_range or default_width
    scales = room_scale_range or default_scales
    if not (1 <= widths[0] <= widths[1] < scales[0] - 3 and scales[0] <= scales[1]):
        raise MapError("generator", "invalid_width_or_scale_ranges")  # noqa: EM101
    rejections: list[Rejection] = []
    for attempt in range(max_attempts):
        final_seed = seed + attempt
        streams = np.random.SeedSequence(final_seed).spawn(3)
        parameter_rng = np.random.default_rng(streams[0])
        geometry_rng = np.random.default_rng(streams[1])
        side = size if size is not None else int(parameter_rng.choice((512, 768, 1024)))
        width = int(np.random.default_rng(streams[2]).integers(widths[0], widths[1] + 1))
        try:
            match template:
                case "rooms":
                    grid = _rooms(side, width, scales, geometry_rng)
                case "corridor":
                    grid = _corridors(side, width, corridor_spacing, geometry_rng)
                case "hybrid":
                    grid = _hybrid(side, width, scales, geometry_rng)
                case unreachable:
                    assert_never(unreachable)
            free_yx = np.argwhere(grid == 0)
            if len(free_yx) < n_agents:
                raise MapError("generator", "insufficient_spawn_space")  # noqa: EM101, TRY301
            y, x = free_yx[0]
            anchor = (int(x), int(y))
            data = MapData(grid, anchor, f"generated:{template}:{final_seed}")
            area = int(connected_component(grid, anchor).sum())
            if area != len(free_yx):
                raise MapError(data.source, "disconnected_free_regions")  # noqa: TRY301
            if area < 0.15 * side**2:
                raise MapError(data.source, "connected_area_below_15_percent")  # noqa: TRY301
            select_spawns(data, n_agents, seed=final_seed)
        except MapError as error:
            rejections.append(Rejection(final_seed, error.reason))
            continue
        record = GenerationRecord(
            GENERATOR_VERSION,
            template,
            shift,
            seed,
            final_seed,
            side,
            width,
            widths,
            scales,
            corridor_spacing,
            area,
            anchor,
            tuple(rejections),
            obstacle_hash(grid),
            canonical_obstacle_hash(grid),
            scenario_hash(grid, anchor),
        )
        return GeneratedMap(data, record)
    raise GenerationError(template, tuple(rejections))


def generate_pool(  # noqa: PLR0913
    output: str | Path,
    count_per_template: int,
    split: str,
    seed: int,
    *,
    shift: Shift = "none",
    templates: tuple[Template, ...] = ("rooms", "corridor", "hybrid"),
    exclude_hashes: frozenset[str] = frozenset(),
) -> Path:
    """Write actual IR2-compatible images and a JSONL manifest, rejecting duplicate geometry."""
    if count_per_template < 1 or split not in ("train", "validation", "test", "ood"):
        raise MapError("generator", "invalid_pool_split_or_count")  # noqa: EM101
    if (split == "ood") != (shift != "none"):
        raise MapError("generator", "ood_split_requires_explicit_single_shift")  # noqa: EM101
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    seen = set(exclude_hashes)
    next_seed = seed
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for template in templates:
            for index in range(count_per_template):
                requested_seed = next_seed
                generated = generate_map(template, next_seed, shift=shift)
                rejected = list(generated.record.rejections)
                while generated.record.canonical_hash in seen:
                    rejected.append(Rejection(generated.record.final_seed, "duplicate_occupancy"))
                    if len(rejected) >= 100:  # noqa: PLR2004
                        raise GenerationError(template, tuple(rejected))
                    generated = generate_map(template, generated.record.final_seed + 1, shift=shift)
                    rejected.extend(generated.record.rejections)
                seen.add(generated.record.canonical_hash)
                filename = f"{split}_{template}_{index:05d}.png"
                pixels = np.where(generated.data.occupancy == 0, 195, 127).astype(np.uint8)
                x, y = generated.data.anchor
                pixels[y, x] = 208
                Image.fromarray(pixels).save(root / filename)
                record = asdict(replace(generated.record, requested_seed=requested_seed))
                record.update(
                    path=filename, split=split, rejections=[asdict(item) for item in rejected]
                )
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
                next_seed = generated.record.final_seed + 1
    return manifest
