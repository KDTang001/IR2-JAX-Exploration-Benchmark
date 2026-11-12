"""IR2-style diagnostic belief images without hidden-map overlays or endpoint interpolation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
from PIL import Image, ImageDraw

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

COLORS: Final = (
    (224, 65, 55),
    (34, 154, 78),
    (20, 170, 205),
    (177, 71, 196),
    (224, 167, 30),
    (50, 90, 210),
)
GRID_COLORS: Final = np.array(((248, 248, 248), (30, 32, 36), (128, 128, 128)), np.uint8)
FRONTIER_COLOR: Final = (245, 50, 50)
LINK_COLOR: Final = (100, 116, 139)
HUD_HEIGHT: Final = 100


@dataclass(frozen=True, slots=True)
class FrameData:
    """Paid-message-overlaid local beliefs and actual compact diagnostics for one instant."""

    local_maps: np.ndarray
    peer_positions: np.ndarray
    peer_events: np.ndarray
    positions: np.ndarray
    balances: np.ndarray
    distances: np.ndarray
    coverage: float
    event: int
    round: int
    valid_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Delivery:
    """Actual post-move endpoints of one successfully delivered paid packet."""

    sender: int
    receiver: int
    start: tuple[int, int]
    end: tuple[int, int]


def frontier_cells(grid: np.ndarray) -> np.ndarray:
    """Mark known-free cells beside unknown cells, excluding the external map boundary."""
    unknown = grid == 2  # noqa: PLR2004
    near = np.zeros(grid.shape, bool)
    near[1:] |= unknown[:-1]
    near[:-1] |= unknown[1:]
    near[:, 1:] |= unknown[:, :-1]
    near[:, :-1] |= unknown[:, 1:]
    return (grid == 0) & near


def _palette_image(image: Image.Image) -> Image.Image:
    """Keep every GIF frame on a fixed small palette rather than retaining RGB frame stacks."""
    colors = [
        *map(tuple, GRID_COLORS.tolist()),
        *COLORS,
        FRONTIER_COLOR,
        LINK_COLOR,
        (0, 0, 0),
        (255, 255, 255),
    ]
    palette = Image.new("P", (1, 1))
    palette.putpalette(
        [channel for color in colors for channel in color] + [0] * (768 - 3 * len(colors))
    )
    return image.quantize(palette=palette, dither=Image.Dither.NONE)


def render_frame(  # noqa: C901, PLR0913
    frame: FrameData,
    paths: Sequence[np.ndarray],
    deliveries: Sequence[Delivery],
    title: str,
    *,
    partial: bool,
    max_rounds: int,
    max_side: int,
) -> dict[str, Image.Image]:
    """Render merged diagnostics and each robot's own belief, trail and paid peer estimates."""
    height, width = frame.valid_shape
    count = len(frame.local_maps)
    scale = max_side / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    views = [("merged", np.min(frame.local_maps, axis=0), None)]
    views.extend((f"robot_{agent + 1}", frame.local_maps[agent], agent) for agent in range(count))
    images = {}
    for label, padded, owner in views:
        grid = padded[:height, :width]
        pixels = GRID_COLORS[grid].copy()
        pixels[frontier_cells(grid)] = FRONTIER_COLOR
        image = Image.new("RGB", (size[0], size[1] + HUD_HEIGHT), (255, 255, 255))
        image.paste(Image.fromarray(pixels).resize(size, Image.Resampling.NEAREST), (0, 0))
        drawing = ImageDraw.Draw(image)

        def point(position: np.ndarray | tuple[int, int]) -> tuple[float, float]:
            return (float(position[0]) + 0.5) * scale, (float(position[1]) + 0.5) * scale

        if owner is None:
            for delivery in deliveries:
                start, end = np.array(point(delivery.start)), np.array(point(delivery.end))
                segments = max(1, int(np.linalg.norm(end - start) / 6))
                for index in range(0, segments, 2):
                    drawing.line(
                        [
                            tuple(start + (end - start) * index / segments),
                            tuple(start + (end - start) * min(index + 1, segments) / segments),
                        ],
                        fill=LINK_COLOR,
                        width=1,
                    )
        agents = range(count) if owner is None else (owner,)
        for agent in agents:
            route = [point(position) for position in paths[agent]]
            if len(route) > 1:
                drawing.line(route, fill=COLORS[agent], width=2)
            x, y = point(frame.positions[agent])
            drawing.polygon(
                ((x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)),
                fill=COLORS[agent],
                outline=(0, 0, 0),
            )
            drawing.text((x + 6, y - 6), str(agent + 1), fill=COLORS[agent])
        if owner is not None:
            for peer in range(count):
                if peer != owner and frame.peer_events[owner, peer] >= 0:
                    x, y = point(frame.peer_positions[owner, peer])
                    drawing.polygon(
                        ((x, y - 4), (x + 4, y + 4), (x - 4, y + 4)), outline=COLORS[peer]
                    )
        view_title = (
            "Merged beliefs (diagnostic)" if owner is None else f"Robot {owner + 1} local belief"
        )
        shown = list(range(count)) if owner is None else [owner]
        legend = (
            "Colors: exact paths | red: frontiers | gray dashes: paid deliveries"
            if owner is None
            else "Own exact path | red: frontiers | triangles: paid peer positions"
        )
        lines = [
            f"{title} | {view_title} {'PARTIAL' if partial else ''}",
            (
                f"Round {frame.round}/{max_rounds}  Event {frame.event}  "
                f"Physical coverage {frame.coverage:.1%}"
            ),
            legend,
            "  ".join(
                f"R{agent + 1}: {int(frame.balances[agent])} B / "
                f"{float(frame.distances[agent]):g} cells"
                for agent in shown[:3]
            ),
            "  ".join(
                f"R{agent + 1}: {int(frame.balances[agent])} B / "
                f"{float(frame.distances[agent]):g} cells"
                for agent in shown[3:]
            ),
        ]
        for row, line in enumerate(lines):
            drawing.text((6, size[1] + 5 + row * 18), line, fill=(0, 0, 0))
        images[label] = _palette_image(image)
    return images


def save_views(
    frames: dict[str, list[Image.Image]], output: Path, stem: str, duration_ms: int
) -> dict[str, dict[str, str | int]]:
    """Write final GIF/PNG files directly; no temporary frame files are created."""
    artifacts = {}
    for label, images in frames.items():
        gif, png = output / f"{stem}_{label}.gif", output / f"{stem}_{label}_last.png"
        images[0].save(
            gif,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
            disposal=2,
        )
        images[-1].convert("RGB").save(png)
        artifacts[label] = {"gif": str(gif), "last_png": str(png), "frames": len(images)}
    return artifacts


def frame_events(start: int, stop: int, agents: int, stride: int, capacity: int) -> np.ndarray:
    """Select actual round boundaries plus both endpoints, thinning only interior frames."""
    if start == stop:
        return np.asarray([start])
    period = agents * stride
    interior = np.arange((start // period + 1) * period, stop, period, dtype=np.int64)
    selected = np.r_[start, interior, stop]
    if len(selected) > capacity:
        selected = selected[
            np.r_[0, np.linspace(1, len(selected) - 2, capacity - 2, dtype=int), len(selected) - 1]
        ]
    return selected
