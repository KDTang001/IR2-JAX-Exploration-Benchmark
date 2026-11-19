"""Write reviewable rollout summaries, real events and final map trajectories."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.rollout import RolloutResult


def write_rollout(  # noqa: PLR0913, PLR0917
    result: RolloutResult,
    config: HostConfig,
    seed: int,
    output: Path,
    elapsed_seconds: float,
    compile_seconds: float,
) -> None:
    """Export only measured active events, retaining physical and cognitive maps separately."""
    output.mkdir(parents=True, exist_ok=True)
    state, trace = result.final_state, result.trace
    active = np.asarray(trace.active)
    valid_events = int(active.sum())
    summary = {
        "seed": seed,
        "config": asdict(config),
        "scope": "micro_environment_smoke",
        "events": valid_events,
        "rounds": int(state.round),
        "coverage": float(np.asarray(state.covered).sum() / np.asarray(state.component).sum()),
        "terminated": bool(state.terminated),
        "truncated": bool(state.truncated),
        "spent_bytes": np.asarray(state.spent_bytes).tolist(),
        "travel_distance": np.asarray(state.travel_distance).tolist(),
        "compile_and_first_seconds": compile_seconds,
        "steady_seconds": elapsed_seconds,
        "active_events_per_second": valid_events / elapsed_seconds,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    arrays = {name: np.asarray(value) for name, value in zip(trace._fields, trace, strict=True)}
    with (output / "events.jsonl").open("w", encoding="utf-8") as stream:
        for index in np.flatnonzero(active):
            record = {name: value[index].tolist() for name, value in arrays.items()}
            stream.write(json.dumps(record) + "\n")
    palette = np.array([[245, 245, 245], [32, 35, 40], [125, 135, 147]], np.uint8)
    colors = [
        (224, 83, 58),
        (40, 126, 214),
        (45, 159, 95),
        (173, 99, 213),
        (220, 153, 32),
        (35, 174, 174),
    ]
    height, width = np.asarray(state.valid_shape).tolist()
    positions = np.asarray(trace.positions)[active]
    maps = np.asarray(state.local_maps)
    union = np.where(np.asarray(state.covered), 0, 2).astype(np.uint8)
    for label, grid, robots in [
        *((f"local_{agent}", maps[agent], (agent,)) for agent in range(config.n_agents)),
        ("physical_union", union, tuple(range(config.n_agents))),
    ]:
        image = Image.fromarray(palette[grid[:height, :width]]).resize(
            (width * 6, height * 6), Image.Resampling.NEAREST
        )
        drawing = ImageDraw.Draw(image)
        for agent in robots:
            points = [(int(x) * 6 + 3, int(y) * 6 + 3) for x, y in positions[:, agent]]
            if len(points) > 1:
                drawing.line(points, fill=colors[agent], width=2)
            if points:
                x, y = points[-1]
                drawing.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colors[agent])
        image.save(output / f"{label}_trajectory.png")
