"""Independent belief/frontier/route pixel oracles and bounded in-memory GIF output."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.config import ConfigError
from ir2_benchmark.sensing.reference import visible_mask
from ir2_benchmark.visualization.recording import RecordingConfig, TrainingRecorder
from ir2_benchmark.visualization.render import (
    COLORS,
    GRID_COLORS,
    FrameData,
    frame_events,
    frontier_cells,
    render_frame,
    save_views,
)


def scene() -> tuple[FrameData, list[np.ndarray]]:
    """Create an L-shaped cell route with an occupied diagonal midpoint."""
    grid = np.ones((12, 12), np.uint8)
    grid[:2, :2] = 2
    grid[2, 2:10] = 0
    grid[2:10, 9] = 0
    grid[8:11, 1:4] = 0
    frame = FrameData(
        np.stack((grid, grid)),
        np.zeros((2, 2, 2), np.int32),
        np.full((2, 2), -1, np.int32),
        np.array([[9, 9], [2, 9]], np.int32),
        np.array([500, 600]),
        np.array([14, 0]),
        0.5,
        2,
        1,
        (12, 12),
    )
    path = np.array([(x, 2) for x in range(2, 10)] + [(9, y) for y in range(3, 10)])
    return frame, [path, np.array([[2, 9]])]


def test_frontier_uses_only_actual_unknown_neighbors_inside_the_valid_map() -> None:
    # Given: an all-free valid map has no unknown space beyond its crop.
    grid = np.zeros((3, 4), np.uint8)
    assert not frontier_cells(grid).any()
    # When: one actual interior unknown cell appears.
    grid[1, 2] = 2
    expected = np.zeros((3, 4), bool)
    expected[0, 2] = expected[1, 1] = expected[1, 3] = expected[2, 2] = True
    # Then: exactly its four known-free neighbors are frontiers.
    np.testing.assert_array_equal(frontier_cells(grid), expected)


def test_rendered_exact_route_does_not_draw_the_endpoint_diagonal_through_walls() -> None:
    # Given: a real four-connected L route with an occupied diagonal midpoint.
    frame, paths = scene()
    # When: drawing all actual cell vertices, rather than joining coarse event endpoints.
    images = render_frame(
        frame, paths, [], "Renderer fixture", partial=False, max_rounds=8, max_side=240
    )
    pixels = np.asarray(images["merged"].convert("RGB"))
    # Then: walls, unknown space and undrawn free space retain independent expected colors.
    np.testing.assert_array_equal(pixels[110, 110], GRID_COLORS[1])
    np.testing.assert_array_equal(pixels[10, 10], GRID_COLORS[2])
    np.testing.assert_array_equal(pixels[170, 30], GRID_COLORS[0])
    assert set(images) == {"merged", "robot_1", "robot_2"}


def test_occluded_wall_stays_unknown_and_visible_wall_pixel_detects_coarse_routes() -> None:
    # Given: the independent sensor oracle at every distinct physical endpoint in the JAX fixture.
    truth = np.ones((24, 24), np.uint8)
    truth[2, 2:19] = truth[2:19, 18] = truth[18, 2:19] = 0
    seen = np.logical_or.reduce(
        [
            visible_mask(truth, np.asarray(point, np.int32), 8)
            for point in ((17, 2), (18, 2), (18, 8))
        ]
    )
    assert not seen[5, 17]
    assert seen[3, 17]
    belief = np.where(seen, truth, 2).astype(np.uint8)
    frame = FrameData(
        np.stack((belief, belief)),
        np.zeros((2, 2, 2), np.int32),
        np.full((2, 2), -1, np.int32),
        np.array([[17, 2], [18, 2]]),
        np.array([500, 600]),
        np.array([14, 0]),
        0.5,
        5,
        2,
        (24, 24),
    )
    exact = np.array(
        [
            (17, 2),
            (18, 2),
            *((18, y) for y in range(3, 9)),
            *((18, y) for y in range(7, 1, -1)),
            (17, 2),
        ]
    )
    # When: the same belief is drawn with exact cells or deliberately incorrect coarse endpoints.
    results = []
    for path in (exact, np.array([(17, 2), (18, 2), (18, 8), (17, 2)])):
        images = render_frame(
            frame,
            [path, np.array([[18, 2]])],
            [],
            "Visibility oracle fixture",
            partial=True,
            max_rounds=8,
            max_side=240,
        )
        results.append(np.asarray(images["merged"].convert("RGB")))
    # Then: unknown stays unknown, and the visible-wall pixel detects the incorrect line.
    np.testing.assert_array_equal(results[0][55, 175], GRID_COLORS[2])
    np.testing.assert_array_equal(results[0][35, 177], GRID_COLORS[1])
    np.testing.assert_array_equal(results[1][35, 177], COLORS[0])


def test_local_view_does_not_reveal_another_robots_true_pose_or_balance() -> None:
    # Given: receiver0 has never received peer1's position.
    frame, paths = scene()
    other = replace(frame, positions=np.array([[9, 9], [3, 8]]), balances=np.array([500, 999]))
    # When: only hidden peer state changes, without any paid receiver knowledge.
    first = render_frame(
        frame, paths, [], "Privacy fixture", partial=False, max_rounds=8, max_side=240
    )
    second = render_frame(
        other, paths, [], "Privacy fixture", partial=False, max_rounds=8, max_side=240
    )
    # Then: only the explicitly privileged merged diagnostic changes.
    np.testing.assert_array_equal(first["robot_1"], second["robot_1"])
    assert not np.array_equal(first["merged"], second["merged"])


def test_gif_and_last_png_are_written_without_temporary_frame_files(tmp_path: Path) -> None:
    # Given: two actual renderer instants with a shared fixed palette.
    frame, paths = scene()
    first = render_frame(frame, paths, [], "GIF fixture", partial=True, max_rounds=8, max_side=240)
    last = render_frame(
        replace(frame, event=4, round=2),
        paths,
        [],
        "GIF fixture",
        partial=True,
        max_rounds=8,
        max_side=240,
    )
    frames = {label: [first[label], last[label]] for label in first}
    # When: final GIFs and final PNGs are encoded from the in-memory image lists.
    artifacts = save_views(frames, tmp_path, "fixture", 500)
    # Then: every view retains its final image and no per-step PNG directory appears.
    assert len(list(tmp_path.iterdir())) == 6  # noqa: PLR2004
    for label, files in artifacts.items():
        with Image.open(files["gif"]) as gif, Image.open(files["last_png"]) as png:
            assert gif.n_frames == 2  # noqa: PLR2004
            assert gif.info["duration"] == 500  # noqa: PLR2004
            gif.seek(gif.n_frames - 1)
            np.testing.assert_array_equal(gif.convert("RGB"), png.convert("RGB"))
            np.testing.assert_array_equal(png, last[label].convert("RGB"))


def test_frame_cap_preserves_initial_terminal_and_only_real_round_boundaries() -> None:
    # Given: a long partial episode whose natural cadence would exceed the frame cap.
    selected = frame_events(17, 10000, 6, 4, 9)
    # Then: endpoints are retained and every interior frame corresponds to a genuine team boundary.
    assert len(selected) == 9  # noqa: PLR2004
    assert (selected[0], selected[-1]) == (17, 10000)
    assert np.all(selected[1:-1] % 24 == 0)
    assert np.all(np.diff(selected) > 0)
    assert len(frame_events(0, 1024, 2, 4, 129)) == 129  # noqa: PLR2004


def test_host_cadence_is_configurable_and_zero_disables_capture(tmp_path: Path) -> None:
    # Given: the original201-episode cadence and an explicit disabled recorder.
    recorder = TrainingRecorder(tmp_path)
    disabled = TrainingRecorder(tmp_path, RecordingConfig(every_episodes=0))
    # Then: selecting the first and cadence multiples requires no simulation allocation.
    assert [index for index in range(403) if recorder.should_record(index)] == [0, 201, 402]
    assert not disabled.should_record(0)
    assert disabled.close(0) == ()
    with pytest.raises(ConfigError, match="recording"):
        RecordingConfig(max_frames=1)
