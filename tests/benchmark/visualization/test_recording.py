"""Exact live-event replay, partial flush and wall-respecting route evidence on JAX."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest
from PIL import Image

from ir2_benchmark.visualization.recording import (
    ModelVersions,
    RecordedEvent,
    RecordingConfig,
    RecordingError,
    TrainingRecorder,
)
from ir2_benchmark.visualization.render import GRID_COLORS

if TYPE_CHECKING:
    from pathlib import Path

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.training.storage import StandardAction
    from ir2_benchmark.types import EnvState


@dataclass(frozen=True, slots=True)
class RealEvents:
    """Only actual step_agent outputs construct the expected recording evidence."""

    initial: EnvState
    final: EnvState
    config: HostConfig
    actions: StandardAction
    trace: RecordedEvent


@pytest.fixture
def real_events() -> RealEvents:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    from ir2_benchmark.config import HostConfig  # noqa: PLC0415
    from ir2_benchmark.env.reset import reset  # noqa: PLC0415
    from ir2_benchmark.env.step import step_agent  # noqa: PLC0415
    from ir2_benchmark.training.storage import StandardAction  # noqa: PLC0415
    from ir2_benchmark.types import MapInput, make_action  # noqa: PLC0415

    grid = jnp.ones((24, 24), jnp.uint8).at[2, 2:19].set(0).at[2:19, 18].set(0).at[18, 2:19].set(0)
    config = HostConfig(
        sensor_radius=8,
        max_distance=20,
        max_rounds=8,
        coverage_target=1,
        fault_mode=1,
        dropout_probability=0.3,
        signal_threshold=-1e6,
    )
    data = MapInput(grid, grid == 0, jnp.asarray([24, 24]), jnp.asarray([17, 2]))
    initial = reset(data, config, jax.random.PRNGKey(7), spawns=jnp.asarray([[17, 2], [18, 2]]))
    actions = StandardAction(
        jnp.asarray([[18, 2], [18, 2], [18, 8], [18, 2], [17, 2]]),
        jnp.asarray([1, 0, 1, 0, 1], bool),
        jnp.asarray([3, 1, 4, 3, 3]),
    )
    step = jax.jit(lambda state, action: step_agent(state, action, config))
    state, records = initial, []
    for index in range(5):
        result = step(
            state, make_action(actions.waypoint[index], actions.send[index], actions.codec[index])
        )
        info, updated = result.info, result.state
        records.append(
            RecordedEvent(
                state.event,
                info.agent_id,
                updated.positions,
                updated.goals,
                updated.balances,
                info.spent,
                info.delivered,
                info.coverage_after,
                result.terminated,
                result.truncated,
            )
        )
        state = updated
    return RealEvents(
        initial, state, config, actions, jax.tree.map(lambda *values: jnp.stack(values), *records)
    )


def test_actual_actions_replay_and_write_merged_and_local_partial_training_gifs(
    tmp_path: Path,
    real_events: RealEvents,
) -> None:
    # Given: real legal actions, including a route turning around a wall between coarse endpoints.
    jax = pytest.importorskip("jax")
    from ir2_benchmark.visualization.replay import replay_step  # noqa: PLC0415

    case = real_events
    kernel = replay_step(case.config)
    compilations = kernel._cache_size()  # noqa: SLF001
    before = jax.tree.map(np.asarray, case.final)
    recorder = TrainingRecorder(
        tmp_path,
        RecordingConfig(frame_stride_rounds=1, max_frames=4, max_side=240),
        training_seed=7,
        route="A",
    )
    assert recorder.begin_episode(
        "pool-lane-sample0", case.initial, case.config, training_events=0, episode_index=0
    )
    # Repeated begin must not replace the real earliest snapshot with a later state.
    assert recorder.begin_episode(
        "pool-lane-sample0", case.final, case.config, training_events=5, episode_index=0
    )
    recorder.append_actions(
        "pool-lane-sample0",
        case.actions,
        np.ones(5, bool),
        case.trace,
        versions=ModelVersions(3, 4, "train"),
    )
    # When: training ends mid-episode and the recorder flushes the actually observed prefix.
    (path,) = recorder.close(training_events=5)
    saved = json.loads(path.read_text(encoding="utf-8"))
    # Then: partial views preserve provenance and leave the training state untouched.
    assert saved["replay_verified"]
    assert saved["partial"]
    assert (saved["initial_event"], saved["end_event"], saved["valid_real_events"]) == (0, 5, 5)
    assert saved["training_events_added_by_replay"] == 0
    assert replay_step(case.config) is kernel
    assert kernel._cache_size() <= compilations + 1  # noqa: SLF001
    assert saved["model_versions"][0]["actor"] == 3  # noqa: PLR2004
    assert set(saved["views"]) == {"merged", "robot_1", "robot_2"}
    for files in saved["views"].values():
        with Image.open(files["gif"]) as image:
            assert 1 <= image.n_frames <= 4  # noqa: PLR2004
    with Image.open(saved["views"]["merged"]["last_png"]) as last:
        pixels = np.asarray(last)
        # Sensor endpoint occlusion keeps cell(x17,y5) unknown even though truth contains a wall.
        np.testing.assert_array_equal(pixels[55, 175], GRID_COLORS[2])
        # Cell(x17,y3) is observed. A coarse endpoint line would color this wall pixel red.
        np.testing.assert_array_equal(pixels[35, 177], GRID_COLORS[1])
    for actual, original in zip(jax.tree.leaves(case.final), jax.tree.leaves(before), strict=True):
        np.testing.assert_array_equal(actual, original)
    assert recorder.close(5) == ()


def test_resume_uses_the_real_mid_episode_snapshot_and_marks_partial(
    tmp_path: Path,
    real_events: RealEvents,
) -> None:
    # Given: the actual state after a finished prefix, without inventing its earlier route history.
    case = real_events
    recorder = TrainingRecorder(tmp_path, RecordingConfig(max_side=120), training_seed=7, route="B")
    recorder.begin_episode(
        "resumed0", case.final, case.config, training_events=5, episode_index=0, partial=True
    )
    # When: a resumed run stops before another real event has been accepted.
    (path,) = recorder.close(5)
    saved = json.loads(path.read_text(encoding="utf-8"))
    # Then: the single honest frame names the real cutoff and does not add fake motion or events.
    assert saved["partial"]
    assert saved["frame_events"] == [5]
    assert saved["valid_real_events"] == 0


def test_replay_mismatch_is_explicit_and_does_not_emit_a_fake_gif(
    tmp_path: Path,
    real_events: RealEvents,
) -> None:
    # Given: real actions but corrupted recorded budget evidence.
    case = real_events
    recorder = TrainingRecorder(tmp_path, RecordingConfig(max_side=120))
    recorder.begin_episode("bad", case.initial, case.config, training_events=0, episode_index=0)
    trace = case.trace._replace(balances=np.asarray(case.trace.balances) + 1)
    recorder.append_actions(
        "bad", case.actions, np.ones(5, bool), trace, versions=ModelVersions(0, 0, "warmup")
    )
    # When: the exact replay disagrees with a recorded real field.
    # Then: the mismatch names its field/event instead of writing plausible replacement frames.
    with pytest.raises(RecordingError, match="replay_mismatch:balances:event=0"):
        recorder.close(5)
    assert list(tmp_path.glob("*.gif")) == []
