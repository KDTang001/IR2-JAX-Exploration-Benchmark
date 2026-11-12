"""Snapshot/delta replay oracles use actual arrays instead of mocked reconstructed frames."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ir2_benchmark.training.replay import MapDelta, ReplayBuffer, ReplayError, ReplayFrame


def frames() -> tuple[ReplayFrame, ...]:
    maps = np.full((2, 5, 5), 2, dtype=np.uint8)
    times = np.full((2, 5, 5), -1, dtype=np.int32)
    inbox = np.zeros((2, 2), dtype=bool)
    result = []
    for event in range(4):
        if event < 2:  # noqa: PLR2004
            maps[0, 1, event + 1] = 0
            times[0, 1, event + 1] = event
        if event == 2:  # noqa: PLR2004
            maps[1, 1, 1] = 0
            inbox[1, 0] = True
        if event == 3:  # noqa: PLR2004
            inbox[1, 0] = False
        result.append(
            ReplayFrame(maps, times, (("event", np.asarray(event)), ("inbox_present", inbox)))
        )
    return tuple(result)


def buffer_with_episode() -> ReplayBuffer:
    states = frames()
    buffer = ReplayBuffer(17, snapshot_interval=2)
    for index in range(3):
        buffer.append(
            episode_id=4,
            before=states[index],
            after=states[index + 1],
            action=np.array([0, 1], dtype=np.float32),
            agent_id=index % 2,
            reward=0.1,
            terminated=False,
            truncated=index == 2,  # noqa: PLR2004
        )
    return buffer


def test_reconstruction_matches_full_history_when_sensing_and_messages_change() -> None:
    # Given: sensing updates, a received map, then inbox consumption.
    expected = frames()
    buffer = buffer_with_episode()
    # When: reconstruct every actual event from periodic snapshots and deltas.
    actual = [buffer.reconstruct(4, index, after=True) for index in range(3)]
    # Then: local maps, timestamps, and inbox state match each full oracle snapshot.
    for frame, reference in zip(actual, expected[1:], strict=True):
        np.testing.assert_array_equal(frame.maps, reference.maps)
        np.testing.assert_array_equal(frame.observed_event, reference.observed_event)
        np.testing.assert_array_equal(
            frame.field("inbox_present"), reference.field("inbox_present")
        )


def test_sampling_resumes_exactly_when_replay_saved_and_loaded(tmp_path: Path) -> None:
    # Given: an advanced replay sampler and persisted numeric archive.
    buffer = buffer_with_episode()
    buffer.sample(length=5, burn_in=1)
    path = tmp_path / "replay.npz"
    buffer.save(path)
    # When: restoring and sampling both original and recovered buffers.
    restored = ReplayBuffer.load(path)
    first = buffer.sample(length=5, burn_in=1)
    second = restored.sample(length=5, burn_in=1)
    # Then: start index and masks agree, with no transition synthesized beyond episode end.
    assert (first.episode_id, first.start_index) == (second.episode_id, second.start_index)
    np.testing.assert_array_equal(first.valid, second.valid)
    assert np.sum(second.valid) <= 3  # noqa: PLR2004
    assert not second.loss_mask[0]
    assert all(entry is None or entry.episode_id == 4 for entry in second.transitions)  # noqa: PLR2004


def test_corrupt_delta_is_reported_when_reconstruction_checksum_disagrees() -> None:
    # Given: a missing delta, as would result from corrupt/incomplete persisted data.
    buffer = buffer_with_episode()
    entry = buffer._episodes[4][0]  # noqa: SLF001
    empty = MapDelta(np.zeros((0, 3), np.int32), np.zeros(0, np.uint8), np.zeros(0, np.int32))
    buffer._episodes[4][0] = replace(entry, delta=empty)  # noqa: SLF001
    # When / Then: corruption cannot silently become a different local history.
    with pytest.raises(ReplayError, match="corrupt_or_missing_replay_delta"):
        buffer.reconstruct(4, 0, after=True)


def test_continuation_is_one_when_transition_only_truncated() -> None:
    # Given / When: an episode reaches the known time limit.
    buffer = buffer_with_episode()
    transition = buffer._episodes[4][-1]  # noqa: SLF001
    # Then: environment continuation differs from the known time cutoff.
    assert transition.truncated
    assert not transition.terminated
    assert transition.continuation == 1


def test_corrupt_next_before_checksum_is_rejected_after_valid_delta() -> None:
    # Given: one valid delta followed by an inconsistent next transition's before checksum.
    buffer = buffer_with_episode()
    entry = buffer._episodes[4][1]  # noqa: SLF001
    buffer._episodes[4][1] = replace(entry, before_checksum="corrupt")  # noqa: SLF001
    # When / Then: reusing the calculated hash still checks the next transition before applying it.
    with pytest.raises(ReplayError, match=r"^corrupt_replay_before_checksum$"):
        buffer.reconstruct(4, 1, after=True)


def test_replay_evicts_complete_old_episodes_when_capacity_reached() -> None:
    # Given: capacity for three events, with one completed two-event episode.
    states = frames()
    buffer = ReplayBuffer(19, snapshot_interval=2, capacity=3)
    for episode in (1, 2):
        for index in range(2):
            buffer.append(
                episode,
                states[index],
                states[index + 1],
                np.zeros(2, np.float32),
                agent_id=index,
                reward=0,
                terminated=False,
                truncated=index == 1,
            )
    # When: the second episode exceeds capacity and old complete data is evicted.
    final = buffer.reconstruct(2, 1, after=True)
    # Then: the retained episode remains exact and no partial delta chain survives eviction.
    assert buffer.size == 2  # noqa: PLR2004
    assert buffer.evicted_events == 2  # noqa: PLR2004
    assert buffer.evicted_episodes == 1
    np.testing.assert_array_equal(final.maps, states[2].maps)
    with pytest.raises(ReplayError, match="unknown_episode"):
        buffer.reconstruct(1, 0)
