"""Host replay with periodic map snapshots and exact event deltas, never GPU map histories."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TypeAlias
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray

from ir2_benchmark.maps.prepare import immutable

Fields: TypeAlias = tuple[tuple[str, NDArray], ...]


def read_field(fields: Fields, name: str) -> NDArray:
    """Read compact replay metadata without reconstructing an unneeded full map."""
    for key, value in fields:
        if key == name:
            return value
    raise ReplayError(f"missing_field:{name}")  # noqa: EM102


@dataclass(frozen=True, slots=True)
class ReplayError(ValueError):
    """Replay continuity, reconstruction, or storage is invalid."""

    reason: str

    def __str__(self) -> str:
        """Describe the concrete typed failure."""
        return self.reason


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    """One reconstructed public state; named fields contain only small non-map arrays."""

    maps: NDArray[np.uint8]
    observed_event: NDArray[np.int32]
    fields: Fields

    def __post_init__(self) -> None:
        """Validate and freeze this value at its input boundary."""
        if self.maps.ndim != 3 or self.maps.shape != self.observed_event.shape:  # noqa: PLR2004
            raise ReplayError("map_and_timestamp_shapes_require_N_H_W")  # noqa: EM101
        object.__setattr__(self, "maps", immutable(np.asarray(self.maps, dtype=np.uint8)))
        object.__setattr__(
            self, "observed_event", immutable(np.asarray(self.observed_event, dtype=np.int32))
        )
        object.__setattr__(
            self,
            "fields",
            tuple((name, immutable(np.asarray(value))) for name, value in self.fields),
        )

    def field(self, name: str) -> NDArray:
        """Look up a required named public tensor, reporting missing metadata explicitly."""
        return read_field(self.fields, name)


@dataclass(frozen=True, slots=True)
class MapDelta:
    """Joint map and observation-time changes; coordinates are [robot,y,x]."""

    coordinates: NDArray[np.int32]
    values: NDArray[np.uint8]
    observed_event: NDArray[np.int32]


def _checksum(maps: NDArray[np.uint8], times: NDArray[np.int32]) -> str:
    """Detect omitted or corrupted deltas instead of silently reconstructing an alternate history."""  # noqa: E501
    return sha256(maps.tobytes() + times.tobytes()).hexdigest()


def _delta(before: ReplayFrame, after: ReplayFrame) -> MapDelta:
    """Store every changed map or timestamp cell exactly once."""
    indices = np.argwhere(
        (before.maps != after.maps) | (before.observed_event != after.observed_event)
    )
    selection = tuple(indices.T)
    return MapDelta(
        immutable(indices.astype(np.int32)),
        immutable(after.maps[selection]),
        immutable(after.observed_event[selection]),
    )


def _apply_delta(maps: NDArray[np.uint8], times: NDArray[np.int32], delta: MapDelta) -> None:
    """Mutate the private reconstruction buffers after parsing delta indices."""
    indices = delta.coordinates
    count = len(indices)
    if (
        indices.shape != (count, 3)
        or delta.values.shape != (count,)
        or delta.observed_event.shape != (count,)
    ):
        raise ReplayError("corrupt_delta_shape")  # noqa: EM101
    if np.any(indices < 0) or np.any(indices >= np.asarray(maps.shape)):
        raise ReplayError("corrupt_delta_coordinates")  # noqa: EM101
    if len(np.unique(indices, axis=0)) != count or np.any(delta.values > 2):  # noqa: PLR2004
        raise ReplayError("corrupt_delta_values_or_duplicates")  # noqa: EM101
    selection = tuple(indices.T)
    maps[selection] = delta.values
    times[selection] = delta.observed_event


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    """Small event metadata and local-map delta; snapshots exist only periodically."""

    episode_id: int
    episode_index: int
    agent_id: int
    action: NDArray[np.float32]
    reward: float
    terminated: bool
    truncated: bool
    before_fields: Fields
    after_fields: Fields
    delta: MapDelta
    snapshot: tuple[NDArray[np.uint8], NDArray[np.int32]] | None
    before_checksum: str
    after_checksum: str
    plan_labels: Fields = ()

    @property
    def continuation(self) -> float:
        """Time limits are known externally; only genuine termination labels zero continuation."""
        return float(not self.terminated)


@dataclass(frozen=True, slots=True)
class ReplaySequence:
    """No cross-episode joins; short tails are masked, with burn-in excluded from losses."""

    episode_id: int
    start_index: int
    transitions: tuple[ReplayTransition | None, ...]
    valid: NDArray[np.bool_]
    loss_mask: NDArray[np.bool_]


class ReplayBuffer:
    """Mutable host accumulator; one working full frame per live episode, sparse persistent history."""  # noqa: E501

    def __init__(self, seed: int, snapshot_interval: int = 32, capacity: int = 4096) -> None:
        """Initialize bounded replay and its independent sampling stream."""
        if snapshot_interval < 1 or capacity < 1:
            raise ReplayError("snapshot_interval_must_be_positive")  # noqa: EM101
        self.snapshot_interval = snapshot_interval
        self.capacity = capacity
        self.evicted_events = 0
        self.evicted_episodes = 0
        self._rng = np.random.default_rng(seed)
        self._episodes: dict[int, list[ReplayTransition]] = {}
        self._last: dict[int, ReplayFrame] = {}

    @property
    def size(self) -> int:
        """Count real event transitions, excluding all sampling padding."""
        return sum(len(episode) for episode in self._episodes.values())

    @property
    def sampling_state(self) -> str:
        """JSON RNG state makes resumed sampling independent of construction seed."""
        return json.dumps(self._rng.bit_generator.state, sort_keys=True)

    def has_sequence(self, burn_in: int) -> bool:
        """Report whether any retained episode can supply a post-burn-in real training label."""
        return any(len(episode) > burn_in for episode in self._episodes.values())

    def restore_sampling_state(self, state: str) -> None:
        """Restore the host sampler stream from a checkpoint."""
        self._rng.bit_generator.state = json.loads(state)

    def append(  # noqa: PLR0913
        self,
        episode_id: int,
        before: ReplayFrame,
        after: ReplayFrame,
        action: NDArray[np.float32],
        *,
        agent_id: int,
        reward: float,
        terminated: bool,
        truncated: bool,
        plan_labels: Fields = (),
    ) -> None:
        """Record one contiguous real event, rejecting missing transitions before compressing maps."""  # noqa: E501
        episode = self._episodes.setdefault(episode_id, [])
        if episode and (episode[-1].terminated or episode[-1].truncated):
            raise ReplayError("cannot_append_to_finished_episode")  # noqa: EM101
        if before.maps.shape != after.maps.shape:
            raise ReplayError("map_shape_changed_inside_transition")  # noqa: EM101
        before_hash = _checksum(before.maps, before.observed_event)
        if episode and before_hash != episode[-1].after_checksum:
            raise ReplayError("missing_transition_between_replay_frames")  # noqa: EM101
        index = len(episode)
        snapshot = (
            (before.maps, before.observed_event) if index % self.snapshot_interval == 0 else None
        )
        episode.append(
            ReplayTransition(
                episode_id,
                index,
                agent_id,
                immutable(np.asarray(action, dtype=np.float32)),
                float(reward),
                bool(terminated),
                bool(truncated),
                before.fields,
                after.fields,
                _delta(before, after),
                snapshot,
                before_hash,
                _checksum(after.maps, after.observed_event),
                plan_labels,
            )
        )
        if terminated or truncated:
            self._last.pop(episode_id, None)
        else:
            self._last[episode_id] = after
        while self.size > self.capacity:
            finished = next(
                (
                    identifier
                    for identifier, rows in self._episodes.items()
                    if identifier != episode_id and (rows[-1].terminated or rows[-1].truncated)
                ),
                None,
            )
            if finished is None:
                raise ReplayError("capacity_exceeded_by_open_episode")  # noqa: EM101
            self.evicted_events += len(self._episodes[finished])
            self.evicted_episodes += 1
            del self._episodes[finished]
            self._last.pop(finished, None)

    def reconstruct(self, episode_id: int, index: int, *, after: bool = False) -> ReplayFrame:
        """Reconstruct from the closest preceding snapshot and verify every replayed delta."""
        if episode_id not in self._episodes or not 0 <= index < len(self._episodes[episode_id]):
            raise ReplayError("unknown_episode_or_event_index")  # noqa: EM101
        episode = self._episodes[episode_id]
        anchor = index - index % self.snapshot_interval
        snapshot = episode[anchor].snapshot
        if snapshot is None:
            raise ReplayError("missing_periodic_snapshot")  # noqa: EM101
        maps, times = snapshot[0].copy(), snapshot[1].copy()
        current_checksum = _checksum(maps, times)
        for current in range(anchor, index + int(after)):
            transition = episode[current]
            if current_checksum != transition.before_checksum:
                raise ReplayError("corrupt_replay_before_checksum")  # noqa: EM101
            _apply_delta(maps, times, transition.delta)
            current_checksum = _checksum(maps, times)
            if current_checksum != transition.after_checksum:
                raise ReplayError("corrupt_or_missing_replay_delta")  # noqa: EM101
        transition = episode[index]
        expected = transition.after_checksum if after else transition.before_checksum
        if current_checksum != expected:
            raise ReplayError("corrupt_reconstructed_frame")  # noqa: EM101
        return ReplayFrame(
            maps, times, transition.after_fields if after else transition.before_fields
        )

    def sample(self, length: int = 32, burn_in: int = 8) -> ReplaySequence:
        """Choose a start within one episode and pad only the unavailable tail."""
        if not self.size or not 0 <= burn_in < length:
            raise ReplayError("sampling_requires_nonempty_replay_and_burn_in_less_than_length")  # noqa: EM101
        ids = sorted(self._episodes)
        episode_id = ids[int(self._rng.integers(len(ids)))]
        episode = self._episodes[episode_id]
        start = int(self._rng.integers(len(episode)))
        count = min(length, len(episode) - start)
        transitions = tuple(episode[start : start + count]) + (None,) * (length - count)
        valid = np.arange(length) < count
        return ReplaySequence(
            episode_id,
            start,
            transitions,
            immutable(valid),
            immutable(valid & (np.arange(length) >= burn_in)),
        )

    def save(self, path: str | Path) -> None:
        """Persist numeric arrays and JSON metadata without pickle or device-resident map histories."""  # noqa: E501
        arrays: dict[str, NDArray] = {}
        metadata = []
        for episode_id, episode in self._episodes.items():
            for entry in episode:
                prefix = f"e{episode_id}_t{entry.episode_index}"
                for name, value in (
                    ("action", entry.action),
                    ("coordinates", entry.delta.coordinates),
                    ("values", entry.delta.values),
                    ("times", entry.delta.observed_event),
                ):
                    arrays[f"{prefix}_{name}"] = value
                field_names = []
                for group, fields in (
                    ("before", entry.before_fields),
                    ("after", entry.after_fields),
                    ("labels", entry.plan_labels),
                ):
                    names = [name for name, _ in fields]
                    field_names.append(names)
                    for name, value in fields:
                        arrays[f"{prefix}_{group}_{name}"] = value
                if entry.snapshot is not None:
                    arrays[f"{prefix}_snapshot_maps"] = entry.snapshot[0]
                    arrays[f"{prefix}_snapshot_times"] = entry.snapshot[1]
                metadata.append(
                    (
                        episode_id,
                        entry.episode_index,
                        entry.agent_id,
                        entry.reward,
                        entry.terminated,
                        entry.truncated,
                        entry.before_checksum,
                        entry.after_checksum,
                        entry.snapshot is not None,
                        field_names,
                    )
                )
        arrays["metadata"] = np.asarray(
            json.dumps(
                {
                    "entries": metadata,
                    "sampling_state": self.sampling_state,
                    "snapshot_interval": self.snapshot_interval,
                    "capacity": self.capacity,
                    "evicted_events": self.evicted_events,
                    "evicted_episodes": self.evicted_episodes,
                }
            )
        )
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        """Restore replay data, sampler state, and open episode frames from a numeric archive."""
        with ZipFile(path) as archive:
            def read_array(name: str) -> NDArray:
                """Read a required numeric member without scanning the archive member list."""
                with archive.open(f"{name}.npy") as member:
                    return np.lib.format.read_array(member, allow_pickle=False)

            metadata = json.loads(str(read_array("metadata")))
            buffer = cls(0, metadata["snapshot_interval"], metadata["capacity"])
            buffer.evicted_events = metadata["evicted_events"]
            buffer.evicted_episodes = metadata["evicted_episodes"]
            for (
                episode_id,
                index,
                agent,
                reward,
                terminated,
                truncated,
                before_hash,
                after_hash,
                snapshot_present,
                names,
            ) in metadata["entries"]:
                prefix = f"e{episode_id}_t{index}"
                fields = [
                    tuple(
                        (name, immutable(read_array(f"{prefix}_{group}_{name}")))
                        for name in group_names
                    )
                    for group, group_names in zip(("before", "after", "labels"), names, strict=True)
                ]
                snapshot = (
                    (
                        immutable(read_array(f"{prefix}_snapshot_maps")),
                        immutable(read_array(f"{prefix}_snapshot_times")),
                    )
                    if snapshot_present
                    else None
                )
                delta = MapDelta(
                    immutable(read_array(f"{prefix}_coordinates")),
                    immutable(read_array(f"{prefix}_values")),
                    immutable(read_array(f"{prefix}_times")),
                )
                entry = ReplayTransition(
                    episode_id,
                    index,
                    agent,
                    immutable(read_array(f"{prefix}_action")),
                    reward,
                    terminated,
                    truncated,
                    fields[0],
                    fields[1],
                    delta,
                    snapshot,
                    before_hash,
                    after_hash,
                    fields[2],
                )
                buffer._episodes.setdefault(episode_id, []).append(entry)
            buffer.restore_sampling_state(metadata["sampling_state"])
            for episode_id, episode in buffer._episodes.items():
                final = episode[-1]
                frame = buffer.reconstruct(episode_id, len(episode) - 1, after=True)
                if not final.terminated and not final.truncated:
                    buffer._last[episode_id] = frame
            return buffer
