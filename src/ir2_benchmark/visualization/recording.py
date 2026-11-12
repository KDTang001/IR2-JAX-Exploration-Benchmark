"""Bounded host capture of selected real episodes, independent of training state and RNG."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np

from ir2_benchmark.config import ConfigError

if TYPE_CHECKING:
    from jax import Array

    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.training.storage import StandardAction
    from ir2_benchmark.types import EnvState


@dataclass(frozen=True, slots=True)
class RecordingConfig:
    """Host-only output settings; these are excluded from model/training identity."""

    every_episodes: int = 201
    frame_stride_rounds: int = 4
    max_frames: int = 129
    duration_ms: int = 500
    max_side: int = 768

    def __post_init__(self) -> None:
        """Validate bounded output capacities before training starts."""
        if (
            self.every_episodes < 0
            or min(self.frame_stride_rounds, self.duration_ms, self.max_side) < 1
            or self.max_frames < 2  # noqa: PLR2004
        ):
            raise ConfigError(
                field="recording",
                detail="nonnegative cadence, positive dimensions/stride/duration; max_frames >= 2",
            )


class RecordedEvent(NamedTuple):
    """Actual before-event IDs and after-event evidence, with a leading event axis [T]."""

    event: Array
    agent_id: Array
    positions: Array
    goals: Array
    balances: Array
    spent: Array
    delivered: Array
    coverage: Array
    terminated: Array
    truncated: Array


@dataclass(frozen=True, slots=True)
class ModelVersions:
    """Actual optimizer/model steps used for a recorded real-action chunk."""

    actor: int
    predictor: int
    phase: Literal["warmup", "train"]
    world_model: int | None = None


@dataclass(frozen=True, slots=True)
class RecordingError(RuntimeError):
    """Recording or replay evidence failed, rather than generating an invented picture."""

    detail: str

    def __str__(self) -> str:
        """Expose the failed evidence check."""
        return self.detail


@dataclass(frozen=True, slots=True)
class CaptureChunk:
    """Only compact host arrays survive along the time axis."""

    waypoint: np.ndarray
    send: np.ndarray
    codec: np.ndarray
    expected: RecordedEvent
    versions: ModelVersions


@dataclass(frozen=True, slots=True)
class EpisodeCapture:
    """Immutable birth snapshot and a deliberately mutable bounded host chunk buffer."""

    episode_id: str
    initial: EnvState
    config: HostConfig
    training_start: int
    episode_index: int
    partial: bool
    chunks: list[CaptureChunk] = field(default_factory=list)


class TrainingRecorder:
    """Record selected live episodes without modifying learner keys, counters or memories."""

    __slots__ = ("_active", "options", "output", "route", "training_seed")

    def __init__(
        self,
        output: Path,
        options: RecordingConfig | None = None,
        *,
        training_seed: int | None = None,
        route: str | None = None,
    ) -> None:
        """Keep output policy separate from checkpoint/training configuration hashes."""
        self.output, self.options = Path(output), options or RecordingConfig()
        self.training_seed, self.route = training_seed, route
        self._active: dict[str, EpisodeCapture] = {}

    def should_record(self, episode_index: int) -> bool:
        """Select the first episode and cadence multiples; zero disables capture completely."""
        return self.options.every_episodes > 0 and episode_index % self.options.every_episodes == 0

    def begin_episode(  # noqa: PLR0913
        self,
        episode_id: str,
        initial_state: EnvState,
        config: HostConfig,
        *,
        training_events: int,
        episode_index: int,
        partial: bool = False,
    ) -> bool:
        """Capture once; a repeated open ID preserves the earliest snapshot and returns True."""
        if episode_id in self._active:
            return True
        if not self.should_record(episode_index):
            return False
        import jax  # noqa: PLC0415

        snapshot = jax.tree.map(
            lambda value: np.asarray(value).copy(), jax.device_get(initial_state)
        )
        self._active[episode_id] = EpisodeCapture(
            episode_id,
            snapshot,
            config,
            training_events,
            episode_index,
            partial or int(snapshot.event) > 0,
        )
        return True

    def append_actions(
        self,
        episode_id: str,
        actions: StandardAction,
        valid: Array,
        expected: RecordedEvent,
        *,
        versions: ModelVersions,
    ) -> None:
        """Copy accepted real events only; skipped IDs never transfer device arrays to the host."""
        capture = self._active.get(episode_id)
        if capture is None:
            return
        indices = np.flatnonzero(np.asarray(valid, bool))
        if not len(indices):
            return
        evidence = RecordedEvent(*(np.asarray(value)[indices].copy() for value in expected))
        count = sum(len(chunk.waypoint) for chunk in capture.chunks)
        start = int(capture.initial.event) + count
        if not np.array_equal(evidence.event, np.arange(start, start + len(indices))):
            detail = "noncontiguous_real_recording_events"
            raise RecordingError(detail)
        if start + len(indices) > capture.config.max_rounds * capture.config.n_agents:
            detail = "recording_exceeds_episode_event_capacity"
            raise RecordingError(detail)
        codec = np.asarray(actions.codec)[indices].copy()
        if np.any((codec < 1) | (codec > 4)):  # noqa: PLR2004
            detail = "only_actual_standard_codecs_are_supported"
            raise RecordingError(detail)
        capture.chunks.append(
            CaptureChunk(
                np.asarray(actions.waypoint)[indices].copy(),
                np.asarray(actions.send)[indices].copy(),
                codec,
                evidence,
                versions,
            )
        )

    def end_episode(self, episode_id: str, reason: str, training_events: int) -> Path | None:
        """Replay/check one captured real episode and write GIFs, last PNGs and metadata."""
        capture = self._active.pop(episode_id, None)
        if capture is None:
            return None
        from ir2_benchmark.visualization.replay import export_capture  # noqa: PLC0415

        return export_capture(
            capture,
            self.output,
            self.options,
            reason,
            training_events,
            self.training_seed,
            self.route,
        )

    def close(self, training_events: int) -> tuple[Path, ...]:
        """Flush each unfinished actual prefix as an explicitly partial training-end clip."""
        return tuple(
            path
            for episode_id in tuple(self._active)
            if (path := self.end_episode(episode_id, "training_end", training_events)) is not None
        )


def checked_event(actual: RecordedEvent, expected: RecordedEvent, index: int) -> None:
    """Require exact discrete evidence and a disclosed float tolerance before writing any GIF."""
    for name, value in zip(expected._fields, expected, strict=True):
        reference, observed = np.asarray(value[index]), np.asarray(getattr(actual, name))
        same = (
            np.allclose(observed, reference, rtol=1e-6, atol=1e-7)
            if np.issubdtype(reference.dtype, np.floating)
            else np.array_equal(observed, reference)
        )
        if not same:
            detail = f"replay_mismatch:{name}:event={int(expected.event[index])}"
            raise RecordingError(detail)
