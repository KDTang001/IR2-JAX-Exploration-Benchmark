"""Model-independent physical-coverage, accounting, and plan-quality metrics."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class MetricError(ValueError):
    """An input trajectory or metric tensor violates the evaluation contract."""

    reason: str

    def __str__(self) -> str:
        """Describe the concrete typed failure."""
        return self.reason


class EpisodeOutcome(StrEnum):
    """Normal exploration outcomes remain distinct from execution failures."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    MODEL_ERROR = "model_error"
    MAP_ERROR = "map_error"
    OOM = "oom"
    ENVIRONMENT_ERROR = "environment_error"


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Times are absent unless success actually occurred; rounds start at one."""

    outcome: EpisodeOutcome
    success: bool
    timeout: bool
    auc: float | None
    final_coverage: float | None
    success_round: int | None
    success_event_offset: int | None
    observed_rounds: int


def _coverage_values(coverage: Sequence[float]) -> NDArray[np.float64]:
    """Parse finite monotone physical-coverage fractions at the API boundary."""
    values = np.asarray(coverage, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise MetricError("coverage_requires_finite_fractions")  # noqa: EM101
    if np.any(np.diff(values) < -1e-12):  # noqa: PLR2004
        raise MetricError("physical_coverage_cannot_decrease")  # noqa: EM101
    return values


def coverage_auc(coverage: Sequence[float], horizon: int, *, terminated: bool = False) -> float:
    """Mean round-end C, holding actual terminal coverage through unexecuted rounds."""
    values = _coverage_values(coverage)
    if horizon < 1 or not 1 <= len(values) <= horizon:
        raise MetricError("coverage_length_must_be_within_horizon")  # noqa: EM101
    if len(values) != horizon and not terminated:
        raise MetricError("unfinished_trajectory_cannot_be_padded_as_timeout")  # noqa: EM101
    return float((values.sum() + (horizon - len(values)) * values[-1]) / horizon)


def summarize_episode(  # noqa: PLR0913
    coverage: Sequence[float],
    *,
    horizon: int,
    n_agents: int,
    outcome: EpisodeOutcome,
    event_coverage: Sequence[float] | None = None,
    threshold: float = 0.99,
) -> EpisodeMetrics:
    """Summarize real end-of-round coverage, optionally locating success inside a round."""
    values = _coverage_values(coverage)
    if not 1 <= n_agents <= 6 or not 0 < threshold <= 1 or horizon < 1 or len(values) > horizon:  # noqa: PLR2004
        raise MetricError("invalid_episode_metric_parameters")  # noqa: EM101
    success = outcome == EpisodeOutcome.SUCCESS
    timeout = outcome == EpisodeOutcome.TIMEOUT
    crossings = np.flatnonzero(values >= threshold)
    if success and not len(crossings):
        raise MetricError("success_requires_observed_threshold_crossing")  # noqa: EM101
    if timeout and len(crossings):
        raise MetricError("timeout_cannot_contain_success")  # noqa: EM101
    first_round = int(crossings[0]) + 1 if success else None
    offset = None
    if event_coverage is not None:
        events = _coverage_values(event_coverage)
        if len(events) > horizon * n_agents:
            raise MetricError("event_trajectory_exceeds_horizon")  # noqa: EM101
        crossed_events = np.flatnonzero(events >= threshold)
        if success:
            if not len(crossed_events):
                raise MetricError("successful_event_trajectory_requires_threshold_crossing")  # noqa: EM101
            event = int(crossed_events[0])
            if event // n_agents + 1 != first_round:
                raise MetricError("event_and_round_success_times_disagree")  # noqa: EM101
            offset = event % n_agents
    auc = coverage_auc(coverage, horizon, terminated=success) if success or timeout else None
    return EpisodeMetrics(
        outcome,
        success,
        timeout,
        auc,
        float(values[-1]) if len(values) else None,
        first_round,
        offset,
        len(values),
    )


def _footprints(
    personal_observed: NDArray[np.bool_],
    component: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Require explicit personal footprints, never local maps that may contain received cells."""
    personal = np.asarray(personal_observed)
    reachable = np.asarray(component)
    if personal.dtype != np.bool_ or reachable.dtype != np.bool_:
        raise MetricError("physical_coverage_requires_boolean_personal_footprints_and_component")  # noqa: EM101
    if (
        reachable.ndim != 2  # noqa: PLR2004
        or personal.ndim != 3  # noqa: PLR2004
        or personal.shape[1:] != reachable.shape
        or not len(personal)
    ):
        raise MetricError("footprints_require_N_H_W_and_matching_component_H_W")  # noqa: EM101
    if not np.any(reachable):
        raise MetricError("coverage_component_cannot_be_empty")  # noqa: EM101
    return personal & reachable[None, :, :]


def physical_coverage(personal_observed: NDArray[np.bool_], component: NDArray[np.bool_]) -> float:
    """Measure the union of independently sensed free cells only within anchor component F."""
    footprints = _footprints(personal_observed, component)
    return float(np.any(footprints, axis=0).sum() / np.count_nonzero(component))


def cross_agent_overlap(
    personal_observed: NDArray[np.bool_],
    component: NDArray[np.bool_],
) -> float | None:
    """Return 1-|union S_i|/sum |S_i|, missing when no personal free cells exist."""
    footprints = _footprints(personal_observed, component)
    total = int(footprints.sum())
    return 1.0 - float(np.any(footprints, axis=0).sum()) / total if total else None


def allocation_concentration(first_observer_counts: Sequence[int]) -> float | None:
    """Sum squared shares of first discoveries; fixed acting order determines ties."""
    counts = np.asarray(first_observer_counts)
    if counts.ndim != 1 or np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise MetricError("first_observer_counts_must_be_nonnegative")  # noqa: EM101
    total = float(counts.sum())
    return float(np.sum((counts / total) ** 2)) if total else None


@dataclass(frozen=True, slots=True)
class CommunicationTotals:
    """Emission includes headers; delivery counts payload copies per receiver."""

    emitted_bytes: int
    emitted_payload_bytes: int
    delivered_payload_bytes: int
    emitted_messages: int
    delivered_messages: int


def communication_totals(
    packet_lengths: Sequence[int],
    payload_lengths: Sequence[int],
    delivered: NDArray[np.bool_],
) -> CommunicationTotals:
    """Sum actual transmissions, including all-lost broadcasts; zero lengths denote no emission."""
    packets = np.asarray(packet_lengths)
    payloads = np.asarray(payload_lengths)
    mask = np.asarray(delivered)
    if (
        packets.ndim != 1
        or packets.shape != payloads.shape
        or mask.ndim != 2  # noqa: PLR2004
        or len(mask) != len(packets)
    ):
        raise MetricError("communication_arrays_require_E_and_E_receivers")  # noqa: EM101
    if (
        not np.issubdtype(packets.dtype, np.integer)
        or not np.issubdtype(payloads.dtype, np.integer)
        or mask.dtype != np.bool_
    ):
        raise MetricError("communication_lengths_require_integer_and_delivery_boolean")  # noqa: EM101
    if np.any((packets < 0) | (packets > 256) | (payloads < 0)) or np.any(payloads > packets):  # noqa: PLR2004
        raise MetricError("invalid_packet_or_payload_length")  # noqa: EM101
    if np.any((packets > 0) & (packets - payloads != 16)):  # noqa: PLR2004
        raise MetricError("emitted_packets_require_16_byte_header")  # noqa: EM101
    if np.any(mask & (packets[:, None] == 0)):
        raise MetricError("unemitted_packets_cannot_be_delivered")  # noqa: EM101
    return CommunicationTotals(
        int(packets.sum()),
        int(payloads.sum()),
        int(np.sum(payloads[:, None] * mask)),
        int(np.count_nonzero(packets)),
        int(mask.sum()),
    )


def _coordinate_scale(valid_shape: tuple[int, int]) -> NDArray[np.float64]:
    """Scale xy by last valid pixel, defining a one-cell dimension's scale as one."""
    h, w = valid_shape
    if min(h, w) < 1:
        raise MetricError("valid_shape_requires_positive_height_width")  # noqa: EM101
    return np.asarray((max(w - 1, 1), max(h - 1, 1)), dtype=np.float64)


def normalize_positions(
    positions: NDArray[np.float64], valid_shape: tuple[int, int]
) -> NDArray[np.float64]:
    """Normalize xy coordinates using the unpadded (height,width)."""
    return np.asarray(positions, dtype=np.float64) / _coordinate_scale(valid_shape)


def denormalize_positions(
    positions: NDArray[np.float64], valid_shape: tuple[int, int]
) -> NDArray[np.float64]:
    """Restore xy coordinates with the same axis order and unpadded dimensions."""
    return np.asarray(positions, dtype=np.float64) * _coordinate_scale(valid_shape)


@dataclass(frozen=True, slots=True)
class PlanErrors:
    """Missing labels remain missing; NLL uses a diagonal Gaussian in input coordinate units."""

    ade: float | None
    fde: float | None
    valid_points: int
    goal_error: float | None = None
    nll: float | None = None
    coverage_one_sigma: float | None = None


def plan_errors(  # noqa: PLR0913
    predicted: NDArray[np.float64],
    target: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    *,
    predicted_goal: NDArray[np.float64] | None = None,
    target_goal: NDArray[np.float64] | None = None,
    stddev: NDArray[np.float64] | None = None,
) -> PlanErrors:
    """Compute ADE/FDE using only valid timesteps, plus optional goal/NLL/calibration metrics."""
    prediction = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    mask = np.asarray(valid_mask)
    if (
        prediction.ndim != 2  # noqa: PLR2004
        or prediction.shape[1] != 2  # noqa: PLR2004
        or prediction.shape != truth.shape
        or mask.shape != prediction.shape[:1]
        or mask.dtype != np.bool_
    ):
        raise MetricError("plan_arrays_require_H_2_and_boolean_mask_H")  # noqa: EM101
    difference = prediction[mask] - truth[mask]
    if not np.all(np.isfinite(difference)):
        raise MetricError("valid_plan_coordinates_must_be_finite")  # noqa: EM101
    count = len(difference)
    distances = np.linalg.norm(difference, axis=-1)
    goal_error = None
    if (predicted_goal is None) != (target_goal is None):
        raise MetricError("goal_error_requires_prediction_and_label")  # noqa: EM101
    if predicted_goal is not None and target_goal is not None:
        p_goal, t_goal = np.asarray(predicted_goal), np.asarray(target_goal)
        if p_goal.shape != (2,) or t_goal.shape != (2,) or not np.all(np.isfinite(p_goal - t_goal)):
            raise MetricError("goals_require_finite_xy")  # noqa: EM101
        goal_error = float(np.linalg.norm(p_goal - t_goal))
    nll = calibration = None
    if stddev is not None:
        sigma = np.asarray(stddev)
        if (
            sigma.shape != prediction.shape
            or np.any(sigma[mask] <= 0)
            or not np.all(np.isfinite(sigma[mask]))
        ):
            raise MetricError("valid_standard_deviations_must_be_positive_finite_H_2")  # noqa: EM101
        if count:
            standardized = difference / sigma[mask]
            nll = float(
                np.mean(
                    np.sum(
                        0.5 * standardized**2 + np.log(sigma[mask]) + 0.5 * np.log(2 * np.pi),
                        axis=-1,
                    )
                )
            )
            calibration = float(np.mean(np.abs(standardized) <= 1))
    return PlanErrors(
        float(distances.mean()) if count else None,
        float(distances[-1]) if count else None,
        count,
        goal_error,
        nll,
        calibration,
    )


@dataclass(frozen=True, slots=True)
class PlanSample:
    """One plan quality observation, separating intention from actual future position targets."""

    ade: float | None
    fde: float | None
    message_age: int | None
    phase: Literal["normal", "fault", "recovery"]
    prediction_target: Literal["intention", "actual_position"]
    goal_error: float | None = None
    nll: float | None = None
    coverage_one_sigma: float | None = None


@dataclass(frozen=True, slots=True)
class PlanGroup:
    """A stratum keyed by exact age, fault phase, target kind, and cold-start status."""

    message_age: int | None
    phase: str
    prediction_target: str
    cold_start: bool
    count: int
    ade: float | None
    fde: float | None
    goal_error: float | None
    nll: float | None
    coverage_one_sigma: float | None
    valid_ade_count: int
    valid_nll_count: int


def _mean_present(values: Sequence[float | None]) -> float | None:
    """Keep an entirely missing measurement missing instead of reporting zero quality error."""
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def grouped_plan_errors(samples: Sequence[PlanSample]) -> tuple[PlanGroup, ...]:
    """Aggregate exact message ages and phases, retaining the no-message cold-start stratum."""
    groups: dict[tuple[int | None, str, str], list[PlanSample]] = {}
    for sample in samples:
        if sample.message_age is not None and sample.message_age < 0:
            raise MetricError("message_age_cannot_be_negative")  # noqa: EM101
        groups.setdefault((sample.message_age, sample.phase, sample.prediction_target), []).append(
            sample
        )
    result: list[PlanGroup] = []
    for (age, phase, target), entries in sorted(
        groups.items(),
        key=lambda pair: (pair[0][1], pair[0][2], -1 if pair[0][0] is None else pair[0][0]),
    ):
        result.append(
            PlanGroup(
                age,
                phase,
                target,
                age is None,
                len(entries),
                _mean_present([entry.ade for entry in entries]),
                _mean_present([entry.fde for entry in entries]),
                _mean_present([entry.goal_error for entry in entries]),
                _mean_present([entry.nll for entry in entries]),
                _mean_present([entry.coverage_one_sigma for entry in entries]),
                sum(entry.ade is not None for entry in entries),
                sum(entry.nll is not None for entry in entries),
            )
        )
    return tuple(result)
