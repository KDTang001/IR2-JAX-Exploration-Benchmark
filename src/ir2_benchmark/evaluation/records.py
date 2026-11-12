"""Checksummed unique result records and resumable artifact verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ir2_benchmark.evaluation.metrics import EpisodeOutcome
from ir2_benchmark.evaluation.scenarios import identity

if TYPE_CHECKING:
    from ir2_benchmark.evaluation.budgets import BudgetMode
    from ir2_benchmark.evaluation.metrics import EpisodeMetrics
    from ir2_benchmark.evaluation.scenarios import EvaluationTask


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """One unique requested task, including execution failures and actual cost."""

    result_id: str
    scenario_id: str
    method_id: str
    route: str
    training_seed: int
    source: str
    family: str
    map_hash: str
    split: str
    n_agents: int
    budget: int
    fault_mode: int
    drop_probability: float
    burst_duration: int
    limited: bool
    outcome: str
    metrics: EpisodeMetrics | None
    emitted_bytes: int
    delivered_payload_bytes: int
    overlap: float | None
    allocation_concentration: float | None
    cold_start_events: int
    recovery_coverage_gain: float | None
    active_events: int
    checkpoint_hash: str
    configuration_hash: str
    trace_file: str
    trace_hash: str
    error: str | None
    plan_diagnostics: bool = False
    map_scenario_hash: str = ""
    decision_rule: str = "unspecified"
    real_training_events: int = 0
    selected_at_events: int | None = None
    budget_mode: BudgetMode = "per_robot"
    requested_team_bytes: int | None = None
    unallocated_bytes: int = 0
    auxiliary_kind: str = ""


@dataclass(frozen=True, slots=True)
class RunnerSummary:
    """Saved tasks are unique; execution errors are retained rather than normal timeouts."""

    requested: int
    executed: int
    resumed: int
    failures: int


def file_hash(path: Path) -> str:
    """Hash a real artifact without loading a large checkpoint/trace into another buffer."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_result(path: Path, result: EvaluationResult) -> None:
    """Atomically replace one result file with a checksum over its complete payload."""
    payload = asdict(result)
    envelope = {"result": payload, "checksum": identity(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_result(path: Path, task: EvaluationTask) -> bool:
    """Resume only an intact matching result and its checksum-verified real trace."""
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    payload = row.get("result", {})
    matching = (
        row.get("checksum") == identity(payload)
        and payload.get("result_id") == task.result_id
        and payload.get("checkpoint_hash") == task.method.checkpoint_hash
        and payload.get("configuration_hash") == task.scenario.config_hash
    )
    if not matching:
        return False
    trace = payload.get("trace_file", "")
    if payload.get("outcome") in (EpisodeOutcome.SUCCESS, EpisodeOutcome.TIMEOUT) and not trace:
        return False
    return not trace or (
        Path(trace).is_file() and file_hash(Path(trace)) == payload.get("trace_hash")
    )
