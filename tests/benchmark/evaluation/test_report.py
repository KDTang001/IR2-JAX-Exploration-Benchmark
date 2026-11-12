"""Hand-computable family weights and missing training repetitions guard report claims."""

import json
from pathlib import Path

import numpy as np

from ir2_benchmark.evaluation.records import file_hash
from ir2_benchmark.evaluation.report import (
    ReportRow,
    family_balanced_mean,
    invalid_training_quotas,
    load_report_rows,
    missing_conditions,
    summarize_method,
)
from ir2_benchmark.evaluation.scenarios import identity


def test_families_are_equal_weight_despite_unequal_map_counts() -> None:
    # Given: one perfect map in family one, three zero-score maps in family two.
    rows = (
        ReportRow("one", "A", 0, "ir2", "one", "m1", auc=1),
        ReportRow("two", "A", 0, "ir2", "two", "m2", auc=0),
        ReportRow("three", "A", 0, "ir2", "two", "m3", auc=0),
        ReportRow("four", "A", 0, "ir2", "two", "m4", auc=0),
    )
    # When: scenarios, maps and then families are aggregated.
    actual = family_balanced_mean(rows)
    # Then: family one contributes half, rather than one quarter of the score.
    assert actual == 0.5  # noqa: PLR2004


def test_missing_training_seeds_remain_explicit_with_many_episodes() -> None:
    # Given: a hundred episodes from one training repetition.
    rows = tuple(
        ReportRow(str(index), "A", 0, "ir2", "rooms", "map", auc=0.4) for index in range(100)
    )
    # When: summarizing against five requested independent training seeds.
    summary = summarize_method("A", rows, (0, 1, 2, 3, 4), repetitions=20)
    # Then: extra episodes do not manufacture four missing trained models.
    assert summary.training_seeds == (0,)
    assert summary.missing_training_seeds == (1, 2, 3, 4)


def test_runtime_errors_have_no_manufactured_auc_or_success_time() -> None:
    # Given: one timeout and one OOM retained in the same method table.
    rows = (
        ReportRow("timeout", "A", 0, "ir2", "rooms", "m1", auc=0.2),
        ReportRow("error", "A", 0, "ir2", "rooms", "m2", outcome="oom", error="device memory"),
    )
    # When: computing available scores with explicit failure counts.
    summary = summarize_method("A", rows, (0,), repetitions=20)
    # Then: neither failure is represented as a successful timed exploration.
    assert summary.success_rate == 0
    assert summary.errors == 1
    assert rows[1].auc is None
    assert rows[1].success_round is None


def test_missing_budget_is_an_incomplete_matrix_condition() -> None:
    # Given: only the zero-budget condition of a two-budget declaration.
    rows = (ReportRow("only-zero", "A", 0, "ir2", "rooms", "m1", budget=0, auc=0.2),)
    declaration = {
        "methods": [{"method_id": "A"}],
        "training_seeds": [0],
        "n_agents": [2],
        "budgets": [0, 2048],
        "conditions": [[0, 0, 64]],
    }
    # When: comparing observed strata with the requested matrix.
    missing = missing_conditions(rows, declaration)
    # Then: absence is explicit, even if a shortened task manifest forgot that budget.
    assert missing == ("A:0:2:2048:0:0:64",)


def test_tampered_trace_cannot_enter_a_complete_report(tmp_path: Path) -> None:
    # Given: an intact checksummed JSON record and its actual trace content hash.
    root = tmp_path / "results"
    root.mkdir()
    trace = tmp_path / "trace.npz"
    np.savez(trace, coverage=np.array([0.25]))
    raw = {
        "result_id": "episode",
        "method_id": "A",
        "training_seed": 0,
        "source": "ir2",
        "family": "rooms",
        "map_hash": "map",
        "outcome": "timeout",
        "metrics": {"auc": 0.25, "final_coverage": 0.25, "success_round": None},
        "trace_file": str(trace),
        "trace_hash": file_hash(trace),
    }
    (root / "episode.json").write_text(
        json.dumps({"result": raw, "checksum": identity(raw)}), encoding="utf-8"
    )
    trace.write_bytes(b"changed-after-result-was-saved")
    # When: report rows are loaded after the trace was modified.
    rows, corrupt = load_report_rows(tmp_path)
    # Then: valid JSON alone cannot certify the episode as complete.
    assert rows == ()
    assert len(corrupt) == 1


def test_short_training_cannot_claim_a_complete_core_quota() -> None:
    # Given: good exploratory scores from a short training run and an early selected best model.
    short = ReportRow("short", "A", 0, "ir2", "rooms", "map", auc=0.8, real_training_events=4096)
    complete_run = ReportRow(
        "best",
        "A",
        1,
        "ir2",
        "rooms",
        "map",
        auc=0.8,
        real_training_events=2000000,
        selected_at_events=1800000,
    )
    declaration = {"methods": [{"method_id": "A"}], "real_training_events": 2000000}
    # When: quota compliance is checked using actual total training counters.
    invalid = invalid_training_quotas((short, complete_run), declaration)
    # Then: the short run is incomplete, while early-best selection from a full run is allowed.
    assert invalid == ("A:0:4096",)
