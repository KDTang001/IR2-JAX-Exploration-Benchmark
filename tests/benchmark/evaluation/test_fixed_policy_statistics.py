"""Scenario repeats cannot supply independent map uncertainty for a fixed policy."""

import pytest

from ir2_benchmark.evaluation.statistics import ReportRow, summarize_method


@pytest.mark.parametrize(
    ("route", "strata", "has_interval"),
    [
        ("baseline", (("ir2", "rooms", 1),), False),
        ("baseline", (("ir2", "rooms", 2),), True),
        ("baseline", (("ir2", "rooms", 2), ("procedural", "rooms", 1)), False),
        ("baseline", (("ir2", "rooms", 2), ("ir2", "caves", 1)), False),
        ("A", (("ir2", "rooms", 1),), True),
        ("B", (("ir2", "rooms", 1),), True),
    ],
)
def test_fixed_policy_interval_requires_two_maps_in_every_valid_stratum(
    route: str, strata: tuple[tuple[str, str, int], ...], *, has_interval: bool
) -> None:
    # Given: twenty scenario seeds per map, sharing one actual policy/training seed.
    training_seed = -1 if route == "baseline" else 0
    rows = tuple(
        ReportRow(
            f"{source}-{family}-{map_index}-{scenario_seed}",
            "method",
            training_seed,
            source,
            family,
            f"map-{map_index}",
            auc=float(map_index),
            route=route,
        )
        for source, family, maps in strata
        for map_index in range(maps)
        for scenario_seed in range(20)
    )
    # When: uncertainty is summarized after averaging repeated scenarios within each map.
    summary = summarize_method("method", rows, (training_seed,), repetitions=100)
    # Then: baseline intervals require two independent maps; A/B semantics stay unchanged.
    assert (summary.ci_low is not None, summary.ci_high is not None) == (
        has_interval,
        has_interval,
    )
