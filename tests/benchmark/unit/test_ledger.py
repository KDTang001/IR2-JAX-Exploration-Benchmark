"""Actual wire length determines a single irreversible attempt charge."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ir2_benchmark.communication.ledger import charge_attempt


@pytest.mark.parametrize(
    ("balance", "requested", "valid", "expected"),
    [
        (50, True, True, (True, 5, 45)),
        (44, True, True, (False, 44, 0)),
        (45, True, True, (True, 0, 45)),
        (50, False, True, (False, 50, 0)),
        (50, True, False, (False, 50, 0)),
    ],
)
def test_attempt_when_budget_and_request_gate_transmission(
    *,
    balance: int,
    requested: bool,
    valid: bool,
    expected: tuple[bool, int, int],
) -> None:
    # Given: a PLAN packet costs 16 + 4 + 4 + 1 + 5*4 bytes, independently of receivers.
    cost = 16 + 4 + 4 + 1 + 5 * 4
    # When: the ledger considers exactly one attempted broadcast.
    eager = charge_attempt(jnp.asarray(balance), cost, requested, valid)
    compiled = jax.jit(charge_attempt)(balance, cost, requested, valid)
    # Then: eager and jit agree on eligibility, balance and charged bytes.
    assert tuple(map(int, eager)) == expected
    assert tuple(map(int, compiled)) == expected


def test_attempt_when_vectorized_budgets_differ() -> None:
    # Given: separate accounts with different message lengths and budgets.
    balances = jnp.asarray([50, 44, 100])
    costs = jnp.asarray([45, 45, 20])
    # When: compiled vmap updates all accounts independently.
    result = jax.jit(jax.vmap(charge_attempt))(
        balances, costs, jnp.ones(3, bool), jnp.ones(3, bool)
    )
    # Then: no padding or other robot's account enters the charge.
    np.testing.assert_array_equal(result.balance, [5, 44, 80])
    np.testing.assert_array_equal(result.spent, [45, 0, 20])
