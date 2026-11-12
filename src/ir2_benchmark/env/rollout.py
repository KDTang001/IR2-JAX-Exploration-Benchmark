"""Temporal scan of ordered live decisions, independently batched by vmap."""

from collections.abc import Callable
from typing import NamedTuple

from jax import Array, lax

from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.diagnostics import PlanDiagnostics
from ir2_benchmark.types import Action, EnvState, LocalObservation, StepResult, make_action

Policy = Callable[[LocalObservation, HostConfig], Action]


class EventTrace(NamedTuple):
    """Compact privileged execution log; actor inputs remain separate."""

    event: Array
    agent_id: Array
    active: Array
    positions: Array
    goals: Array
    balances: Array
    spent: Array
    delivered: Array
    coverage: Array
    reward: Array
    invalid_action: Array
    terminated: Array
    truncated: Array


class RolloutResult(NamedTuple):
    """A final state and compact event trajectory, excluding frozen events in counts."""

    final_state: EnvState
    trace: EventTrace
    plan_diagnostics: PlanDiagnostics | None = None


def rollout(
    initial: EnvState,
    config: HostConfig,
    steps: int | None = None,
    policy: Policy = nearest_frontier_action,
) -> RolloutResult:
    """Observe and choose each current robot's action inside scan, never once per round."""
    length = config.max_rounds * config.n_agents if steps is None else steps

    def event(state: EnvState, _unused: None) -> tuple[EnvState, EventTrace]:
        def live() -> StepResult:
            observation = observe(state, state.current_agent, config)
            action = policy(observation, config)
            return step_agent(state, action, config, observation=observation)

        result = lax.cond(
            state.terminated | state.truncated,
            lambda: step_agent(state, make_action(state.positions[state.current_agent]), config),
            live,
        )
        info, updated = result.info, result.state
        trace = EventTrace(
            state.event,
            info.agent_id,
            info.active,
            updated.positions,
            updated.goals,
            updated.balances,
            info.spent,
            info.delivered,
            info.coverage_after,
            result.reward,
            info.invalid_action,
            result.terminated,
            result.truncated,
        )
        return updated, trace

    final, trace = lax.scan(event, initial, None, length=length)
    return RolloutResult(final, trace)


def scripted_rollout(
    initial: EnvState, actions: Action, config: HostConfig
) -> tuple[EnvState, StepResult]:
    """Diagnostic small-map scan retaining every state for independent per-event tests."""

    def event(state: EnvState, action: Action) -> tuple[EnvState, StepResult]:
        result = step_agent(state, action, config)
        return result.state, result

    return lax.scan(event, initial, actions)
