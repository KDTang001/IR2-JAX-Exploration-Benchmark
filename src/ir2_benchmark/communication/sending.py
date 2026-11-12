"""Sending presets see only the method action and receiver-local observation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from ir2_benchmark.presets import resolve_preset

if TYPE_CHECKING:
    from ir2_benchmark.config import HostConfig
    from ir2_benchmark.types import Action, LocalObservation


def prepare_message(observation: LocalObservation, intent: Action, config: HostConfig) -> Action:
    """A custom sending function cannot modify the chosen movement action."""
    result = resolve_preset("sending", config.send_policy)(observation, intent, config)
    result = result._replace(waypoint=intent.waypoint)
    return (
        result._replace(codec=jnp.int32(config.message_codec)) if config.message_codec else result
    )


def validate_builtin_policy(config: HostConfig, learned_communication: bool) -> None:
    """Built-in A/B heads encode standard broadcast actions, not arbitrary packets."""
    if config.exchange_mode != "broadcast" or config.send_policy not in (
        "action",
        "none",
        "always",
        "periodic",
    ):
        raise ValueError(
            "built-in A/B policies require standard broadcast sending; use a custom method adapter"
        )
    if learned_communication and (config.send_policy != "action" or config.message_codec != 0):
        raise ValueError("fixed sending/codec overrides require learned communication disabled")


def action(observation: LocalObservation, intent: Action, config: HostConfig) -> Action:
    """Respect the method's original communication decision."""
    return intent


def none(observation: LocalObservation, intent: Action, config: HostConfig) -> Action:
    """Disable sending without changing the movement action."""
    return intent._replace(send=jnp.bool_(False))


def always(observation: LocalObservation, intent: Action, config: HostConfig) -> Action:
    """Request one packet at every acting robot's turn."""
    return intent._replace(send=jnp.bool_(True))


def periodic(observation: LocalObservation, intent: Action, config: HostConfig) -> Action:
    """Request packets on every configured number of team rounds."""
    return intent._replace(send=observation.round % config.send_period == 0)
