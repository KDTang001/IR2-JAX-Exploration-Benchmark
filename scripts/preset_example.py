"""Run one custom-message event; no model or training job is created."""

import json

import jax
import jax.numpy as jnp

from ir2_benchmark.communication.propagation import LinkResult
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.rewards import RewardResult
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.presets import preset_manifest, register_preset
from ir2_benchmark.types import MapInput, make_action


def byte_reward(context, config):
    """Example method reward combines coverage and an explicit byte penalty."""
    cost = dict(config.custom_parameters)["byte_cost"] * context.spent
    gain = 100 * (context.coverage_after - context.coverage_before)
    terms = jnp.zeros(6, jnp.float32).at[0].set(gain).at[2].set(-cost)
    return RewardResult(gain - cost, terms)


def custom_link(context, config):
    """An example physical rule; only this environment function sees truth."""
    distance = jnp.linalg.norm((context.end - context.start).astype(jnp.float32))
    return LinkResult(distance <= config.communication_range, jnp.float32(0), jnp.bool_(False))


def local_message(observation, intent, config):
    """Two local coordinates serialized as bytes, using participant codec 128.

    This tiny example is restricted to a map smaller than 256 cells per axis;
    production position messages should reuse the standard coordinate codec.
    """
    payload = jnp.zeros(240, jnp.uint8).at[:2].set(observation.position.astype(jnp.uint8))
    return intent._replace(
        send=jnp.bool_(True), codec=jnp.int32(128), payload=payload, payload_length=jnp.int32(2)
    )


def main():
    register_preset("reward", "example_bytes", byte_reward, version="1")
    register_preset("signal", "example_link", custom_link, version="1")
    register_preset("sending", "example_message", local_message, version="1")
    config = HostConfig(
        sensor_radius=1,
        coverage_target=1,
        reward_model="example_bytes",
        signal_model="example_link",
        send_policy="example_message",
        custom_parameters=(("byte_cost", 0.01),),
    )
    grid = jnp.ones((5, 20), jnp.uint8).at[2, 1:19].set(0)
    state = reset(
        MapInput(grid, grid == 0, jnp.array([5, 20]), jnp.array([1, 2])),
        config,
        jax.random.PRNGKey(0),
        jnp.array([[1, 2], [2, 2]]),
    )
    result = jax.jit(lambda s: step_agent(s, make_action(s.positions[0]), config))(state)
    received = observe(result.state, jnp.int32(1), config)
    # A method decodes only actually received inbox slots. Custom bytes do not
    # update the environment's standard peer positions/maps automatically.
    print(
        json.dumps(
            {
                "spent": int(result.info.spent),
                "reward": float(result.reward),
                "received": bool(received.inbox_present[0]),
                "decoded_xy": received.inbox_buffers[0, 16:18].tolist(),
                "implementations": preset_manifest(config),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
