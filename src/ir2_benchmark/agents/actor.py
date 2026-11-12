"""Shared recurrent actor and exact conditional motion/communication likelihoods."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from ir2_benchmark.models.encoders import ActorFeatures, ObservationEncoder
from ir2_benchmark.models.ir2_policy import GraphPolicy


class ActorOutput(NamedTuple):
    """Type logits condition on the sampled waypoint's locally generated plan."""

    memory: Array
    waypoint_logits: Array
    send_logit: Array
    type_logits: Array


class IndexAction(NamedTuple):
    """Policy indices; the environment receives the indexed waypoint and wire codec."""

    waypoint: Array
    send: Array
    message_type: Array


class PolicyStatistics(NamedTuple):
    """Log probability and entropy include only executed stochastic decisions."""

    log_prob: Array
    entropy: Array


def initial_memory(batch_shape: tuple[int, ...] = (), hidden: int = 256) -> Array:
    """Each environment/robot owns its own carry, with shared network parameters."""
    return jnp.zeros((*batch_shape, hidden), jnp.float32)


def masked_log_probs(logits: Array, mask: Array) -> Array:
    """Respect legal actions; fall back to slot zero only when none is legal."""
    fallback = ~jnp.any(mask, axis=-1)
    valid = mask.at[..., 0].set(mask[..., 0] | fallback)
    logits = logits.at[..., 0].set(jnp.where(fallback, 0, logits[..., 0]))
    return jax.nn.log_softmax(jnp.where(valid, logits, -jnp.inf), axis=-1)


class Actor(nn.Module):
    """Encode locally, choose a waypoint, then condition communication on its route."""

    hidden: int = 256
    message_types: int = 4
    candidate_attention: bool = True

    def setup(self) -> None:
        """Keep the two stages addressable through Flax's method argument."""
        self.encoder = ObservationEncoder(self.hidden)
        self.graph_policy = GraphPolicy()
        self.recurrent = nn.GRUCell(features=self.hidden)
        self.candidate_projection = nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)
        self.candidate_bias = nn.Dense(1, use_bias=False, precision=jax.lax.Precision.HIGHEST)
        self.communication_projection = nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)
        self.gate = nn.Dense(1, precision=jax.lax.Precision.HIGHEST)
        self.types = nn.Dense(self.message_types, precision=jax.lax.Precision.HIGHEST)
        if self.candidate_attention:
            self.neighbor_keys = nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)
            self.neighbor_values = nn.Dense(self.hidden, precision=jax.lax.Precision.HIGHEST)

    def encode(self, features: ActorFeatures, memory: Array) -> tuple[Array, Array]:
        """Return updated carry and motion logits before the route exists."""
        if features.graph is not None:
            graph = features.graph
            nodes = self.graph_policy.encode_graph(
                graph.node_inputs, graph.node_padding_mask, graph.edge_mask
            )
            context, candidates = self.graph_policy.encode_candidates(
                nodes, graph.edge_inputs, graph.current_index, graph.node_padding_mask
            )
            logits = self.graph_policy.pointer_logits(context, candidates, graph.edge_padding_mask)
            with jax.default_matmul_precision("highest"):
                carry, _ = self.recurrent(memory, self.encoder(features, context))
            return carry, logits
        # Flax 0.10.2 GRUCell has no precision field; scope its internal Dense contractions.
        with jax.default_matmul_precision("highest"):
            carry, hidden = self.recurrent(memory, self.encoder(features))
        candidates = nn.tanh(self.candidate_projection(features.candidates))
        logits = jnp.sum(candidates * hidden[..., None, :], -1) / jnp.sqrt(self.hidden)
        logits += self.candidate_bias(features.candidates)[..., 0]
        if self.candidate_attention:
            mask = features.neighbor_mask[..., None, :]
            neighbors = jnp.where(features.neighbor_mask[..., None], features.neighbor_features, 0)
            keys = nn.tanh(self.neighbor_keys(neighbors))
            values = nn.tanh(self.neighbor_values(neighbors))
            scores = jnp.einsum("...ch,...nh->...cn", candidates, keys) / jnp.sqrt(self.hidden)
            weights = jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1)
            weights = jnp.where(mask, weights, 0)
            context = jnp.einsum("...cn,...nh->...ch", weights, values)
            logits += jnp.sum(candidates * context, axis=-1) / jnp.sqrt(self.hidden)
        return carry, jnp.where(features.candidate_mask.at[..., 0].set(True), logits, -jnp.inf)

    def communication(self, hidden: Array, plan_features: Array) -> tuple[Array, Array]:
        """Only local hidden state and the selected local plan reach message heads."""
        features = nn.silu(
            self.communication_projection(jnp.concatenate((hidden, plan_features), -1))
        )
        return self.gate(features)[..., 0], self.types(features)

    def __call__(self, features: ActorFeatures, memory: Array, plan_features: Array) -> ActorOutput:
        """Recompute the sampled conditional distribution with frozen stored features."""
        carry, logits = self.encode(features, memory)
        gate, types = self.communication(carry, plan_features)
        return ActorOutput(carry, logits, gate, types)


def action_statistics(
    output: ActorOutput,
    action: IndexAction,
    candidate_mask: Array,
    *,
    learned_communication: bool = True,
) -> PolicyStatistics:
    """Exclude fixed communication from likelihood and entropy gradients."""
    waypoint_lp = masked_log_probs(output.waypoint_logits, candidate_mask)
    waypoint_log_prob = jnp.take_along_axis(waypoint_lp, action.waypoint[..., None], -1)[..., 0]
    waypoint_entropy = -jnp.sum(
        jnp.exp(waypoint_lp) * jnp.where(jnp.isfinite(waypoint_lp), waypoint_lp, 0), -1
    )
    if not learned_communication:
        return PolicyStatistics(waypoint_log_prob, waypoint_entropy)
    gate_lp = jnp.where(
        action.send, jax.nn.log_sigmoid(output.send_logit), jax.nn.log_sigmoid(-output.send_logit)
    )
    gate_probability = jax.nn.sigmoid(output.send_logit)
    gate_entropy = -(
        gate_probability * jax.nn.log_sigmoid(output.send_logit)
        + (1 - gate_probability) * jax.nn.log_sigmoid(-output.send_logit)
    )
    type_lp = jax.nn.log_softmax(output.type_logits, -1)
    selected_type_lp = jnp.take_along_axis(type_lp, action.message_type[..., None], -1)[..., 0]
    type_entropy = -jnp.sum(jnp.exp(type_lp) * type_lp, -1)
    return PolicyStatistics(
        waypoint_log_prob + gate_lp + jnp.where(action.send, selected_type_lp, 0),
        waypoint_entropy + gate_entropy + jnp.where(action.send, type_entropy, 0),
    )


def sample_communication(output: ActorOutput, waypoint: Array, key: Array) -> IndexAction:
    """Sample communication after the caller constructs its selected local plan."""
    gate_key, type_key = jax.random.split(key)
    return IndexAction(
        waypoint,
        jax.random.bernoulli(gate_key, jax.nn.sigmoid(output.send_logit)),
        jax.random.categorical(type_key, output.type_logits).astype(jnp.int32),
    )
