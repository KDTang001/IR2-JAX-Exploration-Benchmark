"""Native JAX implementation of the original IR2 graph pointer policy.

Inputs use trailing shapes [N,6], [K], [], [N], [K], [N,N], with optional
leading batch dimensions. Masks are True for blocked entries. No graph node
pooling or candidate feature projection is substituted for the IR2 attention.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import Array
from jax.typing import DTypeLike


def action_padding_mask(edge_padding_mask: Array) -> Array:
    """Match PolicyNet's slot-zero exclusion and global all-masked fallback.

    The original tests all rows together when given a batch. A vmapped single
    observation therefore intentionally has a per-observation fallback.
    """
    blocked = edge_padding_mask.astype(bool).at[..., 0].set(True)
    return blocked.at[..., 0].set(~jnp.all(blocked))


def _uniform(key: Array, shape: tuple[int, ...], dtype: DTypeLike = jnp.float32) -> Array:
    bound = shape[-1] ** -0.5
    return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def _linear(features: int, fan_in: int, name: str | None = None) -> nn.Dense:
    """Match torch.nn.Linear's uniform kernel and bias initialization."""
    bound = fan_in**-0.5

    def initialize(key: Array, shape: tuple[int, ...], dtype: DTypeLike = jnp.float32) -> Array:
        return jax.random.uniform(key, shape, dtype, minval=-bound, maxval=bound)

    return nn.Dense(
        features,
        kernel_init=initialize,
        bias_init=initialize,
        precision=jax.lax.Precision.HIGHEST,
        name=name,
    )


class _Attention(nn.Module):
    embedding_dim: int
    n_heads: int

    @nn.compact
    def __call__(self, query: Array, memory: Array, blocked: Array) -> Array:
        width = self.embedding_dim // self.n_heads
        shape = (self.n_heads, self.embedding_dim, width)
        q = jnp.einsum(
            "...qd,hdf->...hqf",
            query,
            self.param("w_query", _uniform, shape),
            precision=jax.lax.Precision.HIGHEST,
        )
        k = jnp.einsum(
            "...kd,hdf->...hkf",
            memory,
            self.param("w_key", _uniform, shape),
            precision=jax.lax.Precision.HIGHEST,
        )
        v = jnp.einsum(
            "...kd,hdf->...hkf",
            memory,
            self.param("w_value", _uniform, shape),
            precision=jax.lax.Precision.HIGHEST,
        )
        scores = (
            jnp.einsum("...hqf,...hkf->...hqk", q, k, precision=jax.lax.Precision.HIGHEST)
            * width**-0.5
        )
        weights = jax.nn.softmax(jnp.where(blocked[..., None, :, :], -1e8, scores), -1)
        heads = jnp.einsum("...hqk,...hkf->...qhf", weights, v, precision=jax.lax.Precision.HIGHEST)
        projection = self.param("w_out", _uniform, (self.n_heads, width, self.embedding_dim))
        return jnp.einsum(
            "...qhf,hfd->...qd", heads, projection, precision=jax.lax.Precision.HIGHEST
        )


class _Layer(nn.Module):
    embedding_dim: int
    n_heads: int

    @nn.compact
    def __call__(self, query: Array, blocked: Array, memory: Array | None = None) -> Array:
        norm = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="normalization1")
        normalized = norm(query)
        keys = normalized if memory is None else norm(memory)
        value = query + _Attention(self.embedding_dim, self.n_heads, name="attention")(
            normalized, keys, blocked
        )
        hidden = nn.LayerNorm(epsilon=1e-5, use_fast_variance=False, name="normalization2")(value)
        hidden = _linear(512, self.embedding_dim, name="ff1")(hidden)
        hidden = nn.relu(hidden)
        return value + _linear(self.embedding_dim, 512, name="ff2")(hidden)


class _Pointer(nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, context: Array, candidates: Array) -> Array:
        shape = (self.embedding_dim, self.embedding_dim)
        q = jnp.matmul(
            context, self.param("w_query", _uniform, shape), precision=jax.lax.Precision.HIGHEST
        )
        k = jnp.matmul(
            candidates, self.param("w_key", _uniform, shape), precision=jax.lax.Precision.HIGHEST
        )
        scores = jnp.einsum("...d,...kd->...k", q, k, precision=jax.lax.Precision.HIGHEST)
        return 10 * jnp.tanh(scores * self.embedding_dim**-0.5)


class GraphPolicy(nn.Module):
    """IR2 PolicyNet; feature methods also support separate communication heads."""

    embedding_dim: int = 128
    n_heads: int = 8
    n_layers: int = 6

    def setup(self) -> None:
        """Build the original encoder, decoder and pointer parameter structure."""
        if self.embedding_dim % self.n_heads:
            message = "embedding_dim must be divisible by n_heads"
            raise ValueError(message)
        self.initial_embedding = _linear(self.embedding_dim, 6)
        self.current_embedding = _linear(self.embedding_dim, self.embedding_dim * 2)
        self.encoder = tuple(
            _Layer(self.embedding_dim, self.n_heads, name=f"encoder_{i}")
            for i in range(self.n_layers)
        )
        self.decoder = _Layer(self.embedding_dim, self.n_heads, name="decoder_0")
        self.pointer = _Pointer(self.embedding_dim)

    def encode_graph(self, node_inputs: Array, node_padding_mask: Array, edge_mask: Array) -> Array:
        """Encode every graph node with adjacency and padding attention masks."""
        features = self.initial_embedding(node_inputs)
        blocked = edge_mask.astype(bool) | node_padding_mask[..., None, :].astype(bool)
        for layer in self.encoder:
            features = layer(features, blocked)
        return features

    def encode_candidates(
        self,
        node_features: Array,
        edge_inputs: Array,
        current_index: Array,
        node_padding_mask: Array,
    ) -> tuple[Array, Array]:
        """Return decoder context [...,D] and gathered graph candidates [...,K,D]."""
        candidates = jnp.take_along_axis(node_features, edge_inputs[..., None], axis=-2)
        current = jnp.take_along_axis(node_features, current_index[..., None, None], axis=-2)
        decoded = self.decoder(current, node_padding_mask[..., None, :].astype(bool), node_features)
        context = self.current_embedding(jnp.concatenate((decoded, current), axis=-1))
        return context[..., 0, :], candidates

    def pointer_logits(self, context: Array, candidates: Array, edge_padding_mask: Array) -> Array:
        """Return clipped pointer scores with original finite action masking."""
        return jnp.where(
            action_padding_mask(edge_padding_mask), -1e8, self.pointer(context, candidates)
        )

    def __call__(  # noqa: PLR0913, PLR0917 - Preserve the original six-array policy interface.
        self,
        node_inputs: Array,
        edge_inputs: Array,
        current_index: Array,
        node_padding_mask: Array,
        edge_padding_mask: Array,
        edge_mask: Array,
    ) -> Array:
        """Return action log probabilities from the original graph observation."""
        nodes = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        context, candidates = self.encode_candidates(
            nodes, edge_inputs, current_index, node_padding_mask
        )
        return jax.nn.log_softmax(
            self.pointer_logits(context, candidates, edge_padding_mask), axis=-1
        )


def convert_torch_policy_weights(
    state_dict: Mapping[str, Any], n_layers: int = 6
) -> dict[str, Any]:
    """Convert an original PolicyNet state_dict to Flax variables without importing torch.

    Linear kernels transpose; attention tensors retain the original head-major
    layout. Accepts CPU/GPU torch tensors or NumPy arrays for offline conversion.
    """

    def array(name: str) -> Array:
        value = state_dict[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return jnp.asarray(value, dtype=jnp.float32)

    def linear(prefix: str) -> dict[str, Array]:
        return {"kernel": array(prefix + ".weight").T, "bias": array(prefix + ".bias")}

    def layer(prefix: str) -> dict[str, Any]:
        result = {
            "attention": {
                name: array(prefix + ".multiHeadAttention." + name)
                for name in ("w_query", "w_key", "w_value", "w_out")
            },
            "ff1": linear(prefix + ".feedForward.0"),
            "ff2": linear(prefix + ".feedForward.2"),
        }
        for name in ("normalization1", "normalization2"):
            result[name] = {
                "scale": array(prefix + "." + name + ".normalizer.weight"),
                "bias": array(prefix + "." + name + ".normalizer.bias"),
            }
        return result

    params = {
        "initial_embedding": linear("initial_embedding"),
        "current_embedding": linear("current_embedding"),
        "pointer": {name: array("pointer." + name) for name in ("w_query", "w_key")},
        "decoder_0": layer("decoder.layers.0"),
    }
    for i in range(n_layers):
        params[f"encoder_{i}"] = layer(f"encoder.layers.{i}")
    return {"params": params}
