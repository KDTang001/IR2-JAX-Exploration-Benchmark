"""Explicit, immutable named implementations resolved before device execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib import import_module
from inspect import getsource
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ir2_benchmark.config import HostConfig


_BUILTINS = {
    "reward": ("coverage_delta", "coverage_distance", "sparse_success", "ir2_individual"),
    "signal": ("attenuation", "ideal", "range", "los_range", "ir2"),
    "sending": ("action", "none", "always", "periodic"),
}
_MODULES = {
    "reward": "ir2_benchmark.env.rewards",
    "signal": "ir2_benchmark.communication.propagation",
    "sending": "ir2_benchmark.communication.sending",
}


@dataclass(frozen=True)
class Implementation:
    """Immutable function identity with explicit user-maintained version."""

    function: Callable[..., object]
    version: str
    source_hash: str


_CUSTOM: dict[tuple[str, str], Implementation] = {}


def register_preset(
    kind: str,
    name: str,
    function: Callable[..., object],
    *,
    version: str,
    source_hash: str | None = None,
) -> None:
    """Register before creating configs/kernels; names cannot be replaced in-process.

    Supply a SHA-256 covering dependencies when inspectable function source is
    insufficient (closures, generated functions or external helper modules).
    """
    if kind not in _BUILTINS or not name or not version or not callable(function):
        raise ValueError("expected reward/signal/sending, a name, callable and version")
    if name in _BUILTINS[kind] or (kind, name) in _CUSTOM:
        raise ValueError(f"preset already registered: {kind}/{name}")
    digest = source_hash or sha256(getsource(function).encode()).hexdigest()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("source_hash must be a lowercase SHA-256")
    _CUSTOM[kind, name] = Implementation(function, version, digest)


def preset_names(kind: str) -> tuple[str, ...]:
    """List built-in names and explicitly registered custom implementations."""
    return _BUILTINS[kind] + tuple(name for category, name in _CUSTOM if category == kind)


@lru_cache
def resolve_preset(kind: str, name: str) -> Callable[..., object]:
    """Called by host/tracing code, never by a device callback."""
    if name in _BUILTINS[kind]:
        return getattr(import_module(_MODULES[kind]), name)
    try:
        return _CUSTOM[kind, name].function
    except KeyError as error:
        raise ValueError(f"unregistered preset: {kind}/{name}") from error


@lru_cache
def _builtin_digest() -> str:
    # Bind helper dependencies as well as the entry point, including dirty sources.
    root = Path(__file__).parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def preset_manifest(config: HostConfig) -> dict[str, dict[str, str]]:
    """JSON-safe identity used by training and evaluation artifact bindings."""
    result = {}
    for kind, name in (
        ("reward", config.reward_model),
        ("signal", config.signal_model),
        ("sending", config.send_policy),
    ):
        custom = _CUSTOM.get((kind, name))
        result[kind] = {
            "name": name,
            "version": custom.version if custom else "1",
            "source_hash": custom.source_hash if custom else _builtin_digest(),
        }
    return result


def verify_preset_manifest(config: HostConfig, recorded: dict | None) -> None:
    """Refuse to resume new artifacts with a different preset implementation."""
    if recorded is not None and recorded != preset_manifest(config):
        raise ValueError("preset implementation mismatch; explicit migration/new run required")
