"""Atomic, configuration-bound Flax byte checkpoints, shared by A and B."""

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile


@dataclass(frozen=True, slots=True)
class CheckpointError(ValueError):
    """Restoring a checkpoint with different experiment semantics is prohibited."""

    expected: str
    actual: str

    def __str__(self) -> str:
        """Report both configuration identities for a concrete recovery failure."""
        return f"checkpoint configuration mismatch: expected {self.expected}, found {self.actual}"


def save_checkpoint(path: str | Path, payload: bytes, config_hash: str) -> None:
    """Atomically replace one file; payload is flax.serialization.to_bytes(state)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    identity = config_hash.encode("ascii")
    with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(struct.pack("<I", len(identity)))
        stream.write(identity)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)


def load_checkpoint(path: str | Path, expected_hash: str) -> bytes:
    """Return bytes for flax.serialization.from_bytes(template, payload)."""
    data = Path(path).read_bytes()
    size = struct.unpack("<I", data[:4])[0]
    actual = data[4 : 4 + size].decode("ascii")
    if actual != expected_hash:
        raise CheckpointError(expected_hash, actual)
    return data[4 + size :]
