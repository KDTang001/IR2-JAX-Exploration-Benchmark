"""Evaluation boundaries distinguish model execution failures from physical environment errors."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelExecutionError(RuntimeError):
    """A checkpoint/model adapter failed before yielding a valid real episode."""

    detail: str

    def __str__(self) -> str:
        """Retain the original error type/message in the task's result record."""
        return self.detail


def is_oom(error: MemoryError | RuntimeError) -> bool:
    """Recognize actual allocator failures without relabeling other runtime errors as OOM."""
    return isinstance(error, MemoryError) or any(
        marker in str(error).lower() for marker in ("out of memory", "resource_exhausted")
    )
