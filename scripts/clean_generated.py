"""Remove only known rebuildable project outputs; preview by default."""

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class UnsafeCleanupPathError(Exception):
    """Refuse a candidate whose resolved path is outside the project."""

    path: Path

    def __str__(self) -> str:
        """Identify the path that failed the confinement check."""
        return f"Cleanup refused outside project: {self.path}"


def main() -> int:
    """Limit deletion to transfer archives and source/test tool caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "server-project.tar.gz",
        root / "server-source.tar.gz",
        root / "server-update.tar.gz",
        root / "ir2-reference.tar.gz",
        root / ".pytest_cache",
        root / ".ruff_cache",
        root / "artifacts/transfer/server-project.tar.gz",
        root / "artifacts/transfer/server-source.tar.gz",
        root / "artifacts/transfer/server-update.tar.gz",
        root / "artifacts/transfer/ir2-reference.tar.gz",
        root / "artifacts/upstream.tar.gz",
    ]
    for parent in (root / "src", root / "tests"):
        if parent.exists():
            candidates.extend(parent.rglob("__pycache__"))
    transfer = root / "artifacts/sync"
    if transfer.exists():
        candidates.extend(transfer.glob("*.zip"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or resolved == root:
            raise UnsafeCleanupPathError(candidate)
        print(f"{'remove' if args.apply else 'would remove'}: {candidate}")  # noqa: T201
        if args.apply:
            if candidate.is_dir():
                shutil.rmtree(candidate)
            else:
                candidate.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
