"""Recover complete map members from a pinned archive without extracting GIFs."""

import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Final

PREFIX: Final = "IR2-Multi-Robot-RL-Exploration-1ef6d224ae18503f4763c3cec20e05a737f95783"


def main() -> int:
    """Extract maps only; finalization separately verifies every file digest."""
    project = Path(__file__).resolve().parents[1]
    destination = (project / "artifacts/source-cache").resolve()
    if not destination.is_relative_to(project):
        return 2
    entered_maps = False
    with tarfile.open(project / "artifacts/upstream.tar.gz", mode="r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            is_map = parts[:2] == (PREFIX, "DungeonMaps")
            if not is_map:
                if entered_maps:
                    return 0
                continue
            entered_maps = True
            target = destination.joinpath(*parts[1:]).resolve()
            if not target.is_relative_to(destination):
                return 2
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                return 2
            content = archive.extractfile(member)
            if content is None:
                return 2
            target.parent.mkdir(parents=True, exist_ok=True)
            with content, target.open("wb") as output:
                shutil.copyfileobj(content, output)
    return 0 if entered_maps else 2


if __name__ == "__main__":
    raise SystemExit(main())
