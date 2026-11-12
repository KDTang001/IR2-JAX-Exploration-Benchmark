"""Verify the pinned upstream checkout, preserve inputs, then remove its cache."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Final

EXPECTED_COMMIT: Final = "1ef6d224ae18503f4763c3cec20e05a737f95783"
EXPECTED_MAP_COUNT: Final = 17299
EXPECTED_MAP_DIGEST: Final = "7734ef412da9903a83e42c9b54b9aa634afb51250e1be17d36be77d5da15c458"


def main() -> int:
    """Finalize only the temporary checkout created for this project."""
    root = Path(__file__).resolve().parents[1]
    cache = root / "artifacts" / "source-cache"
    target = root / "DungeonMaps"
    if cache.is_symlink() or not cache.resolve().is_relative_to(root):
        return 2
    commit = subprocess.run(  # noqa: S603
        ["git", "-C", str(cache), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    maps = cache / "DungeonMaps"
    files = sorted(path for path in maps.rglob("*") if path.is_file())
    count = len(files)
    fingerprint = hashlib.sha256()
    for path in files:
        fingerprint.update(path.relative_to(maps).as_posix().encode())
        fingerprint.update(hashlib.sha256(path.read_bytes()).digest())
    digest = fingerprint.hexdigest()
    valid = commit == EXPECTED_COMMIT and count == EXPECTED_MAP_COUNT
    valid = valid and digest == EXPECTED_MAP_DIGEST
    if not valid or target.exists():
        print(f"Reference finalization refused: commit={commit}, count={count}")  # noqa: T201
        return 2
    reference = root / "reference" / "ir2"
    reference.mkdir(parents=True, exist_ok=True)
    for source in (*cache.glob("*.py"), cache / "LICENSE.txt", cache / "README.md"):
        shutil.copy2(source, reference / source.name)
    maps.rename(target)
    provenance = {
        "repository": "https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration.git",
        "commit": commit,
        "map_count": count,
        "map_tree_sha256": digest,
        "maps": "DungeonMaps",
        "reference": "reference/ir2",
    }
    (root / "artifacts/setup/source-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8",
    )
    shutil.rmtree(cache)
    print(json.dumps(provenance, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
