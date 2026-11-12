"""Explicit manifest sources shared by formal training and evaluation commands."""

import json
from pathlib import Path

from ir2_benchmark.maps.sampler import ManifestSource


def manifest_sources(path: Path) -> tuple[ManifestSource, ...]:
    """Read a source registry; relative roots/manifests are anchored to that registry."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        ManifestSource(
            (path.parent / row["root"]).resolve(),
            (path.parent / row["manifest"]).resolve(),
            row["source"],
        )
        for row in rows
    )
