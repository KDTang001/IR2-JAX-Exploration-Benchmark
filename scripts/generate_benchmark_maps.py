"""Build the declared pools with one-factor narrow-width OOD and joint hash exclusion."""

import json
import time
from pathlib import Path

from ir2_benchmark.maps.generate import generate_pool

root = Path("artifacts/maps")
records = [json.loads(line) for line in (root / "ir2_manifest.jsonl").read_text().splitlines()]
excluded = {row["canonical_hash"] for row in records if row["canonical_hash"]}
summaries = []
for split, count, seed, shift in (
    ("train", 2000, 410000, "none"),
    ("validation", 100, 510000, "none"),
    ("test", 100, 610000, "none"),
    ("ood", 100, 710000, "narrow"),
):
    output = root / "procedural" / split
    if output.exists():
        detail = f"Refusing to overwrite an existing pool: {output}"
        raise FileExistsError(detail)
    start = time.monotonic()
    print(f"Generating {split}: {count} maps per template, seed={seed}, shift={shift}", flush=True)  # noqa: T201
    manifest = generate_pool(
        output, count, split, seed, shift=shift, exclude_hashes=frozenset(excluded)
    )
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    hashes = {row["canonical_hash"] for row in rows}
    if len(rows) != count * 3 or len(hashes) != len(rows) or hashes & excluded:
        detail = f"Pool count or cross-split hash integrity failure: {split}"
        raise RuntimeError(detail)
    excluded.update(hashes)
    summary = {
        "split": split,
        "maps": len(rows),
        "seed": seed,
        "shift": shift,
        "manifest": str(manifest),
        "wall_seconds": time.monotonic() - start,
    }
    summaries.append(summary)
    print(json.dumps(summary), flush=True)  # noqa: T201
    (root / "procedural_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
print("All declared procedural pools generated and disjoint", flush=True)  # noqa: T201
