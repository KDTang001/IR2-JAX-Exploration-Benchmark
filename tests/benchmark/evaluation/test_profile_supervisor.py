"""Full-episode dispatch must not require profiler-only geometry fields."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class CompletedChild:
    """Intercept the launch boundary without executing or initializing any device."""

    pid: int = 12345

    def poll(self) -> int:
        """Expose the already-completed intercepted child."""
        return 0

    def wait(self) -> int:
        """Allow the supervisor to finish its ordinary record-writing path."""
        return 0


@pytest.mark.parametrize("batch_size", [1, 4])
def test_full_episode_dispatch_without_profiler_geometry_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int
) -> None:
    script = Path(__file__).resolve().parents[3] / "scripts/phase6_profile_supervisor.py"
    spec = importlib.util.spec_from_file_location("supervisor_under_test", script)
    assert spec is not None
    assert spec.loader is not None
    supervisor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supervisor)
    source = tmp_path / "source-final/src/ir2_benchmark"
    source.mkdir(parents=True)
    (tmp_path / "final-package-inventory.json").write_text("{}")
    manifest, image = tmp_path / "tasks.jsonl", tmp_path / "map.png"
    manifest.write_bytes(b"dispatch-only fixture; never loaded as an episode\n")
    image.write_bytes(b"dispatch-only hash fixture\n")
    case = {
        "case_id": f"fixed-b{batch_size}",
        "batch_size": batch_size,
        "task_manifest": str(manifest),
        "task_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "map": {"path": str(image)},
        "map_file_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
    }
    (tmp_path / "profile-request.json").write_text(
        json.dumps({"cases": [case], "python": sys.executable})
    )
    (tmp_path / "mini-declaration.json").write_text(
        json.dumps({"cases": [case], "case_timeout_seconds": 60})
    )
    ledger = tmp_path / "gpu-window.json"
    ledger.write_text(json.dumps({"deadline_epoch": time.time() + 600}))
    original_ledger = ledger.read_bytes()
    calls = []

    def intercept(argv: Sequence[str], **_options) -> CompletedChild:  # noqa: ANN003
        calls.append(list(argv))
        return CompletedChild()

    monkeypatch.setattr(supervisor, "query", lambda *_args: "")
    monkeypatch.setattr(supervisor, "which", lambda _name: "/usr/bin/timeout")
    monkeypatch.setattr(supervisor.subprocess, "Popen", intercept)
    monkeypatch.setattr(sys, "argv", [str(script), str(tmp_path), "mini"])
    supervisor.main()

    assert len(calls) == 1
    argv = calls[0]
    assert argv[:3] == ["timeout", "--signal=TERM", "--kill-after=5s"]
    assert "--cost-worker" in argv
    assert "evaluate" in argv
    assert argv[argv.index("--batch-size") + 1] == str(batch_size)
    assert argv[argv.index("--task-manifest") + 1] == str(manifest)
    assert "--bucket" not in argv
    assert "--agents" not in argv
    assert ledger.read_bytes() == original_ledger
    finished = json.loads((tmp_path / "mini-finished.json").read_text())
    assert (finished["recorded"], finished["requested"]) == (1, 1)
