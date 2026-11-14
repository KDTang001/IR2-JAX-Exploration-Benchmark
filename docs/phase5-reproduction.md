# Phase 5 wheel reproduction

This workflow installs the benchmark into two independent Linux environments and
runs the complete **requested Phase 5 manifest**, including its actual episode
limits. It does not run the 2,294,784-episode formal declaration or train an
exploration policy or world model. Map pools and reference weights are external
inputs; the wheel contains Python modules only.

The original diagnostic baseline is commit
`be8949ff8b6d25cee0891c48d128e8fee6cc44d3`. That revision predates
`doctor --runtime-only`. Set `IR2_REV` below to the full accepted Phase 5 commit,
after all production changes are committed. Every source file participates in
the preset identity: changing even the doctor changes newly planned task IDs.
Freeze the manifest with that same installed wheel before execution.

These commands run in Bash on the target Linux server. They leave the existing
`ex` packages unchanged; its Python 3.11.0 executable supplies only the base
interpreter. A venv created with `--system-site-packages` is insufficient for this
acceptance. The clean CPU environment has no CUDA or PyTorch packages. The CUDA
environment obtains CUDA user-space libraries from its own locked dependencies;
the host NVIDIA driver remains external.

## Pin and build

Set `IR2_REV`, `IR2_TASK_MANIFEST`, and `IR2_CORE` to the accepted commit, its
frozen limited workload, and the corresponding report declaration. The manifest
must contain absolute existing map/checkpoint paths. Retain its original JSONL
bytes. Relocating assets requires explicitly regenerating a manifest and recording
the new file identity; do not silently rewrite an interrupted run's inputs.
The Phase 5 evidence inputs are `phase5/evidence/tasks.jsonl` and
`phase5/evidence/core.json` under the chosen external acceptance root: 72
full-512-round tasks over three fixed validation maps (512/N=2, 768/N=4,
1024/N=6), eight baselines, Normal/Dropout-0.3/Burst-128:192, plus six declared
probes. Use the finalized manifest, including any separately documented blocker
corrections. The original core file's relative `base_config` must remain
resolvable; its copy below is evidence, not a relocated report input.

```bash
set -euo pipefail
: "${IR2_REV:?Set the full accepted Phase 5 commit SHA}"
: "${IR2_TASK_MANIFEST:?Set the absolute frozen Phase 5 tasks.jsonl path}"
: "${IR2_CORE:?Set the absolute matching report declaration JSON path}"
export IR2_REPO="${IR2_REPO:-$HOME/Tz/ir2-benchmark}"
export IR2_BASE_PYTHON="${IR2_BASE_PYTHON:-$HOME/anaconda3/envs/ex/bin/python}"
test "$(git -C "$IR2_REPO" rev-parse "$IR2_REV^{commit}")" = "$IR2_REV"
test -f "$IR2_TASK_MANIFEST"
test -f "$IR2_CORE"
mkdir -p "$HOME/Tz/acceptance"
export IR2_RUN
IR2_RUN=$(mktemp -d "$HOME/Tz/acceptance/phase5-wheel.XXXXXXXX")
mkdir -p "$IR2_RUN/source" "$IR2_RUN/wheels" "$IR2_RUN/work" "$IR2_RUN/inputs"
printf '%s\n' "$IR2_REV" > "$IR2_RUN/revision.txt"
git -C "$IR2_REPO" archive --format=tar --output="$IR2_RUN/source.tar" \
  "$IR2_REV" pyproject.toml README.md src configs scripts \
  requirement.txt requirements-server.txt
tar -xf "$IR2_RUN/source.tar" -C "$IR2_RUN/source"
cp "$IR2_TASK_MANIFEST" "$IR2_RUN/inputs/tasks.jsonl"
cp "$IR2_CORE" "$IR2_RUN/inputs/core.json"
sha256sum "$IR2_RUN/source.tar" "$IR2_RUN/inputs/"* > "$IR2_RUN/input-sha256.txt"

# A private uv executable and cache; no installation into ex or shell profile edit.
curl --fail --location https://astral.sh/uv/0.9.9/install.sh \
  --output "$IR2_RUN/install-uv.sh"
UV_UNMANAGED_INSTALL="$IR2_RUN/tools" sh "$IR2_RUN/install-uv.sh"
export UV_CACHE_DIR="$IR2_RUN/uv-cache" UV_PYTHON_DOWNLOADS=never
export PATH="$IR2_RUN/tools:/usr/local/bin:/usr/bin:/bin"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH LD_PRELOAD CUDA_HOME CUDA_PATH
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
uv --version > "$IR2_RUN/uv-version.txt"
"$IR2_BASE_PYTHON" -I -c 'import sys; print(sys.version)' > "$IR2_RUN/base-python.txt"
cd "$IR2_RUN/work"
cat > "$IR2_RUN/build.in" <<'EOF'
hatchling==1.25.0
editables==0.5
EOF
uv pip compile --python "$IR2_BASE_PYTHON" --generate-hashes \
  "$IR2_RUN/build.in" --output-file "$IR2_RUN/build.lock"
uv build --wheel --no-sources --python "$IR2_BASE_PYTHON" \
  --build-constraint "$IR2_RUN/build.lock" \
  --out-dir "$IR2_RUN/wheels" "$IR2_RUN/source" \
  > "$IR2_RUN/build.log" 2>&1
export IR2_WHEEL
IR2_WHEEL=$(find "$IR2_RUN/wheels" -maxdepth 1 -name '*.whl' -print)
test -f "$IR2_WHEEL"
sha256sum "$IR2_WHEEL" > "$IR2_RUN/wheel-sha256.txt"
```

`uv build` uses the committed Hatchling configuration and an isolated build
environment. Its package-directory argument and explicit Python selection work
with an existing interpreter; see the official [build guide](https://docs.astral.sh/uv/guides/package/)
and [environment guide](https://docs.astral.sh/uv/pip/environments/).

## Resolve and install independent CPU/CUDA environments

The direct versions match the established JAX generation. JAX 0.4.35 explicitly
supports jaxlib 0.4.34 and its CUDA extra requires that jaxlib version; see its
[published dependency metadata](https://pypi.org/pypi/jax/0.4.35/json).
Preserve the generated complete hash locks alongside the wheel. A fresh
resolution on a later date may choose different transitive dependencies; replay
the retained locks rather than resolving again to reproduce an accepted run.

```bash
cat > "$IR2_RUN/cpu.in" <<'EOF'
jax==0.4.35
jaxlib==0.4.34
flax==0.10.2
optax==0.2.4
chex==0.1.87
orbax-checkpoint==0.10.2
numpy==2.0.2
pillow==13.0.0
matplotlib==3.9.4
scipy==1.13.1
EOF
cp "$IR2_RUN/cpu.in" "$IR2_RUN/cuda.in"
cat >> "$IR2_RUN/cuda.in" <<'EOF'
jax[cuda12]==0.4.35
jax-cuda12-plugin==0.4.35
jax-cuda12-pjrt==0.4.35
nvidia-cublas-cu12==12.4.5.8
nvidia-cuda-cupti-cu12==12.4.127
nvidia-cuda-nvcc-cu12==12.4.131
nvidia-cuda-runtime-cu12==12.4.127
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.2.1.3
nvidia-cusolver-cu12==11.6.1.9
nvidia-cusparse-cu12==12.3.1.170
nvidia-nccl-cu12==2.21.5
nvidia-nvjitlink-cu12==12.4.127
EOF
for backend in cpu cuda; do
  uv pip compile --python "$IR2_BASE_PYTHON" --generate-hashes \
    --only-binary=:all: "$IR2_RUN/$backend.in" \
    --output-file "$IR2_RUN/$backend.lock" \
    > "$IR2_RUN/$backend-resolve.log" 2>&1
  uv venv --python "$IR2_BASE_PYTHON" "$IR2_RUN/$backend"
  uv pip sync --python "$IR2_RUN/$backend/bin/python" --require-hashes \
    --only-binary=:all: "$IR2_RUN/$backend.lock" \
    > "$IR2_RUN/$backend-install.log" 2>&1
  uv pip install --python "$IR2_RUN/$backend/bin/python" --no-deps "$IR2_WHEEL" \
    >> "$IR2_RUN/$backend-install.log" 2>&1
  uv pip check --python "$IR2_RUN/$backend/bin/python" \
    > "$IR2_RUN/$backend-check.txt" 2>&1
  uv pip freeze --python "$IR2_RUN/$backend/bin/python" \
    > "$IR2_RUN/$backend-freeze.txt"
done
sha256sum "$IR2_RUN/"*.lock >> "$IR2_RUN/input-sha256.txt"
```

Keep any scheduler-assigned `CUDA_VISIBLE_DEVICES`. Record the host driver with
`nvidia-smi`; GPU memory and compiler caches are distinct from Python dependency
isolation. Do not install `requirement.txt` into either clean environment: it
also includes legacy Torch/Ray tools that the NPZ inference path does not need.

## Verify real import origins and run doctor

Run from the new work directory, never the checkout. `-I` excludes the current
directory, user-site packages, and Python environment-variable import overrides.
The checks below require every named runtime module to live inside its selected
venv and require the installed package bytes to match the committed source tree.

```bash
for backend in cpu cuda; do
  export JAX_PLATFORMS="$backend"
  "$IR2_RUN/$backend/bin/python" -I - "$IR2_RUN/source/src/ir2_benchmark" \
    > "$IR2_RUN/$backend-import-origin.json" <<'PY'
import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import site
import sys

prefix = pathlib.Path(sys.prefix).resolve()
assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE
origins = {}
for name in ("ir2_benchmark", "jax", "jaxlib", "flax", "optax", "numpy", "PIL", "matplotlib"):
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    assert path.is_relative_to(prefix), (name, str(path))
    origins[name] = str(path)
package = pathlib.Path(origins["ir2_benchmark"]).parent
trees = []
for root in (pathlib.Path(sys.argv[1]), package):
    trees.append({p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(root.rglob("*.py"))})
assert trees[0] == trees[1]
direct = json.loads(importlib.metadata.distribution("ir2-benchmark").read_text("direct_url.json"))
assert not direct.get("dir_info", {}).get("editable", False)
print(json.dumps({"executable": sys.executable, "prefix": sys.prefix,
                  "base_prefix": sys.base_prefix, "sys_path": sys.path,
                  "origins": origins, "direct_url": direct,
                  "package_source_hashes": trees[1]}, indent=2))
PY
  gpu=()
  if [ "$backend" = cuda ]; then gpu=(--require-gpu); fi
  "$IR2_RUN/$backend/bin/python" -I -m ir2_benchmark.cli doctor \
    --runtime-only "${gpu[@]}" --output "$IR2_RUN/$backend-doctor.json" \
    > "$IR2_RUN/$backend-doctor.log" 2>&1
done
```

Success requires `scan_correct=true`, `inference_correct=true`,
`optimizer_status="skipped_runtime_only"`, `optimizer_updates=0`, and both losses
`null`. CUDA also requires an actual GPU device. This checks compiled
`jit`/`vmap`/`scan` and Flax inference. Optax is imported/versioned but its update
path is explicitly skipped. Omitting `--runtime-only` preserves the legacy 40
toy regression updates; do not use that mode for this no-update acceptance.
The recorded scan timings are not exploration throughput.

## Evaluate, report, and demonstrate interruption/resume

Use the same absolute frozen task file for both installations. Configurations,
map identities, spawn/fault seeds, checkpoints, and implementation bindings are
embedded in it. Supply the report declaration explicitly; its default
checkout-relative path is not usable from an installed wheel. A nonzero
evaluation exit with retained execution failures must be recorded and reported,
not reclassified as successful exploration.

```bash
for backend in cpu cuda; do
  export JAX_PLATFORMS="$backend"
  output="$IR2_RUN/evaluation-$backend"
  status=0
  "$IR2_RUN/$backend/bin/python" -I -m ir2_benchmark.cli evaluate \
    --task-manifest "$IR2_RUN/inputs/tasks.jsonl" --output "$output" \
    --batch-size 1 --max-pending-tasks 16 \
    > "$IR2_RUN/$backend-evaluate.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "$IR2_RUN/$backend-evaluate.exit"
  "$IR2_RUN/$backend/bin/python" -I -m ir2_benchmark.cli report \
    --task-manifest "$IR2_RUN/inputs/tasks.jsonl" --results "$output" \
    --core "$IR2_CORE" --output "$output/report" \
    > "$IR2_RUN/$backend-report.log" 2>&1
done
```

Run the following once per backend in a **new** output directory. It sends SIGINT
only after at least one checksummed result JSON has been atomically published,
records completed IDs and hashes, resumes the unchanged manifest, and then
invokes a second resume. If the first run finishes before the signal, record that
the interruption attempt was inconclusive and repeat with a larger declared
workload. Do not claim interruption from an already completed manifest.

```bash
export backend=cuda                    # Repeat with backend=cpu.
export JAX_PLATFORMS="$backend"
export IR2_INTERRUPT_OUTPUT="$IR2_RUN/interruption-$backend"
"$IR2_RUN/$backend/bin/python" -I - <<'PY'
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

root = pathlib.Path(os.environ["IR2_RUN"])
output = pathlib.Path(os.environ["IR2_INTERRUPT_OUTPUT"])
assert not output.exists()
output.mkdir()
command = [sys.executable, "-I", "-m", "ir2_benchmark.cli", "evaluate",
           "--task-manifest", str(root / "inputs/tasks.jsonl"), "--output", str(output),
           "--batch-size", "1", "--max-pending-tasks", "16"]
with (output / "interrupted.log").open("w") as stream:
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 3600
    while process.poll() is None and not tuple((output / "results").glob("*.json")):
        if time.monotonic() >= deadline:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise SystemExit("No published result before interruption deadline; retain logs")
        time.sleep(0.2)
    if process.poll() is not None:
        raise SystemExit("Run completed before interruption; this attempt is inconclusive")
    process.send_signal(signal.SIGINT)
    interrupted_status = process.wait()
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((output / "results").glob("*.json"))}
requested = sum(bool(line.strip()) for line in (root / "inputs/tasks.jsonl").read_text().splitlines())
assert before and len(before) < requested and interrupted_status != 0
(output / "interruption.json").write_text(json.dumps(
    {"signal": "SIGINT", "exit": interrupted_status, "published": before}, indent=2) + "\n")
for name in ("resumed", "second-resume"):
    with (output / f"{name}.log").open("w") as stream, \
         (output / f"{name}.json").open("w") as summary_file:
        result = subprocess.run(command, stdout=summary_file, stderr=stream)
    (output / f"{name}.exit").write_text(str(result.returncode) + "\n")
    summary = json.loads((output / f"{name}.json").read_text())
    if name == "resumed":
        assert summary["resumed"] >= len(before)
    else:
        assert summary["executed"] == 0 and summary["resumed"] == summary["requested"]
    for name_before, digest in before.items():
        assert hashlib.sha256((output / "results" / name_before).read_bytes()).hexdigest() == digest
PY
"$IR2_RUN/$backend/bin/python" -I -m ir2_benchmark.cli report \
  --task-manifest "$IR2_RUN/inputs/tasks.jsonl" --results "$IR2_INTERRUPT_OUTPUT" \
  --core "$IR2_CORE" --output "$IR2_INTERRUPT_OUTPUT/report" \
  > "$IR2_INTERRUPT_OUTPUT/report.log" 2>&1
```

Completion means the requested manifest has no pending IDs and every retained
record/trace verifies against its task. Valid recorded environment/model/OOM
failures remain failures even when resumed. Keep report completeness scoped to
the requested declaration; this workflow cannot certify the formal matrix.

## Reference checkpoint acquisition

The reference is upstream commit
`1ef6d224ae18503f4763c3cec20e05a737f95783`, path
`model/stage2/checkpoint.pth`, Git blob
`3e99f032d497909507785aae06f6c092748b8e1c`, 52,176,462 bytes, SHA-256
`cd72ac7dbf164b8be089394154794416fc969a47d79c1072c437f6f2aa363cf3`.
It is an external pretrained reference; its upstream map exposure is not
certified. The original [checkpoint source](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/model/stage2/checkpoint.pth)
and [Phase 4 provenance](../research/evidence/phase4-policy-provenance-20261114.json)
provide the existing evidence.

The already verified server asset is
`$HOME/Tz/acceptance/phase4-20261114/reference/policy.npz`, SHA-256
`0a38ab8007bb64f36790d37549e70f7908cc1faba7d17a060ef0f99799193eb8`.
Use that exact file to reproduce its file-based checkpoint identity. To obtain
the original without a large raw-GitHub transfer, extract the verified Git object
from an existing original clone. If absent, fetch that upstream commit first:

```bash
printf '%s  %s\n' \
  0a38ab8007bb64f36790d37549e70f7908cc1faba7d17a060ef0f99799193eb8 \
  "$HOME/Tz/acceptance/phase4-20261114/reference/policy.npz" | sha256sum --check
```

When the historical NPZ is unavailable, regenerate from the verified original
before planning tasks:

```bash
upstream=1ef6d224ae18503f4763c3cec20e05a737f95783
if ! git -C "$IR2_REPO" cat-file -e "$upstream:model/stage2/checkpoint.pth"; then
  git -C "$IR2_REPO" fetch \
    https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration.git "$upstream"
fi
test "$(git -C "$IR2_REPO" rev-parse "$upstream:model/stage2/checkpoint.pth")" \
  = 3e99f032d497909507785aae06f6c092748b8e1c
mkdir -p "$IR2_RUN/reference"
git -C "$IR2_REPO" cat-file blob 3e99f032d497909507785aae06f6c092748b8e1c \
  > "$IR2_RUN/reference/checkpoint.pth"
# Conversion reads the pre-existing Torch in ex; no package installation or updates.
"$IR2_BASE_PYTHON" -I - "$IR2_RUN/reference" <<'PY'
import hashlib
import pathlib
import sys
import numpy as np
import torch

root = pathlib.Path(sys.argv[1])
source = root / "checkpoint.pth"
assert source.stat().st_size == 52176462
assert hashlib.sha256(source.read_bytes()).hexdigest() == \
    "cd72ac7dbf164b8be089394154794416fc969a47d79c1072c437f6f2aa363cf3"
weights = torch.load(source, map_location="cpu", weights_only=True)["policy_model"]
arrays = {name: tensor.detach().cpu().numpy() for name, tensor in weights.items()}
assert len(arrays) == 90 and all(value.dtype == np.float32 for value in arrays.values())
np.savez(root / "policy.npz", **arrays)
with np.load(root / "policy.npz", allow_pickle=False) as restored:
    assert set(restored.files) == set(arrays)
    for name, expected in arrays.items():
        assert restored[name].dtype == expected.dtype
        np.testing.assert_array_equal(restored[name], expected)
print(hashlib.sha256((root / "policy.npz").read_bytes()).hexdigest())
PY
```

NPZ ZIP metadata may change the container SHA-256 when repackaged. The extraction
above proves name, dtype, shape, and element equality against the verified PTH;
it does not promise the historical NPZ ZIP hash. The historical tensor digest
`06aae41095fefc4c651e8c9a4d8cf14df30bcd5de18085fcbcd4f75411754ea3` is
retained as prior evidence, with no undocumented digest-framing assumption.
Choose the checkpoint container before freezing tasks: PTH and repackaged NPZ
have distinct file identities and cannot share an interrupted run's identity.

## Evidence needed for an accepted reproduction

Retain the pinned revision, source/wheel hashes, build log and lock, both full
dependency locks/freezes/checks, actual import-origin JSON, doctor outputs,
external task/core hashes, evaluation exit statuses, reports/release manifests,
and interruption/resume logs with unchanged completed-result hashes. Preserve
the same externally hashed maps and checkpoint for replay.

The current release writer derives a Git root from the installed package path
and has a legacy `runtime_lock` label referring to `ex`. In an installed wheel
those fields do not identify this isolated installation. The adjacent revision,
wheel hash, import-origin evidence, and complete locks above are authoritative;
its package source hashes still describe the actual installed code. This is a
documented metadata limitation, not evidence that a clean installation was run.
Report observed CPU/CUDA results separately from this recipe; commands alone do
not establish successful reproduction.
