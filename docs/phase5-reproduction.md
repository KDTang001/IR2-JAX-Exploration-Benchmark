# Phase 5 wheel reproduction

This workflow installs the benchmark into two independent Linux environments.
The CUDA acceptance completes two unchanged tasks selected from the frozen Phase
5 manifest, each with its actual 512-round limit. The clean CPU acceptance uses
the separately declared microgeometry workload described below, with complete
episodes and the same 512-round limit. It does not run the 2,294,784-episode
formal declaration or train an exploration policy or world model. Map pools and reference weights are external
inputs; the wheel contains Python modules only.

The original diagnostic baseline is commit
`be8949ff8b6d25cee0891c48d128e8fee6cc44d3`. That revision predates
`doctor --runtime-only`. The Phase 5 source commit is
`ade2b705a3caf9e39eeade7ce91df3d4f3945390`; use this full SHA as `IR2_REV` below.
Every source file participates in the preset identity: changing even the doctor changes newly planned task IDs.
Freeze the manifest with that same installed wheel before execution.

These commands run in Bash on the target Linux server. They leave the existing
`ex` packages unchanged; its Python 3.11.0 executable supplies only the base
interpreter. A venv created with `--system-site-packages` is insufficient for this
acceptance. The clean CPU environment has no CUDA or PyTorch packages. The CUDA
environment obtains CUDA user-space libraries from its own locked dependencies;
the host NVIDIA driver remains external.

## Pin and build

The source revision is fixed below. For replay, set `IR2_TASK_MANIFEST` and
`IR2_CORE` to an existing frozen limited workload and its report declaration.
For a fresh asset location, generate them with the portable planning recipe
after installation. The manifest
must contain absolute existing map/checkpoint paths. Retain its original JSONL
bytes. Relocating assets requires explicitly regenerating a manifest and recording
the new file identity; do not silently rewrite an interrupted run's inputs.
The Phase 5 evidence inputs are `phase5/evidence/tasks.jsonl` and
`phase5/evidence/core.json` under the chosen external acceptance root: 72
tasks with the full 512-round limit over three fixed validation maps (512/N=2, 768/N=4,
1024/N=6), eight baselines, Normal/Dropout-0.3/Burst-128:192, plus six declared
probes. Use the finalized manifest, including any separately documented blocker
corrections. The original core file's relative `base_config` must remain
resolvable; its copy below is evidence, not a relocated report input. The clean
CUDA installation proof selects the first two lines (512/N=2, Normal, `nf_off` and
`nf_plan`) without modifying either task. The separate full-episode acceptance
executes the 78-task declaration. Completion of this two-task installed subset
does not establish completion of all 78 tasks.

```bash
set -euo pipefail
export IR2_REV=ade2b705a3caf9e39eeade7ce91df3d4f3945390
export IR2_PUBLIC_REPO=https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark.git
export IR2_AUDIT_BRANCH=audit/benchmark-foundation-2026-11-12
export IR2_REPO="${IR2_REPO:-$HOME/Tz/ir2-benchmark-reproduction-source}"
export IR2_BASE_PYTHON="${IR2_BASE_PYTHON:-$HOME/anaconda3/envs/ex/bin/python}"
if [ ! -d "$IR2_REPO/.git" ]; then
  git clone --filter=blob:none --depth=1 --no-checkout --single-branch --branch "$IR2_AUDIT_BRANCH" \
    "$IR2_PUBLIC_REPO" "$IR2_REPO"
fi
# Main may still be the older baseline; fetch the audit branch explicitly.
git -C "$IR2_REPO" fetch --no-tags --filter=blob:none --depth=1 \
  "$IR2_PUBLIC_REPO" "$IR2_AUDIT_BRANCH"
export IR2_AUDIT_REV
IR2_AUDIT_REV=$(git -C "$IR2_REPO" rev-parse FETCH_HEAD)
git -C "$IR2_REPO" fetch --no-tags --filter=blob:none --depth=1 \
  "$IR2_PUBLIC_REPO" "$IR2_REV"
test "$(git -C "$IR2_REPO" rev-parse "$IR2_REV^{commit}")" = "$IR2_REV"
mkdir -p "$HOME/Tz/acceptance"
export IR2_RUN
IR2_RUN=$(mktemp -d "$HOME/Tz/acceptance/phase5-wheel.XXXXXXXX")
mkdir -p "$IR2_RUN/source" "$IR2_RUN/wheels" "$IR2_RUN/work" "$IR2_RUN/inputs"
printf '%s\n' "$IR2_REV" > "$IR2_RUN/revision.txt"
git -C "$IR2_REPO" archive --format=tar --output="$IR2_RUN/source.tar" \
  "$IR2_REV" pyproject.toml README.md src configs scripts \
  requirement.txt requirements-server.txt
tar -xf "$IR2_RUN/source.tar" -C "$IR2_RUN/source"
printf '%s\n' "$IR2_AUDIT_REV" > "$IR2_RUN/audit-evidence-revision.txt"
sha256sum "$IR2_RUN/source.tar" > "$IR2_RUN/input-sha256.txt"

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
export IR2_REFERENCE="$HOME/Tz/acceptance/phase4-20261114/reference/policy.npz"
printf '%s  %s\n' \
  0a38ab8007bb64f36790d37549e70f7908cc1faba7d17a060ef0f99799193eb8 \
  "$IR2_REFERENCE" | sha256sum --check
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
export IR2_REFERENCE="$IR2_RUN/reference/policy.npz"
```

NPZ ZIP metadata may change the container SHA-256 when repackaged. The extraction
above proves name, dtype, shape, and element equality against the verified PTH;
it does not promise the historical NPZ ZIP hash. The historical tensor digest
`06aae41095fefc4c651e8c9a4d8cf14df30bcd5de18085fcbcd4f75411754ea3` is
retained as prior evidence, with no undocumented digest-framing assumption.
The later independent comparison reads the original Git PTH object and the
existing server NPZ, and matches all 90 names, float32 dtypes, shapes, and raw
tensor-value SHA-256 hashes. Its explicitly framed aggregate digest is
`95b84b9a956fd3a17221a9458a407bc9da32b02349f092ab5836f81c548cd2fb`.
The framing is SHA-256 over `ir2-policy-tensors-v1` followed by a NUL byte, then
for each lexicographically sorted tensor name: an unsigned little-endian
64-bit header length, the ASCII JSON header, an unsigned little-endian 64-bit
raw-byte length, and the C-order raw array bytes. The header is
`{"dtype": array.dtype.str, "name": name, "shape": list(array.shape)}` encoded
with sorted keys, `separators=(",", ":")`, and `ensure_ascii=True`.
The detailed comparison is retained in
`reproduction/reference-tensor-equivalence.json`; no 52 MB PTH copy was needed
on the target server for this comparison.
Choose the checkpoint container before freezing tasks: PTH and repackaged NPZ
have distinct file identities and cannot share an interrupted run's identity.

## Portable planning with the same selected maps

Skip this planning block when replaying an existing absolute-path manifest in
its original asset location. For a fresh clone/location, retrieve the original
selection from the explicitly fetched audit revision. Set `IR2_ASSET_ROOT` to
an existing map tree containing the retained `DungeonMaps/...` paths; obtaining
those immutable assets is separate from installing the wheel. Keep the public
selection evidence unchanged and write a relocated copy outside Git.

```bash
: "${IR2_REFERENCE:?Choose a verified reference NPZ or generate it above}"
export IR2_ASSET_ROOT="${IR2_ASSET_ROOT:-$HOME/Tz/ir2-benchmark}"
git -C "$IR2_REPO" show \
  "$IR2_AUDIT_REV:research/evidence/phase5-representative-selection.json" \
  > "$IR2_RUN/inputs/selection-original.json"
JAX_PLATFORMS=cpu "$IR2_RUN/cpu/bin/python" -I - \
  "$IR2_RUN/inputs/selection-original.json" "$IR2_ASSET_ROOT" \
  "$IR2_RUN/inputs/selection-relocated.json" <<'PY'
import hashlib
import json
import pathlib
import sys
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import load_ir2_map

source = pathlib.Path(sys.argv[1])
selection = json.loads(source.read_text())
old_root = pathlib.Path("/home/user/Tz/ir2-benchmark")
new_root = pathlib.Path(sys.argv[2]).expanduser().resolve()
assert tuple(row["bucket"] for row in selection["maps"]) == (512, 768, 1024)
for row in selection["maps"]:
    relative = pathlib.Path(row["path"]).relative_to(old_root)
    candidate = (new_root / relative).resolve()
    assert candidate.is_relative_to(new_root) and candidate.is_file()
    data = load_ir2_map(candidate)
    assert obstacle_hash(data.occupancy) == row["map_hash"]
    assert scenario_hash(data.occupancy, data.anchor) == row["scenario_hash"]
    row["path"] = str(candidate)
selection["path_rebinding"] = {
    "old_root": str(old_root), "new_root": str(new_root),
    "original_selection_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "rule": "preserve ordered selected map identities; rebase path only",
}
pathlib.Path(sys.argv[3]).write_text(json.dumps(selection, indent=2) + "\n")
PY
# Declaration only: this invocation does not run an episode.
JAX_PLATFORMS=cpu "$IR2_RUN/cpu/bin/python" -I \
  "$IR2_RUN/source/scripts/phase5_full_episodes.py" \
  --selection "$IR2_RUN/inputs/selection-relocated.json" \
  --config "$IR2_RUN/source/configs/benchmark.json" \
  --checkpoint "$IR2_REFERENCE" --output "$IR2_RUN/declared"
export IR2_TASK_MANIFEST="$IR2_RUN/declared/tasks.jsonl"
export IR2_CORE="$IR2_RUN/declared/core.json"
```

Only each selected map's filesystem path changes. The ordered three map hashes,
anchor/scenario hashes, split, source, family, bucket, seeds, configuration, and
method definitions remain fixed. The helper verifies the actual relocated PNGs
before declaration; never choose another map because it performs better.
Scenario/result identities exclude map/checkpoint paths, so path rebinding alone
preserves IDs when weights, config, and package bytes are identical. Repacking a
checkpoint changes its file hash and therefore affected method/result IDs.
The regenerated JSONL and release provenance have a new file hash even when
logical IDs stay equal. Use a new output directory and retain both original and
relocated inputs; do not overwrite public evidence or an interrupted run.

For either an existing manifest or the new declaration, freeze the installed
acceptance subset and record the exact external inputs now:

```bash
: "${IR2_TASK_MANIFEST:?Set the absolute frozen Phase 5 tasks.jsonl path}"
: "${IR2_CORE:?Set the absolute matching report declaration JSON path}"
cp "$IR2_TASK_MANIFEST" "$IR2_RUN/inputs/phase5-tasks.jsonl"
head -n 2 "$IR2_TASK_MANIFEST" > "$IR2_RUN/inputs/tasks.jsonl"
cp "$IR2_CORE" "$IR2_RUN/inputs/core.json"
sha256sum "$IR2_RUN/inputs/"* >> "$IR2_RUN/input-sha256.txt"
```

## Declare the separate clean CPU functional workload

Native 512-bucket CPU execution is costly: a retained 32-event diagnostic
exceeded its 1,800-second bound, and a later four-event nearest-frontier
diagnostic measured about 10.88 seconds of steady execution. Those are cost
diagnostics, not completed native CPU acceptance. Freeze this smaller workload
before observing its outcomes. Its generated corridor, configuration and task
identities are separate from the native Phase 5 matrix; it cannot replace any
of the 78 native tasks. Both methods run complete episodes, ending at the
coverage target or their full 512-round limit, with no event-count truncation.

```bash
JAX_PLATFORMS=cpu "$IR2_RUN/cpu/bin/python" -I - <<'PY'
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np
from PIL import Image
from ir2_benchmark.agents.baselines import make_baseline_method
from ir2_benchmark.config import load_config
from ir2_benchmark.env.reset import from_prepared_map, reset
from ir2_benchmark.evaluation.scenarios import (
    EvaluationMap, make_scenario, tasks, write_task_manifest,
)
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.maps.prepare import load_ir2_map, prepare_map, select_spawns

run = Path(os.environ["IR2_RUN"])
root = run / "cpu-functional"
assert not root.exists()
root.mkdir()
pixels = np.zeros((32, 64), np.uint8)
pixels[10:22, 1:63] = 255
pixels[16, 4] = 208
path = root / "corridor-32x64.png"
Image.fromarray(pixels).save(path)
data = load_ir2_map(path)
config = replace(
    load_config(run / "source/configs/benchmark.json"),
    n_agents=2, agent_bucket=2, sensor_radius=6, max_distance=6, spawn_radius=2,
    graph_nodes_per_axis=16, graph_local_radius=12, graph_global_radius=24,
    graph_capacity=128, graph_padding_size=128, graph_coordinate_scale=64.0,
    graph_utility_radius=6, graph_peer_radius=3, graph_merge_threshold=96,
    graph_unique_radius=3, graph_offshoot_unique_radius=4,
    graph_offshoot_max_radius=8, graph_frontier_protect_radius=12,
    max_rounds=512, coverage_target=0.99,
)
(root / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
selected = EvaluationMap(
    str(path), obstacle_hash(data.occupancy), "validation", "generated",
    "installation-corridor", 64, scenario_hash(data.occupancy, data.anchor),
)
scenario = make_scenario(selected, config, 71, limited=True)
methods = tuple(make_baseline_method("nearest_frontier", mode)
                for mode in ("off", "plan"))
write_task_manifest(root / "tasks.jsonl", tasks((scenario,), methods))
prepared = prepare_map(data, buckets=(64,))
spawns = select_spawns(data, 2, scenario.spawn_seed, config.spawn_radius)
state = jax.block_until_ready(reset(
    from_prepared_map(prepared), config,
    jax.random.PRNGKey(scenario.spawn_seed), spawns=spawns,
))
birth_coverage = float(np.asarray(state.covered).sum() / prepared.component.sum())
assert 0 < birth_coverage < config.coverage_target
(root / "core.json").write_text(json.dumps({
    "scope": "cpu_installation_functional_v1", "limited": True,
    "requested": 2, "round_limit": 512, "training_events": 0,
    "methods": [asdict(method) for method in methods],
}, indent=2) + "\n")
(root / "declaration.json").write_text(json.dumps({
    "declared_before_execution": True, "scenario_seed": 71,
    "map": asdict(selected), "config_hash": scenario.config_hash,
    "spawn_seed": scenario.spawn_seed, "fault_seed": scenario.fault_seed,
    "birth_coverage": birth_coverage, "spawns": spawns.tolist(),
    "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.iterdir() if p.is_file()},
}, indent=2) + "\n")
PY
```

This map has 744 connected free cells, occupancy hash
`ae28141a0ca82ffacdf8c380e10e1e1ecc8aba9ed6f121537e2b73d9c7afffa2`,
anchor `(4,16)`, and birth coverage `0.13978494623655913`. The seed-derived
spawns are `(4,16)` and `(5,15)`. Record nonzero active events, legal actions,
actual outcome, final coverage, and verified event traces after execution.

## Evaluate, report, and demonstrate interruption/resume

Use the native two-task file for CUDA and the separately frozen functional
file for CPU. Configurations,
map identities, spawn/fault seeds, checkpoints, and implementation bindings are
embedded in it. Supply the report declaration explicitly; its default
checkout-relative path is not usable from an installed wheel. A nonzero
evaluation exit with retained execution failures must be recorded and reported,
not reclassified as successful exploration.

Run the following once per backend in a **new** output directory. It sends SIGTERM
only after at least one checksummed result JSON has been atomically published,
records completed IDs and hashes, resumes the unchanged manifest, and then
invokes a second resume. SIGTERM also works when a detached background shell
has inherited an ignored SIGINT disposition; retain the actual signal and
exit status rather than inferring interruption from an attempted signal. If the first run finishes before the signal, record that
the interruption attempt was inconclusive and repeat with a larger declared
workload. Do not claim interruption from an already completed manifest. The
resume completes the requested two-task evaluation; its final report is the
installed evaluation/report proof, so no duplicate full run is necessary.

```bash
export backend=cuda                    # Repeat with backend=cpu.
export JAX_PLATFORMS="$backend"
export IR2_INTERRUPT_OUTPUT="$IR2_RUN/interruption-$backend"
export IR2_ACCEPTANCE_TASKS="$IR2_RUN/inputs/tasks.jsonl"
export IR2_ACCEPTANCE_CORE="$IR2_CORE"
if [ "$backend" = cpu ]; then
  export IR2_ACCEPTANCE_TASKS="$IR2_RUN/cpu-functional/tasks.jsonl"
  export IR2_ACCEPTANCE_CORE="$IR2_RUN/cpu-functional/core.json"
fi
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
           "--task-manifest", os.environ["IR2_ACCEPTANCE_TASKS"], "--output", str(output),
           "--batch-size", "1", "--max-pending-tasks", "16"]
with (output / "interrupted.log").open("w") as stream:
    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 3600
    while process.poll() is None and not tuple((output / "results").glob("*.json")):
        if time.monotonic() >= deadline:
            process.send_signal(signal.SIGTERM)
            process.wait()
            raise SystemExit("No published result before interruption deadline; retain logs")
        time.sleep(0.2)
    if process.poll() is not None:
        raise SystemExit("Run completed before interruption; this attempt is inconclusive")
    process.send_signal(signal.SIGTERM)
    interrupted_status = process.wait()
before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((output / "results").glob("*.json"))}
requested = sum(bool(line.strip()) for line in
                pathlib.Path(os.environ["IR2_ACCEPTANCE_TASKS"]).read_text().splitlines())
assert before and len(before) < requested and interrupted_status != 0
(output / "interruption.json").write_text(json.dumps(
    {"signal": "SIGTERM", "exit": interrupted_status, "published": before}, indent=2) + "\n")
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
  --task-manifest "$IR2_ACCEPTANCE_TASKS" --results "$IR2_INTERRUPT_OUTPUT" \
  --core "$IR2_ACCEPTANCE_CORE" --output "$IR2_INTERRUPT_OUTPUT/report" \
  > "$IR2_INTERRUPT_OUTPUT/report.log" 2>&1
```

Completion means the requested manifest has no pending IDs and every retained
record/trace verifies against its task. Valid recorded environment/model/OOM
failures remain failures even when resumed. Keep report completeness scoped to
the requested declaration; this workflow cannot certify the formal matrix.

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

## Recorded installation evidence, 2026-11-15

The target acceptance root is
`/home/user/Tz/acceptance/phase5-20261114/reproduction`. Its final wheel is
`final-wheels/ir2_benchmark-0.1.0-py3-none-any.whl`, SHA-256
`077cad89fa8825a2aa4aa21fad2ced23de3bbec8a4f02c7f723eeffd60e06f7a`.
`final-source-verification.json` compares all 112 source/build files with
commit `ade2b705a3caf9e39eeade7ce91df3d4f3945390` and the actual live
acceptance snapshot. Each backend's import-origin JSON additionally compares
all 111 installed Python module files with that pinned source. Both environments
have `include-system-site-packages = false`, and their package compatibility
checks pass: 38 CPU packages and 50 CUDA packages.

The actual CPU and NVIDIA H100 doctor outputs both record correct compiled
scan and inference, `optimizer_updates=0`, `optimizer_status="skipped_runtime_only"`,
and null losses. These are functional checks performed while the separate main
acceptance jobs were active; their times are not isolated performance results.

A fresh empty temporary local Git repository fetched the exact public source
pin at `2026-11-14T17:15:30Z`, exit 0, Git tree
`0cdaec36780b374e2c7f741983164dc5fa65ab1b`. The target server's separate
bounded public fetch timed out after 60 seconds, exit 124. The target build
used a transferred Git archive whose 112 file hashes were verified against
that public pin. These are distinct acquisition steps, retained in
`public-source-fetch-local.json`, `logs/public-final-fetch.log`, and
`final-source-verification.json`.

`portable-proof/path-rebinding-proof.json` records an actual relocation check
for all three selected PNGs. Their occupancy and anchor hashes match, and all
78 ordered result/scenario IDs survive regeneration under the installed wheel.
The relocated files are hardlinks to the retained assets; no map is resampled
or substituted.

The earlier `371f42e` wheel was correctly rejected by the original task
implementation bindings after a cosmetic source-format change. Its verification
and rejection logs are retained. The final pin restores the measured snapshot's
exact runtime bytes. Initial detached SIGINT interruption attempts were also
inconclusive because SIGINT was inherited as ignored; complete results from
those attempts do not themselves prove interruption. The accepted signal,
return code, first-result checksum, and both resume summaries must come from
the subsequent controlled attempts.

The controlled CPU attempt is `cpu-evaluation-final-v2`. Both `nf_off` and
`nf_plan` execute 1,024 active events over all 512 rounds, with zero invalid
actions and valid checksummed traces. Both time out with final coverage
`0.5483871102333069`; the birth coverage is `0.13978494623655913`.
These are completed functional episodes, not successful exploration at the
0.99 target. SIGTERM stops the first evaluator with exit `-15` after one
published result; its record hash remains unchanged. The first resume executes
one task and resumes one, and the second resume executes zero and resumes two,
both with zero execution failures. The installed report records
`requested_manifest_complete=true`, `pending_results=0`, and
`formal_complete=false`. Detailed validation is retained in
`cpu-acceptance-validated.json`.
The CPU helper's historical `parent_manifest_sha256` field identifies the native
comparison manifest for context. The CPU microgeometry tasks are independently
declared; their own `manifest_sha256` and declaration are authoritative.

The controlled CUDA attempt is `cuda-evaluation-final-v2`, using the original
first two native tasks without changes. Both reach the 0.99 target from birth
coverage `0.0841974025974026`, with zero invalid actions:

| Method | Active events | Observed rounds | Outcome | Final coverage |
| --- | ---: | ---: | --- | ---: |
| `nf_off` | 681 | 341 | success | 0.9902961254119873 |
| `nf_plan` | 476 | 238 | success | 0.9902961254119873 |

SIGTERM stops the initial CUDA evaluator with exit `-15` after one published
result, at 354.924 seconds. Its SHA-256
`64bad84d481fde60273d40f18c5a0aed28df8b893ec89dd11043e1e303de215b`
is unchanged after both resumes. The first resume executes one and resumes one;
the second executes zero and resumes two, both with zero execution failures.
The installed report has two completed results, zero pending results,
`requested_manifest_complete=true`, `limited=true`, and `formal_complete=false`.

The recorded CPU functional window is `2026-11-14T17:19:59Z` through
`17:26:40Z`; CUDA is `17:25:38Z` through `17:38:23Z`. The CUDA window overlaps
the separate main acceptance workers and is a contended functional run. It
does not support a CPU/GPU performance comparison.

The final evidence verifier exits 0 at `2026-11-14T17:38:24Z` and publishes
`control/reproduction-complete.json`. That marker covers these installed
functional checks only; the separate main 78-task compute has its own gate.
The complete reproduction evidence SHA-256 is
`78263861568722520ce6cbec4006049d2c81f48527e44d755d5dd0866dc0fa09f`.
Compact public copies are indexed in
[the reproduction evidence index](../research/evidence/phase5-reproduction-index.json),
with [the unchanged evidence bytes](../research/evidence/phase5-reproduction.json),
[the completion gate](../research/evidence/phase5-reproduction-gate.json),
[actual installed origins](../research/evidence/phase5-reproduction-cpu-import-origin.json),
[CUDA origins](../research/evidence/phase5-reproduction-cuda-import-origin.json),
[source verification](../research/evidence/phase5-reproduction-source-verification.json),
[installation checks](../research/evidence/phase5-reproduction-installation.json), and
[scoped report fields](../research/evidence/phase5-reproduction-reports.json).
The unchanged [CPU report](../research/evidence/phase5-reproduction-cpu-report.json)
and [CUDA report](../research/evidence/phase5-reproduction-cuda-report.json) are
also retained as compact public artifacts.
Dependencies, wheels, models, and full traces remain external to Git.
