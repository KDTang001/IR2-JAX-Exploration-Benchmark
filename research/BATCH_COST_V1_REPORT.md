# Full-episode B1/B4 resource check: batch-cost-v1

Status at the recorded queue snapshot: **prepared and queued; zero GPU cases
executed**. This document does not certify B4 full-episode parity or throughput.
The server queue continues independently of the interactive SSH connection.

## Authorization and resource policy

The user authorized a **new four-hour maximum** for full-episode B1/B4 checks and
explicitly selected **waiting for an idle H100** to make performance comparable.
The window started on **2026-11-17 at 19:45:02 CST** and ends at **23:45:02 CST**;
waiting and compilation count toward that limit. It cannot be renewed by restarting
the script. The previous Phase 6 window remains closed.

The first recorded queue snapshots found three other GPU applications and
65–98% GPU utilization. No measurement worker was launched in those snapshots.
The queue requires no compute applications and utilization below 10% continuously
for 60 seconds before starting. Other users' processes are never stopped. Each
worker also has an independent GNU timeout, at most 3600 seconds per case and no
later than the global deadline minus the termination reserve. A resource error,
runner failure or wall-clock interruption stops subsequent cases.

The immutable server root is
`/home/user/Tz/acceptance/batch-cost-v1-20261117`. Live state is in
`queue-current.json`, `gpu-window.json`, `mini-current.json` (when running),
`resource-cost-mini/*.record.json` (completed cases), and
`gpu-window-closed.json` (closed window). The reused runner's directory label
`resource-cost-mini` does not change this protocol's identity.

## Frozen workload

This check selects ten **existing** result IDs from `resource-pilot-v1`, preserving
the original JSONL task bytes. Each is run once at B1 and once at B4, yielding
twenty executions in four fresh processes:

| Order | Navigation / communication | Batch limit | Actual groups |
|---|---|---:|---|
| 1 | nearest frontier / mixed | 1 | 1+1+1+1+1 |
| 2 | nearest frontier / mixed | 4 | 4+1 |
| 3 | IR2 adapted / mixed | 4 | 4+1 |
| 4 | IR2 adapted / mixed | 1 | 1+1+1+1+1 |

Each case uses the same ordered maps: IR2 easy, IR2 difficult, procedural rooms,
corridor and hybrid. All five are the previously frozen **validation** selections,
with bucket 768; the two IR2 images are 480×640 and the procedural images are
768×768. This is not a 512/768/1024 cross-bucket study.

N=4, scenario seed 0, 8192 bytes per robot, Normal links, 512 rounds, 99% reachable
physical coverage, graph capacity 400 and existing mixed communication are fixed.
No test/OOD data, new seeds, policy training, world-model training or all-240 pilot
execution is authorized by this check. These are two navigation methods under
one communication setting; no additional algorithm is being claimed.

Identities:

- Parent raw manifest SHA256:
  `714c7454e0789d7447b058a835c77449d356010731a1c8a8a883cb5d0dce478e`.
- Combined ten-task raw JSONL SHA256:
  `ee9eed4b64387ccf71fb1abe6f53a8327ede6f514183770a86c7fb6578462bd4`.
- Frozen 111-file environment package SHA256:
  `402942e2eefc04ad33aa65fe8d8e83e68b2411e05e932264548f8cff3d0a7fbd`.
- Source/publication baseline: `976c6aa79048ef2363c88ec321d1ede7768264bf`.
  The package implementation is unchanged from Phase 6's final source.
- Original IR2 reference NPZ SHA256:
  `0a38ab8007bb64f36790d37549e70f7908cc1faba7d17a060ef0f99799193eb8`.
  Its training-data exposure remains unverified.

The [request](evidence/batch-cost-v1-profile-request.json) binds individual source,
map, checkpoint, manifest and measurement helper hashes; the target-server
[input preflight](evidence/batch-cost-v1-input-preflight.json) verified 124 files.
The [declaration](evidence/batch-cost-v1-mini-declaration.json) specifies case order
and manifest hashes. The [compressed task file](evidence/batch-cost-v1-tasks.jsonl.gz)
contains the five nearest-frontier tasks followed by five IR2-adapted tasks.

## Timing and acceptance

The measurement-only wrapper calls the actual cached evaluation kernel via
`lower(...).compile()` and a synchronized compiled invocation. It records input
readiness, lowering/compilation, device execution, complete batch boundaries,
lane loading, summarization/compressed trace writing, and result JSON writing.
These added synchronization points are disclosed instrumentation; the run does
not claim unmodified-runner steady-state throughput. Boundary measurements are
nested and must not be added together indiscriminately.

Each method's B4 four-lane batch and B1 tail are reported separately. The fair
whole-workload comparison includes the same five episodes on each side and the
extra B1 compilation needed by B4's tail. Count actual **active robot events**;
exclude frozen padding from the numerator while retaining its runtime in costs.
Fresh-process host/JAX peaks and sampled own-process NVML memory are distinct.
Any foreign GPU PID or failed telemetry sample contaminates timing comparability.

The offline analyzer verifies checksums, outcome/accounting/coverage metrics and
all compact event-trace fields. Integer and boolean fields, positions, goals,
message bytes, links, fees and termination must match exactly between B1/B4.
Coverage uses decoded integer covered-cell counts; any reported float discrepancy
must additionally be within two ULPs. Reward tolerance is 3e-5 with zero relative
tolerance. Changed actions, covered-cell counts, packets, deliveries or stopping
events fail parity. Existing graph-state validation remains enabled. Compact
traces do not contain every intermediate full-map tensor, so this does not claim
a direct comparison of every such tensor.

## Checks completed before queuing

- Ruff checks passed for the measurement scripts.
- On the target Linux server, a tiny actual CPU workload completed a two-lane
  batch and one-lane tail, recording separate compilation and execution for both.
  Three tasks executed, zero resumed, zero failures. These are fixtures, not
  formal benchmark episodes or H100 acceptance.
- A closed window was rejected; a changed frozen input was rejected before
  creating a new window. See [CPU preflight](evidence/batch-cost-v1-cpu-preflight.json)
  and its [timing evidence](evidence/batch-cost-v1-cpu-preflight-timings.json).
- The [window](evidence/batch-cost-v1-gpu-window.json),
  [resource policy](evidence/batch-cost-v1-resource-policy.json) and
  [initial queue snapshot](evidence/batch-cost-v1-queue-current.json) are archived.
  Unrelated projects' executable paths are redacted in public snapshots; the
  original bytes and their digests are retained on the server.

No full-episode B4 result or speedup is available in this snapshot. The remaining
decision about the full 240-task pilot and the paper core matrix stays open.

## Reproduction after results exist

Use the frozen source, input paths and manifest identities in the request. For a
new execution obtain a new resource budget and output root; do not restart this
window. The final CLI worker command and actual resource samples are recorded in
each case record. Offline audit, without initializing JAX:

```sh
/home/user/anaconda3/envs/ex/bin/python scripts/analyze_batch_cost.py \
  /home/user/Tz/acceptance/batch-cost-v1-20261117
```

The analyzer writes `batch-cost-audit.json`; its `complete` field must remain
false until all four cases and ten trajectory pairs have actually passed.
