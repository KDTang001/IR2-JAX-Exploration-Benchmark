# Full-episode B1/B4 resource check: batch-cost-v1

Status reviewed on 2026-11-18: **H100 full-episode acceptance remains blocked**.
The repaired dispatcher launched the first B1 case on all five frozen validation
maps. The two IR2 maps succeeded; all three procedural maps failed graph-state
validation. The coordinator stopped before B4 or IR2-adapted execution: **5/20
executions attempted, 2 successful results, 3 environment errors, 0 B1/B4 pairs**.

A bounded diagnostic on the same H100, tasks and configuration localized each
procedural failure to a local graph reaching the declared 400-node limit. No
capacity, action, budget or frozen task was changed. This is not an OOM result,
a valid batch speedup measurement or completion of the 240-task pilot.

## Original authorization and resource policy, November 17

The user authorized a **new four-hour maximum** for full-episode B1/B4 checks and
explicitly selected **waiting for an idle H100** to make performance comparable.
The window started on **2026-11-17 at 19:45:02 CST**, with an original deadline of
**23:45:02 CST**; waiting and compilation count toward that limit. It closed early
on the dispatch failure. It cannot be renewed by restarting
the script. The previous Phase 6 window remains closed.

The first recorded queue snapshots found three other GPU applications and
65–98% GPU utilization. No measurement worker was launched in those snapshots.
The queue requires no compute applications and utilization below 10% continuously
for 60 seconds before starting. Other users' processes are never stopped. Each
worker also has an independent GNU timeout, at most 3600 seconds per case and no
later than the global deadline minus the termination reserve. A resource error,
runner failure or wall-clock interruption stops subsequent cases.

The immutable server root is
`/home/user/Tz/acceptance/batch-cost-v1-20261117`. Final state is in
`queue-current.json`, `gpu-window.json`, and `gpu-window-closed.json`. No
`mini-current.json` worker-launch record or completed case records were created.
The reused runner's directory label
`resource-cost-mini` does not change this protocol's identity.

## Frozen workload

This check selects ten **existing** result IDs from `resource-pilot-v1`, preserving
the original JSONL task bytes. The planned workload runs each once at B1 and once
at B4, yielding twenty executions in four fresh processes:

| Order | Navigation / communication | Batch limit | Planned groups |
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

## November 17 dispatch failure, diagnosed and repaired November 18

The H100 became idle, and the queue observed 60 seconds of continuous availability.
At 20:32:27 CST the parent dispatcher entered full-episode (`mini`) mode. It first
constructed an unused profiler command, reading `case["bucket"]` before reaching
the full-episode command. These manifest entries intentionally have no top-level
`bucket` or `n_agents` fields. The resulting `KeyError: 'bucket'` occurred before
`Popen`. This was a measurement-harness error, with no evidence about environment
correctness or GPU capacity. The previous wrapper-only smoke check missed the
parent dispatch path.

The fix skips profiler command construction in full-episode and comparison modes.
It changes no environment or model code, manifest tasks, budgets or graph rules.
All 124 original frozen inputs remain intact. The original failed root, helper,
request, logs and closed marker have not been overwritten.

Target-server evidence:

- [Failure summary](evidence/batch-cost-v1-dispatch-failure-summary.json),
  [original traceback](evidence/batch-cost-v1-dispatch-last-run-queue-driver.txt),
  [closed-window record](evidence/batch-cost-v1-dispatch-last-run-gpu-window-closed.json)
  and [redacted queue history](evidence/batch-cost-v1-dispatch-queue-public.jsonl.gz).
- The new B1/B4 dispatch regression failed twice on the original helper with this
  exact exception: [before XML](evidence/batch-cost-v1-dispatch-dispatch-before.xml).
  The corrected helper passed both cases:
  [after XML](evidence/batch-cost-v1-dispatch-dispatch-after.xml).
  This check intercepts the launch boundary and does not execute a GPU workload.
- The real parent dispatcher then launched real **CPU** workers on five tiny
  generated fixtures: B1 groups `1+1+1+1+1`, B4 groups `4+1`; ten executions,
  zero failures, five exact compact-trace pairs. See the
  [CPU result](evidence/batch-cost-v1-dispatch-cpu-dispatch-smoke.json) and
  [executed diagnostic script](protocols/phase6-20261118/cpu_dispatch_smoke.py.txt).
  Only the child platform was explicitly changed to CPU in this diagnostic.
  These fixtures are not the frozen full-size validation episodes and do not
  certify H100 performance.
- The repaired helper still rejects the actual closed November 17 window.
  No GPU measurement was restarted during this repair.

The recovery was initially prepared at
`/home/user/Tz/acceptance/batch-cost-v1-recovery-20261118`, with the same ten task
identities and four-case order, plus updated helper hashes. Its
[request](evidence/batch-cost-v1-recovery-profile-request.json) and
[preflight](evidence/batch-cost-v1-recovery-prepared.json) bind 124 verified inputs.
Those preparation snapshots predate authorization and execution; they are kept
unchanged. The subsequent approved request, window and actual results are archived
separately below. The full 240-task pilot and paper core matrix remain pending.

## Authorized H100 recovery, November 18

The user approved a **new maximum four-hour window**, including waiting and
compilation, starting **10:10:05.508 CST**, with an absolute deadline of
**14:10:05.508 CST**. It retained the 60-second idle-GPU gate. The approved request
pins 125 inputs, including the new authorization record. The source package and
all original task bytes remain unchanged; dispatch uses revision `490252a3`.

The first case, `nf_mixed-b1`, ran for **618.395 seconds** and exited with code 1,
without a timeout. The parent closed at **10:21:24.239 CST**, after 678.731 seconds
including the idle wait. Its outcomes were:

| Validation family | Outcome | Active events in valid trace | Final physical coverage |
|---|---|---:|---:|
| IR2 easy | success | 1,971 | 99.7250% |
| IR2 difficult | success | 1,978 | 99.1656% |
| Procedural rooms | environment_error | unavailable | unavailable |
| Procedural corridor | environment_error | unavailable | unavailable |
| Procedural hybrid | environment_error | unavailable | unavailable |

All three errors report `IR2 graph/history capacity exhausted; increase the
declared capacity before running`. Their error envelopes contain null metrics
and an `active_events=0` placeholder; that placeholder does **not** mean the
simulation performed no actions. Failed tasks are not successes or normal
timeouts and are not used for an aggregate event-throughput estimate.

The recorded host process peak was 1,276,997,632 bytes; JAX allocator peak was
456,595,968 bytes. These are distinct host/device measurements for this one B1
process, not per-environment memory, NVML peak or a certified parallel limit.
The GPU was idle before launch. No allocation error was recorded. The offline
full-cohort analyzer correctly rejected this case with
`RuntimeError: case failed: nf_mixed-b1`; no completed paired audit was produced.

Actual execution evidence is under
[the recovery run](evidence/batch-cost-v1-recovery-run-20261118/run/profile-request.json),
including the [closed window](evidence/batch-cost-v1-recovery-run-20261118/run/gpu-window-closed.json)
and [case record](evidence/batch-cost-v1-recovery-run-20261118/run/resource-cost-mini/nf_mixed-b1.record.json).
The server originals remain at
`/home/user/Tz/acceptance/batch-cost-v1-recovery-20261118`.
The two valid traces passed a separate [partial audit](evidence/batch-cost-v1-recovery-run-20261118/partial-audit.json)
of result/trace checksums, task identities, legal actions, integer coverage
monotonicity, AUC, termination and communication/balance accounting. Its
`complete=false` and `cohort_accepted=false` distinguish this from full paired
acceptance. The [audit script](protocols/phase6-20261118/audit_recovery.py.txt)
imports NumPy only, with no environment execution or GPU initialization.

### First-overflow localization, unchanged configuration

A separate diagnostic, bounded to ten minutes and the same absolute 14:10
deadline, used the remaining authorization. It did not reopen the closed parent
ledger. Its jitted B1 kernel follows the same reset, observation, baseline action
and environment transition, stopping at the **first** graph error. This omits
benchmark scoring and trace recording; its runtime is not evaluation throughput.
The diagnostic completed all three procedural maps on the H100.

| Family | Event counter at first failure | Local node counts, four real robots | Largest global node count | Largest history count | Largest outgoing degree |
|---|---:|---|---:|---:|---:|
| rooms | 1,278 | 339 / **400** / 342 / 343 | 341 | 321 | 29 |
| corridor | 1,563 | 399 / 397 / **400** / 397 | 386 | 392 | 30 |
| hybrid | 1,609 | **400** / 396 / 385 / 396 | 362 | 404 | 30 |

`graph_capacity=graph_padding_size=400`, history capacity 513 and permitted degree
30 were unchanged. Padded robot slots in the raw arrays are not real robots.
Each first overflow coincides with a real robot's local graph reaching 400;
global node counts and history remain below their limits, degree does not exceed
30, and no `path_failed` flag is set. The guard in `ir2_graph.py` rejects
`sum(valid) >= graph_padding_size`. This identifies the active limit; it is not
evidence that additional limits could never be reached with larger graphs.

See [diagnostic counts](evidence/batch-cost-v1-recovery-run-20261118/diagnostic/first-overflow/diagnostic.json),
[budget binding](evidence/batch-cost-v1-recovery-run-20261118/diagnostic/budget-link.json)
and [executed script](protocols/phase6-20261118/capacity_diagnostic.py.txt).
The diagnostic's worker took 158.422 seconds including its first compilation.
It exited within its bound. A final server check verified all 125 pinned inputs
again and found no GPU compute processes; see the
[closure check](evidence/batch-cost-v1-recovery-run-20261118/diagnostic/follow-up-closed.json).
No further GPU work is planned in this window; the unused authorization is not
carried forward. Training and the full pilot remain stopped.

### Next gate: declare capacity policy before rerunning

The frozen C400 workload cannot currently provide ten valid B1/B4 pairs. Original
IR2's worker also rejects graphs at its node-padding limit; therefore silently
discarding these procedural maps, truncating graph nodes, or converting errors
to timeouts would not constitute acceptance. This is a benchmark capacity-policy
decision, not a demonstrated JAX performance regression.

The concrete next proposal is a **separate, versioned extended-capacity preset**:
retain the strict C400 result and original preset, increase only graph storage
and node padding together, and retain history, edge construction, the 30-choice
limit, maps, seeds, budgets and communication. A first candidate is C512; it is
not yet proven sufficient or approved as the pilot replacement. Regenerate
configuration hashes/task identities and a new manifest, disclose that this
extends original capacity, and first check complete-episode feasibility and
pre-limit behavior against C400. Any later B1/B4 comparison must use that same
new preset on both sides. Increasing `max_candidates` is excluded: it also changes
graph construction. No such capacity change or replacement pilot has been run.

## Reproduction after results exist

Use the frozen source, input paths and manifest identities in the request. For a
new execution obtain a new resource budget and output root; do not restart this
window. The final CLI worker command and actual resource samples are recorded in
each case record. Offline audit, without initializing JAX:

```sh
/home/user/anaconda3/envs/ex/bin/python scripts/analyze_batch_cost.py \
  /home/user/Tz/acceptance/batch-cost-v1-recovery-20261118
```

This currently **rejects** the failed case before writing a completed audit.
Only a fully valid cohort can produce the corresponding paired acceptance.
To reproduce the limited valid-trace audit from the public archive, copy the
evidence directory to a new destination, remove only its generated
`partial-audit.json`, and run:

```sh
python research/protocols/phase6-20261118/audit_recovery.py.txt \
  /path/to/evidence-copy research/evidence/batch-cost-v1-tasks.jsonl.gz
```

The helper uses exclusive creation for the audit output and cannot overwrite the
published snapshot. The result must retain `complete=false` and
`cohort_accepted=false`. [Artifact identities](evidence/batch-cost-v1-recovery-run-20261118/artifact-index.json)
and the [review](evidence/batch-cost-v1-recovery-run-20261118/review.json) bind these
compact exports; full server originals remain intact.
