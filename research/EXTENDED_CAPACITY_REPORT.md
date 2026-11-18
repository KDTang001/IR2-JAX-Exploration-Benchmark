# C1024 complete-episode and B1/B4 acceptance

Status on 2026-11-18: **PASS in the declared C1024 validation scope**. This report
tracks a separate declared storage extension. The C400 results and original
240-task pilot remain unchanged. See [the execution protocol](EXTENDED_CAPACITY_PROTOCOL_V1.md).

## Scope and preparation

The user approved adding an expanded preset and continuing full-episode pairing.
`configs/benchmark_capacity1024.json` changes only `graph_capacity` and
`graph_padding_size` from 400 to 1024; `configs/benchmark.json` is the original
capacity control. K=30, graph merge threshold 340, history length 513, map lattice,
sensing, action semantics, communication and stopping rules are unchanged.
No environment, model or training source has been modified for this extension.

Ten new validation task identities were frozen from the same five-family,
two-navigation C400 cost subset. Each uses N4, seed0, 8192 bytes per robot, Normal,
mixed communication and 512 rounds/99% coverage. This checks map bucket 768 only,
not all map/robot/budget/fault combinations. The final B1/B4 workload is twenty
executions in four fresh processes, each retaining five-map order and B4's 4+1
batch boundaries.

On target Linux CPU, six focused freeze/identity checks passed. These check that
new task identities preserve map/checkpoint/spawn/fault identities and reject
changes to neighbor count, graph merge threshold, split, original identity or
task scope. The environment's 111-file frozen source remains unchanged.
The new request binds 147 inputs. Server root:
`/home/user/Tz/acceptance/batch-cost-c1024-v1-20261118`.

The [new frozen manifest](evidence/capacity-check-v1-c1024-20261118/capacity-tasks.jsonl)
has SHA256 `5e64e95541f3686899f5c438800b39d2567764feb8b33b876e9004b571a2d0bf`;
the configuration identity is
`7dcf454f6e042f75f4f9d3a4c0279870c3601e7a491dda727ef107474b3201ba`.
The [old/new task mapping](evidence/capacity-check-v1-c1024-20261118/capacity-tasks.capacity.json)
and [CPU results](evidence/capacity-check-v1-c1024-20261118/cpu-tests.xml)
are separate from the original C400 evidence.

## Execution and acceptance gates

The new continuation ledger started at 11:11:15 CST and inherits the existing
absolute deadline **14:10:05.508 CST**. It does not renew the earlier four-hour
authorization or reopen the closed C400 ledger. Preparation, idle waiting and
compilation count toward the same bound. The idle gate is no GPU compute
applications and utilization below 10% for 60 seconds.

The first gate compares 64-event C400/C1024 traces and final logical state for
all ten task pairs on H100. Both graph states must validate. The second gate
requires all four complete-episode cases and ten same-configuration B1/B4 trace
pairs to pass the existing independent analyzer. Successful C400 NF/easy and
NF/difficult traces are additional cross-capacity controls, not B1/B4 pairs.
Timing comparability separately requires clean process telemetry.

The [H100 prefix comparison](evidence/capacity-check-v1-c1024-20261118/prefix-parity.json)
completed all ten pairs successfully in 268.375 seconds including compilation.
All recorded compact trace fields matched exactly, with no nonzero floating
differences. Physical/cognitive state fields were compared except differently
padded node storage; graph comparisons cover valid node order, utilities,
visibility, local/global edges and global node coordinates. Both graph states
passed validation. This is a 64-event behavior check, not a throughput result or
a comparison beyond the original capacity limit.

The first full case, NF/mixed B1, started at **11:16:44 CST**. Subsequent cases
completed in the declared order: NF B1, NF B4, IR2 B4, IR2 B1. The coordinator
closed successfully at **12:05:46 CST**, before the unchanged 14:10 deadline.
The continuation, including idle wait and prefix compilation, took **54m30s**;
the four full-episode worker processes totaled **49m01s**. No GPU work remains.

## Complete-episode results and parity

All **20 executions** produced valid results: **8 early successes, 12 normal
512-round timeouts, zero environment/model errors, zero OOMs and zero illegal
actions**. There were **36,016 active robot events**; frozen scan slots are
excluded from that count. The three procedural NF tasks that failed C400 now
each complete 2,048 active events under C1024, without silent graph truncation.

The target-server [independent full-cohort audit](evidence/capacity-check-v1-c1024-20261118/batch-cost-audit.json)
reports `complete=true`, four cases and ten passed B1/B4 pairs. Every compact
trace field matches exactly, including coverage, rewards, movement, packet bytes,
links, delivery, spending and termination; no floating tolerance was needed.
The audit independently checks result/trace checksums, configuration/map/weight
identity, accounting, integer coverage monotonicity, AUC, actual batch order and
outcomes. All four process records have clean sampled GPU telemetry.

Three existing complete C400 controls also match C1024 exactly: NF/easy,
NF/difficult and IR2/easy. The third uses the previously published cost trace;
no extra GPU repeat was run. The [summary and control checks](evidence/capacity-check-v1-c1024-20261118/capacity-summary.json)
are reproducible with the
[offline summary script](protocols/phase6-20261118/summarize_capacity_check.py.txt).
The [final input check](evidence/capacity-check-v1-c1024-20261118/final-input-preflight.json)
verified all 147 pinned inputs unchanged and observed no remaining GPU compute
processes. Its dependency versions are explicitly a post-run observation.

The following outcomes are identical at B1 and B4. They describe one fixed
validation seed per map and serve execution acceptance, not an algorithm ranking.

| Family | NF outcome | NF final coverage | IR2 adapted outcome | IR2 final coverage |
|---|---|---:|---|---:|
| IR2 easy | success, round 493 | 99.7250% | timeout, round 512 | 98.6336% |
| IR2 difficult | success, round 495 | 99.1656% | success, round 178 | 99.1016% |
| rooms | timeout, round 512 | 55.9156% | timeout, round 512 | 95.4865% |
| corridor | timeout, round 512 | 52.2230% | timeout, round 512 | 59.0457% |
| hybrid | timeout, round 512 | 87.1627% | success, round 266 | 99.2105% |

## Measured complete-workload costs

Each row is a fresh process on the same H100, eight host cores/threads, the same
ordered five-map workload and C1024 configuration. B1 executes `1+1+1+1+1`; B4
executes `4+1`. No competing GPU PID or failed telemetry query was recorded.
These are single cold-process observations, not repeated timing distributions.

| Method | Batch limit | Active events | Process wall s | Lower/compile s | Synchronized device s | End-to-end active events/s |
|---|---:|---:|---:|---:|---:|---:|
| NF mixed | 1 | 10,093 | 782.289 | 51.725 | 724.667 | 12.902 |
| NF mixed | 4 | 10,093 | 642.857 | 105.858 | 531.001 | 15.700 |
| IR2 mixed | 1 | 7,915 | 816.511 | 52.987 | 758.145 | 9.694 |
| IR2 mixed | 4 | 7,915 | 699.487 | 108.993 | 584.234 | 11.315 |

Whole-workload B4 speedup is **1.217× NF / 1.167× IR2** including compilation,
loading and result writes. Device-only speedup is **1.365× / 1.298×**.
The single-lane tail receives its own B1 compilation in the B4 process, so the
four-lane batch alone must not be used to price the complete five-map workload.

| B4 process / actual batch | Active events | Lower/compile s | Device s | Batch boundary s |
|---|---:|---:|---:|---:|
| NF, first four maps | 8,045 | 54.479 | 402.268 | 458.307 |
| NF, one-map tail | 2,048 | 51.379 | 128.733 | 180.545 |
| IR2, first four maps | 6,854 | 56.123 | 436.356 | 494.388 |
| IR2, one-map tail | 1,061 | 52.871 | 147.878 | 201.174 |

Batch-boundary time contains compilation/device time and is nested, not an
additional independent phase to sum. Process wall also includes loading,
initialization and serialization outside those boundaries. Device event rates
count active robot events while retaining frozen-lane runtime in the denominator.

## Process memory and supported scope

| Method / batch limit | Host process peak GiB | JAX live-allocation peak GiB | Sampled own NVML peak GiB |
|---|---:|---:|---:|
| NF B1 | 1.26 | 1.37 | 3.08 |
| NF B4 | 2.00 | 8.31 | 10.60 |
| IR2 B1 | 1.47 | 1.38 | 3.15 |
| IR2 B4 | 2.29 | 8.35 | 10.71 |

These are independent process peaks; B4 includes its compiled single-lane tail.
NVML is sampled every two seconds and includes context/pool overhead, so its
sample maximum is a lower bound on physical peak, not JAX live allocation or
per-environment memory. No maximum parallelism or linear memory increment is
certified. This proves B4 for the declared N4/five-map/768-bucket/Normal/mixed/8KiB
subset. B16, other map/robot buckets and the complete 240-task pilot were not run.

Raw server originals remain under the declared root. The public archive retains
compact traces, case/compilation/memory records and checksummed task identities;
the source package, old C400 failures and Phase 1–5 evidence remain intact.
The [artifact index](evidence/capacity-check-v1-c1024-20261118/artifact-index.json)
binds exported file hashes/sizes; the [review](evidence/capacity-check-v1-c1024-20261118/review.json)
records the final count and scope checks. Compact evidence is approximately
2.3 MB; it contains no training products or full profiler dumps.
Run new measurements only with a fresh output root and an applicable resource
authorization; the closed ledger cannot be restarted.

A runtime capacity error remains an environment error; the runner cannot
silently trim graphs or convert it to a normal episode timeout. The passed gates
above certify only the declared C1024 subset.

## Interpretation

C1024 extends the task beyond the original C400 capacity limit. It is not a
semantics-preserving performance optimization beyond the point where C400 fails.
Within the valid C400 range, behavior preservation is supported by the ten prefix
checks and three complete controls above. IR2 adapted
uses the same external weights with a larger masked graph input; successful
inference does not certify train-only fairness or generalization beyond its
original training graph sizes. This run does not start training or the full pilot.
