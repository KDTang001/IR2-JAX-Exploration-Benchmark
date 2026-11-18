# C1024 complete-episode and B1/B4 acceptance

Status on 2026-11-18: **running; acceptance not yet established**. This report
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

No result, speedup, maximum supported parallelism or complete-cohort acceptance
is claimed until those recorded gates pass. A runtime capacity error remains an
environment error; the runner cannot silently trim graphs or convert it to a
normal episode timeout.

## Interpretation

C1024 extends the task beyond the original C400 capacity limit. It is not a
semantics-preserving performance optimization beyond the point where C400 fails.
Within the valid C400 range, behavior preservation must be measured. IR2 adapted
uses the same external weights with a larger masked graph input; successful
inference does not certify train-only fairness or generalization beyond its
original training graph sizes. This run does not start training or the full pilot.
