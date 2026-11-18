# C1024 complete-episode and B1/B4 acceptance

Status on 2026-11-18: **64-event pre-limit comparison passed; full episodes running**. This report
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
remain scheduled only if all preceding work passes and the absolute time bound
permits it.

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
