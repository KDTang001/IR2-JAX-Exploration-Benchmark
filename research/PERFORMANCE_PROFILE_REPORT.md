# Phase 6 performance profiling

Target: the shared NVIDIA H100, fixed host affinity 0–7, eight BLAS/OpenMP threads,
JAX 0.4.35 / jaxlib 0.4.34, unchanged `ex` environment. No training. Base source:
`d3d6c5fdec831de3076b7ede70e25b72108e59bf`.

November 18 complete-episode follow-up: the repaired H100 runner executed five
nearest-frontier B1 tasks, with two IR2 successes and three procedural graph
capacity errors. First-overflow diagnostics identified the declared 400-node
local limit. B4 was not started; no complete-episode paired throughput or new
maximum batch size is certified. This does not revise the profiler/short-segment
results below. See [full-episode evidence and next capacity gate](BATCH_COST_V1_REPORT.md).

## Measurement and evidence

Each of the six configurations uses a fresh process. The three validation maps
are the preselected Phase 5 maps, with actual geometry 500×500, 480×640 and
1000×1000, padded respectively to 512, 768 and 1024. Logical populations are
N2/N4/N6, with the same static six-slot representation. Maps, checkpoint bytes,
seed 7, config and actions are fixed. These representative cases vary both map
and N; they do not alone identify a causal map-size effect.

Each process measures reset, scripted environment events, nearest-frontier
rollout, frozen IR2 network inference, and IR2 adapted rollout. There are four
events per rollout segment, two warmups and five synchronized timing samples.
Reset units are environments, inference units are graph decisions, and rollout
units are actual active robot events summed across lanes. A segment is not a step.
First-call time includes tracing, compilation and first execution; it is not
reported as pure compiler time. The separate paired harness measures lowering/
compilation directly and uses 32-event segments with ten alternating paired samples.

The profiler runs separately from the steady timers, with a synchronized invocation.
GPU operator categories are joined to optimized HLO metadata using kernel names;
unmapped and copy work remains explicit. Categories can include fused operations.
Their sums describe device-operation durations, not additive wall-time percentages.
Profiler overhead is excluded from the unprofiled sample times.

Raw evidence: `/home/user/Tz/acceptance/phase6-prep-20261117/` on the target server.
The [compressed analysis and raw-file inventory](evidence/phase6-profile-analysis.json.gz)
contains 30 measured phases and SHA256/size identities for 1,047 raw files.
The [actual execution records](evidence/phase6-profile-executions.json) retain argv,
map/weight identities, CPU affinity, per-worker versions and post-run dependency
observations. All sampled baseline GPU records contain at most one compute process.
The JAX traces contain actual `/device:GPU:0` streams despite an optional
TensorFlow Python trace-hook import warning; profiling did not require changing
the approved JAX runtime. Host events were not substituted for GPU measurements.

## Baseline steady measurements

These are initial four-event workloads, not complete-episode costs or training rates.
Memory is the compiler's temporary-buffer plan for that executable, not its process
NVML peak or the physical memory requirement of an entire experiment.

| Bucket / N / B | Reset ms | NF active events/s | IR2 active events/s | Reset temp GiB | NF temp GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 2 / 1 | 3.826 | 23.19 | 22.88 | .708 | .161 |
| 512 / 2 / 4 | 12.478 | 67.49 | 66.32 | 2.833 | .675 |
| 768 / 4 / 1 | 6.274 | 23.52 | 23.04 | 1.532 | .341 |
| 768 / 4 / 4 | 20.009 | 64.33 | 61.64 | 6.129 | 1.448 |
| 1024 / 6 / 1 | 9.013 | 23.93 | 24.07 | 2.662 | .593 |
| 1024 / 6 / 4 | 31.999 | 55.39 | 56.90 | 10.649 | 2.540 |

First-call intervals across these cases: reset 5.32–7.54 s, scripted events
38.76–41.51 s, nearest-frontier 45.90–47.80 s, IR2 inference .96–1.07 s,
IR2 rollout 48.47–50.28 s. Each complete profile process took 171.8–183.9 s,
including HLO/trace export overhead within that process. No persistent compile
cache was enabled. Timers block complete output PyTrees.

## Measured bottlenecks

1. **Graph visibility materializes a map per node.** Optimized HLO attributes
   `u8[1,6,400,512,512]` directly to `_visible`'s batched conditional at original
   `ir2_graph.py:188`. This is a 600 MiB logical buffer. At 1024/N6/B4 the
   corresponding `u8[4,6,400,1024,1024]` is 9.375 GiB. XLA can reuse a temporary
   allocation across values, so individual logical-buffer sizes must not be summed
   to invent a peak. The allocation report and compiled temporary plan independently
   identify the problem.
2. **Graph maintenance dominates the measured device work.** It accounts for
   76.6–85.9% of reset GPU-operation duration and 53.4–63.5% in the measured policy
   rollouts. The four-event NF segments produce 49,613–63,285 GPU operations.
   Numerous small loop kernels, device copies and loop-control transfers remain.
   Removing a large allocation does not remove this dynamic graph workload.
3. **Policy inference is a small part of these workloads.** Frozen IR2 inference
   takes about .73–1.29 ms per batched decision, while the full four-event IR2
segment takes 166–281 ms. Isolated network time cannot be directly multiplied
   and subtracted to assign fused rollout phase percentages. Sensing is also a
   small device-duration category here. The evidence does not support blaming
   neural-network size or the CPU map loader for these synchronized inner workloads.
4. **The scripted probe has a distinct memory cost.** It first observes every
   static robot slot to construct endpoints. At 1024/B4 its compiler temporary
   plan is 12.546 GiB versus about 2.540 GiB for NF rollout. Phase 5's first-scripted
   B16 OOM therefore cannot establish a universal B16 policy-rollout limit.
   Large-map B16 remains unmeasured after optimization; it was not retried.

The profiler covers reset and early exploration. Mature graph/near-exhausted-budget
costs require the full-length validation measurements below; the initial four-event
rate cannot establish whole-episode throughput or explain every late-episode stall.

For the largest representative IR2 rollout (1024/N6/B4), the single captured
four-event invocation has the following HLO-attributed device work. Reset is a
separate invocation measured above. Category boundaries are source scopes, not
isolated component APIs; fused or unmapped work is retained explicitly.

| Source category | Summed device duration ms | Share of device-operation duration |
| --- | ---: | ---: |
| Sensing | 1.915 | .9% |
| Graph maintenance/features | 118.437 | 53.4% |
| Candidate generation/rendezvous observation code | 5.931 | 2.7% |
| Policy inference | 1.447 | .7% |
| Communication encoding/decoding/channel code | 21.960 | 9.9% |
| State/reward update code | 6.852 | 3.1% |
| Copies/memsets | 34.650 | 15.6% |
| Unattributed kernels | 30.702 | 13.8% |

The 221.894 ms sum is not the unprofiled wall time of 281.19 ms and is not a
stage-latency decomposition. Nevertheless, the explicitly attributed graph work,
large allocation and many small loop kernels jointly identify the optimization
priority. No component-level percentage is inferred from the network timer alone.

## Minimal optimization and paired acceptance

The candidate removes only the redundant outer conditional in `_visible`. Its
`nearby` mask already includes node validity; an inactive node has count zero and
the loop returns its unchanged zero buffer. No ray, tie ordering, utility rule,
sensor, communication, reward or PRNG operation is changed. There is no refactor.

The pairing harness extracts the original `_visible` AST verbatim from the frozen
base and compiles it alongside the candidate, sharing every other module and the
same inputs. Reset and actual NF/IR2 mixed-message rollouts are compared leaf by
leaf, with graph validity checked separately. All state arrays, physical coverage,
actions, rewards, packet buffers, fees, deliveries and keys participate. An exact
failure aborts the comparison; two equal invalid graphs cannot pass acceptance.
Normal is checked for all six cases; the 768/N4/B4 case also checks independent
Dropout .3 and a short burst/recovery window. The latter is an optimization probe,
not the pilot's formal round128/duration64 fault condition.

All **22 paired workloads passed**, with zero invalid actions and no graph-state
errors. The committed package at `736ca08659d5753c7a3b9c6aab471004fb0fea82`
matches all 111 frozen source hashes. The optimization is retained on that evidence.
The [paired raw samples, output-leaf hashes and execution records](evidence/phase6-optimization-comparison.json.gz)
are published separately from the original baseline profiles.

| Bucket / N / B | Reset median speedup | NF mixed speedup | IR2 mixed speedup | Reset temp MiB: old → new | NF temp MiB: old → new |
| --- | ---: | ---: | ---: | ---: | ---: |
| 512 / 2 / 1 | 1.268× | 1.009× | 1.011× | 725.16 → 159.31 | 164.94 → 66.22 |
| 512 / 2 / 4 | 1.354× | 1.034× | 1.036× | 2900.59 → 637.22 | 690.73 → 415.27 |
| 768 / 4 / 1 | 1.443× | 1.015× | 1.019× | 1568.92 → 159.31 | 348.93 → 131.11 |
| 768 / 4 / 4 | 1.594× | 1.056× | 1.048× | 6275.66 → 637.22 | 1480.65 → 759.76 |
| 1024 / 6 / 1 | 1.551× | 1.028× | 1.034× | 2726.27 → 176.27 | 607.54 → 247.17 |
| 1024 / 6 / 4 | 1.648× | 1.082× | 1.080× | 10905.15 → 705.15 | 2597.41 → 1324.49 |

Normal mixed-message rollouts improve by **0.9–8.2%** in these median measurements.
Reset speedup is 1.27–1.65×, corresponding to 21.1–39.3% less median latency.
This is primarily an allocation improvement, not an order-of-magnitude rollout
speedup. The 768/B4 Dropout and burst probes also pass exactly: NF ratios
1.0535/1.0539 and IR2 ratios 1.0460/1.0463. Small ratios are descriptive measurements,
not a claim of statistical significance. Compilation always runs old then new;
steady timing alternates order. Persistent compilation caching is disabled.

Compiler memory is reported separately for each executable. Both variants coexist
in the paired process, so its cumulative allocator/NVML peak is not an independent
per-variant peak. The largest reset temporary plan falls from 10.65 to .69 GiB
(93.5%); its NF rollout plan falls from 2.54 to 1.29 GiB. The same map, shapes,
state, actions, message bytes, delivery, rewards and random inputs are used within
each pair. No unsupported per-variant physical peak claim is made.

## Full-episode cost probe

Two tasks were declared before execution, as a subset of the frozen 240 identities:
IR2 validation/easy `img_4775.png`, 480×640 in bucket768, seed0, N4/B1, 8192 bytes
per robot, Normal, mixed communication, both navigation methods. They retain
512 rounds and .99 physical coverage termination. They use the standard evaluation
runner and derived scenario seeds, rather than the profiler's seed7 initial state.

| Method | End-to-end seconds | Active events | End | Final physical coverage | JAX allocation peak MiB | Sampled NVML peak MiB | Host peak MiB |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| NF mixed | 187.96 | 1971 | success, round493 | 99.7250% | 435.44 | 1098 | 1197.71 |
| IR2 adapted mixed | 131.38 | 2048 | normal round512 timeout | 98.6334% | 446.81 | 1170 | 1406.04 |

Each is a fresh process: wall time includes loading, compilation, device execution,
host transfer, validation, compression and result writes. JAX allocation peaks and
host high-water RSS cover that whole process. NVML values are lower bounds sampled
every two seconds and include CUDA/context/pool costs; they are not interchangeable
with live JAX allocated bytes. One success and one normal timeout are two valid
complete tasks. The timeout is not a process time limit or execution failure.

Checksums, trace digests, active-event counts, monotone coverage, round-end AUC,
emitted/spent bytes, per-agent balances, invalid-action absence and frozen charges
were independently recomputed. Both send 32,724 bytes; each robot retains 11 bytes,
which cannot afford another standard packet. The last transmission is event1139;
movement continues for another 831/908 active events. NF's final 77 scan slots are
frozen. See [machine costs and audits](evidence/phase6-mini-cost.json) and the
[pre-execution declaration](evidence/phase6-mini-declaration.json).
The [artifact index](evidence/phase6-mini-artifact-index.json) maps these results to
the two published compact NPZ traces and checksummed result envelopes. Their bytes
match the server hashes; the large profiler captures remain server-side.

Two observations on one map do not price the five-family, two-budget, three-fault
pilot, certify full-length B4 execution, or rank navigation algorithms. Multiplying
these two cold process costs by 240 would give 8.76–12.53 hours as **arithmetic only**;
this is not a measured bound or H100 prediction. Phase 5's 19.8-hour mechanical
estimate is not promoted to a verified cost. The full pilot needs separate approval.

## Reproduction and limits

Use the pinned source snapshots, `scripts/phase6_profile.py`,
`scripts/phase6_compare.py`, the persisted request/deadline, and the recorded command
vectors. Each case uses a fresh process and checks the map file SHA. GPU traces,
HLO, allocation reports and memory profiles remain server-side; the public compact
inventory permits byte-identity verification without copying 2.72 GB into Git.
The measurement method follows JAX's documented requirement to
[synchronize profiler workloads](https://docs.jax.dev/en/latest/profiling.html)
and distinguishes [live device allocation profiles](https://docs.jax.dev/en/latest/device_memory_profiling.html)
from a compiler buffer plan and a sampled NVML peak.

This phase does not repeat the Phase 5 CPU/H100 matrix. CPU functional checks and
wheel installation are separate from GPU throughput acceptance. B1/B4 early
workloads passing does not certify every mature full-length B4 task, maximal
parallelism, full-pilot duration, or formal paper feasibility.
