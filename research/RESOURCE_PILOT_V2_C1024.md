# resource-pilot-v2-C1024

Status on 2026-11-18: **240 tasks frozen; authorized 30-task probe passed scoped execution acceptance**. The user
approved a maximum-four-hour H100 window for the 30-task probe below, including
idle waiting and compilation. This is independent of all closed earlier windows.
No training, all-240 execution, formal test/OOD run or merge is authorized here.

## Capacity decision and immutable predecessors

C1024 is the candidate common specification for the next benchmark, paper
protocol and train-only SAC design. It is not yet certified for every condition
or adopted as a final paper specification. C400 remains the original-capacity
control in `configs/benchmark.json`; old manifests and results stay unchanged.
The candidate is `configs/benchmark_capacity1024.json`: only graph capacity and
node padding change to 1024, retaining K30, merge threshold 340 and history 513.
IR2 external weights are the same reference, with unverified training exposure;
successful larger-input execution is not proof of training equivalence.

## Freeze-only workload

The target-server freeze is at
`/home/user/Tz/acceptance/resource-pilot-v2-c1024-20261118`.
The [240-task compressed manifest](evidence/resource-pilot-v2-c1024-20261118/tasks.jsonl.gz)
has raw JSONL SHA256
`58b0a928d1e0f2af7bf5d549e12c00e8ed44b599b8d8a97628461a90c6c9e524`.
The [30-task probe](evidence/resource-pilot-v2-c1024-20261118/probe-tasks.jsonl)
has SHA256 `d0b7d1aecd8ed79bfb2fc98f86b28cc5d8b01186dcdb2ef9086f5cfe0ffd1d79`.
The [freeze record](evidence/resource-pilot-v2-c1024-20261118/freeze.json) retains
the pre-execution `executed_gpu_tasks=0` snapshot; it is not a live progress field.

Reuse the same minimum-hash validation winner in each of five strata: IR2
easy/difficult and procedural rooms/corridor/hybrid. Recheck image, occupancy,
scenario, canonical/split and source-manifest identities using the existing
validation selector. Formal `evaluation_maps()` remains test/OOD-only.

Freeze **5 maps × seed0 × N4 × budgets {2048,8192} × three independent conditions
× two navigation methods × four communication modes = 240 tasks**. Conditions
are Normal, Dropout .3 and Burst start128/duration64. Navigation is NF or IR2
adapted; communication is off/plan/map/mixed. Max rounds 512, target .99.
All tasks are limited validation observations, not paper test results.

Regenerate scenario, config and task identities. All 240 must differ from C400
v1 while preserving selected maps, spawn/fault seeds and method/checkpoint
identities. The ten previous C1024 Normal/8KiB/mixed identities should match the
corresponding v2 entries; record them as prior engineering exposure, not newly
executed v2 tasks. Do not overwrite or silently migrate the old v1 plan.

## Authorized small probe

The new [window](evidence/resource-pilot-v2-c1024-20261118/gpu-window.json) began
**15:23:46 CST**, with a non-renewable **19:23:46 CST** deadline. The idle gate
passed at 15:24:46; [the window closed](evidence/resource-pilot-v2-c1024-20261118/gpu-window-closed.json)
at **16:42:06 CST**, after 78 minutes 21 seconds including waiting and compilation.
The request binds 140 inputs, including all 111 unchanged environment/model
source files. Earlier closed windows remain untouched.

Select exactly **30 tasks** from v2: five maps × two navigation methods × budget
2048 × Normal/Dropout .3/Burst64, with mixed communication, seed0 and N4. Use six
fresh processes in order Normal NF/IR2, Dropout NF/IR2, Burst NF/IR2. Each uses a
batch limit of four and actual groups 4+1. This is not a new B1/B4 speedup test.
The previous 8KiB Normal cohort is a reference; 8KiB faults remain untested here.

Record actual emitted bytes during the Burst window and the last emission event.
Earlier 8KiB NF traces spend 4,552 bytes per robot before round128; the 2KiB
probe can therefore lose affordable transmissions before the declared burst.
Fault-mask correctness and actual communication exposure are separate findings.
Do not treat zero exposure or equal scores as evidence that Burst is harmless.

Accept complete valid successes or normal 512-round timeouts. Require unchanged
source/map/config/checkpoint identities, legal actions, valid graph states,
consistent wire bytes/delivery/balances, correct independent fault conditions,
physical coverage/AUC and intact result/trace checksums. Record terminal local
and global node counts, history counts, outgoing degrees and sticky overflow/
path-failure flags. Terminal counts are not trajectory maxima; sticky validation
must remain enabled throughout the rollout. Capacity errors stay errors.

The coordinator starts a new persisted deadline immediately before waiting for
an idle GPU: no compute applications and utilization below 10% for 60 seconds.
The four-hour bound includes that wait and compilation. Each child has an
independent 1800-second timeout no later than the global deadline, with termination reserve.
Stop later cases on OOM, execution/identity/graph errors or global cutoff, retain
partial outcomes and never clip nodes, drop maps or renew the window.

## Implementation and release preparation

- [x] Add explicit v1/v2 selection to the existing freeze-only planner; retain v1
  defaults and reject capacity/version mismatches before writing files.
- [x] Add terminal graph diagnostics only to the measurement wrapper, with its
  overhead separately recorded; do not modify environment or model transitions.
- [x] Run focused v1/v2 planning checks and a tiny real measurement-wrapper check
  on target Linux CPU, then freeze the actual 240/30 manifests and hashes there.
- [x] Execute only the authorized 30-task H100 probe and independently audit it.
- [x] Unify the candidate paper/SAC capacity references and record remaining
  gates: zero budget, N2/N6, other buckets, 8KiB faults and full pilot costs.
- [x] Verify PR #1/#2 dependency and source identities, prepare a merge plan and
  upload reports/evidence. Do not merge either branch or start training.

Target Linux CPU evidence: [18 planning checks](evidence/resource-pilot-v2-c1024-20261118/planning-tests.xml)
passed; the real measurement CLI executed five tiny graph fixtures in actual
groups 4+1, recording five valid terminal graph diagnostics. See
[the CPU summary](evidence/resource-pilot-v2-c1024-20261118/cpu-graph-diagnostics.json)
and [reproducible fixture script](protocols/phase6-20261118/cpu_graph_diagnostics.py.txt).
These are CPU fixtures, not formal maps or H100 results. Regenerated v1 task
bytes exactly match the old immutable manifest. All 240 C1024 IDs differ from
C400; selected maps, policy/fault seeds and weight identities remain the same.

The dependency review is in [the release merge plan](RELEASE_MERGE_PLAN.md).
Preparing that plan and uploading this declaration does not merge a PR or
promote C1024 to an unrestricted paper configuration.

## Completed H100 probe

[The target-server CPU audit](evidence/resource-pilot-v2-c1024-20261118/probe-audit.json)
passed all 30 unique tasks: **52,567 active robot events, 12 coverage successes,
18 normal 512-round timeouts**, zero execution errors, illegal actions or OOMs.
Every case used a fresh process with actual groups **4+1**, not padding five
tasks into an alleged B4 group. The [finish marker](evidence/resource-pilot-v2-c1024-20261118/probe-finished.json)
records independent-audit exit code zero. This is limited validation evidence,
not an algorithm-ranking result or execution of the full 240-task pilot.

| Independent condition | Navigation | Success / timeout | Active events | Lower/compile | Synchronized device | Five-map wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Normal | nearest frontier | 2 / 3 | 10,135 | 105.75 s | 790.00 s | 901.57 s |
| Normal | IR2 adapted | 2 / 3 | 7,468 | 109.08 s | 615.11 s | 731.76 s |
| Dropout .3 | nearest frontier | 2 / 3 | 10,058 | 110.12 s | 620.14 s | 738.02 s |
| Dropout .3 | IR2 adapted | 2 / 3 | 7,303 | 110.56 s | 514.07 s | 632.94 s |
| Burst 128–191 | nearest frontier | 2 / 3 | 10,135 | 106.04 s | 793.74 s | 905.77 s |
| Burst 128–191 | IR2 adapted | 2 / 3 | 7,468 | 108.69 s | 613.00 s | 729.96 s |

These are complete cold-process workload costs, including loading, compilation,
execution and server-side result writing. Their sum is 77 minutes 20 seconds;
the resource window additionally includes the idle gate and orchestration.
This is not a new B1/B4 comparison. Fault conditions can change trajectories and
work performed, so their wall-time differences are not an optimization speedup.
Compile/device boundaries and independent process memory peaks are retained in
each case's `.memory.timing.json`, `.memory.json` and `.record.json`.

Across the six independent processes, sampled NVML peaks were **10,852–10,966
MiB** (about 10.60–10.71 GiB); JAX allocator peaks were 8.31–8.35 GiB and host
peak RSS 2.00–2.33 GiB. These are per-process five-map peaks including both B4
and the single-lane tail, not per-environment marginal memory. Two-second GPU
process samples recorded no foreign PID or query failure. Sampled peaks are
not continuous hardware maxima, and different fault workloads are not paired
performance controls.

The highest **terminal** local/global graph counts were **638 / 575**; maximum
terminal history count was 513. These counts are not trajectory maxima. Sticky
overflow/path-failure validation stayed enabled throughout every transition;
no graph capacity or path errors occurred. C1024 is therefore accepted only for
this five-map, N4, bucket768, mixed, 2KiB subset and the earlier documented
8KiB/Normal subset. This does not certify global graph fusion, N2/N6, other
map buckets or all communication modes.

The audit independently regenerated Dropout decisions on **the target CPU**
from the fault seed, event, sender and receiver, and derived Normal/Burst masks
without calling the production fault function. It checked active-event masks,
physical-link-gated delivery, actual wire bytes, per-agent balances, result/trace
checksums, coverage and AUC, as well as all 140 frozen inputs. Windows is used
only for archiving and NumPy postprocessing of the resulting evidence.

## Important exposure finding

All 30 runs emitted **8,176 team bytes**, leaving **4 bytes per real robot**,
which cannot fund another packet. The final emission was event **307** in every
run, i.e. zero-based round **76** (the 77th round). Motion and sensing continued.
The declared Burst begins at round128, event512: its ten trajectories contain
**2,331 active in-window events but zero emitted bytes during that window**.
An early successful episode can finish before the recovery boundary; correctness
is checked only on active events, with terminal freeze preserved. Nine of the
ten Burst episodes have active post-recovery events; the tenth succeeded before
round192. Thus observed restoration must not be claimed for all ten episodes.

Thus the fault schedule was exercised, but no transmitted message was exposed
to the Burst. Equal Normal/Burst coverage cannot establish robustness or the
value of communication recovery. The original fault timing and all ten Burst
tasks remain in the frozen manifest and denominator; none were moved, omitted
or rerun with more budget to improve the result. The portable
[NumPy-only summarizer](protocols/phase6-20261118/summarize_resource_probe.py.txt)
records exposure and compares paired Normal/Burst arrays separately from the
as-run independent audit. Its [derived summary](evidence/resource-pilot-v2-c1024-20261118/probe-summary.json)
confirms all ten paired traces have exactly the same other arrays, including
positions, coverage, rewards, budgets, wire bytes and delivery. It is
postprocessing, not a new environment run.

Before claiming fault effectiveness, separately authorize a condition with
actual transmission exposure, such as the already frozen 8KiB/mixed fault
subset. The broader 240-task costs remain unknown: this probe measured only
mixed/2KiB, not off/PLAN/MAP_DELTA or 8KiB faults. Do not mechanically multiply
these costs into a validated full-pilot or paper-matrix prediction. The paper
protocol remains proposed; train-only SAC is unimplemented and untrained.

The subsequent [8KiB Normal-trace exposure preflight](BURST_EXPOSURE_PREFLIGHT_REPORT.md)
supports all ten frozen mixed/Burst tasks having measurement opportunity: all
have in-window emissions and targeted successful Normal deliveries. Static
overlay identifies 817 potentially blocked receiver copies, not observed Burst
losses. Nine Normal traces also support a post-recovery observation; one ended
successfully earlier. This offline check did not start a GPU window or run any
new episode; the actual 8KiB Burst gate is still pending authorization/execution.

## Evidence and review

The [post-run target check](evidence/resource-pilot-v2-c1024-20261118/postrun-verification.json)
reverified all 140 inputs and 111 package files and observed no remaining worker
GPU process. The [source binding](evidence/resource-pilot-v2-c1024-20261118/source-binding.json)
matches those files and the measured wrappers to Git bytes. The archive's 107
files were transferred with SHA256 verification; the temporary remote bundle
was deleted after extraction. Connection interruptions after measurement did not
cause any episode rerun or window renewal. The server-side wall times above do
not include later evidence transfer to Windows.

See the [artifact inventory](evidence/resource-pilot-v2-c1024-20261118/artifact-index.json)
and [scoped self-review](evidence/resource-pilot-v2-c1024-20261118/review.json).
No Phase1–5 or C400/C1024 predecessor evidence was overwritten. The old v1
manifest and pre-execution freeze record remain immutable.
