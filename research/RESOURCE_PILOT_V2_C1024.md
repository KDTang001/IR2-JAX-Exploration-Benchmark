# resource-pilot-v2-C1024

Status on 2026-11-18: **240 tasks frozen; authorized 30-task probe running**. The user
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
passed at 15:24:46. The request binds 140 inputs, including all 111 unchanged
environment/model source files. Earlier closed windows remain untouched.

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
- [ ] Execute only the authorized 30-task H100 probe and independently audit it.
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
