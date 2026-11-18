# resource-pilot-v2-C1024

Status on 2026-11-18: **preparation; not yet frozen or executed**. The user
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

- [ ] Add explicit v1/v2 selection to the existing freeze-only planner; retain v1
  defaults and reject capacity/version mismatches before writing files.
- [ ] Add terminal graph diagnostics only to the measurement wrapper, with its
  overhead separately recorded; do not modify environment or model transitions.
- [ ] Run focused v1/v2 planning checks and a tiny real measurement-wrapper check
  on target Linux CPU, then freeze the actual 240/30 manifests and hashes there.
- [ ] Execute only the authorized 30-task H100 probe and independently audit it.
- [ ] Unify the candidate paper/SAC capacity references and record remaining
  gates: zero budget, N2/N6, other buckets, 8KiB faults and full pilot costs.
- [ ] Verify PR #1/#2 dependency and source identities, prepare a merge plan and
  upload reports/evidence. Do not merge either branch or start training.
