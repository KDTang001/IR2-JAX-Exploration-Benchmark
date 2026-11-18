# Extended graph capacity v1: execution plan and declaration

Approved scope, 2026-11-18: add a separate capacity preset, preserve C400 as the
original comparison, and finish complete-episode B1/B4 acceptance. Reuse the
current checkout and frozen environment source; do not repeat Phases 1–5.

## Configuration and identity

`configs/benchmark.json` remains the C400 control. Add
`configs/benchmark_capacity1024.json`, changing only `graph_capacity` and
`graph_padding_size` to 1024. This extends storage and masked policy input length,
not the number of candidate actions. Keep K=30, merge threshold 340, history 513,
one-hop actions, sensor, rewards, seeds, maps, budgets and link conditions.
1024 is a declared headroom preset, not a measured minimum or a guarantee that
every possible map fits. It avoids repeatedly increasing capacity after each
observed failure; C512 has not been tested. Node capacity is independent of the
768-pixel map bucket used by these selected maps.

Freeze ten new task identities from the same ten C400 validation tasks: five
families, NF/mixed and IR2-adapted/mixed, N4, seed0, 8192 bytes, Normal, 512 rounds,
99% physical coverage. Keep an explicit old/new identity mapping. This is a
separate capacity acceptance subset, not a rewrite of resource-pilot-v1 or its
240-task manifest, and not a formal test/OOD evaluation.

## Work and necessary checks

- [ ] Add the preset and a small freeze-only script. Reject changes to any other
  config field, non-validation tasks, wrong scope, changed original identities
  and existing output paths. Use existing scenario/task identity functions.
- [ ] Run focused freeze/identity checks on target Linux CPU. No full regression
  suite is needed for a configuration-only environment change.
- [ ] Freeze source, helper, manifest, map and checkpoint hashes before execution.
  Keep all new measurements in a new server root.
- [ ] Run a 64-event C400/C1024 pre-limit H100 comparison for all ten tasks.
  Validate both graph states. Compare actions, physical coverage integer counts,
  links, packets, delivery, balances, termination and rewards. Allow only the
  already declared coverage/reward floating tolerances. Any action mismatch fails.
- [ ] Run the full C1024 five-map workload in four fresh processes: NF B1,
  NF B4, IR2 B4, IR2 B1. B4 must actually use groups 4+1; retain separate timings
  for the full batch and tail. Stop subsequent cases on errors.
- [ ] Reuse the independent full-cohort analyzer: all four cases and ten paired
  traces must pass, with unchanged task/config identities on both batch sizes.
  Compare the successful C400 NF/easy and NF/difficult traces against C1024 too.
- [ ] Publish outcome, compiler/device/end-to-end costs, host/JAX/NVML memory,
  limitations and raw compact evidence; update the existing review branch/PR.

## Resource bound and stopping rules

This continuation uses the user's existing maximum-four-hour authorization and
the new instruction approving capacity expansion. It does not reset the clock:
absolute deadline **2026-11-18 14:10:05.508 CST**. Waiting, CPU preparation and
compilation consume the remaining interval. Keep the old closed ledger intact;
bind the continuation to it in a new output root with that same deadline.
Require no GPU compute processes and utilization below 10% for 60 seconds.
Use eight host cores/threads, no preallocation and fresh processes; cap each full
case at 3600 seconds and all children inside the absolute deadline.

Do not drop maps, truncate nodes, change K or graph rules, reclassify capacity
errors as timeouts, restart the closed C400 window, or claim failed pairs passed.
No training, B16, all-240 pilot or formal paper matrix is included. Any newly
observed capacity/neighbor/padding/float issue is recorded and diagnosed before
proposing a separate change.
