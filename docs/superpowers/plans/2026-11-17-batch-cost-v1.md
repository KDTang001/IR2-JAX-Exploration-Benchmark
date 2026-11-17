# B1/B4 full-episode cost check, 2026-11-17

The user authorized a fresh maximum four-hour H100 window. The closed Phase 6
window stays closed. This is resource preparation, not training or the 240-task
pilot. Reuse the clean `codex/phase6-prep` checkout and frozen environment package.

## Frozen workload

Filter the existing `resource-pilot-v1` manifest, preserving its original bytes
and result IDs: all five selected validation maps, seed 0, N=4, 8192 bytes per
robot, Normal link, `nf_mixed` and `ir2_mixed`. No test/OOD data or new seeds.
Run each method's five maps once with B1 and once with B4: ten unique tasks,
twenty executions, four independent worker processes. Keep the selected map order.
Case order: NF B1, NF B4, IR2 B4, IR2 B1. Do not retry failed cases or lower B4.
The standard 512-round/99% stopping rules and graph capacity 400 remain unchanged.

## Measurement and safeguards

1. Freeze manifests, source/package inventory, checkpoint and map hashes before
   execution, and bind these plus harness hashes in the request.
2. Use a new output root and non-renewable deadline (14400 seconds, including
   compilation and waiting). Require an independent GNU timeout for each child;
   cap each case at 3600 seconds without extending the global deadline.
3. Default to an idle GPU: no compute applications and utilization below 10% for
   60 seconds. Sample every 15 seconds while queued. Shared execution requires the
   pending explicit user answer and a recorded permission; it never renews time.
4. Use eight CPU cores/threads, no preallocation or persistent compilation cache.
   Record other GPU applications throughout execution. Never stop their processes.
5. Reuse the actual evaluation runner. Measurement wrappers outside the package
   record loading, explicit lowering/compilation, synchronized device execution,
   batch boundary overhead, summarization/NPZ writing and result writing. Record
   process host/JAX peaks and sampled NVML memory. Added synchronization is disclosed.
6. Report B4's true four-lane batch separately from its one-lane tail. Count actual
   active robot events, not rollout segments or padded events. Compilation and
   complete process wall time are distinct from kernel execution time.
7. Stop following cases on runner failures, OOM, malformed output or deadline.
   Do not change old results or call wall-clock interruption an episode timeout.

## Acceptance and publication

Compare the same ten result IDs across B1/B4. Exact integer/boolean traces,
movement, messaging/accounting, termination and invalid actions are required.
Compare coverage using decoded integer covered-cell counts and the same component
area. Report float discrepancies separately; rewards may differ only by explained
coverage-rounding error (absolute tolerance 3e-5). A changed action, termination,
delivery or covered-cell count fails the comparison. Existing graph validation
must pass. Independently recheck trace/result checksums and accounting.

Run only harness boundary checks and target-server preflight, not Phases 1–5.
Archive compact traces, raw timing/resource records and identities, write
`research/BATCH_COST_V1_REPORT.md`, update the resource status, and push the review
branch/PR #2. Do not merge main or launch the remaining pilot. If the GPU never
becomes available, publish the bounded queue evidence and pending status honestly.
