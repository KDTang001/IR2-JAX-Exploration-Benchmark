# Phase 6-Prep

Base: `d3d6c5fdec831de3076b7ede70e25b72108e59bf`.
Working branch: `codex/phase6-prep`. Target: the existing shared H100 server.
Status: **complete within the declared preparation scope**. This is benchmark
preparation, with no policy/world-model training or formal test run.

## Scope and resource boundary

The authorized H100 window is at most four hours, including compilation. It began
2026-11-17 15:20:43 CST and expires 19:20:43 CST. The persisted
[deadline](evidence/phase6-gpu-window.json) is reused by later measurements, never
renewed. Workers are isolated, run one at a time, and are terminated at their own
case limit or the global deadline. Each launch checks for competing GPU processes.
All measurements finished at 16:27:13 CST: the conservative first-launch-to-final-
completion interval is **66 min 29 s**, including compilation. The two full-episode
workers also use an independent GNU timeout guard. No measurement remains running;
the unused part of the four-hour window is not automatically reused.
The [closed-window record](evidence/phase6-gpu-window-closed.json) reports zero
remaining workers; a [read-only rejection check](evidence/phase6-closed-guard-proof.json)
confirms that the scheduling entry point refuses to reopen it.

Phase 1–5 records and source snapshots remain untouched. No A/B world-model,
teammate intention, or training implementation was changed. Main was not merged.

## Release and failure handling

Installed-wheel metadata now records actual imported source bytes, distribution
location, accessible wheel archive/digest, the installed dependency inventory,
interpreter, platform and observed runtime information. It does not infer a Git
checkout from arbitrary parent directories, initialize a GPU merely to report
metadata, or claim a hardcoded `ex` runtime lock. Missing revision/lock/archive
evidence remains null with an explicit status. Existing report fields and result
readers remain usable.

The independently installed final wheel passed eleven provenance assertions:
111 package source files equal the frozen LF source snapshot, its import comes
from the new venv, and the wheel digest agrees with the installer archive digest.
Wheel SHA256: `40bac2b46f8615aeb7ca384d2afbea7d43684faaa2ff6bbb4cad9425a6db4cad`.
Package inventory hash: `402942e2eefc04ad33aa65fe8d8e83e68b2411e05e932264548f8cff3d0a7fbd`.
This inventory also matches the local publishable source bytes. See the
[actual installed-wheel proof](evidence/phase6-wheel-verification-final.json).
It deliberately reports no Git revision for a wheel that contains no such claim.
The independent [source-to-Git binding](evidence/phase6-source-binding.json) verifies
all package bytes against commit `736ca08659d5753c7a3b9c6aab471004fb0fea82`.

Allocator `ValueError` messages containing `RESOURCE_EXHAUSTED` survive the model
execution boundary and are classified as OOM by the batch runner and throughput
measurement. Ordinary ValueErrors retain their existing error classification.
Historical result files were not reclassified or rewritten.

## Performance work

All six requested baseline configurations completed five workload phases each:
512/N2, 768/N4, 1024/N6, each B1/B4. The measurements include real GPU traces,
optimized HLO, compiler allocation reports and device/host memory observations.
No large-map B16 attempt was made. The raw 1,047 files (2.72 GB) remain on the
server; the compact public evidence includes their byte sizes and SHA256 hashes.

The first identified allocation problem is a redundant conditional in graph
visibility. Under vmap it broadcasts the map once per graph node. The candidate
production change removes that conditional; inactive nodes already execute a
zero-trip loop and return an all-false result. No graph, sensing, action, reward,
message or random-number rule changes. **22 paired workloads pass exactly**,
including independent Normal/Dropout/burst probes and zero invalid actions.
Reset median speedup is 1.27–1.65×; mixed-message rollout gains are 0.9–8.2%.
The largest reset temporary-buffer plan drops 93.5%, from 10.65 to .69 GiB.
This improves allocation substantially; dynamic graph computation remains the main
runtime bottleneck. Details and limits are in [the performance report](PERFORMANCE_PROFILE_REPORT.md).

## Pilot and paper readiness

The explicit validation-only planner has frozen 240 task identities, covering five
maps, N4, two budgets, three independent fault conditions, and two navigation
methods × four communication strategies. `evaluation_maps()` still selects only
test/OOD for formal evaluation. See [the pilot declaration](RESOURCE_PILOT_V1.md).
The separately predeclared two-episode, full-length validation cost subset completed:
NF mixed succeeded at round493 (187.96 s); IR2 adapted mixed reached the normal
512-round timeout (131.38 s). Both use N4/B1 on one validation map, and include
compilation/loading/result writes. Independent metric/accounting audits pass.
These are **2/240 task identities used for cost only**, with 238 unscheduled;
the full pilot was not launched. One map cannot price or rank the complete matrix.

The external IR2 checkpoint's training exposure cannot be certified against the
new split. It remains an external reference. A train-only SAC baseline is designed,
without implementing or starting training. See [baseline fairness](BASELINE_FAIRNESS_REPORT.md).
The [paper protocol](PAPER_CORE_PROTOCOL_V1.md) fixes proposed comparison/statistical
rules while leaving map count and resource approval pending sufficient cost data.
Neither 58,320 nor 2,294,784 episodes are authorized or started.

## Validation and remaining gates

Targeted tests ran on the target Linux server, using the CPU backend for small
functional checks while the isolated H100 jobs measured performance. The first
source-only export omitted two original IR2 oracle dependencies; those deployment
failures are retained. After supplying the unchanged original files, only the
affected oracle checks were rerun. No production fix was made for an export error.
The aggregate latest result is 62 distinct passing checks, including all 16 graph
parity tests, new OOM boundary cases, release metadata and validation-only planning.

The [test summary](evidence/phase6-targeted-tests-summary.json) links the retained
initial and corrective XML outcomes. The [paired measurements](evidence/phase6-optimization-comparison.json.gz)
and [full cost audit](evidence/phase6-mini-cost.json) contain the final outcomes.

Remaining decisions: separately approve the full validation pilot and its time
budget; use a broader complete-episode cost sample to choose the paper map count
and concurrency; implement/approve the train-only learning baseline before claiming
a fair learned-policy comparison. Full-length B4 and optimized large-map B16 are
not certified by this phase. The old Phase 5 allocation failures remain historical
evidence, not rewritten results or a newly measured maximum parallelism.

## Follow-up: full-episode B1/B4 cost check

The user authorized another maximum four-hour window and explicitly selected an
idle H100. [batch-cost-v1](BATCH_COST_V1_REPORT.md) freezes ten existing validation
tasks, run as twenty B1/B4 executions in four processes. Its new window is
2026-11-17 19:45:02–23:45:02 CST, including waiting and compilation. Target-server
input verification and a tiny CPU check of the measurement wrappers passed.
The archived snapshot is queued behind unrelated GPU work with **zero GPU cases
executed**; full-episode B4 parity and speedup remain unverified. This is not an
approval or execution of the full 240-task pilot, and the old window stays closed.
