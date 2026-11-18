# Phase 6-Prep

Base: `d3d6c5fdec831de3076b7ede70e25b72108e59bf`.
Working branch: `codex/phase6-prep`. Target: the existing shared H100 server.
Status: **complete within the declared preparation scope**. This is benchmark
preparation, with no policy/world-model training or formal test run.

Latest follow-up, November 18: separate C1024 complete-episode acceptance passed
20 executions and ten exact B1/B4 pairs; C400 controls remain unchanged. See
[the final capacity report](EXTENDED_CAPACITY_REPORT.md) for scope and costs.

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
At that initial probe's completion these were **2/240 task identities used for cost only**, with 238 unscheduled;
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
a fair learned-policy comparison. The initial C400 preparation did not certify
full-length B4 or optimized large-map B16; the separate C1024 follow-up is below.
The old Phase 5 allocation failures remain historical
evidence, not rewritten results or a newly measured maximum parallelism.

## Follow-up: full-episode B1/B4 cost check

The user authorized another maximum four-hour window and explicitly selected an
idle H100. [batch-cost-v1](BATCH_COST_V1_REPORT.md) freezes ten existing validation
tasks, run as twenty B1/B4 executions in four processes. Its new window is
2026-11-17 19:45:02–23:45:02 CST, including waiting and compilation. Target-server
input verification and a tiny CPU check of the measurement wrappers passed.
On November 18 the final record showed a dispatcher `KeyError: 'bucket'` at
20:32:27 CST, after the GPU became available and before any worker launched.
**Zero GPU cases executed**. The command-construction bug is repaired; the target
server's two dispatch regressions and a real CPU parent/child B1/B4 smoke check
passed. Full-episode H100 parity and speedup remain unverified. The failed root
and all original inputs remain intact; a new window is required for recovery.
This is not approval or execution of the full 240-task pilot.

### November 18 authorized H100 recovery: capacity gate remains open

The user subsequently approved a new four-hour maximum, 10:10:05–14:10:05 CST,
including idle waiting and compilation. The repaired dispatcher launched the
five-map nearest-frontier B1 case. IR2 easy and difficult succeeded at 1,971 and
1,978 active events. Rooms, corridor and hybrid returned `environment_error`;
the coordinator stopped at 10:21:24 before any B4 or IR2-adapted case. The full
cohort analyzer rejected the failed case as intended. No paired speedup or new
complete-matrix resource estimate is supported.

A ten-minute diagnostic within the same authorization stopped at first overflow
on these three unchanged procedural tasks. Their local graphs reached the
declared 400-node limit at event counters 1,278 / 1,563 / 1,609. History and global
nodes remained below their limits, outgoing degree did not exceed 30, and no path
failure was set. This is a graph-capacity block, not an observed GPU allocation
failure. Both valid traces passed independent checksum, coverage, event and byte
accounting audits. Further GPU work in this window is stopped; no training ran.

See [the full recovery and capacity report](BATCH_COST_V1_REPORT.md) and
[partial audit](evidence/batch-cost-v1-recovery-run-20261118/partial-audit.json).
The strict C400 configuration and all original failed results remain intact.
A larger-capacity preset would require new configuration/task identities and
complete-episode feasibility checks before replacing any pilot proposal. C512
is a proposed first candidate, not an executed or certified replacement. The
240-task pilot and paper experiment scale therefore remain unapproved.

### Separate C1024 continuation, approved November 18

The user approved a separate expanded-capacity preset, retaining C400 as the
original-capacity comparison. C1024 changes only storage and node padding, with
new task identities and the same maps/seeds/weights/K30. Six target-Linux CPU
checks and all ten 64-event C400/C1024 H100 comparisons passed; compact traces
matched exactly. Complete-episode acceptance then passed: 20 valid executions,
36,016 active events, ten exact B1/B4 trace pairs and three exact complete C400
controls. There were eight successes, twelve normal timeouts, zero execution
errors/OOMs/illegal actions, and clean recorded GPU telemetry. All 147 pinned
inputs remained unchanged. The continuation closed at 12:05:46 CST, within the
unchanged 14:10 deadline; no training or further GPU work is active.

Whole-five-map B4 speedup was 1.217× NF and 1.167× IR2 including cold compilation,
loading and writes; device-only ratios were 1.365× and 1.298×. B4 executed actual
4+1 groups, with an independently compiled tail. Sampled own NVML peaks were
10.60/10.71 GiB. These single-process observations certify this subset's B4,
not maximum parallelism, other map buckets or all 240 pilot tasks. The C400
manifest and failed results remain unchanged. See [the expanded-capacity report](EXTENDED_CAPACITY_REPORT.md).
