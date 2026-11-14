# Phase 4 code review

- codeQualityStatus: CLEAR
- recommendation: APPROVE (final limited functional acceptance; terminal audit at end)
- Scope: 1c8b77cdc381bb96ba8a37128f6a8c4c928f6266..c566c970ba878687ca2caa201d62f319c0b55e71 (30 files).
- Goal: eight fixed baseline policies, honest traffic/accounting, strict unified reporting, limited H100 acceptance without training.
- Plan inspected: docs/superpowers/plans/2026-11-14-phase4-baseline-acceptance.md.
- Notepad: no separate notepad supplied.
- No source changes, local simulations, or local tests performed. git diff --check passed.
- Artifact location: toolkit wrapper failed with Windows syntax error; direct Node CLI returned ULW_LOOP_PLAN_MISSING, so prescribed fallback used.

## CRITICAL

None found.

## HIGH

### H1 — New strict reporting still accepts inconsistent task metadata

References: src/ir2_benchmark/evaluation/report.py:90; src/ir2_benchmark/evaluation/records.py:103.

The new report filter calls completed_result, whose matching predicate checks only checksum, result_id, checkpoint_hash, and configuration_hash. The record checksum proves internal consistency, not consistency with the supplied task. A record retaining those four values and its intact trace but containing a different method_id, map_hash, route, training_seed, or limited flag is accepted. Report grouping and completeness subsequently use the unchecked record fields. This defeats the explicit strict task identity requirement and can assign measurements to the wrong method/map or misrepresent scope.

Fix at the shared result boundary so both resume and reporting reject disagreement: bind scenario_id/source/family/map_hash/split/n_agents/budget/fault_mode/drop_probability/burst_duration/limited/map_scenario_hash/budget_mode/requested_team_bytes/unallocated_bytes to the scenario; bind method_id/route/training_seed/checkpoint_hash/decision_rule/real_training_events/selected_at_events to the method; retain configuration_hash verification. Respect the existing heuristic decision_rule normalization and derive auxiliary_kind using the established budget helper. Do not invent parallel identity serialization.

Evidence: the predicate at records.py:103-108 contains none of these comparisons; report.py:102-119 filters/counts accepted rows, while statistics groups on their stored method/map metadata. Existing test_benchmark_reporting.py mismatched case changes configuration_hash only and therefore does not cover this bypass. Add narrow adversarial cases that change meaningful metadata while updating the checksum and leaving the trace intact.

### H2 — Successful records with absent scores can certify completion

References: src/ir2_benchmark/evaluation/report.py:51, :109, :131; src/ir2_benchmark/evaluation/records.py:112.

A success/timeout result with metrics=null or metrics={} and a valid checksum/trace is loaded with auc=None and final_coverage=None. completed_result accepts it. requested_manifest_complete checks only outcome/error/counts, and formal completeness has no required-score check. A structurally incomplete measurement can therefore be resumed and reported as complete, with blank AUC. Nonfinite scores likewise have no boundary rejection.

Require the mandatory measured metrics for success/timeout records at the result-read boundary and reject missing/nonfinite required numeric scores; preserve the intentional absent success time for timeouts and absent metrics for explicit execution errors. Test missing metrics and a nonfinite required score with valid checksum and intact trace. This is necessary boundary parsing, not redundant production validation.

## MEDIUM

No additional concrete production issue found in the inspected baseline routing/masks, real-weight cache, emission/fee separation, fixed-policy map bootstrap, or CLI adapter. Independent replay reads raw wire/positions and original sensor/radio code instead of importing production metric calculations; its production-independent dropout probability reconstruction is limited (it trusts recorded external links for dropout), while the acceptance script separately checks paired exogenous links. Do not describe it as a complete independent stochastic-channel oracle.

## LOW

No blocking style findings. New raw dict/list annotations in manifest/acceptance helpers are weaker than the programming skill's typed-boundary preference; this follows nearby repository patterns and is not evidence of an additional correctness failure. No broad refactor requested.

## Skill-perspective check

Loaded programming/SKILL.md, its Python reference README, and remove-ai-slops/SKILL.md from the available omo 5.1.38 installation. Applied their test relevance, boundary validation, abstraction, and overfit criteria to production and test changes. No deletion-only, prose-pinning, tautological, or requested-removal tests found. Graph route fixtures distinguish graph distance, masks, padding, and ties; accounting integration tests inspect actual serialized packets and ledger differences. Protocol schedule expectations and method identity assertions concern machine-consumed behavior rather than decorative constants. The identified result boundary gaps violate the programming perspective; no separate unnecessary extraction/normalization or material slop violation identified.

## Runtime evidence and limits

Remote suite results were reported by the executor, not independently inspected in this review: first run 152/160; targeted followup 66/66 on CUDA. Paths supplied: /home/user/Tz/acceptance/phase4-20261114/results/unit.xml, followup.xml, followup.runtime.json, matrix-v1.log. These remote artifacts are not present locally and no remote access was requested of this reviewer. Executor subsequently reported real-weight 400-node parity stopped with max log-prob difference 4.0888786e-05 at a 2e-5 tolerance, before the 312-episode matrix. Thus no runtime acceptance or parity approval is asserted. Require inspected H100 acceptance, parity diagnosis, independent replay, and final report artifacts before final release approval. Limited acceptance must remain distinct from formal benchmark completion.

## Blockers

1. Bind accepted result metadata to its requested task (H1).
2. Reject successful/timeout records without finite mandatory measured scores (H2).
3. Complete and inspect the pending remote parity/limited-acceptance/oracle artifacts before claiming runtime acceptance; current review makes no runtime-success claim.


## Targeted re-review — d23e0731787fd45b08fbc211c654882944636cd1

This section supersedes the open H1/H2 findings above; original findings are retained as review history.

- codeQualityStatus: CLEAR
- recommendation: APPROVE (production code changes; not final runtime acceptance)
- Production blockers: none remaining in the targeted identity/defaults/metrics review.
- Runtime release condition: final focused remote suite, 312-episode acceptance, independent replay and final report artifacts remain pending.

H1 FIXED: completed_result now binds every duplicated scenario/method identity field identified above, using result_budget_fields for allocation provenance and the actual writers' heuristic decision_rule normalization. Defaults come from EvaluationResult appended-field defaults; missing required identity remains rejected, and nondefault task provenance cannot be inferred from a missing record field. Compared with both summarize_execution and runner._failed writers. The requested result_binding.py production filename does not exist; the implementation is in records.py and the regression file is tests/benchmark/evaluation/test_result_binding.py.

H2 FIXED: _valid_completion parses the explicit outcome variants, requires complete EpisodeMetrics for success/timeout, rejects absent/nonfinite/out-of-range numeric scores and inconsistent flags/times, and preserves explicit error records with metrics=None. It handles success at reset and intentional absent timeout success times. AUC comparison matches the production metrics writer's float64 accumulation and monotone-coverage semantics. No local runtime was executed.

Skill perspective repeated on the fix: the validation is justified at the persisted-result boundary. Adversarial tests retain valid checksums/traces while changing identity or metric contracts, distinguish nondefault task values from historical defaults, and include valid normal/error/zero-event paths. These are behavioral regressions, not tautological or deletion-only assertions. No new material slop finding.

Evidence independently read locally:

- research/evidence/phase4-unit-initial-20261114.xml: 160 tests, 8 failures, zero errors.
- research/evidence/phase4-unit-followup-20261114.xml: 66 tests, zero failures/errors. Combined by classname/name, the later result supersedes reruns: 178 unique tests and zero remaining failures. These precede the current binding fix and do not certify its pending remote suite.
- research/evidence/phase4-parity-diagnostic-20261114.json: 10 actual graph comparisons, all greedy and mask identities true; maximum FP64 log-probability discrepancy 3.04645197957143e-13, FP32 log discrepancy 0.000133991241455078, FP32 probability discrepancy 3.12626361846924e-05. These artifacts support the numeric-rounding diagnosis; the final acceptance rerun is still required.
- git diff --check c566c970 d23e0731: passed.

## Runtime graph-failure investigation (read-only, cause not yet confirmed)

Executor reports matrix-v2 CLI smoke: seven timeouts and one ir2_mixed failure, before the 312-task matrix. This invalidates any claim of completed runtime acceptance. Static findings:

1. types.py:238-249 raises ValueError for graph overflow/unreachable rendezvous; runner.py:94-107 classifies ValueError as model_error. This is an environment graph failure and should retain that category. Minimal fix: environment-owned GraphStateError(RuntimeError), raised for these two host graph checks, with explicit runner classification if desired. Preserve ValueError handling for genuine model parsing failures; do not broadly reclassify all ValueErrors.
2. env/ir2_observation.py:88-94 requires both received peer-map evidence and received peer position, plus stale contact and >=1% surplus. Mixed supplies both; PLAN-only has no peer-map evidence, MAP-only supplies no position. Thus the isolated mixed failure is consistent with reaching this branch for the first time.
3. navigation/ir2_graph.py:798-833 inserts known peer positions as nodes even when not known-free locally. global_graph_fusion=false does not disable peer-position nodes or peer-local grids. Partial MAP packets can therefore produce an eligible target separated by unknown cells. This is a hypothesis, not a diagnosed defect.
4. Original env.py:235-280 merges entire connected-group maps and positions together. Original env.py:480-516 tries directed A*, symmetric edges, then rebuilt edges with global radius 0..10*SENSOR_RANGE; env.py:557-560 returns failure if still unreachable. The adapter's rendezvous repair ordering is consistent on static inspection. Original failure behavior does not justify silently suppressing this adapted failure.

Required diagnostic capture: first false-to-true path_failed event, not final state alone (the flag latches at ir2_observation.py:122). Save pre-observation receiver LocalFields, raw inbox packets, local_map and peer_maps, peer positions/timestamps, message ages, full GraphMemory, event/agent/position/config; save observation graph before and after repair. Recompute eligible/surplus and identify failed target; compare directed, symmetric and rebuilt symmetric reachability.

Decisive comparison: supply original Graph_generator the exact receiver nodes/global edges and local map (0->255, 1->1, 2->127), same K/radii; run its find_k_neighbor_all_nodes and A* on the same current/nearest-target pair. If both repaired graphs fail, evidence favors a partial-information extension failure rather than JAX A* parity. If original succeeds, compare adjacency and shortest-route edges to locate a port defect. Known-free flood-fill and target-cell state are supporting evidence only: both collision implementations exclude the destination pixel, so UNKNOWN at target alone does not prove no graph path.

No source edits or local runtime performed. Runtime blocker remains open pending first-failure capture and original-code comparison.

## Reachable-only and strict-control re-review — 748aaeb5f72a1dbaf3c83ca3ba2f75b8fb03cfb5

- codeQualityStatus: CLEAR
- recommendation: REQUEST_CHANGES for acceptance-script CLI exit handling below; core production semantics approved.
- User authorization for reachable_only main plus strict_ir2 control was explicitly supplied by the parent.
- CRITICAL: none. HIGH: none in this increment.

MEDIUM — Preserve the real CLI smoke outcome (scripts/phase4_acceptance.py:257).

The new check=False discards subprocess returncode. The subsequent direct run_initial can create the missing 312 results and end with status=passed even if the actual evaluate CLI crashed before producing any records; a crash after result writes is also hidden. This weakens the explicit CLI acceptance requirement. Restore check=True for the reachable-only smoke, or record the returncode and require zero in the final acceptance predicate while preserving diagnostics. Current successful logs may evidence this particular run but do not correct the gate. This finding concerns acceptance tooling, not rendezvous behavior.

Production semantics reviewed: strict_ir2 remains the HostConfig default and passes abort_unreachable=True, preserving stop-after-first-failure and failure latching. reachable_only suppresses failed-peer utility, keeps the unreachable result flags, continues later peers, and always applies own-radius suppression. The observation caller persists repaired adjacency and only latches path_failed for strict mode. Existing overflow handling remains active. The strict control receives the same map/spawn/fault/method/budget and changes only rendezvous mode. The two benchmark JSON configurations were compared structurally and differ only in rendezvous_mode.

Typed GraphStateError(ValueError) now reaches an explicit runner environment_error branch before generic ValueError, retaining existing ValueError compatibility and genuine model-error handling. The focused classification test distinguishes equal-message graph and ordinary ValueError cases.

Parallel runner: spawn avoids forking initialized CUDA state. Per-task IDs are unique and each config-hash/bucket/method group has exactly one owner, so result/temp/trace paths do not collide. Workers share runner_summary.json and may overwrite it transiently, but parent rewrites the complete sum after all workers return; no reader consumes it during execution. Exceptions prevent a passing aggregate. Final resume explicitly verifies all312 intact results and matched failure totals. No additional blocking concurrency issue found within this bounded script.

Counterexample artifacts independently inspected:

- research/evidence/phase4-graph-counterexample-20261114.json reports first failure event12, receiver0 at(6,23), peer1 at(20,18), surplus258, original rendezvous success=false.
- Parsed the paired NPZ using only standard-library ZIP/NPY decoding (no simulator/test execution). It contains10nodes; original_repair and jax_repair arrays are byte-identical. A plain graph traversal of the stored symmetric repair graph reaches9nodes from(6,23) and excludes(20,18). Receiver local_map[18,20] is2 (UNKNOWN). This independently corroborates the reported disconnected target and graph parity, though original function execution itself was performed remotely by the executor.

Skill review: new jit/vmap test uses a failed first peer followed by a reachable peer and compares utility against the existing independent oracle for the reachable peer. It would catch accidental continued abort and missing own-radius suppression; no tautological/deletion-only test or unnecessary production normalization identified.

Validation: git diff --check d23e0731 748aaeb5 passed. The reported130-test GPU XML was not yet available at review time; final313 runtime outcomes, independent replay and final report remain pending. No local runtime, production edits, or commits performed.



## Legal-action re-review — 016e121b (diff from 0baa41f8)

- codeQualityStatus: CLEAR
- recommendation: APPROVE for production code; final runtime acceptance remains pending.
- CRITICAL/HIGH: none found in this increment.
- Production blockers: none remaining.

The previous CLI gate finding is RESOLVED: phase4_acceptance.py:259 again uses check=True for the actual evaluate subprocess.

Reviewed heuristic.py legal fallback: it orders only legal positive-distance candidates by distance, then x/y and slot for deterministic ties. When Dijkstra finds a reachable frontier its selected first hop is retained; otherwise the closest legal edge is used. It waits only when there is no legal moving candidate. This fixes the prior masked-WAIT behavior without changing frontier route selection. The changed no-frontier regression now passes its selected action through actual navigate_edge and checks invalid_action=false.

MEDIUM (nonblocking test relevance): tests/benchmark/unit/test_baselines.py:221-235, test_node_zero_cannot_be_a_future_target_or_transit_node, no longer distinguishes the defect named by the test. Its only moving candidate [3,2] is now expected both for the legal fallback and for an incorrectly accepted path through nodezero. Removing nodezero exclusion would still choose [3,2]. A future fixture should offer a second legal fallback whose first hop differs from the tempting forbidden route; alternatively remove the misleading coverage claim. This is a programming/remove-ai-slops test-perspective finding, not evidence of a current production path violation.

Independent replay additions inspect all active invalid_action flags, require nonacting robots to remain stationary, and check the acting segment with original Graph_generator.check_collision against receiver knowledge reconstructed from own sensing and delivered MAP cells. They do not import production navigation as the oracle. Passing this collision check alone does not prove graph-slot legality; the separate invalid_action check covers the actual executor contract.

Evidence inspected:

- research/evidence/phase4-unit-legal-20261114.xml:36 tests,0 failures/errors/skips.
- research/evidence/phase4-unit-reachable-20261114.xml:130 tests,0 failures/errors/skips.
- research/evidence/phase4-unit-validation-20261114.xml:1 test,0 failures/errors/skips.
- research/evidence/phase4-v3-interruption-20261114.json explicitly marks62 prior records/1984events invalidated for the fallback correction, with nf_plan9 and nf_mixed6 invalid actions, and results_used_in_final_score=false. This was a real runtime defect; prior static approval did not establish absence of invalid actions.
- git diff --check 0baa41f8 016e121b passed.

Final matrix-v4 runs on source016e121b per executor;312 main tasks plus1 strict control, independent audit and final report are still pending at this review. No local simulator/tests, source edits, or commits performed.


## Static closure — a0e2039a / 30302316

- codeQualityStatus: CLEAR solely for pending final runtime evidence.
- recommendation: APPROVE (static code/test/configuration review).
- Open static findings and production blockers: none.

The nodezero test-relevance finding is RESOLVED in a0e2039a. The fixture now offers equally short legal candidates [3,2] and [2,3]; correct coordinate-first fallback selects [2,3], whereas accepting the forbidden frontier/route through nodezero would select [3,2]. Both parametrized goal/transit cases now distinguish the named regression. Independently inspected research/evidence/phase4-unit-node-zero-20261114.xml:2 tests,0 failures/errors/skips.

Compared parsed configs/evaluation/benchmark.json and benchmark_strict_ir2.json from30302316: only base_config differs (../benchmark.json versus ../benchmark_strict_ir2.json). This correctly pairs strict environment mode with its report declaration; all method and matrix definitions remain shared. No production changes since016e121b are claimed or required by these fixes. git diff --check 016e121b 30302316 passed.

Partial matrix progress is not completion evidence. Final312 main tasks plus strict control, independent replay, and final published report still require artifact review. No implementation edits or local simulator/test runs performed.


## Terminal evidence audit — 2026-11-14

This terminal decision supersedes every provisional status and pending-runtime condition above. Earlier findings and failed-run history remain for traceability.

- codeQualityStatus: CLEAR
- recommendation: APPROVE
- reportPath: .omo/evidence/phase4-baseline-code-review.md
- blockers: []
- Approved scope: Phase4 limited functional acceptance and publication of its code/report/evidence on the existing audit branch. This is not approval of formal benchmark completion, policy quality, training, or performance claims.

### Findings by severity

CRITICAL: none remaining.
HIGH: none remaining; full result-task binding and mandatory finite measured metrics were fixed and reviewed.
MEDIUM: none remaining; CLI smoke exit handling and meaningful nodezero regression were fixed and reviewed.
LOW: none requiring action.

### Evidence independently inspected and cross-checked

Read the completed research/BASELINE_EVALUATION_REPORT.md, primary phase4-20261114.json, raw phase4-run-20261114.json, phase4-episodes-20261114.csv, release/source manifests, stages/test summary and all listed test XMLs. Verified every listed evidence file's SHA256 and byte length at audit time; no mismatch. The raw run acceptance object matches the primary evidence exactly.

Recomputed from CSV:312 unique result IDs;8 methods with39 each;39 scenario identity groups each containing all8 methods;288 paired,16 population,8 zero-budget rows. All312 are timeout, with exactly n_agents*max_rounds active events each, totaling10112. Emitted and spent bytes each sum270828; delivered payload bytes sum232513. All eight zero-budget rows have zero emitted/spent bytes. Every row references a known configuration hash, and recomputing every canonical configuration hash from the raw run's configuration map matches its key.

CSV recorded-versus-independent coverage maximum difference is2.7472499164460373e-08; AUC maximum difference is2.4020234912214278e-08, within the reviewed independent replay tolerance. Raw run records passed independent coverage of312 tasks in disjoint281+31 shards with named remote shard hashes, no execution-error records, and zero invalid actions. Positive28-episode and known-invalid negative gates are retained. Raw event traces and full oracle shards intentionally remain at the documented remote paths; this reviewer audited the published compact evidence and implementation, did not rerun the simulator or independently replay remote traces.

Resume evidence is requested312/resumed312/executed0/failures0. Initial304 executions plus8 CLI-resumed records account for all312. The strict control records exactly one environment_error with metrics=None, while its named reachable-only counterpart is present in the CSV as timeout. The main and control outcomes remain separate. Zero-budget trajectory equality is recorded by the acceptance script's explicit per-motion array comparisons; published CSV independently confirms zero costs.

Report scope agrees across artifacts: requested_manifest_complete=true, requested/completed312, pending0, corrupt[], formal_complete=false. All eight conditional CIs are absent. All outcomes being timeout is disclosed prominently; no exploration-success, algorithm-ranking or full2,294,784-episode completion claim is made.

Rebuilt test outcomes directly from XML in stage order, applying only the documented three-case rename:278 raw unique names,275 canonical cases,275 latest passes, no remaining failure/error/skip. Historical failures and invalidated v3 data remain disclosed. Runtime stages record the conda ex interpreter, CUDA:0/GPU arrays and imports from the deployed source path; the separate probe artifact contains the actual assertions used.

Compared233 deployed source-manifest entries against local files: after CRLF-to-LF equivalence, only README.md and .gitattributes differ due to final publication documentation/CSV line-ending changes. All production, configuration, scripts and tests match the deployed manifest. Raw run also records233 post-run remote hash matches. No production changes after016e121b were introduced by later test/configuration-publication commits.

### Review perspectives and scope limits

Programming and remove-ai-slops perspectives ran throughout and were repeated for each targeted fix. No unresolved test-overfit, redundant internal normalization, unjustified abstraction, or typed-boundary finding remains. The legal fallback and reachable-only extension are explicit behavior, supported by distinct regressions and the original-code disconnected-graph counterexample. Strict IR2 remains the default/selectable control.

The report preserves pretrained-weight provenance, unknown upstream split exposure, no new training, requested-versus-formal completion, one-map-per-stratum CI limits, dropout-oracle boundary, and timing exclusions. Therefore the delivered claims are appropriately bounded to functional acceptance. No local simulator/test execution, production edit, commit, or push was performed by this reviewer.
