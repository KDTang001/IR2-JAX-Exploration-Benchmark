# Release recovery final gate

recommendation: APPROVE
blockers: []

originalIntent: Close R1 for frozen PR1/PR2 integration readiness by independently installing the exact candidate wheel and CPU lock and exercising the installed CLI.
desiredOutcome: Evidence sufficient to seek explicit merge approval; no merge authorization, final-main release certification, training, GPU acceptance, or paper completion.
userOutcomeReview: R1 is closed for candidate 298e625d21deb205a095ba78643a5a3d6dd4976c. The new isolated installation exercises the intended installed surface and preserves earlier failed attempts honestly.

## Criteria and evidence

R1: Same frozen source/hash lock, independent CPU installation, actual origins, runtime-only doctor, two-task evaluate/resume/report, truthful provenance.

All paths below are relative to the repository. Recovery prefix: research/evidence/release-install-recovery-20261119/.

- Independently recomputed SHA256 for all 122 files in bundle-index.json: zero mismatches. Read transfer-receipt.json and manual-qa.json; counts were not used as substitutes for raw inspection.
- Compared all 111 release_manifest source hashes directly against git show 298e625:src/ir2_benchmark/<path>: zero mismatches. HEAD changes since 298 contain report/evidence/attributes only.
- Read hashed-install/commands.json, identity.json, installed-origins.json, cpu-dependencies.log, pip-check.log and benchmark-wheel.lock. The interpreter prefix is the new cpu venv, include-system-site-packages=false, observed imports are inside it, all install/check commands exit zero. pip --require-hashes installs the original wheel and original CPU lock. pip-check observes 38 compatible distributions.
- hashed-install/installed-origins.json contains the actual declared SHA 0c2d8e348b391ab769aa707f9deaf4f1cfba29637696391dccd3a73c28d0aff4. hashed-install/reported/release_manifest.json reports that same accessible archive hash, installed_distribution and import_path_match=true; code_revision and runtime_lock remain null. Unrelated cwd Git initialization is visible in commands.json and cannot become package revision in the reviewed release.py.
- Read hashed-install/doctor.json: CPU, correct inference/scan, optimizer_updates=0, null losses, runtime_smoke_only.
- Read hashed-install/evaluate-resume.log and evaluate-second-resume.log: 1 executed/1 resumed then 0 executed/2 resumed, zero failures. Original archived acceptance protocol checks prefix result bytes remain unchanged. Both raw result records are legitimate four-round timeout outcomes. Independently recomputed both referenced NPZ hashes: match.
- Read hashed-install/reported/report.json: requested_manifest_complete=true, limited=true, formal_complete=false. Read hashed-install/result.json and initial/postrun-verification.json: completion, 37 pins verified, no owned active processes. Inspected controller/protocol implementation behind these assertions.
- initial/result.json remains complete=false. initial installed metadata failure is distinct from final hashed installation; no production patch reclassifies it. initial/controller-probe.json records the expected timeout with retained output and inactive descendant. Reviewed streamed file logging/process-group termination in initial/recovery.py.txt.
- Reviewed research/RELEASE_PREFLIGHT_REPORT.md, prior merge-rehearsal.json, history-integrity.json, self-review.json, lane-ledger.json, cpu-final/source-tests.log and the reused test source. Prior 32 passing tests remain source-run evidence only; they are not represented as a fresh installed-wheel test suite.

## Direct programming and remove-ai-slops pass

Loaded both skills and Python criteria; directly inspected production release.py, errors.py, runner.py, relevant diff, test_release_metadata.py and batch runner assertions, plus archived recovery controllers. No new production/test edits exist in this recovery. No deletion-only or requested-removal tests, gratuitous production extraction, new parsing/normalization layers, or assertion edits made to force success were found. Real CLI/origin/resume runs exercise behavior beyond internal mocks. Existing wheel tests distinguish absent/mismatched/matching hashes rather than pin prose. Existing inventory digest assertion is partly self-derived: it proves serialization consistency, not independent inventory truth; the real import checks and separate postrun inventory support R1. This is a nonblocking test-quality note.

Programming notes: archived one-off drivers use dense statements, globals and exec of text slices; these are maintainability limitations if promoted into reusable tooling, not a violation of R1. Top-level broad exception handling records failure and re-raises rather than hiding candidate failures. No request to refactor frozen production is warranted.

Prior self-review.json does not explicitly enumerate both skill perspectives or all overfit categories. This direct pass supplies the missing perspective coverage; report absence is not treated as a candidate failure.

## Evidence gaps and limitations

- No live remote rerun, SSH, installer, training or GPU action performed by this reviewer. Review reproduces local artifact/source hashes and checks remote raw execution records as authorized.
- Downloaded wheels/venv are not in the transferred raw-evidence bundle; install commands, controller hash checks, origins and release outputs record their remote verification. No claim of independently repeating remote installation locally.
- No new certification of all 240 tasks, paper matrix, train-only fairness, broad C1024 maps/N/buckets, or exclusive GPU timings. Historical Phase1–6 acceptance remains limited and source-pinned.
- Earlier NO-GO report is historical and still accurate for its attempt. A new closure statement should accompany it before presenting current readiness.
- Actual final main must be rebuilt and verified after explicit approval and actual integration. Housekeeping App switch/archive remains external and nonblocking.
- No notepad path was supplied or found. No loop plan exists: wrapper initially failed; direct CLI retry returned ULW_LOOP_PLAN_MISSING. This report therefore uses the required fallback path.

Checked artifact paths additionally include hashed-install/hashed_install.py.txt, initial/prior-protocol.py.txt, initial/recovery.py.txt, initial/controller-probe.json, hashed-install/evaluated/results/*.json and hashed-install/evaluated/events/*.npz. Production remained unchanged; only this required reviewer report was written.
