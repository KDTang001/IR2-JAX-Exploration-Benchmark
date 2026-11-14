# Phase 3 code review

codeQualityStatus: WATCH
recommendation: APPROVE
blockers: []

Reviewed dc7bf34ed82ee7290bdcc0db46ba8923f96f150a..408595181968dfe6548f5342411b29b155704bf9: audit CLI, acceptance tests, LF attributes, fixed validation artifacts, and implementation plan. Goal: full original/procedural dataset acceptance, independent geometry/hash/split checks, reproducible generation, train-only sampling and independent frozen validation. No production or test files were modified by this review.

## Findings

- CRITICAL: none.
- HIGH: none.
- MEDIUM: none.
- LOW: scripts/audit_map_dataset.py:1 lacks the programming skill's script usage/dependency metadata and the remove-ai-slops standalone-checker SIZE_OK explanation. This cohesive acceptance CLI is justified; a module split is not needed for this goal. Record its deliberate standalone exception and reproducible invocation when next editing it. No demonstrated runtime defect.

## Skill perspectives

Loaded programming/SKILL.md, programming/references/python/README.md, and remove-ai-slops/SKILL.md from the installed omo 5.1.38 skills. Reviewed tests and audit code for deletion-only assertions, requested-removal tests, tautologies, implementation constant mirroring, brittle prose tests, untyped escape hatches, needless abstraction and unnecessary production parsing. None found. Literal geometry and counts assert observable acceptance contracts. Independent hash encoding and SciPy/Pillow recomputation are justified oracles; generator reuse tests reproducibility. No production parsing was introduced. Only the low script-hygiene deviation above remains.

## Evidence inspected

Local evidence root: C:/Users/15491/AppData/Local/Temp/ir2-phase3-results-20261114. Remote origin: /home/user/Tz/acceptance/phase3-20261114.

- results/dataset.json and dataset.log: 24,199 images, 6,900 procedural seed reproductions, 13,877 retained unique canonical maps, zero cross-source overlap, empty issues; five manifest digests recorded.
- results/run.json: actual conda ex Python, H100, GPU backend, isolated source import, successful full audit and initial failing test run. Original failure remains preserved.
- results/tests.log, phase3.xml, tests-normalization.log, phase3-normalization.xml, tests-final.log and phase3-final.xml: initial 34 passed / 1 failed checksum; intermediate preset source identity mismatch; final same failed test passes. This supports 35 unique passing cases across runs, not a single final all-green suite.
- source-final-manifest.json: reviewed commit identity and final validation SHA-256 0112548680511201eb0ef714429761da90de16951ade41578341111f7cb6b3b1. results/normalization.json records CRLF-to-LF-only normalization and unchanged dataset PNGs/manifests. results/phase3-final.runtime.json records conda ex and GPU import origin.
- supplement.py and results/supplement.json: actual directory/manifest origin and family checks, hash-first validation selection, 13 actual train source/family/bucket strata loaded on CUDA, occupancy/scenario/bucket corruptions rejected, same 25 validation maps/seeds/N/budget/round limit preserved.
- validation-before.json: production validator rejected the stale configuration before migration.

The independent main CLI trusts manifest origin/family and procedural split labels; the inspected supplemental directory checks close that acceptance gap for this dataset. Future pools containing duplicate_occupancy generation retries would require pool-aware reproduction rather than generate_map(requested_seed) alone; no such mismatch occurred in this fixed dataset.

## Scope and limits

No local runtime tests or training were executed, respecting the remote-only verification constraint. This reviewer inspected recorded remote artifacts rather than rerunning SSH jobs. Lint/typecheck/security results were not supplied in this review bundle and are not claimed. Aggregate research report was still being authored and is outside this reviewed commit. No notepad path was supplied. The toolkit wrapper failed; direct Node CLI status returned ULW_LOOP_PLAN_MISSING, so this required report uses the fallback evidence path.
