# Release merge preparation

Status on 2026-11-18: **plan only; neither PR nor main was merged**.

## Verified dependency

[PR #1](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/pull/1) is Draft
from `audit/benchmark-foundation-2026-11-12` into `main`; the public PR page was
checked directly. [PR #2](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/pull/2)
is Draft from `codex/phase6-prep` into that audit branch; metadata reported no
merge conflict at the observed 7b36a288 head. Mergeability is not a release-test
result or permission to merge.

The [read-only remote-ref snapshot](evidence/resource-pilot-v2-c1024-20261118/release-refs.json)
confirms main at `ca663f725c335351ed5c8bec45960aa1478d5e36`, audit/PR1 at
`d3d6c5fdec831de3076b7ede70e25b72108e59bf`, and PR2 at
`7b36a288b0f9d190da9105920803254657cd616c` before this continuation is uploaded.
The local merge base of audit and the continuation is exactly d3d6c5f, so the
dependency order follows from the branch ancestry, not just their PR numbers.
Later pushed heads require a fresh metadata check before any merge.

## Concrete integration order for later approval

1. Freeze the exact proposed PR2 head, confirm all changed environment/package
   source bytes against accepted evidence, and review supported presets and new
   measurement-only changes. Preserve every old manifest/error/result artifact.
2. The separate v2 30-task probe now passed scoped execution/audit acceptance.
   Its 2KiB/mixed Burst cohort had zero in-window emitted bytes; retain that
   exposure limit in release notes. C1024 remains a candidate for wider
   conditions; do not call all 240 or the paper matrix done.
3. After explicit merge approval, integrate PR1 first. Prefer a merge preserving
   its audit ancestry if repository policy allows it; do not publish/tag that
   intermediate tree as the completed release because PR2 contains metadata/OOM
   corrections. A repository requiring squash needs an explicit revised plan.
4. Refresh main, retarget PR2 to main, inspect the resulting diff/conflicts and
   exact source identity, then integrate PR2. Do not blindly merge PR2 into the
   stale audit branch and assume main has been updated.
5. Build/install the release wheel from the final integrated tree, verify its
   actual import/source/distribution identity and doctor output. Reuse exact
   unchanged source acceptance; rerun only checks affected by any conflict fix.
   Publish the release identity only after this final-tree check.

No retarget, Draft-state change, merge, force-push or default-branch modification
was performed as part of preparation. The package's 111 source files are unchanged
by the v2 work; planner/version guards and optional graph telemetry are outside
the installed environment/model package. Existing Phase 6 wheel evidence remains
historical evidence, not a newly executed final-main installation check.

## Capacity and paper boundary

C400 stays in `configs/benchmark.json` as the original-capacity control. C1024 is
the common candidate in the next pilot, paper protocol and train-only SAC design.
The old formal `configs/evaluation/benchmark.json` remains a C400 declaration;
it must not be silently rewritten or executed as a C1024 paper protocol.
Release notes must name supported ranges and explicit configuration selection.

Publishing usable benchmark code can precede the paper experiments. Freezing
paper capacity/map count still needs coverage of zero budget, N2/N6, other buckets
and the unmeasured budget/fault/method combinations, with costs and failure rules.
The train-only SAC proposal requires separate implementation and training-memory/
gradient-replay acceptance before any training approval. No world-model work or
training is included in this merge plan.
