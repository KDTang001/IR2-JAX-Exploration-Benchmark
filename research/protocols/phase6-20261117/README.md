# Phase 6 measurement protocol sources

These small text archives preserve executed orchestration bytes. Large GPU traces,
optimized HLO, buffer assignments and memory profiles remain under
`/home/user/Tz/acceptance/phase6-prep-20261117` on the target server. Their sizes and
SHA256 identities are in the compressed public profile/comparison evidence.

| Archive | Role | SHA256 |
| --- | --- | --- |
| `phase6_comparison_supervisor.executed.py.txt` | Actual six-case paired supervisor | `2083b6a9282c389335f0470ddb471a1c5fadff271b0ceb3bb82c3d85aae36926` |
| `phase6_mini_supervisor.executed.py.txt` | Actual two full-episode workers, GNU timeout and memory recording | `7f9b090aa9cd635e29e5deb2639e2a8e3ab35e4e694d623954548a93d9ec2fc6` |
| `after_comparison.py.txt` | Gate: six successful cases and 22 exact pairs before the two-task cost subset | `7bb06c01ac836d8ff4d30a033f946cd6573403f75c88dcdfceab87892cc06087` |

The actual GPU workers are published as normal scripts:

- `scripts/phase6_profile.py`: SHA256 `16a78c099ecf44070ea3c1d9f116f4aae393aeaab6d4cf0c8604f980a3671bc1`.
- `scripts/phase6_compare.py`: SHA256 `29cf37e50ad511842214edb559ca7d5ce7dabc1cb4c9d519948751563ceefb76`.
- The latest `scripts/phase6_profile_supervisor.py` additionally rejects reuse of
  the closed window; its executed predecessor above is preserved separately.

`close_window.py.txt` and `export_evidence.py.txt` are the actual post-run stdlib
programs used to verify completion and export small identities; they do not import
JAX or start another GPU job. `.gitattributes` preserves archive bytes on Windows.
The original baseline controller was later extended; actual baseline worker argv,
source inventory, map identities, runtime versions and process isolation are in
`research/evidence/phase6-profile-executions.json`. The unchanged worker source is
pinned above; do not mistake unused legacy argv copied into the private request
for the recorded commands that actually ran.

For a new reproduction, export base `d3d6c5f` to a new `source-baseline`, export
package commit `736ca08659d5753c7a3b9c6aab471004fb0fea82` to `source-final`, and
verify their file inventories. Copy the normal measurement scripts and construct
the six-case request from the published execution records, remapping map/checkpoint
paths while preserving file/content hashes, N, B, seed7, four-event segments,
two warmups and five repetitions. Use a new output root and a separately authorized
window. Never delete an old ledger to extend its deadline. The paired worker uses
32 events and ten alternating timing samples, plus the declared short fault probes.

The frozen pilot planner is separate and planning-only. It accepts a user-specific
source registry/checkpoint path. Remapped absolute paths change manifest **bytes**;
scenario/result identities remain content/config/source-bound. The published frozen
manifest retains its original server paths and digest. The two compact full-episode
traces and checksummed result envelopes have a public artifact index, so their event
counts, byte balances and coverage AUC can be checked without downloading all profiles.
