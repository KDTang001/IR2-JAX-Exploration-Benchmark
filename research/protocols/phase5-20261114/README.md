# Archived Phase 5 execution helpers

These are byte-for-byte copies of the external helpers used on the target server.
Their hashes are bound in the published evidence; `.py.txt` marks these as source
archives rather than current executable entry points. They add no benchmark environment
or policy behavior. Fixed absolute paths, process IDs and deadlines describe that
specific acceptance run. Use [the portable wheel workflow](../../../docs/phase5-reproduction.md)
and the public CLI for a new installation or evaluation.

- `performance_v2.py.txt`: versioned four-event measurements, phase checkpoints,
  fresh worker processes, host RSS and GPU process sampling. Its hash is recorded
  in the [requested manifest](../../evidence/phase5-performance-v2-requested-manifest.json).
- `aggregate_long.py.txt`: read-only frozen-event, low-budget, recovery and file-size
  aggregation after independent original-IR2 replay.
- `post_long_supervisor.py.txt`: requires all 78 intact episodes, independent replay,
  public resume/report, aggregation and device quiescence before the performance
  gate. The last two hashes appear in the [long-run gate](../../evidence/phase5-long-gate.json).
- `resume_performance_v2.py.txt`: keeps the original first-pair results and budget
  unchanged, recording a separate scheduling amendment before dispatching the
  remaining 52 cases within the original two-day total resource window. See the
  [amendment](../../evidence/phase5-performance-v2-schedule-resumption-v1.json).
- `recover_performance_v2.py.txt`: preserves the six existing records and the
  fatal idle-preflight evidence, then waits and resamples on the three exact
  resource-preflight failures without changing thresholds or timed functions.
  The fixed request remains 54; the declared remaining order prioritizes matched
  B1 cases and GPU scaling within the same final deadline. See the
  [recovery declaration](../../evidence/phase5-performance-v2-idle-recovery-schedule.json).
- `supplement_performance_v2_cpu.py.txt`: after the user continued on November 17,
  runs only the two previously unexecuted CPU cases in separate storage. The
  original 52 case files and closed-window metadata are hash-checked unchanged.
  The frozen measurement wrapper/inputs are reused; only storage/progress scope
  changes. Each case is capped at 30 minutes, with a 90 minute total supplementary
  window including identical idle checks. See the
  [supplement declaration](../../evidence/phase5-performance-v2-cpu-supplement-declaration.json).

The supplement ended on November 17 at 10:44:52 CST with two per-case deadline
stops. All 54 original identities now have execution records, with 37 complete
five-phase measurements, six logged GPU OOMs and 11 CPU deadline stops. See the
[combined ledger](../../evidence/phase5-performance-v2-combined-summary.json)
and [supplement integrity proof](../../evidence/phase5-performance-v2-cpu-supplement-integrity.json).
Neither the original resource window nor incomplete timing phases are rewritten.

The measurement wrapper calls the same committed throughput functions; checkpoint
writes and resource sampling occur outside each timed callable. Whole-case memory
peaks include compilation and all phases. The four-event segments are initial
workloads, not mature full episodes or complete mixed-message cycles.
