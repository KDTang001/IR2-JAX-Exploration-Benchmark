# Communication-limited exploration benchmark

**2026-11-10 cleanup:** All experiment runs, checkpoints, replay outputs and recordings were removed from the local and server project at the user's request. No previous run can currently be resumed. Map datasets/manifests, source, configuration and tests remain. Run descriptions below document historical settings, not an active training job; see the [current discussion](../spec/spec-design-benchmark-status-and-discussion.md) before starting a new experiment.

The independent `ir2_benchmark` package runs alongside the original IR2 code. The original graph reference now uses deterministic coordinate ordering for equal-distance neighbors, set traversal and A* ties, as approved for paired comparison. The approved runtime is the existing Linux server conda environment `ex`, Python 3.11, JAX 0.4.35. Follow [server-running.md](server-running.md) for environment preparation; do not replace that environment with a workstation training installation.

## IR2 navigation and observation contract

Use `configs/receiver_ir2_graph.json` with `configs/a/receiver_ir2_graph.json` for the IR2 graph baseline. `grid_path` and `ir2_one_hop` remain explicitly named legacy modes; neither is the faithful IR2 graph implementation. Existing legacy checkpoints cannot be resumed under the new configuration.

The active graph-policy training configuration uses learning rate `1e-5`, matching the scale in original `parameter.py`. An initial `3e-4` run exceeded the PPO KL target and subsequently saturated the pointer, so its checkpoint was discarded. Reusing another legacy method file requires setting its graph-policy learning rate explicitly; matching observation shapes alone does not validate its training settings.

In `ir2_graph`, actions select one outgoing graph edge and execute that complete edge. The actor receives the original full graph: six node features (normalized x/y, exploration utility, rendezvous utility, visit guidepost, occupancy), adjacency, current node, neighbor indices and padding masks. The six-layer graph attention encoder, decoder and pointer follow the original policy. Budget and received-message fields feed the separate communication extension; this baseline does not add their scores to the original navigation logits.

The port retains original ray sensing, frontier reduction, utility visibility sets, local/global graph maintenance, route history, graph merging and offshoot rules. Coordinate sorting resolves original implementation-dependent ties consistently in both references. The original stage-2 scale is selected explicitly in the new configuration. Graph capacity exhaustion or failed rendezvous repair raises an error; neither silently substitutes a truncated graph.

The benchmark retains its own rewards, spawn schedule, termination and paid-message protocol. Rendezvous uses the original utility formula with receiver-owned map caches and received-contact evidence. A peer with only PLAN/position messages has no received map belief: its map-surplus utility stays zero until an actual map payload arrives. The original protocol exchanged pose and map together, so this missing-map state is an explicit benchmark extension. Thus navigation and observation rules are aligned given the same knowledge; the complete benchmark is not claimed to reproduce the original communication MDP. Normal links, random loss and burst disconnection remain separate conditions.

`configs/receiver_ir2_graph_fusion.json` enables teammate graph fusion and must be paired with a communication method that sends MAP_DELTA, such as `configs/a/receiver_map_communication.json`. Only successfully delivered map payloads add teammate map/graph knowledge. PLAN messages cannot import that knowledge. Own route history is retained with either fusion setting.

```bash
conda activate ex
cd ~/Tz/ir2-benchmark
ir2-benchmark doctor --require-gpu --output artifacts/doctor.json
python -m pytest tests/benchmark -q
```

`configs/micro.json` and `configs/default.json` are legacy development configurations, retained for older experiments. The active IR2 graph baseline uses `configs/receiver_ir2_graph.json`: native geometry, radius 80, 512 rounds, 30 neighbor slots, 131072 bytes per robot and Normal links. One-hop movement is not clipped to the legacy movement cap. Micro results remain development checks and are excluded from formal completeness claims.

## Map preparation and source registry

```bash
ir2-benchmark maps audit --root DungeonMaps --output artifacts/maps/ir2_manifest.jsonl
python scripts/generate_benchmark_maps.py
```

Map hashes, rejected generation seeds, actual image paths and source splits are persisted in manifests. A failed map is not replaced. Native occupancy and its spawn anchor are checked during evaluation; changing the anchor changes the scenario identity. Padding adds occupied cells on the right/bottom and does not resize the original map.

The pool script reproduces the deployed 6000/300/300/300 counts with initial seeds 410000/510000/610000/710000, excludes canonical duplicates against IR2 and every previously generated split, and selects only narrow widths for OOD. It refuses to overwrite an existing pool. The single-pool `maps generate` command remains available for development; separate invocations do not by themselves enforce cross-pool exclusions.

Create `artifacts/maps/sources.json` with the following shape. Both `root` and `manifest` paths are resolved relative to this JSON file; absolute paths are also accepted.

```json
[
  {"source":"ir2","root":"../../DungeonMaps","manifest":"ir2_manifest.jsonl"},
  {"source":"procedural","root":"procedural/train","manifest":"procedural/train/manifest.jsonl"},
  {"source":"procedural","root":"procedural/validation","manifest":"procedural/validation/manifest.jsonl"},
  {"source":"procedural","root":"procedural/test","manifest":"procedural/test/manifest.jsonl"},
  {"source":"procedural","root":"procedural/ood","manifest":"procedural/ood/manifest.jsonl"}
]
```

Training selects only `train`; model selection selects only `validation`; formal evaluation selects retained `test`/`ood`. The formal sampler dispatches a deterministic balanced schedule across native `(map bucket,N)` pools, sharing neural parameters across pools. Each 108-entry schedule cycle balances source 50/50, IR2 families, procedural templates/buckets and N=2/4/6. It never pads all maps into a single substitute training geometry.

## Real development runs

```bash
ir2-benchmark rollout --config configs/micro.json --seed 7 --output artifacts/environment-smoke
ir2-benchmark train-a --suite micro --config configs/micro.json --seed 0 --events 4096 --output artifacts/a-smoke
ir2-benchmark train-b --suite micro --config configs/micro.json --seed 0 --events 4096 --output artifacts/b-smoke
ir2-benchmark evaluate --checkpoint artifacts/a-smoke/checkpoint --suite micro --output artifacts/evaluation-a-smoke
ir2-benchmark evaluate --checkpoint artifacts/b-smoke/checkpoint --suite micro --output artifacts/evaluation-b-smoke
ir2-benchmark report --results artifacts/evaluation-a-smoke --output artifacts/report-a-smoke
ir2-benchmark throughput --config configs/micro.json --steps 32 --output artifacts/throughput-smoke
```

The rollout produces `summary.json`, `events.jsonl`, local-map trajectory images and a physical-union trajectory image. Evaluation creates a fresh development map by default (seed 10000), executes the restored policy in the real environment and saves checksum-bound per-task results/traces. A/B use stochastic decisions with fixed policy seeds; the heuristic is deterministic conditional on its scenario. Policy and fault random streams are independent.

### GIFs from actual training

The training CLI records episode indices 0, 201, 402, and so on into each run's `gifs/` directory. `--gif-every-episodes 1` records every episode; `--gif-every-episodes 0` disables capture. `--gif-frame-stride-rounds` defaults to 4 and `--gif-max-frames` to 129, including the initial and final frames. Programmatic training defaults to `recording=None` so existing callers remain unchanged.

```bash
ir2-benchmark train-a --suite micro --config configs/micro.json --seed 0 --events 4096 --gif-every-episodes 1 --output artifacts/a-gif-smoke
ir2-benchmark train-b --suite micro --config configs/micro.json --seed 0 --events 4096 --gif-every-episodes 1 --output artifacts/b-gif-smoke
```

Each selected episode saves one merged-belief GIF, one local-belief GIF per active robot, each view's last PNG, and a shared JSON metadata file. Unknown space is gray, known free space white, known walls dark, frontiers red and executed routes/robots colored. Gray dashed links in the merged view show actual paid deliveries since the preceding frame. Local views use only that receiver's own map plus paid-message overlays and show peer positions only when received; other robots' true positions/routes are reserved for the explicitly diagnostic merged view. The HUD names the actual `warmup` or `train` phase and actor optimizer step, physical coverage, round and byte balances.

Capture retains one initial environment snapshot and compact actual standard-codec actions/evidence. A cached exact environment replay reconstructs executed motion under the recorded navigation mode, checks positions/goals/balances/bytes/deliveries/coverage/terminal flags against the captured real trace, and never calls the actor or adds training events. IR2 graph mode draws the executed one-hop edges; legacy grid-path mode reconstructs BFS cells. No time-indexed native map stack or temporary frame PNGs are produced. Metadata includes the actual map hash, raw fault key, training seed, event range, parameter-version chunks and rendering time. A resumed mid-episode snapshot and a training-end prefix are marked `partial`; their pre-cutoff route history is not invented. These are diagnostic training artifacts, separate from fresh evaluation and formal scores.

Only explicit CLI method flags override a method file. Method-file values override suite defaults; unknown fields fail. A complete A method file contains formal quotas, so a reduced development run must explicitly supply its reduced suite/events/warmup:

```bash
ir2-benchmark train-a --config configs/micro.json --method-config configs/a/fixed_zoh.json --suite micro --events 4096 --warmup-events 256 --seed 0 --output artifacts/a-fixed-zoh-smoke
ir2-benchmark train-b --config configs/micro.json --method-config configs/b/fixed_zoh.json --suite micro --events 4096 --seed 0 --output artifacts/b-fixed-zoh-smoke
```

`experiment.json` records the effective settings, seed and configuration hash. A resumes from its checkpoint file; B resumes from its run directory. Restore with the original seed and unchanged method/environment/manifest settings. Completed evaluation tasks resume only when their result and trace checksums match. `evaluate --task-manifest ... --retry-errors --output ...` retries saved execution errors; ordinary exploration timeouts remain normal recorded outcomes.

## Fixed validation and formal training

Freeze the same model-selection workload before either method starts. The default subset selects the first five occupancy hashes in each of five source/family strata, one fixed scenario per map, yielding 25 validation scenarios. Selection metadata records the rule, hash, seeds and full environment configuration.

```bash
ir2-benchmark validation-manifest --config configs/receiver_ir2_graph.json --map-sources artifacts/maps/sources.json --maps-per-family 5 --seed 0 --output artifacts/maps/validation-scenarios.jsonl
ir2-benchmark train-a --suite formal --config configs/receiver_ir2_graph.json --method-config configs/a/fixed_zoh.json --map-sources artifacts/maps/sources.json --validation-manifest artifacts/maps/validation-scenarios.jsonl --seed 0 --output artifacts/formal/A_fixed_zoh_seed0
ir2-benchmark train-b --suite formal --config configs/receiver_ir2_graph.json --method-config configs/b/fixed_zoh.json --map-sources artifacts/maps/sources.json --validation-manifest artifacts/maps/validation-scenarios.jsonl --seed 0 --output artifacts/formal/B_fixed_zoh_seed0
```

Formal defaults use two million effective real events per trained repetition; warmup counts toward this quota. Each 100000-event milestone evaluates the fixed validation workload using actual environment transitions and selects the best mean coverage AUC. `checkpoint` retains resume state; `best/checkpoint` is selected for independent evaluation. Best metadata distinguishes the selection event from the run's actual total training events. A changed validation manifest is rejected. The core matrix repeats the eight configurations in `configs/evaluation/core.json` for all five declared training seeds; these commands alone do not complete that matrix.

## Paired evaluation and reports

For one selected checkpoint:

```bash
ir2-benchmark evaluate --suite formal --checkpoint artifacts/formal/A_fixed_zoh_seed0/best/checkpoint --map-sources artifacts/maps/sources.json --output artifacts/formal-evaluation
ir2-benchmark report --results artifacts/formal-evaluation --map-sources artifacts/maps/sources.json --output artifacts/formal-report
```

To dispatch multiple trained repetitions, supply `--method-manifest` with a JSON array of records such as `{"checkpoint":"/absolute/run/best/checkpoint","route":"A","training_seed":0}`. The actual communication/estimator settings and event counters are read from each checkpoint's metadata. `--plan-only` persists `tasks.jsonl` without executing it. Run that exact workload later using `evaluate --task-manifest artifacts/formal-evaluation/tasks.jsonl --output artifacts/formal-evaluation`.

Evaluation defaults to `--batch-size 1`. Set an explicit batch size to run compatible independent scenarios through the same cached `jit(vmap(...))` kernel:

```bash
ir2-benchmark evaluate --task-manifest artifacts/formal-evaluation/tasks.jsonl --batch-size 2 --max-pending-tasks 128 --output artifacts/formal-evaluation
```

Grouping uses the native map bucket, environment configuration and route's inference structure. Map arrays, checkpoint weights, spawn positions and independent policy/fault keys are dynamic inputs; compatible checkpoints can share a batch. The runner retains at most `--max-pending-tasks` pending metadata records and one batch window of cached parameter bundles. Checkpoint and metadata hashes are verified when loading cached weights. Partial groups run at their actual size when the pending limit is reached or the manifest ends. Task IDs, per-task traces and resume checks are unchanged. A failed map/checkpoint lane is recorded separately before computation; a device batch failure is recorded for every affected lane without a serial retry or OOM fallback.

The core suite uses every retained map, three scenario seeds per map, the declared N/budgets, and separate Normal/Dropout/Burst conditions. `--limit-scenarios` creates explicit `limited=true` results. Scenario identity binds occupancy, anchor, split, spawn/fault seeds and complete configuration; result identity also binds the checkpoint. Older task manifests lacking the anchor field must be regenerated rather than silently reusing their results.

Reports include `report.json`, `report.md`, episode/failure/cost/cold-start tables, `condition_scores.csv`, plan-quality tables and paired recovery comparisons. Primary coverage/budget figures keep split, N and fault probability/duration separate; coverage-over-time also separates byte budgets. Overall method scores aggregate only core per-robot conditions. Scores average scenarios within maps, maps within families and families within each training seed. Confidence intervals resample training seeds/maps. Runtime/model/map/OOM errors remain separate from exploration timeout; missing success times and missing plan predictions remain missing.

### Budget and auxiliary diagnostics

`core.json` retains the eight learned methods and per-robot budgets. The separately declared auxiliary blocks use Normal conditions, N=2/4/6 and the same retained test/OOD maps and scenario seeds. Fixed-team and unlimited diagnostics reuse `A_learned_predict` and `B_learned_predict` weights. The no-neighbor-plan controls are `A_learned_none` and `B_learned_none`; the real-replay control is `B_real_updates_learned_predict`. Each learned control still requires five full-quota trained repetitions. The nearest-frontier heuristic has training seed -1 and requires no training quota. Zero communication reuses measured per-robot budget-0 results, without another rollout or algorithm.

```bash
ir2-benchmark train-a --suite formal --config configs/receiver_ir2_graph.json --method-config configs/a/no_neighbor_plan.json --map-sources artifacts/maps/sources.json --validation-manifest artifacts/maps/validation-scenarios.jsonl --seed 0 --output artifacts/formal/A_learned_none_seed0
ir2-benchmark train-b --suite formal --config configs/receiver_ir2_graph.json --method-config configs/b/no_neighbor_plan.json --map-sources artifacts/maps/sources.json --validation-manifest artifacts/maps/validation-scenarios.jsonl --seed 0 --output artifacts/formal/B_learned_none_seed0
ir2-benchmark train-b --suite formal --config configs/receiver_ir2_graph.json --method-config configs/b/real_updates.json --map-sources artifacts/maps/sources.json --validation-manifest artifacts/maps/validation-scenarios.jsonl --seed 0 --output artifacts/formal/B_real_updates_learned_predict_seed0
ir2-benchmark evaluate --route heuristic --suite formal --config configs/receiver_ir2_graph.json --suite-config configs/evaluation/auxiliary.json --map-sources artifacts/maps/sources.json --plan-only --output artifacts/plans/heuristic
```

`configs/evaluation/team_budget.json` requests team totals 8192/32768/131072 bytes. Initial equal balances are `floor(total/N)`; N=6 receives 1365/5461/21845 bytes each, with two bytes unallocated in every case. N=2/4 leaves no remainder. `costs.csv` keeps requested team bytes, actual allocated team bytes and unallocated bytes separately. Team curves use requested total bytes on the horizontal axis. `unlimited.json` is explicitly labeled `budget_mode=unlimited`; its balance is `T*256` per robot, which can pay one maximum-size wire packet on every own action. The ordinary ledger and single-packet limit remain in force.

To freeze one combined workload, make `core-methods.json` list the forty core checkpoints and `auxiliary-methods.json` list the fifteen additional learned-control checkpoints plus `{"route":"heuristic"}`. Each learned entry uses the checkpoint/route/training-seed record format above. Auxiliary declarations select only their named methods; they do not multiply every control by every core budget/fault condition.

```bash
ir2-benchmark evaluate --suite formal --config configs/receiver_ir2_graph.json --method-manifest core-methods.json --map-sources artifacts/maps/sources.json --plan-only --output artifacts/plans/core
ir2-benchmark evaluate --suite formal --config configs/receiver_ir2_graph.json --method-manifest auxiliary-methods.json --suite-config configs/evaluation/auxiliary.json --map-sources artifacts/maps/sources.json --plan-only --output artifacts/plans/auxiliary
ir2-benchmark evaluate --suite formal --config configs/receiver_ir2_graph.json --method-manifest core-methods.json --suite-config configs/evaluation/team_budget.json --map-sources artifacts/maps/sources.json --plan-only --output artifacts/plans/team
ir2-benchmark evaluate --suite formal --config configs/receiver_ir2_graph.json --method-manifest core-methods.json --suite-config configs/evaluation/unlimited.json --map-sources artifacts/maps/sources.json --plan-only --output artifacts/plans/unlimited
cat artifacts/plans/{core,auxiliary,team,unlimited}/tasks.jsonl > artifacts/plans/all-tasks.jsonl
ir2-benchmark evaluate --task-manifest artifacts/plans/all-tasks.jsonl --batch-size 2 --output artifacts/combined-evaluation
ir2-benchmark report --results artifacts/combined-evaluation --map-sources artifacts/maps/sources.json --task-manifest artifacts/plans/all-tasks.jsonl --output artifacts/combined-report
```

Core rankings and root coverage curves use only core per-robot results. `auxiliary/` contains separate method/budget-mode/condition tables and figures; `zero_communication/` is the shared zero-budget view. `core_complete`, `auxiliary_complete` and aggregate `formal_complete` are separate. Missing declared auxiliary cases keep the aggregate false even when core is complete. Existing per-robot scenario/result IDs and checksum-verified older result envelopes remain readable. These entry points declare work; they do not establish that any formal training or evaluation matrix has run. Formal completeness also requires `report --map-sources artifacts/maps/sources.json` and the baseline configuration referenced by `core.json` (`base_config`, resolved relative to that declaration). The report independently checks every retained map identity, split, anchor and declared scenario seed in each core/auxiliary condition, using compact bitsets while reading the actual task manifest. Duplicate tasks cannot fill omitted map/seed blocks. Geometry, horizon, candidate count, target and six-slot capacity must match the baseline; declared N/budget/fault axes may vary, and unused burst scheduling metadata is ignored outside Burst. Registry, source-manifest, declaration and baseline hashes, retained-map counts, total missing blocks and bounded examples are recorded in both report and release metadata. Without the independent registry or baseline, older artifacts still produce plots/statistics, but completion flags remain false with an unverified scope.

`plan_quality.csv` uses estimates captured at the actor's actual decision cutoff against already-generated intentions. ADE/FDE/goal errors use map cells, NLL uses normalized coordinates, and ZOH has no invented density. Age, Normal/fault/recovery and cold-start strata are retained. `recovery.csv` compares actual Burst and paired Normal coverage gains in fault and recovery windows and explicitly names missing pairs.

`release_manifest.json` records code/dependency/configuration/map/scenario/checkpoint hashes, available real/imagined counters, errors and pending result IDs. Formal completeness requires the complete core matrix, all five training repetitions and actual two-million-event quotas; a short trained model cannot satisfy that condition.

## Synchronized throughput

The throughput command measures real reset, scripted observation/navigation/movement/sensing/traffic events, and heuristic policy rollout segments. It separates compile/first call, runs three warmups and ten synchronized samples, and excludes frozen events. Policy throughput is segments/s with the event length recorded; environment throughput is effective agent events/s and derived team rounds/s.

```bash
ir2-benchmark throughput --config configs/receiver_ir2_graph.json --suite formal --map-sources artifacts/maps/sources.json --output artifacts/throughput-formal
ir2-benchmark throughput --config configs/micro.json --model-checkpoint artifacts/b-smoke/checkpoint --output artifacts/throughput-b-smoke
```

Formal timing covers native 512²/768²/1024² representatives, N=2/4/6 and batch=1/4/16. An actual OOM is retained and stops larger batches for that geometry/N. `--limit-cases` and alternate `--batches` are explicit reduced profiles. GPU platform/device and JAX allocator peak memory method are recorded; CPU results are labeled CPU. B checkpoint hooks measure actual world-model optimizer updates and closed-loop imagined events; without a supplied checkpoint these quantities remain unmeasured. Benchmarking under concurrent training is not an isolated performance measurement.

Code checks, micro learning, formal training, formal evaluation and performance measurement are separate acceptance stages. Check the saved logs and completeness fields rather than inferring completion from the existence of a command, manifest or checkpoint.
