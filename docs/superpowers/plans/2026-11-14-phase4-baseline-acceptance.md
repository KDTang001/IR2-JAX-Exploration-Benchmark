# Phase 4 基线与统一评测验收清单

**Goal:** 用已验收的 JAX IR2 图环境，完成固定基线的小型配对评测、独立指标重算和错误/恢复验收，发布报告与证据。

**Architecture:** 复用现有 task/scenario manifest、bounded batch runner 和 report。新增 `baseline` 推理 route，与暂停的 A/B 研究配置分开；两种导航 × 四种固定通信共八种显式方法。补齐实际发射字节遥测，区分成功计费与物理流量；不启动训练。

**Tech Stack:** 服务器 conda ex，JAX/CUDA H100，NumPy/Pillow/SciPy，原版 IR2 PyTorch PolicyNet 仅用于权重转换/推理对照。无需新增依赖。

## 1. 方法与接口

- [ ] 最近前沿：只使用接收方本地图，按已知图最短距离选 utility>0 的可达节点，执行通往该节点的第一条合法边，坐标解并列。修正当前 heuristic 的最远邻边 fallback。
- [ ] IR2 adapted：从固定上游 stage2 策略张量转换到既有 `GraphPolicy`，greedy pointer，沿用同一合法一跳和观测。原训练数据暴露未知，明确标为外部预训练参考。
- [ ] 通信 off / plan / map / mixed；后三者每四次自身行动发送一次，mixed 循环 PLAN/MAP/MAP/POSITION。统一 codec、预算、信号和故障，不共享隐藏真值。
- [ ] `Method.route="baseline"`；decision_rule 为 nearest_frontier/ir2_adapted，communication 为 off/plan/map/mixed，training_seed=-1。IR2 任务绑定真实 checkpoint hash；nearest 不加载假权重。
- [ ] `agents/baselines.py` 提供 `BaselineSpec`、`make_baseline_method`、`spec_from_method`、`load_policy_parameters`、`get_baseline_kernel`；kernel 参数与既有 evaluate kernel 相同。`batching.BaselineModel(spec)` 由 parameter cache 识别。

## 2. 发射与结果契约

- [ ] `StepInfo.emitted_length` 由 channel eligible×实际包长确定；成功扣费失败的发送仍记录发射量。
- [ ] `EventTrace` 追加 emitted_length、packet_buffer、physical_links、external_links，旧构造可缺省；baseline 的真实 rollout 始终填入。
- [ ] `summarize_execution` 用发射长度算流量、spent 算扣费；保存实际 wire 和链路掩码。缺少成功计费遥测时显式拒绝，不能把免费失败发送计成零流量。
- [ ] `EvaluationResult`/`ReportRow` 记录可缺省的 spent_bytes；成本表据此计算预算利用率，未知费用保持未知；保留 overlap/分配集中度。

## 3. 专用配置和统一报告

- [ ] 新建 `configs/benchmark.json`、`configs/evaluation/benchmark.json`、`benchmark-methods.json`，保留旧 A/B 配置。
- [ ] CLI `evaluate --method-manifest` 支持 baseline 条目与外部 `--reference-checkpoint`；执行前冻结同一 scenario 集合与八个方法身份。
- [ ] baseline 不归入 no_neighbor_plan 辅助组，不要求学习预测器的 plan telemetry；同一 scenario 的地图、出生、故障种子和隔离身份在方法之间一致。
- [ ] 报告同时列 requested_manifest_complete 与完整正式矩阵状态；limited 小验收不能变成正式成绩。固定策略的区间按地图 bootstrap，不能用 episode 充当独立训练重复。

## 4. 远程验收

- [ ] 原权重出处：1ef6d224ae18503f4763c3cec20e05a737f95783，model/stage2/checkpoint.pth，52,176,462 字节，SHA256 cd72ac7dbf164b8be089394154794416fc969a47d79c1072c437f6f2aa363cf3。
- [ ] 仅提取 policy_model 的90个原始float32张量；用 tensor digest 验证传输链，权重只留服务器验收 reference 目录，不上传模型文件。
- [ ] 服务器运行相关既有 evaluation/metric 测试和必要新增回归；本地只做静态检查，不恢复 A/B 训练。
- [ ] 小型配对矩阵包含八个方法、Normal/Dropout/Burst 独立条件、固定地图与种子；至少覆盖 native 地图与可手算开发场景，所有限额写入任务。
- [ ] 独立脚本从 PNG、原版 sensor、raw positions/wire/交付记录重算物理覆盖、AUC、费用/流量、overlap 和分配集中度，不调用生产指标函数充当 oracle。
- [ ] 检查 resume 不重复计分，缺失/损坏结果不能完成矩阵，map/model/env/OOM 错误显式区分，actual IR2 权重 Torch/JAX 推理对照通过。

## 5. 分工边界与交付

基线工作负责 agents/heuristic.py、agents/baselines.py、evaluation/batching.py；账本工作负责 types.py、env/step.py、env/rollout.py、evaluation/episode.py、records.py；报告工作负责 evaluation/report.py、statistics.py、scope.py、budgets.py、tables.py、plan_quality.py。主执行负责 CLI、scenario route、配置、独立 oracle、远程运行与最终集成。各工作只新增各自必要测试，不运行本地仿真，不提交他人修改。

- [ ] `research/BASELINE_EVALUATION_REPORT.md`、`research/evidence/phase4-20261114.json` 及小型实际结果汇总。
- [ ] 生产源码变化结束后显式更新固定验证场景的实现摘要，保留原25张地图/种子/N/8KiB预算。
- [ ] 只读复核、更新总审计/路线图，清理本轮临时转运文件，推送原审计分支。
