# IR2-JAX Benchmark：首轮代码审计报告

- 日期：2026-11-12
- 审计对象：公开仓库 `KDTang001/IR2-JAX-Exploration-Benchmark`，`main` 分支快照
- 审计方式：GitHub 文件树、README、配置、源码和测试静态阅读
- 运行状态：**未执行本地/服务器测试；未连接 SSH；未启动训练；未修改生产源码**
- 本报告状态：初步静态审计，不是完整验收报告

## 1. 执行摘要

项目已经具备较完整的研究工程构件：JAX/Flax/Optax 运行时检查、IR2 图策略移植、通信包编码和账本、信号模型、地图审计与生成、A/B 训练代码、配对评测、统计报告、吞吐测量和多层测试。

当前最重要的发现不是缺少更多算法，而是需要先确认**公开仓库的可安装/可执行闭环**。在当前 Git Tree 中，`src/ir2_benchmark/cli.py` 和其他模块引用 `ir2_benchmark.env.*`，但递归树没有列出 `src/ir2_benchmark/env/` 下任何受版本控制的文件；根目录的原版 `env.py` 不能自动提供 `ir2_benchmark.env.reset`、`step`、`observe` 等包模块。这是 P0 发布阻断风险，必须在服务器或干净环境中验证并修复。

另一个重要结论是：仓库当前包含 A/B 与预测器代码，但用户已明确现阶段暂停世界模型想象训练和世界模型队友意图预测，优先完成 Benchmark。因此本阶段不扩展或重构这些研究模块；保留现有代码，避免其影响基础环境和基线验收。

## 2. 范围和证据边界

本次静态阅读重点包括：

- `README.md`、`pyproject.toml`、`requirement.txt`、`.gitignore`
- `configs/receiver_ir2_graph.json`、`configs/receiver_ir2_graph_fusion.json`、`configs/evaluation/core.json`
- `agents/actor.py`、`agents/fixed_comm.py`
- `models/encoders.py`、`models/ir2_policy.py`、`models/plan_predictor.py`、`models/world_model.py`
- `communication/channel.py`、`ledger.py`、`faults.py`、`signal.py`、`codecs/standard.py`
- `navigation/one_hop.py`、`jax_path.py`、`ir2_graph.py`
- `maps/prepare.py`、`manifest.py`、`generate.py`、`sampler.py`
- `evaluation/metrics.py`、`runner.py`、`statistics.py`、`workload.py`、`release.py`、`throughput.py`
- A/B 训练入口、部分集成/学习/评测测试

静态审计不能证明函数在真实 JAX 运行时正确，也不能证明论文级实验已经完成。README 和历史状态文档中记载的历史训练结果，不能作为当前公开仓库已有检查点或可恢复训练的证据。

## 3. 发现清单

### P0 — 发布阻断或核心可信度风险

#### AUD-001：CLI 引用的环境包未出现在当前 Git Tree

证据：
- `src/ir2_benchmark/cli.py` 导入 `ir2_benchmark.env.artifacts`、`ir2_benchmark.env.reset`、`ir2_benchmark.env.rollout`、`ir2_benchmark.env.scenarios`。
- `src/ir2_benchmark/commands/train.py` 导入 `ir2_benchmark.env.scenarios`。
- `tests/benchmark/integration/test_event_step.py` 导入 `ir2_benchmark.env.observe`、`ir2_benchmark.env.reset`、`ir2_benchmark.env.step`。
- 当前递归 Git Tree 中没有 `src/ir2_benchmark/env/` 文件；只有根目录的原版 `env.py`。

影响：若树快照准确，安装 `src/ir2_benchmark` 后上述模块无法解析，CLI、训练和集成测试可能无法导入。

建议：在服务器 `ex` 环境执行 `python -c "import ir2_benchmark.env.reset"`、`python -m pytest tests/benchmark/integration/test_event_step.py -q` 和 `ir2-benchmark --help`。如果确认失败，先找回缺失源码或修复包路径，再进行其他工作。不要以根目录原版 `env.py` 替代新 JAX 环境模块。

状态：**静态证据明确，运行结论待执行验证。**

#### AUD-002：正式 Benchmark 尚无当前版本的运行证据

README 说明训练输出、检查点和录像不随公开快照提供；讨论文档也明确正式训练矩阵未完成。命令入口、测试文件和历史运行记录不等于正式结果。

影响：论文不能声称完整矩阵、学习收益、稳定吞吐或正式模型已完成。

建议：修复导入后从小型真实环境闭环开始，按验收阶段逐步建立新实验身份和结果。

状态：**文档明确；正式实验尚待重新执行。**

#### AUD-003：IR2 一致性范围必须精确表述

项目保留 IR2 图策略、六维节点特征、图注意力、pointer、传感器/图维护对照，但 Benchmark 新增了独立预算、包协议、奖励/结束规则、出生与计分区域等机制。信号模型文档也明确区分新解析模型与 IR2 原数值模型。

影响：不能把“局部观测/图策略对齐”写成“整个 IR2 MDP 完全复现”。

建议：发布一张逐组件 parity matrix：完全一致、数值/实现适配、Benchmark 扩展、明确不模拟，并为每一项链接测试和参考实现。

状态：**边界已在文档中部分说明；需统一成论文级矩阵。**

### P1 — 论文可信度和实验设计风险

#### AUD-004：默认 131072 字节/机器人不构成强受限通信实验

`configs/receiver_ir2_graph.json` 默认每机器人预算为 131072 字节。以 512 次自身行动、45 字节 PLAN 估算，总发送成本为 23040 字节，预算明显充足。

建议：正式报告必须覆盖预算受限区间，并报告实际发射字节、送达负载、预算耗尽比例、消息年龄；将 per-robot budget 与 team-total budget 分开命名，避免混淆。

#### AUD-005：通信价值和联合策略不是当前 Benchmark 必需功能

当前 Actor 有通信 gate/type head；图分支的 waypoint logits 由 GraphPolicy 直接产生，GRU/通信分支不直接回馈 pointer。该结构是现有策略设计事实，不是 Benchmark 环境缺陷。

建议：本阶段不改成新型联合决策网络。Benchmark 只需提供公平、方法中立的动作/通信接口，并把现有学习策略作为一种基线候选。

#### AUD-006：标准通信动作空间和自定义协议能力需区分

`communication/sending.py` 对内置 A/B 策略限制为标准广播及若干预设；自定义 codec/定向交换需方法适配器。不能仅凭环境支持自定义 payload，就宣称内置学习算法已支持学习任意消息和接收者。

建议：论文明确区分环境 API 能力、标准基线能力和学习策略能力。

#### AUD-007：信号模型不应混称为 IR2 信号模型

`communication/signal.py` 使用格子线段交长的解析衰减，并说明这是新模型而非 IR2 数值复现；角点接触零长度，不计障碍距离。IR2 原版 Bresenham 像素计数应单独作为 parity/control 模型。

建议：同时提供命名清晰的 `ir2` 与 `analytic_attenuation` 预设，报告不同信号模型的敏感性；逐条验证边界、角点、共址、阈值等案例。

#### AUD-008：地图划分与地图族均衡需发布级独立审计

`maps/manifest.py` 已有 canonical hash、八种旋转/镜像去重和 test 优先逻辑；`sampler.py` 要求多个来源/模板/桶分层。设计不错，但仍需验证真实 manifest、文件可访问性、split 数量、anchor 以及近重复地图。

建议：发布地图 registry、完整排除原因、哈希和 split summary；正式训练只用 train，模型选择只用 validation，最终报告只用 test/OOD。

#### AUD-009：统计单位与置信区间需要明确

`evaluation/statistics.py` 按场景→地图→地图族→训练种子进行均衡汇总，并做训练种子/地图层级 bootstrap。这比把每个 episode 当独立样本更合理。

建议：在论文中预先声明 estimand、层级抽样单位、配对方式和缺失任务处理；确认所有方法在同一地图/场景配对。

#### AUD-010：A/B 训练配置不能混进当前 Benchmark 最小验收

`configs/evaluation/core.json` 定义了 A/B × fixed/learned × ZOH/predict 八组核心配置，并声明 2M 事件、5 个训练种子。用户现阶段明确暂停世界模型训练效率与队友意图预测两项研究。

建议：本阶段 Benchmark 验收优先采用确定性启发式、无通信、固定通信、原版 IR2 策略和必要的训练型 baseline。A/B 世界模型研究不作为本阶段 blocker；旧配置保留但不宣称完成。

### P2 — 完善和论文呈现

- 端到端吞吐需与单纯采集吞吐分开；报告编译、reset、step、策略推理、记录/录像和保存开销。
- 运行时 doctor 仅验证 JAX/Flax/Optax 基础依赖，不等于探索环境性能。
- 录像应同时展示每个机器人局部 belief 和团队物理覆盖，并标注实际消息送达与预算。
- 文档需要清理历史时间线，给每个结果附上代码 commit、配置哈希、地图清单哈希和训练/评估身份。
- 公开发布需有安装测试、最小可复现示例、数据许可证/来源说明和版本化 release manifest。

## 4. 设计中值得保留的强项

1. `EnvState` 与 `LocalObservation` 分离，明确不把全局真值直接传给局部 Actor。
2. 通信包采用真实 uint8 编码、固定包头和 payload 长度。
3. attempt/success 两种扣费模式明确区分。
4. 随机故障使用 purpose/counter-keyed RNG，避免因策略发送次数改变故障随机流。
5. 物理覆盖使用 personal footprints，而非可能融合了接收地图的 local map。
6. 地图哈希区分原始方向身份与旋转/镜像 canonical grouping。
7. 评测 runner 保留 map/model/OOM 错误，不将执行失败伪装为 timeout。
8. release manifest 尝试绑定代码、依赖、配置、地图和检查点身份。

这些是 Benchmark 论文应该重点保留并提供实验证据的基础。

## 5. 建议的审计结论

当前项目应定位为：**功能构件丰富、研究方向明确，但公开快照仍需通过可安装性、环境闭环和正式实验验收的开发中 Benchmark。**

当前第一优先级不是增加新算法，而是：
1. 验证/修复 `ir2_benchmark.env.*` 包导入。
2. 在指定 Linux/JAX 环境完成 doctor、最小 rollout 和核心测试。
3. 独立验证通信/账本/故障/地图指标。
4. 重新建立无通信、固定通信和 IR2 参考基线。
5. 冻结 Benchmark 规范和正式评测协议。

## 6. 尚未完成的审计

- 全部环境状态转移与 reset/observe 代码逐行审查（受缺失包文件问题阻塞）。
- 完整通信接收/解码/图融合链路的事件级审查。
- 所有单元/集成/学习测试实际运行。
- 所有地图清单和 6900 张程序化地图的内容审计。
- CPU/GPU 配对性能测量。
- 独立干净环境安装测试。
- 外部相关工作系统检索和 novelty matrix。
