# IR2-JAX Benchmark 实施路线图

- 版本：Draft 0.1
- 日期：2026-11-12
- 当前状态：审计分支；Phase 1–4 声明范围的目标服务器功能验收已通过，下一步 Phase 5 性能与工程复现。以下原始路线与历史记录以最后一条进度更新为准。

## 原则
1. 先修复会阻断 import/rollout 的问题，再谈性能。
2. 保留原版 IR2 参考实现。
3. 暂停世界模型想象训练和世界模型队友意图预测。
4. 不启动正式大规模训练，直到环境、通信、地图和评测验收通过。
5. 所有修改在独立分支、逐项测试和可回滚提交中完成。

## Phase 0 — 审计与运行入口恢复（当前）
工作：
- 验证 Git Tree 中缺失的 `src/ir2_benchmark/env/*` 是否导致包导入失败。
- 确认 Python/JAX/Flax/Optax 依赖和 editable install。
- 检查 CLI help、doctor、单场 rollout 的导入路径。
- 建立静态依赖图和核心测试清单。

退出条件：
- `import ir2_benchmark` 成功；
- `import ir2_benchmark.env.reset`、`observe`、`step` 等项目模块可解析；
- `ir2-benchmark --help` 和 doctor 正常；
- 一场 micro rollout 完成并产生事件日志；
- 记录当前 main commit、依赖和环境信息。

## Phase 1 — 环境与 IR2 parity
工作：
- reset/observe/step 时序；
- 物理覆盖与局部知识隔离；
- 传感器逐格 parity；
- IR2 graph node/edge/utility parity；
- 一跳动作与旧导航模式边界；
- 确定性和容量溢出。

退出条件：
- 核心几何和事件集成测试通过；
- parity matrix 有明确证据链接；
- 所有 Benchmark 扩展均与 IR2 原始行为分开说明。

## Phase 2 — 通信和故障
工作：
- 标准 codec round-trip 和畸形包；
- 真实 wire byte 账本；
- 收件箱新旧消息覆盖；
- map delta 接收后图融合；
- 信号模型与外部故障独立；
- Normal/Dropout/Burst 配对轨迹。

退出条件：
- 每个事件的发送、费用、送达和知识变化可独立重算；
- 不存在隐式地图共享；
- 预算/故障语义与文档一致。

## Phase 3 — 地图数据和采样
工作：
- IR2 原始数据全量审计；
- 程序化地图 manifest/生成器复现；
- canonical duplicate 和跨 split 泄漏；
- anchor/component/valid_shape 核验；
- train/validation/test/OOD registry。

退出条件：
- 每个保留/排除地图均有原因；
- 文件 hash 与 registry 一致；
- 正式 sampler 只采 train；
- validation 固定并独立于 test。

## Phase 4 — 基线和评测闭环
工作：
- zero communication；
- fixed PLAN；
- fixed MAP_DELTA；
- periodic mixed-message baseline；
- nearest frontier；
- 原版 IR2 adapted policy；
- 统一任务 manifest、错误分类、指标重算和报告。

退出条件：
- 每个基线能完成小型配对评测；
- 结果由独立代码重算一致；
- 统计单位、缺失任务和置信区间固定。

## Phase 5 — 性能与工程可复现
工作：
- CPU/GPU 同 workload 配对；
- native map buckets、N、batch scaling；
- compile/reset/step/policy/update/save/record 分段计时；
- 峰值显存与 host memory；
- 恢复、重跑、manifest/checkpoint hash；
- clean install 文档和 CI。

退出条件：
- 报告所有性能口径；
- OOM/error 不被隐藏；
- 关键运行可从 manifest 和配置复现。

## Phase 6 — 论文实验和发布
工作：
- 冻结代码、数据、配置和 workload；
- 正式 baseline 重复；
- 地图族、机器人数量、预算和故障分层分析；
- Ablation 与敏感性；
- 失败案例、局限和威胁有效性；
- release manifest、citation、license、quickstart。

退出条件：
- 论文中的每张表/图都能追溯到原始任务结果；
- 所有声称的矩阵都有完整性证明；
- Benchmark 可由第三方按 quickstart 运行。

## 当前立即执行的下一步
1. 取得服务器可用的安全远程执行方式；不要在仓库或脚本中保存密码。
2. 执行 Phase 0 的导入/CLI/doctor/micro rollout 检查。
3. 如果发现缺失模块，先恢复正确源码，不临时造一个空包绕过错误。
4. Phase 0 通过后，开始 Phase 1；不提前启动正式训练。


## 进度更新：环境包发布遗漏已恢复（2026-11-12）

用户确认提交 `1431ad6a42eda555d4c34a5eacac06dbc61282ac` 已将原有九个 `src/ir2_benchmark/env/` 模块恢复到本审计分支，并修复 Windows 大小写不敏感忽略规则导致的发布遗漏。详细证据见 `ENV_PUBLICATION_RECOVERY.md`。

已完成的 Phase 0 子项（仅限 Windows CPU 独立 wheel 环境）：
- wheel 构建及非 editable 安装；
- `pip check`；
- 八个功能模块从已安装 wheel 导入；
- CLI help；
- 78 项指定单元/集成测试；
- `ir2_one_hop` micro rollout：256 事件、128 轮，正常 truncated。

仍未完成的 Phase 0/1 项：
- Linux/CUDA 独立安装与运行；
- 完整 `ir2_graph` 原尺寸场景验收；
- 完整传感器、图节点/边/utility parity 测试重跑；
- 全量测试套件与干净环境 CI；
- 完整地图审计及正式评测工作负载验证。

因此 Phase 0 目前状态为 **部分通过（Windows CPU packaging/import/micro smoke）**，不能标记为整体完成。下一工作优先进入 Phase 1 的独立 parity 证据核验，同时等待 Linux/CUDA 环境可用；正式训练仍保持暂停。

## 进度更新：Phase 1 首批 parity 证据（2026-11-12）

上述待办中的指定传感器、图结构 parity 重跑已完成：Windows CPU 下 39 项通过、无跳过，包含原尺寸地图 16 步感知与图维护轨迹，以及事件/隔离检查。详见 [IR2_PARITY_REPORT.md](IR2_PARITY_REPORT.md) 和其中的机器可读证据。

这不是完整 `ir2_graph` 多机器人 episode 验收。非零会合效用、长期容量处理和付费地图融合的联合隔离仍待核验，Linux/CUDA 与后续阶段也未完成。本次没有修改生产代码或恢复训练。

## 进度更新：Phase 1 核心环境验收完成（2026-11-13）

在声明的 Windows CPU 机制与接口范围内，Phase 1 验收完成：73 项不同测试通过、无失败、无跳过，包含四张原尺寸地图各 32 步原版差分轨迹、会合数值对照、双机器人付费地图连续事件、容量/路径失败拒绝计分、成功与超时冻结、非零会合信息隔离。修复参考图排序函数的空邻居输入崩溃；JAX 环境算法未变。完整范围和原始失败/复验结果见 [IR2_PARITY_REPORT.md](IR2_PARITY_REPORT.md)。

下一阶段为 Phase 2 通信协议与故障模型。Linux/CUDA、全地图/多种子扩展和正式实验分别留在后续复现、数据集与实验阶段；本轮未恢复训练。

## 进度更新：Phase 2 通信与故障验收完成（2026-11-14）

在声明的 Windows CPU 范围内，118 项通信测试与 28 项 Phase 1 相关集成回归通过，去重共 146 项。Normal、Dropout、Burst 分别执行，逐事件独立重算真实字节、扣费、送达和知识；覆盖两种账本、5 种物理预设、正式 Burst 起止边界、付费转发/地图融合与畸形包拒绝。

修复 MAP_DELTA 接收时只按数组尺寸而未按 valid_shape 过滤的缺陷，真实观察、想象观察与录像共用真实地图边界。规范、原始失败与复验结果见 [COMMUNICATION_FAULT_REPORT.md](COMMUNICATION_FAULT_REPORT.md)。本轮未训练或修改正式实验矩阵。

下一阶段为 Phase 3 地图数据集与划分审计。Linux/GPU 和正式大规模评测仍未完成。

## 进度更新：目标服务器 Phase 1/2 功能补验（2026-11-14）

此前只做 Windows CPU 本地验收而遗漏用户指定服务器，阶段完成的验证安排不完整。现已在服务器 conda `ex`、NVIDIA H100、强制 CUDA 后端下，用审计提交 `1e0af0ed` 的同一源码快照补验：Phase 1 73 项、Phase 2 118 项全部通过，均无失败或跳过。28 项相邻回归包含在 Phase 1 中，两个阶段共 191 项不同测试。

上传前后的文件摘要、实际导入路径、JIT 数组所在设备和测试结束后的源码一致性均已核对。四张原尺寸地图的逐步记录及 Normal/Dropout/Burst × attempt/success 六组通信记录与本地证据完全一致。初次 launcher 导入旧 editable 安装的预检已中止，正式结果含快照路径与 CUDA 断言。详见 [远程验收证据](evidence/remote-acceptance-20261114.json)。

目标服务器功能验收已补足；Linux/GPU 吞吐、全地图/种子矩阵和正式实验仍待后续阶段。没有启动训练或扩展生产代码。下一步仍为 Phase 3 地图数据集与划分审计。

## 进度更新：Phase 3 数据集与采样验收完成（2026-11-14）

目标服务器 `ex` 上全量核对 17,299 张 IR2 与 6,900 张程序化 PNG，逐一验证占据/anchor/component/尺寸/三个 hash，独立重算 IR2 test 优先去重及训练族验证留出。全部 6,900 个程序化实例按 requested_seed 重现，13,877 个保留 canonical 全局唯一，跨来源交集为零。

合并保留 train 11,625、validation 924、test 1,028、ood 300。实际 sampler 只从 train 调度；evaluation reader 只读 1,328 张 test/ood；固定验证仍使用原来 25 张 validation 地图、同一组出生/故障种子、N、轮数上限和 8 KiB 预算。修复旧验证场景的配置身份与 CRLF/LF 导出/源码绑定问题；没有改 PNG、地图划分或训练算法。

远程最终去重 35 项测试通过、无失败或跳过；另外将实际 13 个来源/族/桶组合加载到 CUDA，核对三类错误元数据显式拒绝。IR2 difficult 本身跨 512/768/1024 桶，已经按实际 registry 验证。所有初次失败与辅助检查错误假设均保留。见 [完整报告](MAP_DATASET_REPORT.md)、[机器证据](evidence/phase3-20261114.json)、[逐文件 SHA-256](evidence/phase3-data-files-20261114.sha256)和[只读复核](evidence/phase3-code-review-20261114.md)。

下一步为 Phase 4 基线与统一评测，先冻结 Benchmark 专用配置与小型配对验收。训练未恢复；吞吐和正式实验仍在后续阶段。

## 进度更新：Phase 4 基线与统一评测完成（2026-11-14）

在服务器 `ex` + H100/CUDA 上完成最近前沿/原版权重 IR2 adapted × off/PLAN/MAP/mixed 八基线的 312 个限定任务，覆盖三类独立链路条件、开发与原尺寸地图、N=2/4/6、零预算；10,112 个有效 agent 事件无执行错误或非法动作。所有轨迹的原版传感器/信号/碰撞、物理覆盖、通信账本和指标独立重算通过。续跑全部复用 312 个完整结果，未新增 rollout；分阶段相关检查去重 275 项最终通过。

实测部分地图与位置消息触发原版同样会失败的会合分支，经用户确认增加主配置 `reachable_only`，不可达队友暂不贡献会合效用；`strict_ir2` 保留原默认和匹配的环境/评测配置，同种子反例按预期返回环境错误。最近前沿没有可达前沿时选择最近合法邻边，避免被 IR2 屏蔽的等待动作。实际发射字节与支付预算分开记录；错配任务、缺少/非有限指标和损坏 trace 不能被当成完成。

报告区分 requested manifest 完成与 formal 完成，本轮正式完成标志保持 false；每个地图族仅一张地图，未发布置信区间或排名。IR2 原权重仅为训练暴露未知的外部预训练参考，没有开展数据公平的重训。旧 A/B 研究配置和代码继续保留、暂停训练。详情、失败历史和逐任务值见 [Phase 4 报告](BASELINE_EVALUATION_REPORT.md)及其配套证据。

下一步 Phase 5：同实现/地图/并行数的 CPU–GPU 比较、冷编译与稳态分段计时、显存/主存和清洁环境复现。当前没有启动训练，也没有执行完整正式矩阵。
