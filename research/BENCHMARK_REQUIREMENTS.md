# IR2-JAX Benchmark 需求规格

- 版本：Draft 0.1
- 日期：2026-11-12
- 状态：待用户审阅；不是实现批准或完成声明

## 1. 范围

本规格只定义当前 Benchmark 基础设施的目标。暂不开展：
- 世界模型生成想象轨迹以提升训练效率；
- 世界模型队友意图预测；
- 新的自适应通信/通信价值联合决策方法研究。

已有 A/B、PlanPredictor、WorldModel 源码保留，不作为当前阶段必须完成的功能，也不删除。

## 2. Benchmark 研究目标

提供一个研究型、可复现的 JAX 原生通信受限多机器人探索平台，使研究者能够在统一任务、地图、信息权限、通信成本和评测指标下比较探索方法。

Benchmark 不应仅以 JAX 加速作为唯一贡献。主要研究价值应来自：
1. 方法中立的部分可观测多机器人探索环境；
2. 事件级、可审计的付费通信协议；
3. 真实链路和独立故障条件；
4. 可追溯地图数据集及 OOD 划分；
5. 统一基线、指标和统计方法；
6. 可复现运行身份与 JAX 性能测量。

## 3. 功能需求

### BR-01 环境状态和信息权限
- 全局真值仅用于仿真物理转移、链路判定、团队物理覆盖和集中式评测。
- 局部策略只能读取自身感知、历史记忆和实际送达消息。
- 接收地图不增加物理覆盖计数。
- 所有机器人事件顺序和事件时钟必须明确且可重放。
- reset/observe/step 必须具备独立确定性测试。

验收：改变未观测区域真值时，局部观测不变；没有合法地图包送达时，接收方地图/图知识不得变化。

### BR-02 探索与导航语义
- 地图坐标、占据编码、出生位置、计分连通分量、感知边界和运动规则必须文档化。
- IR2 parity 与 Benchmark extension 分开标注。
- 一跳图动作与旧 grid/BFS 模式不得混为同一基线。
- 不允许静默截断图、路径队列或合法候选。
- 固定随机种子和确定性排序应产生可重现结果。

验收：传感器逐格对照；图节点/边/utility 对照；动作合法性、穿墙、越界和终止规则测试。

### BR-03 通信协议
- 包头、payload、字段精度、最大长度和序列化字节必须明确。
- 所有消息成本基于真实编码字节，而不是张量尺寸。
- 发送、物理可达、外部故障、送达、收件箱覆盖、解码和知识更新分开记录。
- 预算不足和无效包不能扣费；attempt 与 success 模式分别定义。
- 不允许免费 ACK、自动多跳或隐藏全图访问。
- 位置/目标/计划的生成时间戳不能伪装成接收时刻。

验收：对每种 codec 做 round-trip、边界和畸形包测试；对每次发射独立重算费用；核对失败送达仍按 attempt 模式收费。

### BR-04 通信链路与故障
- 物理信号模型与外部故障必须独立。
- Normal、Dropout、Burst 分开评估，不在主矩阵中混合。
- 故障随机数由 scenario/event/sender/receiver 等语义键确定，不依赖策略历史发送次数。
- 局部断联不停止运动/感知、不清空记忆、不免费通知故障类别。

验收：固定场景下逐链路重算可达性和丢包；改变方法发送频率不改变相同事件的外部故障样本。

### BR-05 地图数据和划分
- 每张地图必须有源路径、来源、地图族、尺寸、anchor、occupancy hash、canonical hash、scenario hash、split 和排除原因。
- IR2 与程序化地图分别报告。
- 训练、验证、测试和 OOD 必须严格隔离。
- 旋转/镜像重复需审计；测试优先于训练保留。
- 生成器版本、种子、参数、失败种子和最终地图哈希必须可追溯。
- 不允许读取失败时静默替换地图。

验收：独立重建 registry，重新核对文件存在性、哈希、split、anchor、连通区域和重复组。

### BR-06 基线
第一版优先：
1. 无通信/零预算；
2. 固定周期标准消息；
3. 固定 PLAN；
4. 固定 MAP_DELTA；
5. nearest-frontier 等确定性启发式；
6. 原版 IR2 策略在本 Benchmark 协议下的适配基线；
7. 至少一个可复现的学习式探索基线（若正式论文需要）。

A/B 世界模型想象训练和队友意图预测不纳入当前阶段必须完成的基线。

验收：所有基线使用同一地图、出生种子、预算定义、链路条件、完成判据和评测代码；不允许跨论文直接比较未经重跑的分数。

### BR-07 统一指标
主指标：
- round-end physical coverage AUC；
- success rate；
- completion rounds/events（仅成功 episode 定义）；
- final physical coverage。

次指标：
- travel distance；
- emitted wire bytes；
- delivered payload bytes；
- budget utilization/exhaustion；
- message age；
- physical overlap/redundancy；
- per-robot first-discovery share；
- failure/recovery coverage gain。

执行错误（OOM、map error、model error）必须与探索 timeout 分开报告。指标的分母、事件时间、padding 和终止后处理必须有精确定义。

验收：从原始轨迹独立重算指标，报告值与重算值一致。

### BR-08 性能
分别测量：
- 安装/初始化；
- JIT 编译与首次调用；
- reset；
- scripted environment events；
- heuristic rollout；
- learned policy inference；
- end-to-end training update；
- checkpoint/recording。

必须同步设备计时，报告地图原尺寸、有效尺寸、N、batch、设备型号、软件版本和峰值内存测量方法。纯采集吞吐不得称为训练吞吐。

### BR-09 可复现与发布
每个 run/evaluation 必须记录：
- git commit；
- source/config/dependency hashes；
- map registry/scenario manifest hashes；
- training seed/scenario seed；
- method identity；
- checkpoint hash；
- effective real events；
- evaluation workload identity；
- errors/pending tasks；
- device/runtime metadata。

验收：在干净环境按公开说明完成 doctor、micro rollout、baseline evaluation 和 report。

## 4. 非功能需求
- JAX 核心尽量使用纯函数、显式状态和显式 PRNG。
- 固定形状/掩码需有容量溢出检查。
- 不为性能优化牺牲地图/通信语义。
- 不在正式测试集调参。
- 代码修改应小步、可回滚、有对应测试。
- benchmark 核心 API 不依赖特定学习算法。

## 5. 当前交付门槛
Release Candidate 前必须通过 BR-01 至 BR-09 的适用验收。论文实验可以按研究范围分层，但任何缩减矩阵都必须明确标记为 limited，不得宣称完整矩阵完成。
