# IR2-JAX Benchmark 实验协议

- 版本：Draft 0.1
- 日期：2026-11-12
- 状态：预注册草案；参数档位待预实验确认

## 1. 基本原则
- 所有方法共享同一任务定义、地图、spawn、事件顺序、通信协议和计分规则。
- 正式测试集不用于模型选择、参数调整或预算选择。
- 训练种子与场景种子分离。
- 每个方法在相同场景身份上配对运行。
- OOM、模型执行错误、地图错误和 timeout 分开记录。
- 训练数据量、推理算力、实际通信成本分别报告。

## 2. 研究问题
RQ1：JAX 环境是否能在保持明确 IR2 探索语义对照的同时支持高效并行仿真？

RQ2：不同通信预算和消息类型如何影响团队覆盖效率、重复探索和通信成本？

RQ3：在随机丢包与局部突发断联下，现有探索策略的性能如何退化和恢复？

RQ4：IR2 与程序化地图、不同地图族和不同机器人数量之间是否存在泛化差异？

RQ5：JAX 环境在地图尺寸、机器人数量和 batch size 增长时的吞吐与内存如何变化？

## 3. 基线协议
首轮方法集合：
- B0 Zero communication；
- B1 Fixed PLAN；
- B2 Fixed MAP_DELTA；
- B3 Fixed periodic mixed message schedule；
- B4 Nearest-frontier heuristic；
- B5 Original IR2 policy adapted to the benchmark protocol。

若 B5 无法直接兼容新环境，需明确适配层和不等价部分；不能用原论文数字代替。学习式 A 可作为附加基线，但暂不把世界模型 B/意图预测纳入当前正式范围。

## 4. 场景因素
建议核心维度：
- active robots N ∈ {2,4,6}；
- budget per robot ∈ {0, 2048, 8192, 32768} bytes；
- link condition ∈ {Normal, Dropout, Burst}；
- Dropout probability ∈ {0.1,0.3,0.5}；
- Burst duration ∈ {64,128} rounds；
- map source ∈ {IR2, procedural}；
- map family ∈ {easy, difficult, complex, corridor, hybrid, rooms, generated-corridor, generated-hybrid}；
- map split ∈ {validation, test, OOD}，严格分开报告。

这些是设计档位，不是所有维度全笛卡尔积。正式矩阵应先定义主效应和交互作用，避免无必要的实验爆炸。

## 5. 配对与随机数
- 每个场景记录 map hash、canonical hash、anchor、scenario hash。
- 所有方法复用同一 map/spawn seed。
- 外部故障 RNG 使用 scenario/event/sender/receiver 等语义键，不能依赖发送次数。
- 对策略随机性使用独立 policy seed；方法间使用相同种子编号，但不要求不同策略消耗相同数量随机数。
- 对机器人编号偏差，增加起点与编号置换的敏感性实验。

## 6. 主指标
- Coverage AUC：以全队轮次结束时的物理覆盖率为序列，完成后按终止覆盖率延续到统一 horizon。
- Success rate：达到预先声明的物理覆盖阈值比例。
- Completion time：仅对成功 episode 报告轮次/事件。
- Final physical coverage。
- Travel distance。
- Emitted wire bytes 与 delivered payload bytes。
- Budget utilization/exhaustion。
- Message age。
- Physical overlap/redundancy。
- Failure-window 与 recovery-window coverage gain。

需要在最终协议中明确 AUC 使用 round-end 还是 event-level；当前代码 `coverage_auc` 以 round-end 序列为输入，`summarize_episode` 可用 event trajectory 定位成功事件。

## 7. 统计分析
- 先在同一地图的多个场景种子内平均；
- 再按地图平均；
- 再按地图族等权；
- 训练型方法先在每个训练种子内聚合，再在训练种子层面汇总；
- 使用 training seed 与 map 的层级 bootstrap 构造区间；
- 同时报告绝对差值、相对差值和置信区间；
- 不把大量同一地图 episode 当作独立样本扩大显著性；
- 所有失败任务保留在 workload completeness 中，不静默删除。

训练型基线的训练重复数需要根据资源预算确定；如果沿用 5 个训练种子，必须如实报告五次独立训练是否完成。

## 8. 性能协议
分别报告：
1. 首次 JIT 编译；
2. warm reset；
3. scripted environment agent-events/s；
4. team rounds/s；
5. heuristic rollout；
6. learned policy inference；
7. complete training updates/s 或 effective real events/s；
8. checkpoint/save/record 开销。

固定代码 commit、地图哈希、有效地图尺寸、bucket、N、batch、设备型号、驱动、JAX/XLA 版本、精度、allocator 和 warmup/repeat 数。所有异步设备结果需 `block_until_ready` 后计时。

CPU/GPU 对比必须同一算法、同一地图、同一配置和同一事件数。不要用旧原版 CPU 吞吐与新 JAX GPU 吞吐直接计算加速比。

## 9. 正式实验冻结
开始正式矩阵前，冻结：
- git commit；
- dependency lock/环境快照；
- config hashes；
- map registry hash；
- validation manifest hash；
- scenario manifest hash；
- baseline identities；
- metric definitions；
- training seeds；
- test workload declaration。

测试集只运行一次冻结版本；若修复导致重跑，记录原因、旧版本和新版本，不覆盖旧结果。
