# Benchmark Evaluation and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 产出可恢复执行、包含失败案例和成本核算的统一实验报告。

**Architecture:** 环境事件日志汇总为场景结果，固定清单驱动独立条件评测，按训练种子与地图分层统计。吞吐测试与算法质量结果分开。

**Tech Stack:** Python、NumPy、JAX、JSONL、pytest、Matplotlib；图片作为独立科研产物生成，不用截图代替数据。

---

## V1：指标与微型 oracle（A/B 训练前实现）

创建 evaluation/metrics.py；测试 unit/test_metrics.py。

- [ ] 手工轨迹 C=[0.25,0.5,0.75,1] 验证 AUC=0.625；若第 2 轮达到 0.99，则剩余轮保持 0.99 而不是 1。非成功案例保留 timeout，不生成假的成功时间。
- [ ] 两机器人亲自观察集合 S0={a,b}、S1={b,c}，重复度必须 1-3/4=0.25；转发地图不增加 S_i 或联合物理覆盖。验证集中度 p=[0.75,0.25] 为 0.625。
- [ ] 计费统计区分发送含头字节与成功接收载荷字节；广播被 3 人接收时接收量可大于发送载荷，不能错误判为账本异常。
- [ ] 计划 ADE/FDE 只用有效长度 mask，位置归一化与还原一致；按消息年龄、正常/故障/恢复阶段分别聚合，冷启动单列而非丢弃。
- [ ] `uv run pytest tests/benchmark/unit/test_metrics.py -q`。指标函数不依赖模型类，供启发式、A 和 B 共用。

## V2：固定场景清单与配对评测

创建 evaluation/scenarios.py、evaluation/runner.py；测试 integration/test_scenarios.py。

- [ ] 每条场景记录 map_hash、split、spawn_seed、fault_seed、N、budget、fault_mode、drop_probability 或 isolation_ids/start/duration、T、config_hash。Normal 中无额外丢包，Dropout 无 Burst，Burst 无随机丢包。
- [ ] 配对方法用相同清单，不以全局 RNG 依次消耗来生成场景。固定执行顺序不变，编号重分配另列诊断清单。
- [ ] 一个完成场景写一条不可重复的 result_id；断点恢复跳过已有且校验完整的结果。模型报错、地图错误、OOM 与环境 timeout 使用不同状态，不把运行错误当正常探索失败，也不自动过滤错误地图。
- [ ] 测试中断后继续不会重复计数；方法先后运行顺序变化不改变场景定义；缺少一个预算条件时报告不完整矩阵而非输出看似全量结果。
- [ ] `uv run ir2-benchmark evaluate --checkpoint artifacts/a-smoke/checkpoint --suite micro --output artifacts/evaluation-smoke`；确认实际执行真实环境，而不是读取训练日志冒充测试。运行 `uv run pytest tests/benchmark/integration/test_scenarios.py -q`。

## V3：公平矩阵、统计和负结果

创建 evaluation/report.py、configs/evaluation/core.json、tests/benchmark/unit/test_report.py；增加 README benchmark 入口说明。

- [ ] 核心配置明确 A/B×固定/学习通信×ZOH/预测 8 组；训练额度均为 2M 有效真实事件，5 个正式训练种子，预热计入额度。配置数量和模型选择规则记录，不静默给予某组更多搜索。
- [ ] 原测试地图每张 3 个场景种子；程序化同分布和结构变化测试分开。开发时可指定 `--limit-scenarios`，结果带 limited=true，不能用开发子集冒充正式全量。
- [ ] 先按地图内场景均值，再地图族内地图等权，再族间等权。置信区间按训练种子/地图分层重采样；一个训练模型上 10000 个 episode 不视为 10000 次独立训练重复。
- [ ] 为失败案例、实际发射字节、无消息冷启动、局部故障恢复分别生成表。收益未显著或不如 ZOH 的结果照常保留；成功率低时不得只画成功条件时间。
- [ ] 使用 2 个地图族、一族 1 张/另一族 3 张的手算 fixture，验证总分不是按 4 张图直接加权。用缺失训练种子 fixture 验证报告标记不完整。
- [ ] `uv run ir2-benchmark report --results artifacts/evaluation-smoke --output artifacts/report-smoke`，产出机器可读 JSON/CSV、覆盖与预算曲线、失败/成本表及 Markdown 报告；运行 `uv run pytest tests/benchmark/unit/test_report.py -q`。

## V4：吞吐、发布清单和最终验收

创建 evaluation/throughput.py、tests/benchmark/integration/test_release.py，文档 docs/benchmark-running.md。

- [ ] 环境吞吐测试锁定动作脚本并包含感知、导航、账本与观测，不仅测一个数组加法。分别报告 reset、有效 agent-event/s、team-round/s、带策略 rollout/s、world-model update/s、imagined-event/s。
- [ ] 编译和首次调用单独计时，至少 3 次预热、10 次同步计时，使用 `block_until_ready` 等待输出 PyTree 的设备数组。吞吐分母是实际未终止事件数，不把填充步计入。
- [ ] 每种地图桶与 N 测 batch={1,4,16}，显存不足记录 OOM 后停止增加该档，不能排除大地图只展示最高速度。记录设备、驱动、版本、精度和峰值显存测量方法。
- [ ] `uv run ir2-benchmark throughput --config configs/micro.json --output artifacts/throughput-smoke`；CLI 返回 JSON 数据和统计。GPU 不可用时显式标记 CPU，不能沿用 GPU 文件名或结论。
- [ ] 发布清单含锁文件、版本/代码修订、地图/场景哈希、完整配置、模型选择规则、checkpoint 哈希、训练真实/想象计数、运行错误清单和结果目录。README 写出 doctor→maps→rollout→train-a/train-b→evaluate→report 的完整命令。
- [ ] `uv run pytest tests/benchmark -q` 后，从干净独立环境按文档复现一次微型 A 和 B 训练及评测；核对同 seed 的配置、计数、轨迹和结果容差。正式全量实验未执行完时，发布状态保持 incomplete，列出剩余 result_id。

## 收尾证据

每个开发阶段提供最少一个真实 CLI 运行的退出码和产物路径。最终报告区分：代码检查通过、微型学习通过、正式训练完成、正式评测完成、性能测量完成。五者不可互相替代。方法未显示优势是合法研究结果；缺少应执行的实验是未完成工作。
