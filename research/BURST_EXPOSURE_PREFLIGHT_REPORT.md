# 8 KiB / mixed：Burst 离线暴露预审

日期：2026-11-18。输入基线：`f9bb88e4e585f225bd335c5a535d69373b7dadea`。
状态：**离线暴露机会预审通过；真实 Burst 效应尚未验证**。
本步未连接目标 SSH 机器、初始化 JAX/GPU、运行环境或启动训练；没有使用任何已关闭的 GPU 窗口。

## 结果

已归档的十个 8 KiB/Normal 任务均有同身份的 B1、B4 完整 H100 轨迹。重新核验全部
20 条轨迹的 SHA256、20 份结果校验和及旧归档索引；十对 B1/B4 的轨迹哈希和除路径外
的结果字段一致。统计按十个唯一任务计算，使用 B4 轨迹，没有将重复执行算作额外样本。

与已冻结 v2 的十个 8 KiB/mixed/Burst 任务逐项核对地图、Validation 划分、出生和故障种子、
方法/权重、预算、N4、768 桶及 C1024 配置；环境配置仅 `fault_mode` 不同。没有重新选图、
移动第 128–191 零基轮的故障窗口或改变隔离机器人。原 240 项及所有旧证据均保留。

- **10/10 有实际发送，也有位于未来屏蔽边上的 Normal 成功投递**，支持进入真实 Burst 小实验。
- 第 128 零基轮开始前，所有任务、每台机器人都还剩 **3,640 字节**，不存在之前 2 KiB
  在窗口前已无法发送的问题。
- 窗口内合计 **628 次发射、88,944 字节**；Normal 实际送达 **978 个接收方副本**。
  对这些已发生的投递静态叠加冻结 Burst 掩码，有 **817 个接收方副本**落在将被阻断的边上，
  对应 **424 次不同的发射事件**。817 不是发射次数，也不是真实 Burst 实验中已经观测到的损失。
- **9/10** 的 Normal 轨迹在第 192 零基轮以后仍有发送，并在相关机器人对之间有投递。
  另一项 `ir2/difficult × IR2 adapted` 在事件709后成功结束，尚未到恢复边界；不视为失败。
  其真实 Burst 回合是否仍提前成功，必须由新实验确定。
- 暴露强度不均：`procedural/rooms × IR2 adapted` 只有 **2 个**可受阻接收方副本。
  保留这一弱暴露任务及十项固定分母，不根据结果移除或换图。

| 地图族 | 导航 | 窗口发射字节 | Normal 投递副本 | 静态可受阻副本 | 第192轮后相关边的 Normal 投递副本 |
| --- | --- | ---: | ---: | ---: | ---: |
| IR2 easy | 最近前沿 | 9,104 | 191 | 159 | 144 |
| IR2 easy | IR2 adapted | 9,104 | 35 | 35 | 72 |
| IR2 difficult | 最近前沿 | 9,104 | 72 | 70 | 60 |
| IR2 difficult | IR2 adapted | 7,008 | 100 | 79 | 0：此前已成功 |
| procedural rooms | 最近前沿 | 9,104 | 162 | 130 | 60 |
| procedural rooms | IR2 adapted | 9,104 | 6 | 2 | 16 |
| procedural corridor | 最近前沿 | 9,104 | 189 | 157 | 144 |
| procedural corridor | IR2 adapted | 9,104 | 14 | 13 | 13 |
| procedural hybrid | 最近前沿 | 9,104 | 183 | 151 | 150 |
| procedural hybrid | IR2 adapted | 9,104 | 26 | 21 | 10 |

九条 Normal 轨迹完整经历窗口，窗口末每台还剩 1,364 字节；提前结束的 difficult/IR2
轨迹只覆盖其中198个活跃机器人事件，最后观测余额为每台1,888字节。这不是其第192轮余额。
十条轨迹共覆盖窗口内2,502个活跃机器人事件；恢复边界后有10,386个活跃事件、48,548字节
发射及669个相关边投递副本。这些都是 **Normal 观测**，不能称作真实断联后的恢复效果。

## 计算口径及边界

每轮四个机器人事件，窗口是 `512 <= event < 768`。使用实际 `emitted_length`、
`physical_links`、`delivered` 与 `spent`；先验证 Normal 投递确为实际发射与物理链路的交集，
再按冻结任务的 `isolation_ids` 标记发送方或接收方被隔离的边。

```text
static_blocked_copy[event, receiver]
  = normal_delivered[event, receiver]
    AND event_in_frozen_burst_window
    AND (sender_is_isolated OR receiver_is_isolated)
```

发送字节按每次发射只计一次；接收方副本可一对多。静态可受阻副本的 wire-byte 加权合计
115,939字节、payload 加权合计102,867字节，均不是额外发射或预算扣费。预算按 attempt
规则核对发射与扣费相等，余额使用窗口开始前的状态，而非首次窗口事件完成后的状态。

**这是在已观察 Normal 轨迹上叠加固定掩码，不是 Burst 环境 rollout。** 真实断联可能
从第一次失去消息起改变局部地图、队友估计、动作、物理链路和后续发送，所以本预审不能预测
真实丢失副本数、覆盖差、预算效率或恢复收益，也不能把817称为真实损失的上下界。
本步只回答这十项固定场景是否具有测量通信效应的机会。

## 已准备的下一道门槛，尚未执行

从原240项清单直接抽取的[十项任务清单](evidence/burst-exposure-preflight-20261118/selected-burst-tasks.jsonl)
保留原行字节及任务ID，SHA256：
`6499107bab45f48063b2772f73277235998f42c24311dd31aada8593411e07db`。
这是选择与身份记录，**不是新 GPU 授权、全量先导启动或正式论文协议**。Dropout 不在该十项中。

未来真实 Burst 验收应使用其本身的实际发射与物理链路计算可投递副本，再区分外部掩码
阻断的副本与实际送达；不能只用 Normal 总投递减去 Burst 总投递，误将路线变化全算成丢包。
逐项记录窗口发送/扣费、实际可受阻副本、恢复后活跃事件与投递、正常成功或超时，以及
相同 Normal 身份的配对差异。真实运行若出现零暴露，也要保留并解释，不事后换图或挪动窗口。

可在源码、配置、地图、权重和种子身份未变化时，沿用已验收 Normal 轨迹进行行为配对；
跨窗口墙钟不能当成严格性能对照。旧输入无法匹配时，应另行声明所需对照和预算，不能偷偷补跑。
不要求每个方法都探索成功，也不要求性能差必须显著。代码发布整合仍按既有 PR1→PR2
计划及明确合并批准执行；本步没有合并、改基分支或修改 Draft 状态。

## 复现与证据

- [逐场 JSON、输入哈希及运行身份](evidence/burst-exposure-preflight-20261118/preflight.json)
- [纯 NumPy 分析协议](protocols/phase6-20261118/audit_burst_exposure_preflight.py.txt)
- [审查与完整性记录](evidence/burst-exposure-preflight-20261118/review.json)
- [本次输出哈希索引](evidence/burst-exposure-preflight-20261118/artifact-index.json)
- 原始数据：[C1024 完整回合归档](evidence/capacity-check-v1-c1024-20261118/artifact-index.json)
  及 [v2 冻结清单](evidence/resource-pilot-v2-c1024-20261118/tasks.jsonl.gz)

在仓库根目录运行，输出必须使用不存在的新目录，避免覆盖任何证据：

```bash
python research/protocols/phase6-20261118/audit_burst_exposure_preflight.py.txt . NEW_OUTPUT_DIRECTORY
```

分析读取约1.9 MB压缩历史轨迹，数值处理耗时约0.06秒（不含解释器启动）；运行时为本机
Python3.13.5/NumPy2.1.3。历史环境计算来自目标 H100；本机没有重新运行环境，且未导入
JAX、PyTorch 或 benchmark 包。本次只新增分析协议、结果和报告，没有改变生产环境或模型。
自审还用显式逐事件、逐真实接收者循环复核十项投递计数和边界余额，结果一致；44个输入
文件逐一匹配工作区与已发布基线的 Git blob。该核对不等于重新执行或独立审计原始环境轨迹。
