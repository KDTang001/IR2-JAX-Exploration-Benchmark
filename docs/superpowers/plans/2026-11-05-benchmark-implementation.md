# Communication-limited Exploration Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现已批准的 benchmark，包括 JAX 环境、A/B 两条学习路线和公平、可复现的退化评测。

**Architecture:** 新增独立 ir2_benchmark 包，原 IR2 实现保留为只读参考。按有序机器人事件推进单环境，批量轴只用于独立环境；真实环境与想象模型共用消息、预算和动作契约。

**Tech Stack:** Python 3.12、JAX、Flax Linen、Optax、NumPy/Pillow、pytest；uv 管理独立环境与锁文件。CPU 做独立参考与小规模验证，Linux/WSL2 NVIDIA 运行训练。

**2026-11-05 执行环境覆盖：** 用户后续指定 `<user>@<server>`、conda `ex`、`~/Tz`。实际采用 `~/Tz/ir2-benchmark`、Python 3.11、H100；保留现有 conda 环境，使用 `python -m pip` 在该环境内安装受现有依赖快照约束的包。下文及子计划的 `uv run python`/`uv run pytest`/`uv run ir2-benchmark` 命令分别替换为 `python`/`python -m pytest`/`ir2-benchmark`；不再新建本机或 uv 环境。安装入口与清理规则见 `docs/server-running.md`。

---

## 使用与状态

根目录：E:/method/IR2-Multi-Robot-RL-Exploration。本文所有相对文件路径均相对于该根目录；文件树中的新路径是待创建的目标，不声称已经存在。在当前合适的 checkout 中工作，无需为了命名新建 worktree。未获请求时不自动提交或推送 Git。

本计划引用的两个 superpowers 执行辅助技能未在当前技能目录发现；这不是项目代码依赖。默认由当前会话按下面的检查清单顺序执行，采用已可用的工具和测试，不宣称已加载缺失技能，不自行启动其他 agent。

本轮仅完成计划。所有实现项初始未勾选。代码、训练、GPU 可用性和性能结果都不能因存在计划而标记完成。

当前事实：Windows Python 为 3.13.5，没有 JAX/Flax/Optax；有 NumPy、Pillow、pytest；uv 未在 Windows PATH 检出。已有 WSL2 Ubuntu-20.04，检查时为停止状态。GPU 为 RTX 4080 Laptop、12282 MiB。WSL 内 GPU、Python、uv 和依赖尚未检查，不能把原生 Windows CPU 测试结果当作 GPU 结果。

## 子计划与依赖

1. [共享环境与协议](2026-11-05-benchmark-environment.md)：运行时、契约、地图、感知、导航、通信、事件循环及微型真实回放。
2. [A 路线](2026-11-05-benchmark-route-a.md)：循环策略、计划预测、真实交互 PPO 与通信消融。
3. [B 路线](2026-11-05-benchmark-route-b.md)：世界动态模型、闭合想象状态、想象 actor-critic 与无想象对照。
4. [统一评测与交付](2026-11-05-benchmark-evaluation.md)：指标、配对场景、矩阵、吞吐和复现入口。基础指标先于 A/B 使用，最终矩阵在两条路线通过后执行。

四个子计划是一个交付的分解，完成环境或 A 不代表整个请求完成。预实验和正式矩阵分开记录，不能用微型地图训练替代正式评估。

## 统一文件与契约

```
pyproject.toml                     # 独立包、CLI、pytest testpaths
uv.lock                           # 实测解析出的依赖锁，不手写版本猜测
src/ir2_benchmark/
  config.py                       # 主机端不可变配置及解析
  types.py                        # JAX 状态/动作/观测 PyTree
  maps/{manifest,generate,prepare}.py
  sensing/{reference,jax_sensor}.py
  navigation/{reference,jax_path,candidates}.py
  communication/{ledger,signal,faults,channel}.py
  codecs/{packet,standard}.py
  env/{reset,observe,step,rollout}.py
  agents/{heuristic,actor,critic,fixed_comm}.py
  models/{plan_predictor,world_model}.py
  training/{storage,plan_losses,ppo,train_a,replay,world_losses,imagination,dreamer,train_b}.py
  evaluation/{metrics,scenarios,runner,report,throughput}.py
  cli.py
configs/{micro,default}.json
tests/benchmark/{unit,integration,learning}/
artifacts/                         # 实验产物，忽略大文件，仅报告清单入库
```

坐标统一 int32 [x,y]，地图下标 [y,x]；外部传输坐标 uint16，小端。FREE=0、OCCUPIED=1、UNKNOWN=2；真值仅允许前两类。本地地图 [N,H,W]，真实地图 [H,W]，位置 [N,2]，预算 int32 [N]。单个发包缓冲 uint8 [256]，长度 int32；收件箱 [N,N,256] 以接收者/直接发送者索引，自环无效。逻辑 N 与静态桶 N 分开，填充机器人永不行动或发包。

HostConfig 校验 N≤6、坐标范围、合法桶、R/d_max/T、消息长度。JAX 内使用已验证静态配置，无 Python 异常控制正常转移。状态含 event、round、current_agent、terminated、truncated、PRNG、覆盖足迹和每机器人认知；不存 Python 图或对象引用。

```
reset(map_data, config, key) -> EnvState
observe(state, agent_id, config) -> LocalObservation
step_agent(state, action, config, key) -> StepResult
policy_step(params, memory, observation, key) -> (Action, new_memory, PolicyInfo)
```

observe 不消费消息、不更改状态，step_agent 返回的新状态才标记已消费消息。策略构造计划及锁定载荷只读 LocalObservation；独立 HostInfo/PrivilegedState 不进入 actor。

## 通用开发步骤

每个任务先写列出的行为断言，运行对应命令确认尚未实现的行为失败，再实现目标模块并重新运行。失败应来自真实缺陷或缺失接口，不能用 mock 返回期望值制造通过。只在相关测试通过后进行该任务的 CLI/训练场景。当前步骤产物、命令、退出码和证据路径记录到 artifacts/development/evidence.jsonl；对原文件的用户修改不回退。

计划中的伪批量替代不可接受：所有环境热点必须可 jit/vmap，不得用 Python 环境循环或 CPU callback 代替核心推进。代码注释仅解释约束，不复述代码。

## 设计覆盖与验收门槛

| 设计部分 | 负责任务 | 通过标准 |
| --- | --- | --- |
| 栅格、遮挡、移动、航点 | E3–E5 | 独立参考逐格/逐路径对照 |
| 固定顺序和消息时序 | E7–E8 | 同轮后行动可收信，前行动不可重做决策 |
| 字节成本与故障 | E2、E6 | 失败扣费、预算非负、隔离双向正确 |
| 数据划分与生成 | E3 | 全量 manifest，无静默替换或跨集合重复 |
| A 真实训练 | A1–A5 | 真实轨迹、PPO 更新、计划标签权限检查 |
| B 想象训练 | B1–B5 | 不读隐藏真值的想象、actor 更新及真实环境测试 |
| 公平矩阵与指标 | V1–V4 | 配对条件、真实交互额度、失败案例及置信区间 |
| 性能与部署 | E1、V4 | GPU 验证、同步计时、编译与稳定吞吐分开 |

## 执行边界

- [ ] 共享环境可独立运行并提供 JSON 与轨迹图。
- [ ] A 固定种子的小地图训练/验证闭环完成。
- [ ] B 固定种子的小地图训练/验证闭环及想象更新证据完成。
- [ ] 预实验默认值经验证集检查后冻结，保留变更理由。
- [ ] 核心正式矩阵、负结果、运行成本和复现说明齐备。

性能与算法优势是测量结果，不作为可伪造的完成标准。正式矩阵成本超出本机能力时保留可恢复清单和已完成结果，明确哪些运行未执行，不把排队/配置完成说成实验完成。
