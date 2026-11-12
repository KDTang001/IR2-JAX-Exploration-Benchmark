# Route A: Plan-assisted Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在真实仿真交互中训练探索与通信策略，并验证邻居计划预测和 ZOH 的差异。

**Architecture:** 共享参数循环 actor 只读 LocalObservation，集中 critic 读取独立特权输入。计划模型按接收者—邻居保留记忆；采样及一次 PPO 更新期间冻结预测器。

**Tech Stack:** JAX、Flax Linen、Optax、pytest；依赖环境子计划 E1–E8 和评测 V1。所有文件路径相对总计划根目录。

---

## A1：策略、critic 与通信动作

创建 agents/actor.py、agents/critic.py，测试 learning/test_actor.py。

- [ ] 定义 actor 输入：局部地图编码、[K,features] 候选及 mask、自身位置/余额、消息与年龄、[N-1,H,2] 计划估计及有效 mask。actor 参数共享，memory 按环境和机器人分开；critic 接口另收 PrivilegedState。
- [ ] 构建隐藏维度 256 的循环网络，输出航点分类、发送 Bernoulli、给定发送的消息类型分类。路线信息在选择航点后由本地导航器产生，类型头只读本地特征和该计划。无发送时类型 log_prob 与 entropy 均置零。
- [ ] 核心概率运算按下面契约实现，等待始终合法以防所有候选被 mask：

```python
masked_logits = jnp.where(valid_actions, logits, -jnp.inf)
log_probs = jax.nn.log_softmax(masked_logits)
joint_log_prob = waypoint_log_prob + gate_log_prob + jnp.where(send, type_log_prob, 0.0)
```

- [ ] 测试重复 seed 动作一致；被 mask 的航点概率为零；更改另一个机器人 memory 不改变当前机器人 memory；改变 privileged critic 输入不改变 actor 输出。
- [ ] 固定通信对照覆盖通信动作，不为强制门控/类型生成 PPO 梯度，但保持运动策略和观测接口相同。运行 `uv run pytest tests/benchmark/learning/test_actor.py -q`。

## A2：计划估计与标签时间

创建 models/plan_predictor.py、training/plan_losses.py；测试 learning/test_plan_predictor.py、learning/test_plan_labels.py。

- [ ] 定义 PlanEstimate 含目标位置、H 个计划位置、位置方差、有效长度分布、最后消息事件和 missing 标记。sender_plan_label 只存在训练批次 target 字段，不能在模型 input PyTree 中。
- [ ] 实现 ZOH：首次未收消息为无效，不填真实邻居起点；新 PLAN 到达后替换缓存；仅时间流逝不把时间戳改成当前时刻。POSITION 消息不能伪造完整 PLAN。
- [ ] 建立随机循环模型：deterministic=256、categorical=16×16；按事件用接收者可见输入推进先验，有对应有效测量才更新测量条件。参数共享，各接收者—邻居对的状态独立。
- [ ] 坐标归一化用地图有效宽高而非 padding 桶；方差设正下界。位置损失计算有效 mask 内的 Gaussian NLL，另加有效长度交叉熵与 KL，缺失标签掩码不产生零坐标监督。解码目标与路径是最新意图，不用未来实际轨迹顶替。
- [ ] 手工事件日志：0 在 e=0 生成 P0；1 在 e=1 更新前读取的 0 标签为 P0；0 下一次更新 P1 后，之后的截面才使用 P1。对未发送 P1 的接收者，训练 target 可变但 forward 输入不得变。此测试用两个不同 target 求前向输出应一致、loss 可不同。
- [ ] `uv run pytest tests/benchmark/learning/test_plan_predictor.py tests/benchmark/learning/test_plan_labels.py -q`。额外在固定小型轨迹集上拟合，确认有效样本损失下降并保存初末数值，而非只检查 loss 有限。

## A3：真实采样存储与事件 GAE

创建 training/storage.py、training/ppo.py；测试 learning/test_returns.py、learning/test_storage.py。

- [ ] 记录每事件 actor_id、LocalObservation 的采样特征、候选表、mask、动作、旧 log_prob、旧 value、reward、终止/截断、有效标记、采样时 memory 和预测特征版本。不能在 PPO 更新时用新预测器重新生成旧特征。
- [ ] 用非对称二步奖励 r=[1,2]、V=[0,0,0]、gamma=0.5、lambda=1、最后真实终止，手算 advantages=[2,2]。padding 事件不改变真实 prefix 的回报；终止后 bootstrap=0，采样段截断仍用下一事件 value，任务时间上限遵循有限任务剩余时限规则。
- [ ] 按事件反向 scan 计算 delta 和 GAE。每事件只有当前 actor 的 log_prob 被优化，所有事件 team reward 进入共享 critic；发送为 false 时类型头无梯度。
- [ ] 实现 ratio/clipped surrogate、value loss、有效动作 entropy、梯度裁剪。固定参数重算同一批采样时 ratio 应近似 1；故意更换旧候选表应被 batch 契约检查拒绝。
- [ ] `uv run pytest tests/benchmark/learning/test_returns.py tests/benchmark/learning/test_storage.py -q`，保存一个确定性批次的手算/实际回报对照。

## A4：训练调度与恢复

创建 training/train_a.py；CLI 增加 train-a、evaluate；测试 learning/test_train_a.py。

- [ ] 启发式预热采 100000 有效事件；微型冒烟配置覆盖为 256，不修改正式默认。真实事件计数包含预热，done padding 不计数。
- [ ] 每轮：固定预测参数 → rollout 128 事件 → PPO 更新 4 epoch → 更新计划模型 → 下一轮。逻辑批量通过微批/梯度累积适配显存，不隐式改变样本总额。
- [ ] checkpoint 原子写入临时文件后重命名，保存 actor/critic/预测器参数与优化器、PRNG、采样环境状态或明确边界重置策略、计数器和配置哈希。恢复时配置不兼容须报错，不悄悄换预算或地图集。
- [ ] 同一个固定训练批次更新前后 actor 参数必须变化；固定通信组的未训练门控头可不变。验证 actor 优化器没有接受 PlanLabel/PrivilegedState 作为输入特征。
- [ ] 运行 `uv run ir2-benchmark train-a --config configs/micro.json --seed 11 --events 4096 --output artifacts/a-smoke`，随后 `uv run ir2-benchmark evaluate --checkpoint artifacts/a-smoke/checkpoint --suite micro --output artifacts/a-smoke/eval`。要求有真实 episode、事件/预算计数、checkpoint 和独立验证结果，不能只执行随机前向。
- [ ] 停在批次边界恢复，与未中断运行的下一批 PRNG/计数/参数更新对照；运行 `uv run pytest tests/benchmark/learning/test_train_a.py -q`。

## A5：四组对照与诊断

创建 configs/a/{fixed_zoh,fixed_predict,learned_zoh,learned_predict}.json；测试 integration/test_a_comparison.py。

- [ ] 配置由共同基础配置派生，只改 communication_mode 与 neighbor_estimator。对比声明真实事件额度、地图划分、预算、奖励和感知规则的配置哈希字段一致。
- [ ] 先在微型清单上运行四组，同一组种子/地图/故障清单。分别报告覆盖、成功率、实际字节、计划误差及消息年龄，不用实际通信量一致作为隐含前提。
- [ ] actor 使用的计划误差只在有训练/评测 target 时统计；无消息冷启动样本不能被丢掉以美化平均误差。按年龄和 Burst 阶段分桶。
- [ ] 运行 `uv run pytest tests/benchmark/integration/test_a_comparison.py -q`；完成 artifacts/a-smoke/comparison.json 与回放。优势未出现时保留结果，检查信息权限与账本后才调验证参数。
