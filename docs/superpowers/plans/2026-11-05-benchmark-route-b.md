# Route B: Imagined Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 训练世界动态模型，用其想象轨迹更新分散 actor，并在真实仿真测试集评测。

**Architecture:** 中心 RSSM 仅用于训练动态预测/critic，各机器人 actor 保持本地观测接口。预算和时间精确推进，动态与送达预测更新模拟本地状态，随后重建航点候选。

**Tech Stack:** JAX、Flax Linen、Optax；复用 E1–E8、A1/A2 接口和 V1。算法参考 DreamerV3，但此多机器人适配不声称是官方实现。

---

## B1：真实回放与训练序列

创建 training/replay.py；测试 learning/test_replay.py。

- [ ] 主机回放使用周期性局部地图快照和事件增量，保存计划标签、消息、位置、动作、reward 和终止。GPU 不存 [batch,time,N,H,W] 的全量地图历史。
- [ ] 事件序列长度 32，前 8 步用于烧入；采样不得把不同 episode 的最后/第一帧拼成连续转移。包括短 episode 时使用有效 mask，不把 padding 学成静止转移。
- [ ] 用 5×5 人工地图依次感知两格、收到一条地图消息、再消费 inbox；从快照加增量重建每个事件状态，与原始完整记录逐字段一致。
- [ ] 时间截断与真正覆盖结束用不同标记；模型 continuation target 只对真实 terminated 置零，任务上限由已知时限控制。
- [ ] `uv run pytest tests/benchmark/learning/test_replay.py -q`。验证同 seed 恢复后的采样索引一致，以及有缺失/损坏增量时明确报错。

## B2：动作条件世界模型

创建 models/world_model.py、training/world_losses.py；测试 learning/test_world_model.py。

- [ ] 固定初始维度 deter=256、stoch=16×16，输入事件 actor_id、动作及历史状态。训练测量编码可读取集中监督，actor 不直接消费该隐状态。
- [ ] 输出头：当前行动者位置/局部观测增量、每接收方送达概率、team reward、continuation。地图头按当前行动者 2R+1 窗口瓦片解码；未预测格保留既有认知，不能补为已知自由。
- [ ] 总损失为 masked 动态 KL、局部栅格交叉熵、位置损失、送达 BCE、reward 回归及 continuation BCE。所有 mask 分母为有效样本数；无有效样本贡献零而非 NaN。
- [ ] 纯函数模型前向参数包含 obs/action/history，不允许 EnvState 真值地图作为 imagine 参数。监督训练可有 privileged_target，但只能进入损失和训练测量编码。
- [ ] 用同隐状态、两个不同合法动作验证预测发生变化，并通过真实轨迹小集拟合验证学习；不同随机噪声下只要求分布一致而非单样本相同。
- [ ] `uv run pytest tests/benchmark/learning/test_world_model.py -q`，保存 held-out 轨迹的 1/3/5 轮位置、局部图、奖励、送达误差，不能只报训练重建误差。

## B3：闭合的想象状态转移

创建 training/imagination.py；测试 learning/test_imagination.py。

- [ ] 定义 ImaginedState：中心 latent、每机器人模拟本地地图/位置/余额/inbox/局部记忆、当前事件及有效标记。初始化于真实事件前截面；真值地图、未来故障日程和未来实际消息不复制进 actor 可读字段。
- [ ] 每 imagined event 顺序执行：生成当前机器人 local_obs → actor 动作与本地规划 → 锁定载荷 → 世界模型预测动态 → 精确预算/容量约束 → 预测送达门控 → 更新接收者 inbox → 重新计算下一机器人候选 → 下一 event。
- [ ] 运动预测约束为当前规划路径内至多 d_max 的可达点；不能从预测方差随机跳进已知障碍。新局部观测由模型生成，地图消息内容只来自模拟发送者本地地图。
- [ ] 测试余额 44 时 45 字节包永不送达，即便送达头概率为 1；余额 50、送达头为 0 时余额变 5。使用小型可控预测输出作为转移输入，不 mock 整个 imagine 返回值。
- [ ] 设计包含地图增量的两步例子，验证第二步候选会变化；只复用第一步候选的错误实现必须被该测试检出。改变存于评测器、未传入 imagine 的真实地图不得改变想象输出。
- [ ] `uv run pytest tests/benchmark/learning/test_imagination.py -q`，jit/scan/vmap 对 5*N 事件核验 agent_id 顺序与终止冻结。

## B4：想象 actor-critic 更新

创建 training/dreamer.py、training/train_b.py；测试 learning/test_imagined_update.py。

- [ ] 将模型参数冻结，用世界模型预测 reward/continue 与精确时限构造 lambda-return，默认 gamma=0.99、lambda=0.95；送达和航点为离散事件，使用 score-function policy loss，不宣称对环境 argmax 自动反传。
- [ ] 核心 actor loss 按有效当前行动者聚合：

```python
advantage = jax.lax.stop_gradient(returns - values)
weighted = valid * continuation_weight
actor_loss = -(weighted * joint_log_prob * advantage).sum() / jnp.maximum(weighted.sum(), 1)
```

补充配置声明的 entropy 项；critic 回归 returns，critic 可读中心 latent，actor 输入保持 A1 的 LocalObservation。不能给 actor 拼上 critic 的全局特征。
- [ ] 在无真实环境 step 的测试更新中，使用非零优势的想象轨迹验证 actor 参数改变、世界模型参数不变；如果误用真实 rollout 更新，测试应无法获得所需环境对象并失败。
- [ ] 完成训练循环：采真实事件 → 模型更新 → 从回放采起点 → 想象 5*N 事件 → actor/critic 更新。计数器独立记录 real_events、imagined_events、model_updates、actor_updates。
- [ ] 执行 `uv run ir2-benchmark train-b --config configs/micro.json --seed 11 --events 4096 --output artifacts/b-smoke`，输出 checkpoint 和计数，随后使用真实环境 evaluate。想象 return 上升不能替代真实测试。
- [ ] `uv run pytest tests/benchmark/learning/test_imagined_update.py -q`，保存一次想象专属更新前后参数差及真实环境调用计数为零的证据。

## B5：对照、模型偏差与恢复

创建 configs/b/{fixed_zoh,fixed_predict,learned_zoh,learned_predict,real_updates}.json；测试 integration/test_b_comparison.py。

- [ ] B-ZOH 禁止给 actor 提供预测邻居计划，但保留环境动态模型来模拟交互；报告这不是完全无模型基线。固定通信组发送动作不参与 actor 梯度。
- [ ] real_updates 保持相同世界模型训练与本地输入，只将 actor/critic 训练轨迹改为真实回放；记录所有优化差异，以判断想象训练的贡献。
- [ ] 对比真实起点的短期开环预测与真实执行，用共同固定动作脚本而非不同策略动作，把模型误差与动作差异分开。观察 1/3/5 轮误差增长。
- [ ] checkpoint 包含世界模型、局部计划模型、actor/critic、所有优化器、PRNG、回放索引与采样状态；微批和序列烧入设置写配置哈希。
- [ ] 运行 `uv run pytest tests/benchmark/integration/test_b_comparison.py -q`，并生成 artifacts/b-smoke/comparison.json。A、B 在相同真实事件预算下用 V2 同一清单测试，报告计算成本，不以训练时间相同偷换样本效率。
