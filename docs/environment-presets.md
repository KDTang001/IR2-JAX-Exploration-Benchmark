# 环境预设和方法自定义

旧 JSON 配置无需增加字段：仍为团队覆盖增量奖励、障碍衰减信号、一跳广播、有效尝试扣费，由动作决定发送内容和时机。出生与 99% 团队物理覆盖结束规则不变。

## 配置选择

以下字段可添加到现有环境 JSON；不是训练器配置：

```json
{
  "reward_model": "coverage_distance",
  "reward_coverage_weight": 100.0,
  "reward_distance_weight": 0.01,
  "signal_model": "los_range",
  "communication_range": 100.0,
  "exchange_mode": "broadcast",
  "charge_mode": "attempt",
  "send_policy": "periodic",
  "send_period": 4,
  "message_codec": 3
}
```

`send_period` 以每个机器人的行动轮次计，首次轮次 0 发送。周期内仍需通过预算、物理链路和故障检查。

| 字段 | 选项 |
|---|---|
| reward_model | coverage_delta / coverage_distance / sparse_success / ir2_individual / 已注册自定义名称 |
| signal_model | attenuation / ideal / range / los_range / ir2 / 已注册自定义名称 |
| exchange_mode | broadcast / addressed |
| charge_mode | attempt / success |
| send_policy | action / none / always / periodic / 已注册自定义名称 |
| message_codec | 0 保留方法动作；1 位置、2 目标、3 计划、4 地图增量 |

**奖励**：`coverage_delta` 默认严格保持 `100 × 覆盖率增量 − 1/(机器人数量 × 512)`；可配置覆盖权重和 `reward_event_cost`。`coverage_distance` 默认使用覆盖增量和距离成本，事件成本默认 0。`sparse_success` 仅成功事件发放 `reward_success_bonus`（默认 40），超时无成功奖励，结束后的调用奖励为 0。

`ir2_individual` 要求 `navigation_mode=ir2_graph`，使用原版动作前目标节点效用、团队合并图消失的旧前沿数和实际距离：`消失前沿/25 + 探索效用/50 + 会合效用 − 距离/512`。**它不包含原版整轮结束后给每个机器人加 40 的团队奖励，也不改变本 benchmark 的完成判据。** 因此没有发布名为 `ir2` 的“完整原版奖励”预设。默认系数与原版公式对照，状态来源仍遵循本 benchmark 的付费通信约束。

**物理信号**：`range` 的距离边界包含等号；`los_range` 还要求线段穿过障碍的长度为 0，单独角点接触不阻断。二者及 ideal 不定义 dBm，诊断字段 `signal_strength_valid=False`。`attenuation` 保留现有线段/格子交长计算；`ir2` 使用原版 skimage Bresenham 像素计数，排除起点、包含终点，阈值使用严格大于。

两个衰减模型支持 `transmit_power`、`reference_loss`、`gamma_free`、`gamma_obstacle`、`reference_distance`、`signal_threshold`。IR2 额外支持 `signal_noise_free_min/max` 与 `signal_noise_obstacle_min/max`；两个噪声参数每个 episode 只采样一次，所有链路和时刻复用。随机数生成器为 JAX，跨 NumPy/Python 和 JAX 的相同整数种子不保证噪声样本相同；固定噪声参数可直接数值对照。

正常、随机丢包、突发断联仍由原有 `fault_mode` 单独选择，不互相混合。ideal 仅取消物理信号限制，不取消预算或故障。

## 地址、账本和消息内容

addressed 模式要求 `make_action(..., recipients=布尔数组)`，数组长度为 `config.n_slots`。只选一个即单播，可同时选择多个。选择自己、非活跃槽位或空集合会使包无效，不扣费、不送达；broadcast 模式不能偷偷接受定向地址。自定义方法需自行输出接收者，内置 A/B 尚无接收者动作头。

有效发送总费用为 16 字节包头加实际负载长度，一次广播或定向多接收者发送只收费一次。attempt 模式失败也收费；success 模式仅至少一人实际收到才收费。两种模式都预先检查足额预算，不能用 success 模式透支发送。物理送达后即视为成功，即使接收者随后不使用或丢弃旧消息。未送达但有效的尝试仍推进消息序号和 MAP 游标。

标准编码器位于 `ir2_benchmark.codecs.standard`，自定义内容使用 codec 128–255 和最多 240 个负载字节。更大内容需自行分包，每包都计费。转发使用现有 `forward_packet` 并再次显式发送；不会自动多跳扩散。

自定义负载原样进入已送达收件箱，接收方法自行解码。仅标准 MAP_DELTA（包括显式转发的 MAP_DELTA）能够更新环境中的地图知识并参与图融合。特征或 PLAN 不会获得队友地图。学习特征必须序列化，不能通过环境外共享张量绕过字节预算。

MAP_DELTA 的 wire 坐标按 uint16 校验；接收时仅合并 `0 <= x < valid_shape[1]`、`0 <= y < valid_shape[0]` 的条目，真实地图外的静态 padding 保持未知。越界条目仍占用发射字节，不退款。直接调用 `merge_local_messages` 的自定义 padded-map 方法须传入 `valid_shape=...`；省略时视整个数组为未 padding 地图。

## 自定义函数契约

在创建配置和编译前调用 `register_preset(kind, name, function, version="1")`。kind 为 reward、signal 或 sending；禁止覆盖已有名称，不从 JSON 执行或导入任意代码。命令行运行自定义方法时，由方法的 Python 入口注册，再调用环境或训练 API。

- reward：`function(RewardContext, HostConfig) -> RewardResult`。返回 float32 标量及固定 `(6,)` 分项；内置分项依次为覆盖、距离、事件、成功、前沿、节点效用。自定义方法记录自己的分项解释。可用统计量包括前后覆盖、距离、花费和成功；图统计仅在 `reward_graph_statistics=true` 或 ir2_individual 时计算，开启要求 ir2_graph。
- signal：`function(LinkContext, HostConfig) -> LinkResult`。输入真实占据图、两个端点、独立事件/接收者 key 及 episode_key，输出可用性、强度和强度有效标志。只有物理模型获得真实地图。
- sending：`function(LocalObservation, Action, HostConfig) -> Action`。只提供局部信息；可以决定内容、时机和接收者。返回的 waypoint 不会覆盖原方法运动动作。接收端通过 LocalObservation 的 inbox 字段解码自定义内容。

自定义数值参数用 `custom_parameters: [["byte_cost", 0.01]]`，载入时转为可哈希的元组。更复杂的方法状态由方法自己显式维护并生成 Action，不在这些无状态钩子里使用可变全局状态。

函数必须为纯 JAX 运算、固定数组形状，随机性使用传入 key。更换实现或容量可能重新编译。注册表仅在主机/追踪阶段解析，不在设备上回调 Python。返回类型/形状不匹配会在编译时失败。

内置 A/B 训练器支持标准广播消息；选择 none/always/periodic 或固定 message_codec 时必须关闭学习通信头。固定策略已接入真实采集、动作记录和 B 的想象通信账本。自定义负载、定向接收者或自定义发送函数需方法自己的策略/回放适配器，不能直接塞进现有固定维度的 A/B 动作编码。

可运行的最小例子（只执行一个环境事件，不训练）：

```sh
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/preset_example.py
```

## 实验记录与恢复

训练配置和评估场景记录展开配置、预设名称/版本、源码 SHA-256。内置实现摘要覆盖包内源码依赖；自定义默认摘要覆盖函数源码，涉及外部依赖或闭包时必须传入包含这些依赖的 `source_hash` 并维护版本。新产物的恢复/读取检查摘要，不匹配时要求显式迁移或新实验，避免无声改变环境语义。

评估的原始指标保持统一。正式比较固定物理信号、故障、容量、扣费等环境条件；方法可以选择训练奖励和协议允许的消息策略。不同 reward_model 的累计回报不能直接比较。使用 `StepInfo.reward_terms/physical_links/external_links/signal_strength/signal_strength_valid` 检查变化来源。
