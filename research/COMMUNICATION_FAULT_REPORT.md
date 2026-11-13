# Phase 2：通信协议与故障模型验收

日期：2026-11-14。分支：`audit/benchmark-foundation-2026-11-12`。基线提交：`10a6fa5e31e465bc3bddbeb872ed325f716810a7`。

## 1. 结论与验收范围

**Phase 2 在声明的 Windows CPU 范围内验收通过。** Normal、Dropout、Burst 分别作为独立条件运行，未混合故障。完成标准协议、真实字节账本、收件箱、物理信号/外部故障分离、接收知识和付费地图融合检查。

最终通信验收集合 **118 项通过、无失败、无跳过**；另有 **28 项 Phase 1 相关集成测试复验通过**。合计 **146 个不同测试**，不是完整全仓库测试。检查不包含训练、正式地图矩阵或 Linux/GPU 性能；这些仍属于后续阶段。

发现并修复一个实际缺陷：接收 MAP_DELTA 时只检查数组尺寸，未检查真实地图 `valid_shape`，可能把批量 padding 写成已知栅格。三条接收调用链已统一修复，保留失败与复验记录。

测试清单、版本、源码 SHA-256、初次失败、每事件 wire bytes/费用/送达记录见 [机器可读证据](evidence/phase2-20261114.json)。前一阶段结论见 [IR2_PARITY_REPORT.md](IR2_PARITY_REPORT.md)。

## 2. 协议、账本与收件箱

源码：[packet.py](../src/ir2_benchmark/codecs/packet.py)、[standard.py](../src/ir2_benchmark/codecs/standard.py)、[channel.py](../src/ir2_benchmark/communication/channel.py)、[ledger.py](../src/ir2_benchmark/communication/ledger.py)。

| 项目 | 当前规范与实际检查 |
|---|---|
| 包头 | 16 字节，小端 `<BBBBIIHBB`；version、codec、sender、flags、sequence、generated_event、payload_length、recipient_mask、reserved |
| 容量 | 载荷最多 240 字节，整包最多 256；静态数组 padding 不计费 |
| POSITION / GOAL | 两个 uint16 坐标，整包 20 字节；非对称及 65535 坐标与 Python struct 对照 |
| PLAN | source、goal、有效长度及固定 H 个坐标；H=5 时 45 字节，尾部保持点仍传输并收费 |
| MAP_DELTA | count 加每条 x/y/state；整包 `17+5*count`，最多 47 条即 252 字节；仅自由/障碍状态合法 |
| 自定义消息 | codec 128–255，内容按原字节进入付费收件箱；不自动获得地图语义 |
| 转发 | 单层 6 字节 origin 封装；保留原作者/时间，每次显式发射再次收费；PLAN 转发为 51 字节；超容量及嵌套封装无效 |
| 结构错误 | 错误版本、flags/reserved、身份、长度、地址、PLAN 长度、MAP count/state 等被拒绝；实际 channel 不扣费、不修改 inbox |
| 广播 | 只发给其他有效机器人，一次发送只收费一次；inactive padding 不接收 |
| 定向交换 | `addressed` 才允许显式收件者；空集合、自身和非活跃地址无效。不能用 broadcast 偷换定向地址 |
| 收件箱 | `[receiver, direct_sender]` 单槽；新包按外层 generated_event、sequence 比较，只覆盖更新的未读包；记录 overwrite |
| 消费与缓存 | observe 为纯函数；step 只消费行动机器人的 unread 标志，已解码知识持续保留，不清空其他机器人收件箱 |

结构校验只看公开字节，不查隐藏真值。全 uint16 坐标在 wire 层合法；越过真实地图边界的地图条目由接收层忽略。协议未提供内容真实性认证；不能把“格式合法”表述为地图内容一定真实。

旧包物理送达仍计入 delivered，但可能不覆盖更新的未读包。已消费槽位可接受后到的旧包，解码的 POSITION/GOAL/PLAN 缓存按原作者生成时间拒绝倒退；本地地图按格时间门控。队友地图缓存不是动态占据图的时序融合器，本项目采用静态地图假设。

### 两种扣费预设

- `attempt`：请求、结构、身份和余额全部通过即按整包实际字节扣一次；全部丢包或无物理接收者也收费。
- `success`：仍须预先足额预算，至少一位接收者物理送达才收费；随后被应用丢弃或未读槽拒绝覆盖不退款。
- 预算不足、无效包、不请求发送均不扣费；接收不收费，零预算仍能接收、移动、感知。
- 有效但未送达的发送仍推进 sequence 和 MAP 轮转游标；没有 ACK，不假装已成功送达，也不自动重发。

默认规范为 attempt；success、addressed、自定义消息属于明确的 Benchmark 预设/扩展，比较方法时须固定环境配置，不能跨账本模式直接比较分数。

## 3. 三个独立故障条件

源码：[faults.py](../src/ir2_benchmark/communication/faults.py)、[config.py](../src/ir2_benchmark/config.py)、[step.py](../src/ir2_benchmark/env/step.py)。

| 条件 | 实现与验收 |
|---|---|
| Normal (`fault_mode=0`) | 只应用基础物理链路；外部 allowance 全真；配置拒绝非零 dropout_probability |
| Dropout (`fault_mode=1`) | 在基础可达的有向接收链路上额外丢包；独立接收者采样；不应用 Burst 隔离 |
| Burst (`fault_mode=2`) | 一次选取 floor(N/2) 个活跃机器人；窗口内屏蔽其入站和出站；其他机器人间仍可通信；配置拒绝非零 dropout_probability |

窗口使用全队轮次：`burst_start <= round < burst_start+burst_duration`。真实 step 检查了 r=127、128、窗口末轮和恢复首轮，覆盖开始 128、持续 64/128 轮两档，且分别检查隔离组内外的发送者。

N=2/3/4/6 的隔离选择与独立随机排序对照，数量为 floor(N/2)，inactive 槽不入选。连续事件测试验证断联不停止移动或扫描、不清空既有知识；恢复只放开新消息，不自动补发断联期间丢失的包。

故障 key 按 root key → purpose=FAULT → episode → event → sender → receiver 逐级 fold_in；不使用成功次数、sequence、MAP 游标或策略采样次数。现有测试核验调用历史无关、用途分离及 `jit(vmap)` 的逐接收者一致性。新增轨迹逐事件用独立展开的 key 配方重算 Dropout 结果。

另外在 p=0.1/0.3/0.5 下分别检查 4,096 个事件×4 个接收者的边际丢包率和两两联合频率。检查为固定种子的统计 sanity check，不是随机独立性的数学证明。实际频率见证据 JSON。

源码追踪确认内置 A 采集使用 `env.key`、B 采集使用环境 fault key；评估分别传入 spawn/policy 和 fault key。自定义调用者仍须保持固定场景 fault key，不能按动作修改 `step_agent(key=...)` 来改变未来故障样本。

## 4. 物理信号与外部故障分离

源码：[propagation.py](../src/ir2_benchmark/communication/propagation.py)、[signal.py](../src/ir2_benchmark/communication/signal.py)、参考 [ss_realistic_model.py](../ss_realistic_model.py)。

| 信号预设 | 边界与对照 |
|---|---|
| ideal | 物理可达全部为真，仍受预算、地址、活跃状态及故障约束 |
| range | 欧氏距离 `<= communication_range`；无有效 dBm 强度 |
| los_range | 距离条件加连续几何墙体长度为零；只触墙角不阻断 |
| attenuation | 连续线段与整数中心格的交长分段衰减；障碍/自由长度与逐墙格 slab oracle 对照；严格强度大于阈值 |
| ir2 | 复现 skimage.draw.line 像素计数，排除源格、包含终点；自由距离=欧氏距离−障碍格数；严格强度大于阈值 |

`attenuation` 是 Benchmark 连续几何模型；`ir2` 是原版像素计数模型。二者遇墙时可能产生不同强度/连接，必须作为不同预设，不能声称数值等价。

本轮重跑原有独立几何与 IR2 128 个线段对照；IR2 同时与原版布尔可达性和独立强度算式比较（强度容差 2e-5）。测试固定噪声值以对齐数值；随机 X_g/K 使用 JAX episode key 每个 episode 采样一次，各方向和事件复用。Python random 与 JAX 的同整数种子不保证产生相同样本，不能宣称同种子原版随机数逐位一致。

新增 **5 种物理模型×3 个独立故障条件=15 个真实 step 组合**，逐项对比 physical_links、external_links、最终 delivered 和费用。最终送达是结构/预算、地址、活跃及非自身掩码，与物理、外部掩码的交集。故障不能改变物理强度；ideal 不能取消外部故障。强度与这些掩码保留在诊断 StepInfo，不作为 actor 免费可用的连接矩阵。

## 5. 独立逐事件重算轨迹

测试：[test_phase2_acceptance.py](../tests/benchmark/integration/test_phase2_acceptance.py)。

每个独立条件使用相同真值、出生、root key、固定运动指令：4 个有效机器人、6 个静态槽、8 轮共 32 个事件，执行一格往返移动。轮次依次轮换 POSITION、GOAL、PLAN、MAP_DELTA。Burst 缩短为 `[2,4)` 轮以在小夹具内覆盖开始、持续和恢复；Formal 128 轮边界另有实际 step 检查。

每个条件分别运行 attempt/success 两个账本预设，共 6 条完整轨迹：

| 条件 | 事件数 | 接收方送达次数 | attempt 总发射字节 | success 总计费字节 |
|---|---:|---:|---:|---:|
| Normal | 32 | 96 | 1,856 | 1,856 |
| Dropout (p=0.3) | 32 | 66 | 1,816 | 1,796 |
| Burst | 32 | 76 | 1,456 | 1,252 |

一次广播可产生多位接收者送达，送达次数不是发送次数。success 列是费用，不是实际发射流量；失败发射在该预设中可免费，因此两者需要分开报告。MAP 内容随接收知识变化，包长也会变化，上表字节数不能用于宣称某个故障条件“提高了通信效率”。

Host oracle 不调用生产 codec/ledger/fault 解码函数：使用 Python struct 解析真实发送字节，独立重算费用和余额、消息可用时刻、生成时间、PLAN 缓存、地图逐格知识、MAP 选择/轮转及每接收者送达。单位半径感知在夹具中用手工五格扫描重算。每步核对知识和物理足迹，证明确实发生移动、接收地图不变成物理覆盖、padding 不参与通信。

这组精确通信 oracle 使用 grid_path 控制运动以获得可独立手算的轨迹，不作为 IR2 图导航 parity 的新证据。IR2 图模式的付费融合六组合、图观察、历史及独立事件回归另外重跑，覆盖 fusion 开/关、PLAN 代替 MAP、丢包和余额不足。

## 6. COMM-001：MAP_DELTA 真实边界缺陷与最小修复

复现条件：数组 12×20、真实有效地图 8×12，向接收者发送 `[5,3]` 与 `[15,3]` 两个地图条目。原实现接收两个条目；后者虽然位于数组中，却已在真实地图之外。首次运行对应断言失败：本应未知的 `[3,15]` 实际变为自由。

修复在 `merge_local_messages` 增加 keyword-only 可选 `valid_shape`，局部地图和队友地图缓存都按真实边界过滤。真实 observe、imagined_observe、录像 frame_fields 三个现有调用点均传入真实形状。无该参数的独立调用按未 padding 数组处理，兼容既有五参数调用；第三方 padded map 调用必须显式传入形状。

回归检查确认：合法条目保留，越界条目在三条路径中保持未知；整包仍扣 27 字节；没有为该修复增加状态字段、修改 wire 协议、改变故障采样或启动 B 训练。对 B 代码的一行修改只是传递同一个接收边界参数。

## 7. 运行与证据边界

运行环境沿用 Phase 1 的独立 Windows Python 3.11 CPU 虚拟环境。修复后的检查设置 `PYTHONPATH` 指向当前分支 src，避免误测旧 wheel；版本和实际导入路径记录在证据中。ruff 仅装入临时审计环境用于格式检查，没有改项目依赖。

从仓库根目录，PowerShell 运行最终通信集合：

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$env:JAX_PLATFORMS = 'cpu'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:MPLBACKEND = 'Agg'
$env:IR2_PHASE2_EVIDENCE = "$env:TEMP/ir2-phase2-events"
python -m pytest tests/benchmark/unit/test_packet.py tests/benchmark/unit/test_channel.py tests/benchmark/unit/test_ledger.py tests/benchmark/unit/test_signal.py tests/benchmark/integration/test_presets.py tests/benchmark/integration/test_phase2_acceptance.py -k 'not collector' -q -p no:cacheprovider
```

相邻阶段回归为 `test_event_step.py`、`test_ir2_contract.py`、`test_phase1_acceptance.py`，排除会训练模型的 `graph_training` 案例。与新增部分测试分批执行并按测试 ID 去重，146 不包含重复运行次数。旧 Matplotlib/Pyparsing 弃用警告不当作数值失败，也不为消除它们扩大修改。

本次通过表示：所列包结构、预算规则、独立故障条件与接收知识变化有可复算证据。自定义算法可以沿用标准编码器或 opaque codec，但应固定预设身份、遵守字节容量及接收边界。正式公平评测、完整地图划分、方法基线和跨设备复现仍需后续阶段验收。

下一阶段为 **Phase 3：地图数据集、划分、重复与泄漏审计**。
