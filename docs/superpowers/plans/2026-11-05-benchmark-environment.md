# Shared JAX Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付独立可运行、按字节扣费、具备三类链路条件的 JAX 栅格探索环境。

**Architecture:** 主机侧准备地图和配置，纯函数执行感知、导航、发包与状态转移。每机器人事件顺序推进，独立环境批量化；另有 NumPy 参考实现作为 oracle。

**Tech Stack:** Python 3.12、JAX、NumPy、Pillow、pytest、uv。根目录与统一契约见总计划。

---

## E1：隔离运行时与包入口

创建 pyproject.toml、src/ir2_benchmark/__init__.py、src/ir2_benchmark/cli.py、configs/micro.json、tests/benchmark/test_runtime.py；修改 .gitignore 仅新增 artifacts 大产物规则。pytest testpaths 只指 tests/benchmark，不能误收集原 IR2 的 test_driver.py 并启动 Ray。

- [ ] 只读执行 `wsl -d Ubuntu-20.04 -- nvidia-smi` 与 Linux Python/uv 发现；记录驱动、内存、系统版本和工作目录映射。失败保留日志；CPU 正确性任务可继续，不宣称 GPU 可用。
- [ ] 在独立 .venv 配置 Python 3.12。依赖采用 jax、flax、optax、numpy、pillow，开发依赖 pytest、ruff；Linux GPU 增加 jax 的官方 CUDA extra，并在选择前核对驱动兼容性。用 uv 解析并写入锁文件，不修改原 IR2 的 Anaconda 环境。
- [ ] 最小 pyproject 配置包括下面内容，依赖通过 `uv add` 加入，锁定实测版本。

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ir2-benchmark"
version = "0.1.0"
requires-python = ">=3.12,<3.14"

[project.scripts]
ir2-benchmark = "ir2_benchmark.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests/benchmark"]
```

- [ ] 运行 `uv run python -c "import jax; print(jax.devices()); print(jax.jit(lambda x: x + 1)(1).block_until_ready())"`，必须打印设备和 2；独立 GPU 验证需至少一个 gpu 平台设备。运行 `uv run pytest --collect-only -q`，列表不得含旧 test_driver。
- [ ] CLI 用 argparse，先提供 doctor 子命令，输出 Python/JAX/平台/设备 JSON，不将失败默默转换为 gpu=false 后返回成功；指定 `--require-gpu` 时无 GPU 返回非零。

## E2：编码协议与预算内核

创建 codecs/packet.py、codecs/standard.py、communication/ledger.py；测试 unit/test_packet.py、unit/test_ledger.py。

- [ ] 写独立 Python struct oracle：`struct.Struct('<BBBBIIHBB')` 的大小须为 16。PLAN 的 H=5 载荷为 29 字节，完整包为 45 字节。测试大小来自字段算术而非调用生产 encoder 计算期望。
- [ ] 实现 header 编解码、固定 uint8 缓冲及实际长度；非法 codec、保留位非零、长度不符在主机输入处拒绝。JAX 内返回 valid 掩码，不抛 Python 异常。
- [ ] 编写 POSITION、GOAL、PLAN、MAP_DELTA 的真实字节往返断言，包含非对称坐标 (3,11)、坐标 65535、小端序、过长载荷及零条目。CUSTOM 只验证容量与协议结构。
- [ ] 实现账本公式：

```python
eligible = requested & packet_valid & (cost <= balance)
next_balance = jnp.where(eligible, balance - cost, balance)
spent = jnp.where(eligible, cost, 0)
```

spent 不依赖 delivery_mask；测试输入 balance=50、cost=45、无人收到，余额必须为 5；balance=44 时余额仍 44 且不能发送；一次广播送达 3 人仍只扣 45。
- [ ] 运行 `uv run pytest tests/benchmark/unit/test_packet.py tests/benchmark/unit/test_ledger.py -q`。对 jit 和 eager 均验证余额和实际字节相同，检查缓冲填充不计费。

## E3：地图清单、生成与出生区域

创建 maps/manifest.py、maps/prepare.py、maps/generate.py，测试 unit/test_maps.py；CLI 增加 maps audit、maps generate。

- [ ] 测试一张损坏图像、一张无起点标记、一张只有不足 N 个可达自由格的图像；必须返回具体错误，不替换其他地图。IR2 标记由像素值 208 识别，选择确定性的标记区域代表自由格，记录选点算法；不复刻原代码固定取第 128 个像素的脆弱索引。
- [ ] 对实际地图逐张读取，产出路径、形状、占据哈希、起点、连通面积、划分和排除理由。图像转换规则复核 env.py，避免把标记误转为墙。padding 右/下为障碍，保留 valid_shape。
- [ ] 测试同图不同文件名、旋转图、相同障碍但不同起点。划分隔离以占据栅格的旋转/镜像规范化哈希为分组，场景 ID 另含起点；因此不能靠换起点把相同地图放进训练和测试。按设计的测试优先规则剔除训练冲突。
- [ ] 生成器分别实现房间门洞、走廊、混合结构；用 flood-fill 核验连通面积和出生空间。固定 seed 产生完全相同栅格，不同模板形成可观察的结构差异。被拒 seed 和最终 seed 都写清单。
- [ ] 在 tests/benchmark/unit/test_maps.py 中验证 prepare 后原有效区域逐格不变，地图桶外一律 OCCUPIED；同 seed 的起点编号分配可复现。
- [ ] 运行 `uv run ir2-benchmark maps audit --root DungeonMaps --output artifacts/maps/ir2_manifest.jsonl`，对所有文件计数并列出排除项；只生成 manifest，不覆盖原图。运行 `uv run pytest tests/benchmark/unit/test_maps.py -q`。

## E4：遮挡感知与边界

创建 sensing/reference.py、sensing/jax_sensor.py，测试 unit/test_sensing.py。

- [ ] 先构造 7×7 手工地图：机器人 (1,3)，墙 (3,3)，(2,3) 可见、墙可见、(4,3) 不可见；R=1 时距离大于 1 的格未知。
- [ ] NumPy oracle 独立实现连续线段与单元格相交判定，不调用 JAX 的射线辅助函数。补充斜角夹墙、地图边缘、同格、圆半径边界和 padding 场景。
- [ ] JAX 实现预计算相对目标与 supercover 格序列，运行时平移索引、裁剪并用有效性掩码避免边界截断产生虚假格；按每目标是否存在先前阻挡决定可见性。
- [ ] 仅更新 visible 对应格，本地未知保留。重观察不重复增加团队首次覆盖计数；发现障碍不计入自由覆盖。
- [ ] 运行 `uv run pytest tests/benchmark/unit/test_sensing.py -q`，全部手工地图与随机小图在 reference/eager/jit/vmap 下逐格一致。测试观测操作不改变输入数组。

## E5：路径、移动与候选航点

创建 navigation/reference.py、navigation/jax_path.py、navigation/candidates.py，测试 unit/test_navigation.py。

- [ ] 手工 U 形墙地图验证路径绕墙，不能用欧氏直线距离当实际长度；未知格不可穿越，目标等于自己返回等待，孤立目标返回无效。
- [ ] 参考 BFS 用 deque；JAX 用容量 H*W 的数组队列、visited、parent，扩展顺序固定为上、右、下、左。每节点首次发现时写父节点，禁止重复入队。路径回溯同样有 H*W 的容量界限。
- [ ] 路径移动最多 d_max 边：长度 7、d_max=2 必须只走 2；距离 1 则只走 1。实际距离按边数累加。未到目标可保留目标至下轮。
- [ ] 候选表最多 64 个加等待，包含有效旧目标，其余只从已知可达自由格确定性采样；若不足使用 mask，不把 padding 坐标当真实目标。环境仍接受表外的合法航点。
- [ ] 用独立参考 BFS 逐例验证 JAX 的终点、长度和不可达结果；运行 `uv run pytest tests/benchmark/unit/test_navigation.py -q`，再对同地图多目标 vmap，禁止 host callback。

## E6：信号、故障与广播送达

创建 communication/signal.py、communication/faults.py、communication/channel.py；测试 unit/test_signal.py、unit/test_channel.py。

- [ ] 独立几何测试水平、竖直、对角线段穿墙长度之和等于总长度，共址结果有限；阈值相等时不可达。使用设计稿给出的常量和分段损耗式。
- [ ] 测试 Normal 不附加外部丢包，Dropout p=0 全通/p=1 全丢，Burst 指定节点的入站和出站均断而其余链路保持基础判定。
- [ ] 使用 `jax.random.fold_in` 按用途、episode、event、sender、receiver 生成随机流；多发或少发消息不能改变另一事件的抽样。分别测初始采样、策略采样、故障采样不复用 key。
- [ ] 收件箱 [receiver,sender] 更新只发生在有效送达；同源较新的消息覆盖旧包并增加覆盖计数。无自动转发；转发须等该机器人动作，重新付费并保留原时间字段。
- [ ] 运行 `uv run pytest tests/benchmark/unit/test_signal.py tests/benchmark/unit/test_channel.py -q`。固定广播源验证三个接收者送达可不同，扣费只发生一次。

## E7：纯状态与有序事件

创建 config.py、types.py、env/reset.py、env/observe.py、env/step.py；测试 integration/test_event_step.py。

- [ ] 创建不可变 HostConfig 和 JAX NamedTuple 状态。reset 做出生感知和覆盖初始化，预算独立。所有状态字段在 total shape 静态时可作为 PyTree 传入 jit。
- [ ] 编写两机器人测试：0 在本轮发送的新消息，1 观察时可见；1 发给 0 的消息必须等 0 下次回合才能影响动作。整个轮次的动作不可提前由旧 obs 一次算好。
- [ ] 按获批设计实施“决策并锁定载荷 → 移动 → 终点感知 → 后移动位置判链路并发送 → 覆盖奖励”。地图增量锁定于移动前，新发现内容不能本回合自动插入包。
- [ ] observe 重复两次须得到相同输出且 state 不变；step 才清理已消费 inbox。局部 actor 不得包含真实全队位置、故障掩码和联合覆盖。
- [ ] 覆盖完成立即冻结剩余事件；时间上限标为 truncated，不与 terminated 混用。测试终止状态下无新增位置、字节和奖励。
- [ ] `uv run pytest tests/benchmark/integration/test_event_step.py -q`。使用不同隐藏远处地图但相同可见局部状态验证 observe 不变；该测试不能改变当前传感器实际可见区域。

## E8：批量运行和真实 CLI 场景

创建 env/rollout.py、agents/heuristic.py、agents/fixed_comm.py；CLI 增加 rollout；测试 integration/test_rollout.py。

- [ ] 实现最近前沿的路径距离选择与固定通信循环。无前沿时等待，不用真实地图选择未知目标。
- [ ] 用 scan 沿事件时间推进，当前机器人先调用 observe 再选动作；环境批次 vmap，不同 N/地图桶分开编译。终止事件用掩码冻结并从吞吐有效步数排除。
- [ ] 固定动作脚本下 comparing reference/eager/jit/batch，每事件比较位置、地图、预算、收件箱、覆盖和 done。随机小地图至少覆盖 2/4/6 台机器人。
- [ ] 执行 `uv run ir2-benchmark rollout --config configs/micro.json --seed 7 --output artifacts/environment-smoke`。必须产出 summary.json、events.jsonl 和本地/联合地图轨迹图，人工检查墙后未知及局部隔离行为。
- [ ] 执行 `uv run pytest tests/benchmark/unit tests/benchmark/integration -q`，记录耗时。通过后标记环境子计划完成，A/B 和正式实验保持未完成。
