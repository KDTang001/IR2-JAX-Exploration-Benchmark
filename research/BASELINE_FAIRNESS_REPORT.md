# Phase6-Prep：基线公平性与 train-only 学习基线方案

11 月 18 日容量扩展说明：C1024 配置只扩大图存储及节点 padding，IR2 adapted
仍使用同一组外部预训练权重。更大的输入图属于额外适配范围；本轮通过执行验收，仍不
认证其训练数据公平性或对超出原训练图规模的泛化能力。该项不新增算法、不启动训练，
详见[容量验收报告](EXTENDED_CAPACITY_REPORT.md)。

日期：2026-11-17。工作基线 `d3d6c5f`；仅研究与协议设计。**没有训练、没有下载模型二进制、没有新算法，也没有恢复 A/B、世界模型或预测器。** 本文把本轮直接核查、历史证据、推断和待实现方案分开。

## 1. 可支持的公平性结论

当前组合是**两种导航 × 四种固定通信 = 八个配置，不是八种独立算法**。最近前沿不训练；`ir2_*` 是 external pretrained reference，使用原版权重但在本 Benchmark 的消息、预算和 `reachable_only` 规则下执行，故称 IR2 adapted。`training_seed=-1` 和 `real_training_events=0` 仅指本 Benchmark 未新增训练。当前适配器还明确拒绝非零训练事件/其他训练种子的固定基线身份；将来重训必须新增独立方法身份，不能冒用这八个历史 ID。[当前实现](../src/ir2_benchmark/agents/baselines.py)

**原版权重的训练暴露对新划分未经认证。** 不应称它为 benchmark-train-only 公平学习基线，也不能因存在重叠机会便称已经证明测试泄漏。检查原始训练入口只能说明代码允许读什么，不能重建已发布权重究竟读过什么。论文把它作为外部预训练参考单列，学习基线公平性主结论须等下面的从头训练方案实现、审计并训练完成后才成立。

## 2. 上游来源和本轮核查边界

上游固定提交为 `1ef6d224ae18503f4763c3cec20e05a737f95783`。本轮以本地 Git 对象直接读源码，并访问官方仓库固定提交 raw 源码与作者论文。GitHub tree 浏览入口失败，raw 源码成功；未以搜索摘要代替源码。[官方 parameter.py](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/parameter.py)、[官方 driver.py](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/driver.py)

IR2 论文描述通过强化学习与 curriculum learning 学习断连探索和重新会合之间的权衡；这支持方法来源，不提供本 Benchmark 划分的逐图暴露证明。当前可访问版本为 v3；不能把后来的论文版本自动视为固定权重的训练运行记录。[IR2 作者论文](https://arxiv.org/abs/2409.04730v3)

| 身份 | 历史证据中记录的值 |
| --- | --- |
| 发布文件 | `model/stage2/checkpoint.pth`，52,176,462 bytes |
| PTH SHA256 | `cd72ac7dbf164b8be089394154794416fc969a47d79c1072c437f6f2aa363cf3` |
| 仅 policy_model 的 NPZ SHA256 | `0a38ab8007bb64f36790d37549e70f7908cc1faba7d17a060ef0f99799193eb8` |
| 参数内容摘要 | `06aae41095fefc4c651e8c9a4d8cf14df30bcd5de18085fcbcd4f75411754ea3` |
| 历史参数核对 | 90 个 float32 张量，逐名/形状/类型/内容核对 |

本轮直接读取了 [Phase4 provenance](evidence/phase4-policy-provenance-20261114.json) 和 [Phase5 reproduction](evidence/phase5-reproduction.json)，确认其中记录的身份；**没有重跑模型二进制相等性检验**，上述历史结果不作为新的实测。参数身份即使相等，也不能认证训练集身份。

### 原始数据入口：观察与推断

- `parameter.py`（不是 README 中误写的 `parameters.py`）设置 `TRAIN_SET_DIR= DungeonMaps/train/ + TRAIN_SET_NAME`，默认 easy；difficult 分支另有参数。没有当前 benchmark train/validation/test 的 manifest 约束。
- `runner.py` 随机选机器人数量，把 episode_number 传给 Worker；Worker 把 global_step 传成 `Env(map_index=...)`。Env 读取目录文件名、逆序排序，再用 episode index 对文件数取模。它不是按当前训练 manifest 随机抽图。读图失败还存在退到目录第一张的分支，未来 train-only 入口应明确拒绝失败并留证。[runner](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/runner.py)、[worker](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/multi_robot_worker.py)、[env](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/env.py)
- 新验证集含原 train 目录保留图；旧数据审计还记录原训练候选与测试代表几何冲突而排除的 `test_priority` 项。因此从旧目录训练存在接触新验证集或测试等价几何的机会。这是**数据入口风险推断**，不是已发布 checkpoint 的实际暴露事实。[地图审计说明](MAP_DATASET_REPORT.md)
- 缺失：该 checkpoint 的每阶段数据文件摘要、实际读取/回放样本日志、训练种子、阶段继承链、模型选择记录。当前源码默认 `LOAD_MODEL=False` 不能证明发布 stage2 从未继承 stage1，也不能证明实际运行使用了当前默认参数。没有这些运行级证据，暴露结论保持 unverified。

### 原始 SAC 与网络：可复用的学习基线来源

原网络输入六维节点特征：二维坐标、探索 utility、rendezvous utility、guidepost、occupied-node；带节点/边屏蔽的图编码器为六层、八头、128 维，单层 decoder 后 pointer 输出合法邻边离散动作，K=30。[model.py](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/model.py)、[观测构造](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/multi_robot_worker.py#L151-L244)

driver 实现 replay 上的离散 SAC：双 Q 取小值、目标 Q、策略概率加权的熵正则目标，以及可学习 log_alpha。buffer 容量 10,000、热身 2,000、batch 128，每个成功回收的作业满足条件后更新八批；gamma=.995，网络 Adam lr=1e-5，alpha lr=1e-4。目标计数初始 1，每批增一，八批结束后 `>64` 则硬拷贝并重置 1（通常每64批）；注释里的“1024”不是执行条件。StepLR 对象虽创建/保存，driver 没有 `.step()` 调用。原版每32 episode编号保存；异步发出编号不等同完成样本量。[执行代码](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/1ef6d224ae18503f4763c3cec20e05a737f95783/driver.py#L165-L310)

SAC 的方法出处为最大熵、离策略 actor-critic；此处离散图动作和具体更新日程以 IR2 源码为准，不声称原 SAC 连续动作实验就是本协议。[SAC 原论文](https://arxiv.org/abs/1801.01290v2)

## 3. 首选候选：IR2-SAC train-only C1024 v1（NOT IMPLEMENTED / NOT TRAINED）

这是既有 IR2/SAC 的可复现重训基线提案，不提出方法论文新算法。它需后续独立实现和训练授权；本次四小时资源窗不包含训练。建议优先完成一个共同导航策略，然后同一 checkpoint 配四种固定通信，不为每种通信挑选不同权重。

容量已统一为 `configs/benchmark_capacity1024.json` 的候选规格，与下一版论文方案及
resource-pilot-v2-C1024 对齐。旧 C400 训练设计尚未实现或训练，不存在可重标的权重；
C400 仍是独立原始容量对照。只有 C1024 适用范围与训练资源预检通过后才冻结训练配置，
不能把 N4/Normal/8KiB 的验收扩大为 N2/N6、零预算、全部故障和桶的训练资格证明。

| 项目 | 待实现但明确的候选约定 |
| --- | --- |
| 数据资格 | 只读 registry 中 `split=train` 且 exclusion_reason 为空的 11,625 图（IR2 5,625 + procedural 6,000）；拒绝 split/占据/场景/文件哈希不符，禁止目录 fallback |
| 采样 | 复用已审计 TrainingMapSampler 的均衡来源、族/桶和 N=2/4/6 调度；保存每个 episode_id→map/scenario/hash 映射。不得按 test 成绩调整采样 |
| 初始化 | 三个独立训练种子 17001/17002/17003；actor、两 Q 随机初始化，target 从对应 Q 拷贝；空 replay、全新 optimizer，log_alpha=-2；禁止载入外部 IR2 或 A/B 权重 |
| 观测/动作 | 候选 `benchmark_capacity1024.json`：`ir2_graph` 六维局部图、合法一跳邻边、K=30、1024节点存储/padding；相同128维六层 actor/Q；训练 categorical sampling，评测 masked greedy；不把真值地图或隐藏队友状态输入 actor/critic |
| 环境 | 固定 benchmark传感器80、实际尺寸/原出生规则、reachable_only、关闭global graph fusion；512轮/99%物理覆盖终止。它是 benchmark-adapted 重训，不是原训练完全复现 |
| 通信分布 | 每个训练episode按 episode_id 确定性均衡 off/plan/map/mixed；非零预算2048/8192均衡；normal/dropout .3/burst64均衡；N=2/4/6均衡。发送周期和编码器与固定基线相同，不学习通信策略 |
| 奖励来源 | 复用原env表达式：new_frontiers/25 + new_pose_explore_util/50 + new_pose_rendezvous_util - distance/512，加完成团队奖励40。移植前逐项定义与benchmark状态的映射并用独立小场景验证，禁止未经声明改成另一奖励；不向网络暴露奖励计算用的真值状态 |
| 更新 | replay 10,000，热身2,000，batch128；每个完成且合法的训练episode回收后八批，无放回批内抽样；双Q/alpha目标、gamma、学习率、梯度裁剪（actor5/Q2000）和每64批target硬更新沿用上文；固定学习率，无隐式scheduler.step |
| 顺序与随机性 | 建议确定性的同步采集批次，按episode_id入库更新，固定批次宽度并记入run身份；保存Python/NumPy/Torch/CUDA RNG、所有采样索引及actor版本。此调度与原Ray异步有区别，必须报告 |
| 提议训练上限 | 每种子最多1,000,000实际active robot transitions或100,000更新批，先到即停；精确执行时不越界。另需批准硬件墙钟上限；本数字是可审议起点，不是收敛/成本保证 |
| checkpoint | step0，随后每5,000更新批和终态；保存actor/Q/target/alpha/optimizer/replay/RNG/环境/调度计数及hash；崩溃恢复不得只恢复actor |
| 模型选择 | 924张validation池中，五个来源/族各按(occupancy_hash,scenario_hash,path)取前5，共25图；场景种子0/1/2，N=4、8192、normal、mixed。每checkpoint同样75回合，无梯度、无replay写入；按族等权地图AUC选择，每训练种子独立选，精确并列取较早更新批 |
| 显存与有效批次 | 有效 batch 仍为128；候选使用固定 microbatch8 累积16次后每个优化器只更新一次，target/alpha/actor版本以有效更新批计数。须先检验损失/梯度/目标更新与完整批的数值一致性并测前反向显存；8尚未证明足够，不按OOM自动改批次或更新频率。本轮不做训练预检 |
| 评测身份 | 新 `ir2_sac_trainonly_c1024_v1_*` 身份，实际训练种子/事件数/数据和checkpoint摘要；不得用 `training_seed=-1` 或旧 C400 提案身份；在独立扩展矩阵中计训练重复成本，不悄悄加入旧八配置矩阵 |

924是资格池，不是声称每次用924图调参。25图规则需发布新选择清单与hash，不假定已有25图manifest的完整配置和此方案相同。正式 test/OOD 不用于超参、奖励、checkpoint选择、早停、性能优化或训练。若训练不足上限或不收敛，报告实际预算与学习曲线，不宣称SAC已被公平击败。

**实施前验收门槛**：非法split注入应被拒绝；逐读取记录均属于冻结train allowlist；验证无optimizer/replay变化；与上游同一小批的loss/gradient/target更新数值对照；存档恢复后下一批采样/更新一致；所有训练seed包括失败者均留存。当前未实现或执行这些验收。

## 4. 直接重算的输入身份与复查方法

本轮解析实际五个JSONL（不从报告抄总数），仅计无exclusion_reason记录，得到 train=11,625、validation=924、test=1,028、ood=300。这里只复查清单和字节身份，没有重算24,199张PNG的几何审计。

| 相对 artifacts/maps 路径 | 文件SHA256 |
| --- | --- |
| ir2_manifest_enriched.jsonl | `c60e0fbe288e0c206f8dcad348b3c8f77e73534f33a0faa0eb072b4e8e7761d8` |
| procedural/train/manifest.jsonl | `0f9ca254710d1895925012468772d8d6e46a6c7c403e01cb3f3d3e361ae86546` |
| procedural/validation/manifest.jsonl | `730340a96e9d920fa79a29d4108ed572fc8d1a0e93ede762e822fff711386c1d` |
| procedural/test/manifest.jsonl | `d97bf5f0cf30029008b52c5e515e7e17708967d41da46e75fe032af20148039f` |
| procedural/ood/manifest.jsonl | `72fe2955e9848e4e688df21da6cb03b434aefea5566813694600495413f674b3` |

可复查调用（仓库根；仅读取，不运行训练模块）：

```python
import collections, hashlib, json
from pathlib import Path
files = [Path('artifacts/maps/ir2_manifest_enriched.jsonl'),
         *sorted(Path('artifacts/maps/procedural').glob('*/manifest.jsonl'))]
counts = collections.Counter()
for p in files:
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    counts.update(r['split'] for r in rows if not r.get('exclusion_reason'))
    print(p, hashlib.sha256(p.read_bytes()).hexdigest())
assert dict(counts) == {'train':11625, 'validation':924, 'test':1028, 'ood':300}
print(dict(counts))
```

直接执行 `git diff 1ef6d224ae18503f4763c3cec20e05a737f95783 -- driver.py parameter.py model.py env.py runner.py multi_robot_worker.py`：driver/parameter/model/runner/worker无差异；env唯一差异是 `np.lib.pad`→`np.pad`。`git cat-file -t`返回commit。上游Git blob SHA256：driver=`33f8c8f0f2576d9ae1aa792cd83c97b266b54a9a3cf6715e051b245fe66e15d1`；parameter=`ce43fa019b3e9442b11fdbda9864e20a5e912cb974db7694fa951e76d78187e9`；model=`2a0c4d91497fc5b5db981e2b91f1b8ed4ed11550bfaee3a67c5d6d1a967fd637`。原始代码与当前适配器均直接阅读，未执行driver（导入会初始化Ray）。

未来训练运行需生成独立 train allowlist SHA256（排序、UTF-8/LF、逐行canonical JSON规则固定）、所有数据PNG和六个registry/manifest文件摘要、Git commit/dirty patch/依赖锁/运行命令、逐读取与回放样本账本、checkpoint链和验证选择记录。以上五个文件hash不是尚未生成的train-only allowlist hash。正式论文还需决定是否增加传统协作探索算法；当前八配置不能声称覆盖所有传统多机器人基线。
