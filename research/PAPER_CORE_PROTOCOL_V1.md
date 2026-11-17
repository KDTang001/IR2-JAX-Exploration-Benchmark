# PAPER_CORE_PROTOCOL_V1 — 待批准的论文核心协议

日期：2026-11-17；状态 **PROPOSED / NOT FROZEN / NOT RUN**。这是Phase6-Prep的可审议协议，不是正式评测启动声明。原Phase1–5证据不可改写；旧2,294,784项矩阵、240项validation资源先导和58,320项论文草案均没有因此得到执行批准。当前H100四小时仅用于profiler、必要的配对优化验证和极小validation成本子集，包含编译时间，**不含训练或正式test/OOD试验**。

## 1. 问题、比较对象与结论边界

主问题：在相同导航、地图、出生、预算和故障条件下，固定消息调度对探索覆盖和通信成本的影响；并比较两种导航对这些条件的敏感性。最近前沿与外部预训练IR2 adapted配off/PLAN/MAP_DELTA/mixed，合计八个配置。外部IR2的暴露未经新划分认证，其结果仅作外部参考；不能写成train-only公平学习比较。[公平性与独立重训提案](BASELINE_FAIRNESS_REPORT.md)

不把原IR2论文的协议/奖励/性能直接套到本Benchmark。固定通信不代表学习通信策略；本阶段没有新方法算法。若以后批准train-only SAC，新增独立方法/种子/预算扩展版本，不能把三次训练复制计成三次场景重复，也不能混用历史`ir2_*`结果身份。

## 2. 地图冻结规则（本轮不选出正式清单）

候选池当前为test 1,028、OOD 300；清单文件hash和本轮直接重算见公平性报告。九个统计strata：IR2 test complex/corridor/hybrid；procedural test rooms/corridor/hybrid；procedural narrow OOD rooms/corridor/hybrid。OOD保持narrow变化，不额外混入large_rooms。

建议版本名 `paper-core-v1`，正式机器manifest尚未生成。对每个stratum内合格记录按 `(occupancy_hash, scenario_hash, relative_path)` 升序，取前m张；m需资源与统计要求批准后才写入。禁止按方法得分、完成速度或困难程度挑选/替换地图。地图有效尺寸和桶记录原值，不裁剪缩放来满足成本。先验分层若要另加尺寸桶配额，应先修订版本并说明改变的总体。

未来manifest必须包含：source/family/split、相对路径、PNG文件SHA256、occupancy/canonical/scenario hash、anchor算法/坐标、component_area、valid_shape/bucket、来源清单hash、选择规则版本、m、全部scenario配置与种子。排序后canonical JSON（UTF-8、LF、固定key顺序及分隔符）计算manifest SHA256，并另存schema/version与生成工具Git身份。**本文件没有正式地图清单hash；不得把输入池hash当成已批准的选择hash。**

Phase4工程验收曾使用部分test/OOD地图，既有证据保留且不用于调参。正式清单冻结时标注 prior_engineering_exposure=true/false 和历史用途，不能称整个池从未运行过；新性能优化与资源先导只使用validation。如果旧工程检查造成任何方法选择，应披露并在批准前处理，不事后悄悄换图。

## 3. 候选核心因子与准确算术

| 因子 | 提议固定值/规则 |
| --- | --- |
| 地图 | 九strata各m张；旧草案m=10仅供预算讨论 |
| 场景seed | 0、1、2；不是训练seed |
| N | 2、4、6，逐N单独报告 |
| 每机器人预算 | 0、2048、8192 bytes；attempt扣费，broadcast |
| 正常 | external fault关闭，但仍保留物理IR2信号门限 |
| Dropout | p=.3；与Burst分开运行，保留底层故障采样语义 |
| Burst | 开始round128、持续64round，p=0；隔离机器人由fault_seed排序选择floor(N/2)，记录身份 |
| 导航×通信 | 2×4=8配置，通信每4次自身动作一次，mixed为PLAN→MAP_DELTA→MAP_DELTA→POSITION |
| 时长/终止 | 最多512轮；以anchor可达自由区域物理覆盖≥.99为成功，成功后冻结；不是总PNG面积 |
| 共同环境 | `configs/benchmark.json`语义，IR2图K=30、400节点、传感器80、reachable_only、关闭global graph fusion；发布时绑定精确配置/实现hash |

Normal、Dropout、Burst是三项独立条件，不形成混合故障交叉。`make_scenario`目前以map_hash、map_scenario_hash、scenario_seed的identity摘要派生spawn_seed/fault_seed；正式manifest必须存实际整数与isolated robot列表，使八配置共享同一场景，不能依赖全局RNG或方法完成顺序。[现有场景构造](../src/ir2_benchmark/evaluation/scenarios.py)

每张地图为3 seeds ×3 N ×3 budgets ×3 faults ×8配置 =648 episodes；九族各m张为 **5,832m**。m=10才是58,320。旧完整声明为1328×3×3×4×6×8=2,294,784。validation旧候选5图×1seed×1N×2budgets×3faults×8=240。它们都是组合计数，不是运行证明或资源承诺。

零预算保留作为无通信一致性对照；同导航四种通信在零预算下预期相同，必须真实核验，不把预期相同当成八次独立算法证据。若未来折叠重复执行，必须预先声明任务别名、结果复用规则和减少后的实际工作量，不能在旧分母中算作真实执行。

## 4. 指标与失败分类

主终点为512轮物理覆盖AUC；按现有实现取每轮末覆盖的均值，提前成功后延续实际末值至512轮，不填1.0。覆盖分母限anchor连通自由区域，padding与断开区域排除；接收地图不能伪造传感器物理探索足迹。[metrics实现](../src/ir2_benchmark/evaluation/metrics.py)

次终点：最终覆盖、99%成功率、成功轮次/轮内事件偏移、实际active robot events、移动距离、重复探索（如使用，必须绑定现有定义）、emitted/spent/delivered bytes及预算利用率。delivered按接收副本计，不等于单次物理发射。N/预算/故障分别报告，不把总通信费用或机器人事件误作团队轮次。距离和完成时间同时报告成功率，不能仅在各自成功子集上宣称更快。

512轮合法未完成是探索timeout；墙钟截止、OOM、map/model/environment_error是执行失败，metric保持缺失，不能记作0分或成功。每个配置列requested/success/timeout/error/pending及配对完整数。主配对推断只在预声明完整匹配块上计算，并同时报告缺失模式；若失败与方法相关，不把complete-case估计解释为全体效应。必要时给0..1 AUC缺失上下界，或只出工程结果。正式完成须声明清单全部有有效终态，不能用“尝试过”代替有效计分完成。

## 5. 统计单位与预声明比较

地图是外推重复单位；先在每张地图上平均三个场景seed，再族内等权平均地图，最后等权平均strata。主报告分别列test六族与OOD三族；九族宏平均只作附加汇总。N、预算和fault的效应单独列，不静默混成一个总体。相同地图/seed的八配置保持配对。

主比较为每导航下PLAN/MAP_DELTA/mixed各相对off的AUC差，在各N×非零预算×fault层报告；零预算为一致性检查，不用于挑最佳策略。不要事后选择“最佳通信”再与off比较。导航间配对差为外部参考的次级比较，受训练暴露限制。

固定checkpoint条件下，建议分层配对bootstrap 10,000次（分析seed=20261117），在每个stratum内重采样地图，所有方法和场景seed一并保留；报告95%区间、效应量和各层实际地图数。每族一图不能估计族内地图不确定性。任何显著性断言应预先把同一主比较表的全部检验作为一个family并用Holm校正，不能选择性报告显著格子。bootstrap/检验代码和模拟自检尚待实现，不声称已经验证。

若加入从头训练结果，至少报告每个训练seed的选中checkpoint、效果和失败；地图和训练seed是不同层级。使用保留配对的两层/交叉重采样并单列seed敏感性，不把同一地图上重复rollout当独立样本。三个训练seed仍不足以保证训练随机性估计稳定，不能通过扩大场景数掩盖这一点。

最终m须根据validation上的方差/配对差结构、最小有意义AUC差和期望区间精度决定，并与资源配额核对。当前没有功效/精度计算结果；m=10不是已证明足够的样本量。不可查看formal test分数后再以显著性决定是否追加地图。

## 6. 资源估算与停止规则

当前四小时H100窗口的停止时间由独立调度声明记录；编译、加载、预热、剖析、配对旧/新实现检查、落盘均计入。达到截止停止本轮自有worker，保存部分证据及退出状态，不因无结果而重开窗口。不启动240先导，更不启动58,320或2,294,784正式矩阵。只允许先声明极小validation子集的地图/配置/事件界限/每项超时，算子profile不能混入论文得分。

Phase5初始四事件吞吐不等于成熟图完整回合吞吐，B不能线性外推；既有长回合有共享设备/编译影响。本轮已完成六配置/30阶段profiler、22项优化配对，以及一张validation/easy地图的两条完整成本轨迹：N4/B1、8192、Normal、mixed，NF 187.96秒成功，IR2 adapted 131.38秒正常512轮超时，均包含编译/加载/落盘。见[性能与成本报告](PERFORMANCE_PROFILE_REPORT.md)。这两个数不足以对五族、全部预算/故障或正式矩阵给出实测预测，也不认证历史19.8小时/200天外推。240项清单已冻结，但只有其中两项作为独立计价子集执行；其余238项未安排。

正式预算应按实际批次调度估计：`T = setup_compile + sum_over_batches(t_batch[stratum,bucket,N,method,budget,fault]) + io_analysis + retry_reserve`。优先使用完整512轮或提前正常终止的validation回合墙钟；同时记录事件数、峰值显存/RSS、批量B、同期负载。微片段只提供阶段成本，不足以填完整回合t_batch。若只能拿到小片段，成本结论保持“尚不足以批准正式矩阵”。

条件算术示例（不是实测）：若某已批准运行窗为H秒，至少预留20%给失败/落盘/分析，已测冷启动C秒，则episode配额近似`floor((0.8H-C)*r_effective)`；r必须来自适用桶与长回合的端到端实测，不能用前四事件吞吐替代。对于58,320项，两天48小时且忽略C时仍需至少0.421875 episodes/s（按80%可用时间）。这一速率只是所需门槛，不是H100能达到的预测；重训与验证选择另计。4小时当前授权不等于formal H=4小时。

OOM、非法动作、数据/源码hash不符、指标回放不一致立即停止相应任务及受影响队列并留证；只有基础设施暂时失败可按预声明次数重试，同身份新attempt保留旧结果，不能从多个成绩择优。正式每项墙钟上限、总窗、GPU并发与B按成本审议后写入机器声明，现在不捏造这些数值。任何窗口截止的未执行任务保留pending，不缩分母。

存储预算需以新子集的实际结果/压缩trace字节估计并保留上界余量，不能默认旧均值适用于不同通信预算。全量复制权重不必要；checkpoint只存引用hash，原始trace按任务ID落盘。所有新工作用`phase6-prep/<attempt-id>/`或未来`paper-core-v1/<manifest-hash>/<attempt-id>/`独立目录和身份，禁止覆盖Phase1–5。

## 7. 批准前需要的具体输入与交付

1. 新profiler及必要优化的配对语义证据；每桶/相关N/B的内存边界，以及足够完整的validation回合端到端成本。若四小时只完成profiler，单独申请未来成本测量范围，不能自动扩展。
2. 决定论文主张范围与是否必须先实现train-only SAC/其他传统协作算法；给新增训练独立预算，当前外部参考不替代公平学习基线。
3. 批准m、各层精度目标、主比较family、可承受缺失规则；若资源不足，先声明更小工程研究或增加资源，不发布不足支撑的算法排名。
4. 批准正式运行总墙钟/GPU数量/每项截止/并发、完整成本和存储预算；明确是否另行运行240validation先导及其身份。
5. 生成并审阅完整地图/任务manifest、scenario seeds、checkpoint/配置/源码/依赖hash、数据暴露注记、统计脚本版本；记录最终approval ID/时间，之后才允许正式执行。

上述输入尚未齐备，正式清单和批准记录未生成。当前可交付的是规则与边界；不是声称论文核心协议已冻结或正式实验已完成。
