# Phase 3：地图数据集、划分与采样验收

日期：2026-11-14。分支：`audit/benchmark-foundation-2026-11-12`。环境：用户指定服务器的 conda `ex`，Linux / H100。

## 1. 结论和证据

**Phase 3 在下述固定数据与接口范围内验收通过。** 全量核对 24,199 张 PNG，重现全部 6,900 张程序化地图及生成记录；保留 13,877 张 D4 canonical 唯一地图，未发现保留集合内重复或跨来源冲突。IR2 划分和排除记录全部符合独立重算的规则。

远程功能检查最终去重 **35 项通过、0 项失败、0 项跳过**。原始失败与针对性复验均保留，35 是不同测试 ID 的最终状态，不是一次全绿的运行。另实际调用训练加载器，将全部 **13 种来源/地图族/尺寸桶组合**送入 CUDA，并检查 3 类篡改元数据均被拒绝。

- [机器可读证据](evidence/phase3-20261114.json)：源码、运行环境、原始日志、失败/复验、全量数据结果、实际加载与固定验证迁移。
- [全部 PNG 的 SHA-256 清单](evidence/phase3-data-files-20261114.sha256)：24,199 个相对仓库根目录的路径，包含排除地图。
- [只读代码复核](evidence/phase3-code-review-20261114.md)：APPROVE，无阻断项；脚本独立运行元数据等低优先级文档建议保留。
- [实际 registry](../artifacts/maps/sources.json)、[IR2 完整逐文件清单](../artifacts/maps/ir2_manifest_enriched.jsonl)和四个程序化 manifest 是数据划分的事实依据。本次未改 PNG 或这些地图的 split。

本阶段没有进行训练、方法排名、全任务评测或硬件吞吐测试。图像/连通域/生成重算使用服务器 CPU，环境数组和实际 sampler 加载使用 H100；没有用 Windows CPU 结果代替目标服务器验收。

## 2. IR2 全量审计与排除

| 原始目录 | 原图数 | 保留 train | validation | test | excluded |
|---|---:|---:|---:|---:|---:|
| train/easy | 9,999 | 2,700 | 300 | 0 | 6,999 |
| train/difficult | 6,000 | 2,925 | 324 | 0 | 2,751 |
| test/complex | 500 | 0 | 0 | 223 | 277 |
| test/corridor | 400 | 0 | 0 | 400 | 0 |
| test/hybrid | 400 | 0 | 0 | 105 | 295 |
| 合计 | **17,299** | **5,625** | **624** | **728** | **10,322** |

排除原因逐项复核：

| 原因 | 数量 | 含义 |
|---|---:|---|
| missing_marker_208 | 4,999 | PNG 可以解码，但没有规则要求的 208 出生标记；不猜测替代起点 |
| duplicate_occupancy | 4,997 | D4 等价占据重复；清单指向被保留的代表路径 |
| test_priority | 326 | 原训练目录与测试代表几何冲突；测试保留、训练候选排除 |

没有发现额外损坏图像、无法容纳 6 个出生点的保留地图，或未列明的新排除项。全部 IR2 像素取值的并集为 `{127,195,203,208}`；占据判定保持 `pixel <= 150`，203 属于自由格，但不充当 208 标记。

独立分组忽略出生标记，只对占据栅格的四种旋转及其镜像共八种变换取最小摘要。摘要包含高/宽与行优先 packbits，非方形地图旋转时尺寸一同变化。同几何、不同标记属于同组；定向占据与 anchor 的 scenario hash 另外保留。先选原 test 代表，再按路径字典序确定代表；保留的原训练候选按每族 canonical hash 排序，取 `floor(数量/10)` 为 validation。

IR2 easy 去重后 3,000 张，分出 300 张验证；difficult 去重后 3,249 张，分出 324 张验证。未使用测试得分或训练结果调整任何划分。

## 3. 程序化地图和 OOD

| split | rooms | corridor | hybrid | 合计 | 池起始种子 | 结构参数 |
|---|---:|---:|---:|---:|---:|---|
| train | 2,000 | 2,000 | 2,000 | 6,000 | 410000 | width 8..16，room scale 40..120 |
| validation | 100 | 100 | 100 | 300 | 510000 | 同训练范围，新实例 |
| test | 100 | 100 | 100 | 300 | 610000 | 同训练范围，新实例 |
| ood | 100 | 100 | 100 | 300 | 710000 | 仅 width 改为 4..7，room scale 保持 40..120 |

使用 `ir2-procedural-v1`。从每条记录的 **requested_seed** 重算生成器，逐字段核对 final_seed、尺寸、宽度、参数范围、anchor、连通面积、三个 hash 和拒绝历史；全部 6,900 条一致。没有保存重复的 PNG 池。

全部程序化地图的自由区域连通，面积至少为有效面积的 15%，可容纳 6 台机器人。全部四个池内部、池之间以及与 IR2 全部可计算 canonical 的原图之间交集均为零。只分开随机种子并不足以支持这一结论，本次还核对了实际几何。

当前 6,900 条记录的拒绝历史均为空。若未来 `generate_pool` 因 `duplicate_occupancy` 跳过种子，需要使用池级 seen 集合重放；单独调用 `generate_map(requested_seed)` 不能完整还原那类记录。本次通过只覆盖当前固定池。`large_rooms` 仍是生成器可选扩展，现有正式 OOD 池采用 narrow；没有把两类变化混在同一结果中。

## 4. 原尺寸、anchor、component 与 padding

所有保留地图的实际有效尺寸和桶：

| 来源 | 有效尺寸 H×W | 编译桶 | 保留数 |
|---|---|---:|---:|
| IR2 | 500×500 | 512² | 606 |
| IR2 | 480×640 | 768² | 5,400 |
| IR2 | 1000×1000 | 1024² | 971 |
| 程序化 | 512×512 | 512² | 2,224 |
| 程序化 | 768×768 | 768² | 2,337 |
| 程序化 | 1024×1024 | 1024² | 2,339 |

**IR2 difficult 实际跨三种尺寸。** 其保留数据分别为 500×500 的 501 张、480×640 的 2,000 张、1000×1000 的 748 张；不能依据一个样本把整个 difficult 族固定为 512 桶。现有 sampler 按实际 bucket 分发，本次补充检查按完整 registry 的 13 个组合核实，修正了验收辅助脚本最初只预期 11 个组合的错误假设。

anchor 按已声明的 `208-nearest-centroid-row-major-v1`：取 208 像素中离其质心最近的像素，以行优先顺序解决并列；不是 IR2 旧代码的第 127 个标记像素，属于既有 Benchmark 扩展。独立 SciPy 四邻接连通域结果与所有可用 anchor 的 manifest 面积一致。所有保留地图的 anchor 连通区域恰好覆盖全部自由格；无 marker 的排除文件不具有可审计的 anchor，不能据此声明其 anchor 连通性。

只在右/下补障碍，不缩放或裁剪。本阶段检查包括：

- 现有 CPU host 准备逻辑与独立 BFS、只读数组、错误地图显式拒绝。
- H100 上 512/768/1024 三种桶 × N=2/4/6，实际 `jit(vmap(reset))`；出生位置可重现、彼此不同且位于同一合法区域，inactive padding 机器人没有物理足迹。
- 用手算 45 格 anchor 区域与另一间 42 格断开房间验证完成条件：见完前一间后以 45/45 完成，断开房间和 padding 不进入分母。
- 通过实际训练加载器核对 13 个真实数据组合的 valid_shape、anchor、component 面积和 CUDA 设备，拒绝占据 hash、scenario hash 或 bucket 不一致的输入。

## 5. registry、正式训练采样与固定验证

合并后的保留数量为：train **11,625**、validation **924**、test **1,028**、ood **300**，共 **13,877**。五条 source registry 分别指向 IR2 总 manifest 和程序化 train/validation/test/ood。目录标签 origin/family/split 也与实际路径独立核对；没有仅信任行内的 split 字段。

`TrainingMapSampler` 只取 `split=train` 且未排除的行，真实可选记录共 11,625 条。108 次调度的来源精确为 IR2 54、程序化 54，N=2/4/6 各 36 次；每个 N 内 IR2 easy/difficult 各 9 次，程序化每个族/桶组合各 2 次。IR2 在族内按保留地图取样，其桶分布不强制均匀。已有测试还核对调用顺序独立性、迁移目录后的身份、curriculum 后恢复平衡及缺少 strata 时拒绝。

实际 `evaluation_maps` 读取 test/ood 共 1,328 张，test 1,028、ood 300，不含 train/validation。固定模型选择清单每个来源/族取 validation 中最小的五个占据 hash：IR2 easy/difficult 各 5，程序化三个族各 5，共 25 场。每张地图的资格和场景身份都已核对。

### DATA-001：修复失效的固定验证工作负载

原有清单仍只存早期配置字段，服务器实际 `validate_manifest` 报 `requires validation-only rows and intact configuration hashes`。本次按 `configs/receiver_ir2_graph.json` 的完整配置重建场景，保留原来的 8,192 字节验证预算：

- 同一组 25 个地图相对路径、占据/anchor scenario hash 保持一致。
- 出生/故障种子、N、预算、max_rounds 全部保持一致。
- 显式采用 `ir2_graph` 和 K=30（max_candidates=29），写入当前完整配置与实现摘要；更新 scenario_id/config_hash 和 selection manifest_hash。

公开清单的路径仍相对仓库根目录。服务器旧文件使用过绝对路径，迁移核对先按服务器项目根归一化；公开分支旧/新路径逐字段比较也一致。没有启动模型选择或读取测试得分。

### DATA-002：固定换行与实现绑定

第一次部署使用 Windows 默认 `git archive`，自动将 LF 改为 CRLF。验收测试捕获了清单字节摘要变化。校正后又捕获 preset 对 CRLF 源码的旧绑定与 LF 源码不一致；两次失败都保留。

通过 `.gitattributes` 为 `artifacts/maps` 下的 JSON/JSONL、`src/**/*.py` 和发布的 SHA-256 清单固定 LF，PNG 未设置文本转换。按 Git 中的精确 LF 源码重新绑定验证清单；没有放松配置或 preset 完整性检查。当前包实现摘要为 `bfe667f1cf6fa9515e6828935889c309b4984faea36af2fb45948ee19f17aaee`。Windows `core.autocrlf=true` 下导出清单的 LF 和摘要已另行核对。

后续修改受摘要约束的源码时，固定场景绑定也需显式更新并记录，不能手工绕过校验继续使用旧场景。固定的地图划分和选择规则不因此改变。

## 6. 实际验收与复现

全量几何/种子审计源码：[`scripts/audit_map_dataset.py`](../scripts/audit_map_dataset.py)。先用 Pillow/SciPy 和独立 hash 编码核对，再调用生成器做全量种子复现，不调用生产 map preparation 来充当自己的几何 oracle。16 路服务器 CPU 本次耗时 **199.63 秒**，包含生成重算；这是数据验收耗时，不是环境 step/s。

| 远程 pytest 集合 | 不同测试最终通过数 |
|---|---:|
| test_maps.py | 17 |
| test_sampler.py | 4 |
| test_validation.py | 3 |
| test_phase3_acceptance.py | 11 |
| 合计 | **35** |

首轮 34 通过、1 失败；LF 校正后同一项发现实现摘要不一致；最终只重跑该失败项并通过，按测试 ID 去重为 35。其余通过项没有为文档修改反复重跑。补充实际加载、目录/validation 选择检查与失败辅助假设均单独记入 JSON，不混入 pytest 数量。

环境为 Python 3.11.0、JAX 0.4.35、jaxlib 0.4.34、NumPy 2.0.2、SciPy 1.13.1、Pillow 13.0.0、pytest 8.4.2，H100 驱动 550.163.01。`PYTHONPATH` 明确指向审计快照，pytest probe 实际断言导入路径和 JIT 数组位于 `cuda:0`。最终 228 个快照文件逐项核对，原始失败运行与最终 LF 来源均有单独身份记录。原始日志保留在服务器 `/home/user/Tz/acceptance/phase3-20261114/results`。

从完整仓库根目录核对所有 PNG：

```bash
sha256sum --check research/evidence/phase3-data-files-20261114.sha256
```

全量重放使用证据中的 compact reference 固定 PNG Git tree 和六个 registry/manifest blob 身份；先将 JSON 中 `reproduction.compact_git_reference` 保存为 gzip JSON，再执行：

```bash
python - <<'PY'
import gzip
import json
with open("research/evidence/phase3-20261114.json", encoding="utf-8") as stream:
    reference = json.load(stream)["reproduction"]["compact_git_reference"]
with gzip.open("/tmp/ir2-phase3-reference.json.gz", "wt", encoding="utf-8") as stream:
    json.dump(reference, stream)
PY
export PYTHONPATH="$PWD/src"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/audit_map_dataset.py --data-root "$PWD" --output /tmp/ir2-phase3-audit --workers 16 --reference-git-inputs /tmp/ir2-phase3-reference.json.gz
python -m pytest tests/benchmark/unit/test_maps.py tests/benchmark/evaluation/test_sampler.py tests/benchmark/evaluation/test_validation.py tests/benchmark/integration/test_phase3_acceptance.py -q -p no:cacheprovider
```

实际外置 runner、CUDA/source probe 和补充核验脚本保存在机器可读证据中；审计 CLI 依赖现有项目和 `ex` 环境，未新增依赖或独立框架。

## 7. 范围边界和下一阶段

D4 分组检查覆盖旋转/镜像等价，不声明排除任意平移、局部编辑或感知相似布局。宽度 OOD 的难度和方法效果尚需实际评测。本次也不验证未来自定义生成器/地图池或完整正式实验矩阵。

下一阶段是 **Phase 4：基线与统一评测闭环**。应先固定 Benchmark 专用配置与基线，使用本阶段冻结的训练/验证/测试边界，再进行小型配对评测。训练仍保持暂停。
