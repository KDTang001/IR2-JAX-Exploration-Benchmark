# Phase 3 地图验收实施清单

**Goal:** 在目标服务器核实全部 IR2/程序化数据、划分和出生/覆盖契约，发布可核对的文件摘要和报告。

**Architecture:** 复用已批准的数据设计、现有 manifest、生成器和 sampler。独立 PNG/四邻接连通域/哈希 oracle 全量核对数据；JAX CUDA 检查实际 map loading/reset；只修复实际阻断。用户已授权开始执行，本清单在当前审计分支内直接实施。

**Tech Stack:** 服务器 conda ex，Python/NumPy/Pillow/SciPy、JAX CUDA、pytest。CPU 负责图像与独立 oracle，H100 执行环境数组计算。不启动训练。

## 1. 固定输入

- [x] 核对 `artifacts/maps/sources.json`、五个主 manifest 和原始文件数量。
- [x] 阅读 `maps/{manifest,prepare,generate,sampler}.py`、validation/evaluation 加载代码和已有相关测试。
- [x] 从 Git 固定输入 blob 身份；服务器直接复用现有 PNG，逐文件核对，避免重复传输地图池。
- [x] 保存服务器版本、实际 import 路径与源文件摘要。

## 2. 全量数据 oracle

文件：`scripts/audit_map_dataset.py`（独立审计，不替换环境实现）。

- [x] 逐一解码 17,299 张 IR2 与 6,900 张程序化地图；检查形状、像素、占据、marker anchor、4 连通区域和 bucket。
- [x] 独立重算定向占据、D4 canonical、scenario hash，核对全部 manifest 字段。
- [x] 独立重算 IR2 test 优先去重及各训练族 hash-first 10% 验证划分。
- [x] 检查所有保留 train/validation/test/OOD 集合两两 canonical 交集和程序化/原始全部数据交集。
- [x] 用记录的 requested_seed 重现全部 6,900 张程序化地图及参数/拒绝历史，记录任何失败；不写另一套 PNG。
- [x] 输出逐文件 SHA-256 清单和聚合证据，完整保留排除原因。

运行入口：

```bash
PYTHONPATH=src python scripts/audit_map_dataset.py --data-root /home/user/Tz/ir2-benchmark --output /home/user/Tz/acceptance/phase3/results --workers 16
```

## 3. sampler、validation 与 CUDA

文件：现有 `tests/benchmark/{unit/test_maps,evaluation/test_sampler,evaluation/test_validation}.py`；必要新增验收放 `tests/benchmark/integration/test_phase3_acceptance.py`。

- [x] 用真实 registry 验证 sampler 的 train-only 选择、108 项平衡周期和 relocation 身份。
- [x] 核实 25 张固定验证地图均来自 validation，并且与 train/test/OOD 独立；实际执行 `validate_manifest`。
- [x] 若旧验证场景配置/哈希失效，记录原始失败，再按 `configs/receiver_ir2_graph.json` 重建相同地图集合，不恢复训练。
- [x] 在 CUDA 上对 512/768/1024 桶和 N=2/4/6 检查 reset 重现、不同且同连通区域的出生、有效边界、padding 和物理覆盖分母。
- [x] 运行相应既有测试及新增数据/输入边界回归；不为文档改动重复运行环境全集。

```bash
JAX_PLATFORMS=cuda PYTHONPATH=src XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest tests/benchmark/unit/test_maps.py tests/benchmark/evaluation/test_sampler.py tests/benchmark/evaluation/test_validation.py tests/benchmark/integration/test_phase3_acceptance.py -q -p no:cacheprovider
```

## 4. 交付

- [x] `research/MAP_DATASET_REPORT.md`：全量统计、继承与扩展、发现/修复、证据范围和剩余阶段。
- [x] `research/evidence/phase3-20261114.json`：机器可读检查、失败/复验、源码和环境身份。
- [x] `research/evidence/phase3-data-files-20261114.sha256`：可从仓库根目录核对的文件清单。
- [x] 更新总审计和路线图，推送既有审计分支，保留远程必要原始证据并清理本轮临时转运包。
