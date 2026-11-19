# AUD-001：环境包发布遗漏恢复记录

日期：2026-11-12。目标分支：`audit/benchmark-foundation-2026-11-12`。

## 根因与恢复来源

公开初始提交 `ca663f725c335351ed5c8bec45960aa1478d5e36` 未跟踪 `src/ir2_benchmark/env/`，因此该快照不能提供 CLI 所需的 JAX 环境包。这是实际发布缺陷，不是根目录 `env.py` 可以补偿的导入差异。

本机原目录和既有服务器项目中均存在完整的九个文件。通过 SSH 逐文件读取并比较 SHA-256，九个文件与本机原文件逐字节一致。恢复直接使用这些文件，没有从原版 IR2 重建，也没有修改环境语义。

原 `.gitignore` 中，`!src/ir2_benchmark/env/` 位于后续 `ENV/` 之前。在本机 Windows Git 的忽略大小写规则下，后者重新忽略了整个源码目录。修复将源码例外移到所有虚拟环境规则之后，并明确排除该目录内的 Python 缓存。进一步检查 `src`、`scripts`、`tests`、`configs`，未发现其他被忽略的源码。

| 恢复文件 | 服务器原文件 SHA-256 |
|---|---|
| `__init__.py` | `a204314f4cf1d5cc030caa7a3e20b19c93ce583f6273fcba129d86c891363e89` |
| `artifacts.py` | `e7e0adf999c043340cd44df840deeeb60a00c42e95bb4dd3b256c96f1571c349` |
| `ir2_observation.py` | `d39f48c6a57c471954d9d9562f704e4a7d90998736c581e833279f311a172b13` |
| `observe.py` | `85b0746af467d1b609940d614b9f433ab6097ea0604b2ae745d9fe26e5dc2cc7` |
| `reset.py` | `92b2d47cc4f4fbb63e981e0de03f45e6149250ead5417c820a64ffbec9fdc12b` |
| `rewards.py` | `63679ac961d0e8bc3b57f2edf583fc0a90b3c603b04356e0750ee17d1ba32e6a` |
| `rollout.py` | `0625324e68ff14f803ad7dee2a8e2cea38957505521f357ec396211067061a4f` |
| `scenarios.py` | `a59cc65b775edc8a144c794f962f966438b157ea8e8931f26d732087e13b82c6` |
| `step.py` | `f520bec270646564bd4350835528e381b2e2628d8f6f497768c1dcf1ac69e0ea` |

Git 对 `artifacts.py`、`reset.py`、`step.py` 的 CRLF 换行进行了标准 LF 归一化；其 Git blob 哈希与上表原始字节不同。九个暂存文件在统一换行后均与原实现内容相同。

## 验证范围

从 Git 暂存树导出源码、配置和测试到独立目录，而非直接从当前工作区运行。在全新的 Windows Python 3.11.0 CPU 虚拟环境中，以 `requirement.txt` 约束安装基准运行时及打包／测试工具，构建 wheel 并安装。JAX 为 0.4.35，jaxlib 为 0.4.34，Flax 为 0.10.2，Optax 为 0.2.4。验证包中包含九个模块，导入实际来自已安装 wheel，而非旧服务器包或本机 `src`。

服务器文件比对成功；后续独立运行连接出现 SSH 握手超时，因此不能声称本次完成了服务器 Linux/CUDA 运行验证。此修复不安装原版 PyTorch/Ray 全套工具，不构成完整 `requirement.txt` 或 CUDA 依赖矩阵验收。

实际结果：

| 检查 | 结果 |
|---|---|
| 构建 wheel | 成功；`ir2_benchmark-0.1.0-py3-none-any.whl` |
| wheel 文件审计 | 环境目录含九个 `.py` 文件，包括包初始化文件 |
| 非 editable 安装 | 成功，安装到全新虚拟环境 |
| `pip check` | `No broken requirements found.` |
| 八个功能模块导入 | 全部成功，并断言模块路径位于该虚拟环境内 |
| `python -m ir2_benchmark.cli --help` | 退出码 0 |
| 三个单元测试文件＋三个集成测试文件 | **78 passed in 61.98s**，无跳过／失败 |
| micro rollout | 退出码 0；256 事件、128 轮；正常时间截断 |

wheel SHA-256：`1926cfe3ab258a02c2d59440d9ca3d9f01b483b050303f32fadc7629880caa23`。该摘要标识本次构建，其他打包器版本或构建元数据可能产生不同 wheel 摘要。

本次测试命令：

```bash
python -m pytest \
  tests/benchmark/unit/test_ledger.py \
  tests/benchmark/unit/test_packet.py \
  tests/benchmark/unit/test_channel.py \
  tests/benchmark/integration/test_event_step.py \
  tests/benchmark/integration/test_rollout.py \
  tests/benchmark/integration/test_one_hop.py \
  -q -p no:cacheprovider
python -m ir2_benchmark.cli rollout --config configs/micro.json --seed 7 --output <temporary-output-directory>
```

micro 使用配置所选择的简化 `ir2_one_hop`，不是原尺寸 `ir2_graph` 验收。末覆盖率 0.44889779559118237，`terminated=false`、`truncated=true`，两台机器人各花费 2044 字节；这是环境运行检查，不是学习策略成绩。没有启动正式训练，也未重跑完整传感器／图 parity 或完整地图审计。

## 本轮未改变的范围

- 不改信号模型、不统一其不同数学语义；`attenuation` 和 `ir2` 仍需分别验收。
- 不修改策略网络、B 路线或正式实验矩阵。
- `ir2_graph` 与简化 `ir2_one_hop` 都执行一跳邻接端点完整到达；二者的图生成、邻接构造与观测不同。旧 `grid_path` 才是沿全图路径按距离截断移动的另一类规则。不能把三个模式视为同一个基线，也不能把 `ir2_graph` 误写成不是一跳运动。
- 原始审计文字保留为历史静态快照，新增记录说明已核实与已修复的部分，不将全部审计条目标为通过。
- 本次只更新独立审计分支，不覆盖或合并 `main`。
