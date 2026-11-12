# IR2-JAX Benchmark 实施路线图

- 版本：Draft 0.1
- 日期：2026-11-12
- 当前状态：审计分支；尚未开始生产代码修改

## 原则
1. 先修复会阻断 import/rollout 的问题，再谈性能。
2. 保留原版 IR2 参考实现。
3. 暂停世界模型想象训练和世界模型队友意图预测。
4. 不启动正式大规模训练，直到环境、通信、地图和评测验收通过。
5. 所有修改在独立分支、逐项测试和可回滚提交中完成。

## Phase 0 — 审计与运行入口恢复（当前）
工作：
- 验证 Git Tree 中缺失的 `src/ir2_benchmark/env/*` 是否导致包导入失败。
- 确认 Python/JAX/Flax/Optax 依赖和 editable install。
- 检查 CLI help、doctor、单场 rollout 的导入路径。
- 建立静态依赖图和核心测试清单。

退出条件：
- `import ir2_benchmark` 成功；
- `import ir2_benchmark.env.reset`、`observe`、`step` 等项目模块可解析；
- `ir2-benchmark --help` 和 doctor 正常；
- 一场 micro rollout 完成并产生事件日志；
- 记录当前 main commit、依赖和环境信息。

## Phase 1 — 环境与 IR2 parity
工作：
- reset/observe/step 时序；
- 物理覆盖与局部知识隔离；
- 传感器逐格 parity；
- IR2 graph node/edge/utility parity；
- 一跳动作与旧导航模式边界；
- 确定性和容量溢出。

退出条件：
- 核心几何和事件集成测试通过；
- parity matrix 有明确证据链接；
- 所有 Benchmark 扩展均与 IR2 原始行为分开说明。

## Phase 2 — 通信和故障
工作：
- 标准 codec round-trip 和畸形包；
- 真实 wire byte 账本；
- 收件箱新旧消息覆盖；
- map delta 接收后图融合；
- 信号模型与外部故障独立；
- Normal/Dropout/Burst 配对轨迹。

退出条件：
- 每个事件的发送、费用、送达和知识变化可独立重算；
- 不存在隐式地图共享；
- 预算/故障语义与文档一致。

## Phase 3 — 地图数据和采样
工作：
- IR2 原始数据全量审计；
- 程序化地图 manifest/生成器复现；
- canonical duplicate 和跨 split 泄漏；
- anchor/component/valid_shape 核验；
- train/validation/test/OOD registry。

退出条件：
- 每个保留/排除地图均有原因；
- 文件 hash 与 registry 一致；
- 正式 sampler 只采 train；
- validation 固定并独立于 test。

## Phase 4 — 基线和评测闭环
工作：
- zero communication；
- fixed PLAN；
- fixed MAP_DELTA；
- periodic mixed-message baseline；
- nearest frontier；
- 原版 IR2 adapted policy；
- 统一任务 manifest、错误分类、指标重算和报告。

退出条件：
- 每个基线能完成小型配对评测；
- 结果由独立代码重算一致；
- 统计单位、缺失任务和置信区间固定。

## Phase 5 — 性能与工程可复现
工作：
- CPU/GPU 同 workload 配对；
- native map buckets、N、batch scaling；
- compile/reset/step/policy/update/save/record 分段计时；
- 峰值显存与 host memory；
- 恢复、重跑、manifest/checkpoint hash；
- clean install 文档和 CI。

退出条件：
- 报告所有性能口径；
- OOM/error 不被隐藏；
- 关键运行可从 manifest 和配置复现。

## Phase 6 — 论文实验和发布
工作：
- 冻结代码、数据、配置和 workload；
- 正式 baseline 重复；
- 地图族、机器人数量、预算和故障分层分析；
- Ablation 与敏感性；
- 失败案例、局限和威胁有效性；
- release manifest、citation、license、quickstart。

退出条件：
- 论文中的每张表/图都能追溯到原始任务结果；
- 所有声称的矩阵都有完整性证明；
- Benchmark 可由第三方按 quickstart 运行。

## 当前立即执行的下一步
1. 取得服务器可用的安全远程执行方式；不要在仓库或脚本中保存密码。
2. 执行 Phase 0 的导入/CLI/doctor/micro rollout 检查。
3. 如果发现缺失模块，先恢复正确源码，不临时造一个空包绕过错误。
4. Phase 0 通过后，开始 Phase 1；不提前启动正式训练。


## 进度更新：环境包发布遗漏已恢复（2026-11-12）

用户确认提交 `1431ad6a42eda555d4c34a5eacac06dbc61282ac` 已将原有九个 `src/ir2_benchmark/env/` 模块恢复到本审计分支，并修复 Windows 大小写不敏感忽略规则导致的发布遗漏。详细证据见 `ENV_PUBLICATION_RECOVERY.md`。

已完成的 Phase 0 子项（仅限 Windows CPU 独立 wheel 环境）：
- wheel 构建及非 editable 安装；
- `pip check`；
- 八个功能模块从已安装 wheel 导入；
- CLI help；
- 78 项指定单元/集成测试；
- `ir2_one_hop` micro rollout：256 事件、128 轮，正常 truncated。

仍未完成的 Phase 0/1 项：
- Linux/CUDA 独立安装与运行；
- 完整 `ir2_graph` 原尺寸场景验收；
- 完整传感器、图节点/边/utility parity 测试重跑；
- 全量测试套件与干净环境 CI；
- 完整地图审计及正式评测工作负载验证。

因此 Phase 0 目前状态为 **部分通过（Windows CPU packaging/import/micro smoke）**，不能标记为整体完成。下一工作优先进入 Phase 1 的独立 parity 证据核验，同时等待 Linux/CUDA 环境可用；正式训练仍保持暂停。

## 进度更新：Phase 1 首批 parity 证据（2026-11-12）

上述待办中的指定传感器、图结构 parity 重跑已完成：Windows CPU 下 39 项通过、无跳过，包含原尺寸地图 16 步感知与图维护轨迹，以及事件/隔离检查。详见 [IR2_PARITY_REPORT.md](IR2_PARITY_REPORT.md) 和其中的机器可读证据。

这不是完整 `ir2_graph` 多机器人 episode 验收。非零会合效用、长期容量处理和付费地图融合的联合隔离仍待核验，Linux/CUDA 与后续阶段也未完成。本次没有修改生产代码或恢复训练。
