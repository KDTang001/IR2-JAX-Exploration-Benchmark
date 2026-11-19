# PR #1 / #2 合并前终审（冻结 298e625d）

日期：2026-11-19。审查对象为 PR #1 的 d3d6c5fdec831de3076b7ede70e25b72108e59bf 与 PR #2 的 298e625d21deb205a095ba78643a5a3d6dd4976c。此报告对应代码整合门槛，不是合并授权、正式发布完成或论文实验完成。

## 后续补验

2026-11-19：同一冻结候选的独立CPU安装、CLI/续跑及声明哈希验收已通过，只读终审APPROVE，R1已解除；详见[补验与当前GO边界](RELEASE_INSTALL_RECOVERY_REPORT.md)。以下NO-GO为首轮历史记录，原失败证据保留；PR及main仍未合并。

## 首轮判定（历史）

**NO-GO（发布验收证据未闭环），不是认定候选环境源码发生了新的功能错误。**

合并内容预演、源码/历史完整性、完整 wheel 打包及目标 Linux CPU 的32项定向回归通过。但是独立 CPU 环境的依赖安装未在有界尝试内完成，新的独立 wheel 导入、CLI、来源身份与续跑检查尚未运行；不能把历史安装或 ex 的源码回归代替这一门槛。当前不建议实际合并/发布，保持两项 PR 为 Draft。

阻断项 **R1**：完成同一冻结源码/哈希锁的独立安装，随后执行已声明的安装后检查。uv 和 pip 分别在一次600秒尝试中超时；根因尚未完全确定。包索引请求200、可见锁定版本，只证明索引可达，不证明所有 wheel 的获取和依赖闭包已经验收。没有据此修改生产源码、依赖版本或已有实验结果。

未修改生产源码、候选配置、训练/世界模型/队友预测模块或旧证据；未启动 GPU、训练、240 项或正式矩阵；未实际合并、retarget、改变 Draft 状态或移动 main/两项 PR 的 head。

## 冻结与合并预演

| 对象 | 固定身份 |
| --- | --- |
| main | ca663f725c335351ed5c8bec45960aa1478d5e36 |
| PR #1 head | d3d6c5fdec831de3076b7ede70e25b72108e59bf |
| PR #2 head | 298e625d21deb205a095ba78643a5a3d6dd4976c |
| PR #1 预演内容树 | 817d465163b5a4fb4e7060345cfa9c2bba74cd02 |
| PR #2 后的预演内容树 | 4c4c7cd43b0ff017cc84b2e0d97d906226dd7a53 |

使用 `git merge-tree --write-tree --messages <left> <right>`，未执行 `git merge` 或生成/发布合并提交。ca 是 d3 的祖先，d3 是 298 的祖先；两步 exit=0、无冲突，最终内容树等于 298。适用前提是 PR #1 以保留审计祖先的方式整合，然后将 PR #2 重新定向到更新后的 main。没有核实或改变仓库的平台合并策略；若必须 squash/rebase，需另行核对实际内容/祖先及 PR diff。预演内容树不是未来 main 的提交身份。

## 源码、配置与历史完整性

- 当前111个包源码文件逐一匹配已验收 H100 清单，聚合摘要为 `402942e2eefc04ad33aa65fe8d8e83e68b2411e05e932264548f8cff3d0a7fbd`。
- PR #2 的生产包差异仅涉及 episode、errors、release、runner、throughput、ir2_graph 六个文件；检查 OOM 分类、来源身份和已验收图计算优化的调用边界。新增容量/探针脚本不修改 A/B 或预测器。
- 五份后续归档索引的314个文件均存在且 SHA256 一致；PR #2 未修改/删除此前 raw evidence。核对五张代表 PNG 及三份相关清单，共8个绑定，均匹配服务器身份。此处不是重新执行 Phase1–5 全量测试或重算全部历史轨迹。
- README、research 顶层与 docs 顶层的289个相对 Markdown 链接均存在。范围不包括外部 URL 可用性或所有原始数据许可的独立法律审查。
- C400 与 C1024 配置仅 graph_capacity、graph_padding_size 不同。旧 C400 正式配置/240项清单保留；新容量不沿用其任务身份。

## 目标 Linux CPU 安装与实际 CLI 验证

服务器：`/home/user/Tz/acceptance/release-review-20261119-pip`。包构建成功；wheel SHA256为 `0c2d8e348b391ab769aa707f9deaf4f1cfba29637696391dccd3a73c28d0aff4`，111个源码文件均与冻结清单一致，包含 LICENSE.txt 和 NOTICE。构建源码压缩归档 SHA256为 `5746e913fd4df4369223c2063febb27686807580c1573ab3b6cfed546a2b598b`。

复用的构建锁 SHA256为 `cdd5df9ab42696d6d0b89d3989e976377da518c8443786ab2f6be5c6b60e6b7e`；37项 CPU 依赖锁 SHA256为 `e20b0aaaabadc9f5db169d07b2f7bda185f2497fff28b394c5554c78026d4e24`。锁文件与版本均未重新生成或修改。构建缓存与包索引缓存的使用均在原始协议中保留，不声称全流程冷下载性能。

| 手动 QA | 结果 | 原始记录 |
| --- | --- | --- |
| 冻结引用、Draft/未合并状态 | PASS；结束时 head/base 未变 | [结束快照](evidence/release-preflight-20261119/end-refs-and-ci.json) |
| PR1→PR2 内容树预演 | PASS；无冲突、最终树等于298 | [merge-tree记录](evidence/release-preflight-20261119/merge-rehearsal.json) |
| 源码与历史索引/地图绑定 | PASS；111源码、314归档文件、8绑定 | [完整性记录](evidence/release-preflight-20261119/history-integrity.json) |
| wheel 完整打包、源码/许可证/NOTICE | PASS | [wheel核对](evidence/release-preflight-20261119/cpu-final/wheel-inspection.json) |
| 独立依赖安装 | INCOMPLETE；uv/pip各一次600秒超时 | [uv超时](evidence/release-preflight-20261119/uv-install/continuation-controller.log)、[pip超时](evidence/release-preflight-20261119/cpu-final/controller.log) |
| 独立 wheel 实际导入与来源身份 | NOT RUN；依赖安装阻断 | [安装状态](evidence/release-preflight-20261119/cpu-final/clean-install-status.json) |
| 独立 wheel doctor/evaluate/resume/report | NOT RUN；依赖安装阻断 | [安装状态](evidence/release-preflight-20261119/cpu-final/clean-install-status.json) |
| 目标 Linux CPU 定向回归 | PASS：32/32，失败/错误/跳过均0 | [JUnit](evidence/release-preflight-20261119/cpu-final/source-tests.xml)、[日志](evidence/release-preflight-20261119/cpu-final/source-tests.log) |
| 停止后源码与自有进程 | PASS；111源码未变，自有活动进程为0 | [结束核验](evidence/release-preflight-20261119/cpu-final/postrun-verification.json) |

定向回归实际导入本次冻结的 `source/src/ir2_benchmark`，使用服务器 ex 的现有依赖、强制 CPU，覆盖 release metadata、batch runner、report release 及 runtime-only CLI；其中含小地图真实批量轨迹/尾批和错误分类检查。它不是独立 wheel 安装证明，也不是重新运行 Phase1–5。runtime-only 测试不进行优化器更新。独立安装夹具的两项 CLI 任务未执行，不能报作已完成。

全轮强制 `JAX_PLATFORMS=cpu`、`CUDA_VISIBLE_DEVICES=""`；没有新GPU实验。未对 ex 安装依赖：它只提供基础解释器与源码回归运行时。生产代码未改。人工 QA 未闭环，因此没有启动附加 gate reviewer，也没有给出 reviewer approval。GitHub未返回两提交的 commit statuses 或 PR 触发 workflow runs，**这不是 CI 通过**；端点范围见结束快照。

预先声明但尚未执行的 CLI 安装夹具计划使用单独的24×24地图、32桶、N2、C64、最多4轮、两种最近前沿通信配置。其目标仅是验证安装/评测/续跑接口，不是新 C1024 完整任务认证。原尺寸完整回合结论来自已冻结的历史 H100 证据。

### 预演输入与基础设施问题

初次许可证断言把 Git LF 摘要与 Windows archive 的 CRLF 字节直接比较，已明确为预演输入问题；初始源码归档还遗漏 NOTICE，因此未用于最终验收。改为 `git -c core.autocrlf=false -c core.eol=lf archive ... LICENSE.txt NOTICE ...` 后，许可证与归属说明匹配冻结 Git blob。生产源码未改。

准备尝试中主动停止了不完整构建输入及未完成的依赖获取；离线缓存对 PyPI 缺 Hatchling archive、对 Aliyun 缺 absl-py 版本，不能作为完整离线安装源。正确归档后，uv 在线同步在600秒超时；再用 pip --python 指定全新 venv、同一完整 hash lock、--no-deps 后计划完整 pip-check，仍在600秒超时。未进入 pip-check 或独立安装后的 CLI 检查。后续只运行了有明确来源断言的源码定向回归。

根目录的 controller.log 没有保留超时子进程的完整增量下载输出，因此不能据此断言具体哪个包或网络环节是根因。时间上限是本轮验收脚本的界限，不是证明依赖永远无法安装。所有准备失败/中止与后续检查单独保留，不修改或改算为成功。

[机器 QA 矩阵](evidence/release-preflight-20261119/qa-matrix.json)、[精确提交审查记录](evidence/release-preflight-20261119/lane-ledger.json)、[91项原始记录索引](evidence/release-preflight-20261119/collected-index.json)与[传输核验](evidence/release-preflight-20261119/transfer-verification.json)可交叉检查。原始记录分为许可证检查、遗漏NOTICE的准备、网络/离线尝试、uv在线尝试与最终pip/源码回归；前面的无效准备不算生产构建通过。

## 已通过的历史范围与禁止外推

| 身份 | 已有证据支持 | 仍不能宣称 |
| --- | --- | --- |
| C400 | 原始容量控制；保留3张程序化地图的容量失败 | 所有正式长度任务均可完成 |
| C1024 Normal/8KiB/mixed | 五图、N4、768桶；20次执行、10对 B1/B4 精确一致 | 所有地图/并行规模均适用 |
| C1024 2KiB/mixed | Normal/Dropout .3/Burst 各独立；30次有效执行 | Burst 抗断联有效性（窗口发送为0） |
| C1024 8KiB/mixed/Burst | 10次有效；91,040窗口发射字节；923可投递接收副本=784受阻+139送达；10/10前缀相同、8/10路径变化、9/10恢复相关边投递 | 普遍抗断联、断联必然有害/有益或算法排名 |
| IR2 adapted | 外部预训练参考，允许在已声明适配环境中推理 | Benchmark train-only 公平学习基线；C1024 等同原训练输入范围 |
| 性能 | 有界独立测量与本次观察成本 | 将混有其他GPU PID的 Burst IR2 时间解释为独占 H100 速度 |

八个配置是两种导航×四种固定通信，不是八种独立算法。240项清单未全量执行；论文核心协议、全矩阵规模、公平 train-only SAC 及更多 N/桶/预算/故障/通信方式均尚未完成。零预算、N2/N6 与其它大桶的广泛 C1024 认证也未由这轮补齐。

## 实际合并前后的要求

1. 只有得到用户明确合并批准后，才按 PR #1 → 刷新 main → retarget/重审 PR #2 → PR #2 的顺序操作。审查通过不自动改变两项 Draft PR。
2. 实际平台策略、实际最终 diff 与内容树须重新核实；任何源码/配置冲突修复会使对应旧 QA 覆盖失效。
3. 从实际最终 main 的提交重新构建/安装 wheel，核对真实导入、源码/依赖/wheel身份、CLI、runtime-only doctor、评测/续跑。不能把本报告的298 wheel摘要登记为未知的未来main发布包。
4. 原始错误和限定范围必须保留在发布说明中；新实验需要独立任务身份与新的资源授权，不因代码整合而启动训练或240/论文矩阵。

本轮报告与证据保存在独立 `codex/release-review-20261119` 分支；PR #2 仍冻结298e625d，PR #1 与 main 不变。报告分支不是新的生产候选或第三个待合并功能 PR。

解除 R1 的最小后续范围是：只补齐这版独立 CPU 依赖安装/闭包，核对实际导入位置，执行 runtime-only doctor、预定两项安装夹具及结果不变的续跑/report、真实 wheel/来源身份。既有32项定向回归按精确未变源码保留；如只有下载/安装流程调整，不重做 Phase1–5或H100 rollout。R1解除后再完成发布 gate；本报告不请求或预先授予实际合并许可。

## 收尾与保存

自有进程检查为空后，已清理本轮新目录中的展开源码、临时 venv、工作目录和下载缓存；保留原始日志、锁、协议、wheel、源码归档和证据包。仅删除本轮明确创建的路径，不删除项目源码或旧验收结果。见[清理记录](evidence/release-preflight-20261119/cleanup-receipt.json)。后续补验应使用新输出目录，从冻结源码/归档重新准备，不覆盖本轮失败记录。
