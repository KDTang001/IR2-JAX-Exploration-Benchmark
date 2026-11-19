# IR2-JAX 发布记录 v1

2026-11-19，Phase 1–5 和 Phase 6-Prep 已合入 `main`。本记录绑定完成合并时的工程基线
[`98aa3206dd28af091e245ff9963759f8279f0221`](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/commit/98aa3206dd28af091e245ff9963759f8279f0221)。
v1 是记录版本，wheel 包版本为 `ir2-benchmark 0.1.0`。

[PR #1](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/pull/1)
先合入 `main`，产生合并提交 `e7f94821340c512cb260e6ae398b928416367ebc`。
[PR #2](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/pull/2)
改为指向更新后的 `main`，435 个文件的变更与已经审查的差异逐项一致，随后合并为
`98aa3206`。最终 Git 内容树为 `4c4c7cd43b0ff017cc84b2e0d97d906226dd7a53`，
与先前独立验收的 `298e625d` 候选相同。两次均使用保留父提交的合并方式。

## 构建和来源

目标服务器从最终提交的 Git 归档重新构建 `ir2_benchmark-0.1.0-py3-none-any.whl`。
归档覆盖 `src`、`configs`、`pyproject.toml`、`LICENSE.txt` 和 `NOTICE`，其 Git
归档头记录最终合并提交。首次离线构建的缓存缺少指定构建包。
重试将六项构建依赖按原锁下载并逐包核对 SHA-256，再离线构建成功。

| 对象 | SHA-256 |
| --- | --- |
| 最终提交的构建输入归档 | `4005518219c9379e981d32ed31eaedf385ccde9db1a7957cfc25b88fa2c2fbfe` |
| 重新构建的 wheel | `0c2d8e348b391ab769aa707f9deaf4f1cfba29637696391dccd3a73c28d0aff4` |
| 111 项源码路径与摘要清单的规范 JSON（`code_hash`） | `402942e2eefc04ad33aa65fe8d8e83e68b2411e05e932264548f8cff3d0a7fbd` |
| [构建依赖锁](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/blob/7e75f754ce6909fbfd8af2edafc9daf1902dcdd7/research/evidence/release-install-recovery-20261119/hashed-install/build.lock) | `cdd5df9ab42696d6d0b89d3989e976377da518c8443786ab2f6be5c6b60e6b7e` |
| [CPU 依赖锁](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/blob/7e75f754ce6909fbfd8af2edafc9daf1902dcdd7/research/evidence/release-install-recovery-20261119/hashed-install/cpu.lock) | `e20b0aaaabadc9f5db169d07b2f7bda185f2497fff28b394c5554c78026d4e24` |

重新构建的 wheel 与候选 wheel 的 SHA-256 一致。归档、构建产物和实际安装目录中的
111 个 Python 文件均与最终 Git 提交的文件字节匹配。`code_hash` 是以排序键序列化
`source_hashes` 后计算的 SHA-256，可用 Python 的
`hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest()` 重算。
[构建身份](evidence/release-record-v1/build-identity.json) · [发布来源清单](evidence/release-record-v1/reported/release_manifest.json)

安装使用 pip 的 `--require-hashes`。安装器写入 `direct_url.json` 的归档哈希与实际 wheel 一致，
来源记录返回 `verified_archive_hash`。来源清单中的 `code_revision=null` 表示这个 wheel
没有嵌入可声明的 Git 提交身份。最终 Git 提交由构建归档、内容树及源码逐文件摘要绑定。
同理，`runtime_lock=null` 表示安装包未嵌入运行时锁身份。实际采用的依赖锁及其
摘要记录在构建证据中。[实际导入路径及安装器记录](evidence/release-record-v1/installed-origins.json)

## 最终 Linux CPU 验收

隔离环境使用 CPython 3.11.0，37 项锁定依赖和 `ir2-benchmark` 均从新建环境导入或核对。
主要版本为 JAX 0.4.35、jaxlib 0.4.34、Flax 0.10.2、Optax 0.2.4、NumPy 2.0.2。
CPU 依赖来自已核验的本地 wheelhouse 并使用 `--no-index` 安装。
全部实际版本见[依赖清单](evidence/release-record-v1/cpu-freeze.log)。

| 检查 | 实际结果 |
| --- | --- |
| wheel 安装与依赖检查 | 离线 CPU 依赖安装 9.807 秒，依赖检查通过，模块从隔离环境导入 |
| CLI help 和 `doctor --runtime-only` | 通过，JAX 设备为 CPU，扫描与推理检查为真，优化器更新 0 |
| `evaluate` 与续跑 | 两项限 4 轮的小地图夹具，均正常超时，执行错误 0。首次执行 1 项，补齐执行 1 项并复用原结果，再次续跑复用 2 项 |
| `report` 和证据复核 | 限定清单完整，两条结果和轨迹的哈希核验通过 |

完整结果见 [CPU doctor](evidence/release-record-v1/doctor.json)、
[最后一次续跑](evidence/release-record-v1/evaluated/runner_summary.json)、
[评测报告数据](evidence/release-record-v1/reported/report.json)和
[最终机器验收摘要](evidence/release-record-v1/final-verification.json)。
本记录整理时从服务器逐字节复制了这八份文件，核对了传输 SHA、最终提交的 111 项
源码哈希及实际依赖版本。可在 `research/evidence/release-record-v1` 下执行
`sha256sum -c SHA256SUMS` 核对原件。服务器保留 wheel、构建输入、日志及完整原始结果，
合并前的验收失败另见[原发布审查记录](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/blob/769737844afdf29c7c862d4290ee0bb4d74e8835/research/RELEASE_PREFLIGHT_REPORT.md)
和[候选包补验](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/blob/7e75f754ce6909fbfd8af2edafc9daf1902dcdd7/research/RELEASE_INSTALL_RECOVERY_REPORT.md)。

本记录的最终验收范围为目标 Linux CPU 的安装、来源和最小 CLI 流程，优化器更新为零。
C400 继续作为原始容量对照，C1024 继续作为已在限定条件下验证的候选。Phase 7 待
冻结论文样本量、比较指标、C1024 适用范围及仅使用训练集的 SAC 基线安排。全量 240 项
先导和正式实验尚未执行。
