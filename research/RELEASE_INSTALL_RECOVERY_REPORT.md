# 独立 CPU 安装补验与发布门槛 R1 闭环

日期：2026-11-19。生产候选仍冻结 **298e625d21deb205a095ba78643a5a3d6dd4976c**。

**结论：R1 已解除；人工 QA PASS、只读终审 APPROVE，具备申请明确合并批准的技术条件。没有实际合并或发布 main。** 首轮 [NO-GO 记录](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/blob/769737844afdf29c7c862d4290ee0bb4d74e8835/research/RELEASE_PREFLIGHT_REPORT.md)及其全部失败证据保留。本轮没有修改 src、生产配置或依赖版本，没有使用 GPU、训练或执行240项/论文矩阵。

## 1. 如何解除阻断

确认了三个互相独立的事实：

1. 原控制脚本用 capture_output 等待退出后才保存日志，TimeoutExpired 会跳过输出和终态写入，并可能遗留安装器子进程。本轮只修复验收控制协议：输出直接写日志、每项命令有独立进程组和期限、超时后终止并回收、失败与断言始终写结果。一次受控1秒超时保留了事先输出的标记，实际后代进程已退出。[受控验证](evidence/release-install-recovery-20261119/initial/controller-probe.json)
2. 在目标服务器查询官方 PyPI，原 CPU 锁的37项都存在兼容 Python3.11/Linux 且 SHA 匹配的 wheel，总计197,594,430字节。4路取包后逐文件验证大小与SHA，再在全新 venv 中离线安装、检查依赖闭包。没有重解析或替换原锁。旧超时逐包输出缺失，不能从旧日志恢复具体卡在哪个包或操作，不把推测写作唯一根因。[取包身份与记录](evidence/release-install-recovery-20261119/initial/wheelhouse.json)
3. 第一套恢复环境实际导入/doctor/评测/续跑/report 均通过，但 uv 的本地 wheel 安装生成 archive_info={}，发布模块如实返回 archive_hashed_no_declared_hash，故严格声明哈希断言失败。这条 result=false 保留。第二套干净环境通过 pip --require-hashes 和显式本地 wheel SHA 安装，由安装器真实生成声明哈希；没有手改 direct_url.json 或伪造 Git revision。[旧结果](evidence/release-install-recovery-20261119/initial/result.json) · [最终结果](evidence/release-install-recovery-20261119/hashed-install/result.json) · [wheel哈希要求](evidence/release-install-recovery-20261119/hashed-install/benchmark-wheel.lock)

## 2. 冻结身份与隔离

| 对象 | 身份/核对结果 |
| --- | --- |
| 生产候选 | 298e625d21deb205a095ba78643a5a3d6dd4976c |
| 内容树 | 4c4c7cd43b0ff017cc84b2e0d97d906226dd7a53 |
| 111个源码文件聚合摘要 | 402942e2eefc04ad33aa65fe8d8e83e68b2411e05e932264548f8cff3d0a7fbd |
| wheel SHA256 | 0c2d8e348b391ab769aa707f9deaf4f1cfba29637696391dccd3a73c28d0aff4 |
| 原CPU依赖锁 SHA256 | e20b0aaaabadc9f5db169d07b2f7bda185f2497fff28b394c5554c78026d4e24 |
| 实际运行时 | 目标Linux，Python3.11.0，37项锁定依赖+benchmark共38个distribution |
| 最终安装位置 | /home/user/Tz/acceptance/release-install-recovery-20261119/hashed-install/cpu |
| 隔离检查 | 无system-site-packages、用户site关闭、无editable；ir2/JAX/jaxlib/Flax/Optax/NumPy/Pillow/Matplotlib均从新venv导入；无CUDA/Torch发行包 |
| 来源声明 | 实际与安装器声明wheel SHA一致；wheel_status=verified_archive_hash；code_revision/runtime_lock保持真实缺失值None，锁的外部核验另记 |

ex 只提供基础解释器；依赖安装明确指定新的 venv，没有向 ex 安装依赖。全轮强制 JAX_PLATFORMS=cpu、CUDA_VISIBLE_DEVICES为空；doctor实际设备为CPU、优化器更新0、loss为空。[导入路径](evidence/release-install-recovery-20261119/hashed-install/installed-origins.json) · [doctor](evidence/release-install-recovery-20261119/hashed-install/doctor.json) · [真实来源清单](evidence/release-install-recovery-20261119/hashed-install/reported/release_manifest.json)

## 3. 实际 CLI 验收

夹具明确是24×24合成地图、32桶、N2、C64、最多4轮，最近前沿 off/mixed 两项，不是新的原尺寸 C1024 验收。初次恢复和最终环境各执行两项；最终环境两项均为合法探索超时，执行错误0。合计四项安装夹具执行，不能称作四张独立地图或论文实验。

| 检查 | 实际结果 | 证据 |
| --- | --- | --- |
| 依赖与导入 | 37项实际版本吻合；111源码与冻结Git及wheel吻合 | [独立复核](evidence/release-install-recovery-20261119/initial/postrun-verification.json) |
| CLI help / runtime-only doctor | exit0、CPU、零优化器更新 | [命令/耗时](evidence/release-install-recovery-20261119/hashed-install/commands.json) |
| 首段评测 | 实际执行1项、无错误 | [prefix输出](evidence/release-install-recovery-20261119/hashed-install/evaluate-prefix.log) |
| 补齐与续跑 | 执行1项、复用1项，原结果字节不变 | [resume输出](evidence/release-install-recovery-20261119/hashed-install/evaluate-resume.log) |
| 再次续跑 | 执行0项、复用2项、无错误 | [第二次resume](evidence/release-install-recovery-20261119/hashed-install/evaluate-second-resume.log) |
| report / 来源 | 两项requested完整、limited=true、formal_complete=false；忽略无关工作目录Git；归档哈希声明与实际一致 | [report](evidence/release-install-recovery-20261119/hashed-install/reported/report.json) |
| 结束核验 | 两个结果及轨迹SHA校验通过，自有活动任务0 | [postrun](evidence/release-install-recovery-20261119/initial/postrun-verification.json) |
| 既有定向回归 | 复用精确未变候选的32项远程CPU通过记录，不重新跑Phase1–5 | [原JUnit](evidence/release-preflight-20261119/cpu-final/source-tests.xml) |
| 只读终审 | APPROVE；复核122项归档哈希、111源码及以上实际证据 | [终审](evidence/release-install-recovery-20261119/gate-review.md) |

取包实际耗时 380.675 秒；最终环境离线依赖安装 9.806 秒。以上是本次安装的观察成本，包含缓存/网络条件，不是独占GPU吞吐、训练速度或普遍硬件性能结论。完整实际参数保留在 commands.json；取包和安装分开计量，避免将总等待当作环境步进速度。

## 4. 保存、复用与边界

- [机器QA矩阵](evidence/release-install-recovery-20261119/manual-qa.json)、[精确提交审查账本](evidence/release-install-recovery-20261119/lane-ledger.json)和[122项原始文件索引](evidence/release-install-recovery-20261119/bundle-index.json)绑定两套环境及原失败。源码、wheel、CPU锁均未改变。
- 复用方法是：选择新输出目录和不继承系统site的venv；从SHA已核对的wheelhouse用 --no-index/--require-hashes 安装原cpu.lock；给benchmark wheel单独写 file-URL + SHA 的要求并由pip安装；执行相同导入、doctor、两项评测/续跑/report。真实执行协议和参数已归档，不覆盖旧结果，也不手改安装元数据。依赖包可继续用于最终main复验，无须传回本机。
- C400继续作为原始容量对照、保留容量失败；C1024仍限于此前声明的五图/N4/768等已验收范围。8KiB Burst实际阻断784个接收副本及混合AUC方向不变，不能推出普遍鲁棒性或排名；共享GPU成本仍不是独占性能。八配置仍是两导航×四通信；外部IR2训练暴露未经认证、SAC未重训、全240项及论文矩阵未启动。
- 本机3个临时辅助脚本和对应缓存已核对不存在。审查工作树仍为App当前主检出，归档需要先在界面切回原项目或用户选择保留；没有通过手动删除绕过保护。此项本机工作目录收尾与R1技术验收分开记录。

验收完成后，已退役本轮两套临时venv并清理展开源码、工作目录和uv缓存；保留服务器上的已核验wheelhouse、候选wheel和完整证据供最终main复验。安装位置与导入路径是验收当时的实际记录，不表示临时venv仍在线。见[清理记录](evidence/release-install-recovery-20261119/cleanup-receipt.json)。

## 5. 合并仍需明确批准

本次GO只解除冻结候选的发布前技术阻断。PR1和PR2仍Draft/unmerged，main仍ca663f7；[引用快照](evidence/release-install-recovery-20261119/release-refs.json)。后续需获明确批准，再依赖顺序整合PR1、刷新main、retarget/复审PR2、整合PR2；如果平台只允许squash等方式，先重验实际内容树和差异。从实际最终main重新构建并验真实导入/来源/CLI，不能把当前候选wheel身份直接冒充未来main发布身份。没有因本轮通过而启动GPU、训练、资源先导或论文实验。
