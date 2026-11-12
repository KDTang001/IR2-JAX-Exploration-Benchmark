# 服务器运行约定

用户已指定服务器 <user>@<server>、conda 环境 ex、父目录 ~/Tz。项目部署目录为 ~/Tz/ir2-benchmark。凭据不写入项目或脚本。

2026-11-05 实测：Ubuntu 22.04.5、H100 81559 MiB、驱动 550.163.01；ex 为 Python 3.11.0，原有 PyTorch 2.6.0+cu124、NumPy 2.0.2、cuDNN 9.1.0.70，安装前 pip check 无冲突。

此用户指令替代早期计划中的本机 WSL、Python 3.12 与 uv 环境安排。开发源码兼容 Python 3.11，服务器直接在 ex 中管理依赖，不升级原环境的 Python，也不另建替代环境。

## 安装与验证

```bash
conda activate ex
cd ~/Tz/ir2-benchmark
bash scripts/setup_server.sh
```

安装脚本先保存 artifacts/setup/ex-before.txt，以此限制现有包版本，安装 requirements-server.txt，再安装当前项目并运行 pip check、GPU doctor 和测试。版本选择优先兼容现有 CUDA 12.4/cuDNN 9.1 与 PyTorch。首次安装若解析冲突应分析明确冲突，不能去掉约束强行覆盖已有包。

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
ir2-benchmark doctor --require-gpu --output artifacts/setup/doctor.json
python -m pytest tests/benchmark -q
```

doctor 的批量累加和小型回归训练用于验证 jit/vmap/scan、Flax 和 Optax；输出明确标记 runtime_smoke_only。不得将它的耗时作为探索环境 steps/s，也不能称为 A/B 训练结果。运行时验证通过后才开始环境和模型开发验收。

## 产物和清理

2026-11-10 最新用户指令覆盖下面的历史保留约定：已删除全部实验目录、检查点、回放、GIF、日志与临时副本，服务器释放 3,269,284,100 字节；未删除地图、数据清单、源码、配置、测试或依赖环境。当前无训练任务，也没有可恢复的旧实验检查点。后续实验从新的显式运行开始。

- 历史默认约定为保留原始地图、源码、配置、环境快照、checkpoint、评测结果和失败日志；本次明确要求删除的训练产物已按上述最新指令清理。
- 只清理由本项目创建并可重建的上传压缩包、安装临时目录、pytest/编译缓存和中断的临时文件；删除前确认解析路径位于本项目内。
- 不清理 ~/Tz 之外的文件、不做全局 pip/conda cache purge、不结束其他 GPU 进程。
- 按需上传源码和地图，不复制旧 GIF、旧模型权重、.git 或无关运行日志。
- 依赖安装日志位于 artifacts/setup；安装成功后记录 ex-after.txt 与 GPU 验证结果。

所有正式实验仍按已批准设计和四个实施子计划推进。尚未执行的训练、评测和吞吐实验不能标记完成。

## 本次部署验证

- GPU doctor 使用 NVIDIA H100，通过 jit/vmap/scan 的逐值检查。
- 实际安装版本：JAX 0.4.35、jaxlib 0.4.34、Flax 0.10.2、Optax 0.2.4。
- Flax/Optax 回归烟雾训练的损失从 0.66894394 降至 0.01381547；这只是依赖链验证，不是探索策略训练。
- 两项运行时测试通过；仅 CPU 时指定 --require-gpu 按预期返回 1。
- 安装后 pip check 无冲突，PyTorch 仍为 2.6.0+cu124，CUDA 张量求和通过。
- 地图共 17299 个文件，文件路径与内容的汇总 SHA-256 为 7734ef412da9903a83e42c9b54b9aa634afb51250e1be17d36be77d5da15c458，与本地一致。
- 核心环境与 A/B 训练代码已实现，并继续进行真实训练、恢复与评估回归；具体证据见 [实现与验证状态](implementation-status.md)。完整正式训练与评测矩阵尚未完成，runtime doctor 不代表这些交付已完成。
