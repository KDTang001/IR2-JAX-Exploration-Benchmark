# IR2 Multi-Robot RL Exploration

## JAX benchmark installation

This is an in-development derivative benchmark, not an official IR2 release or a validated trained model. Original upstream attribution appears below. Experiment weights and recordings are not included.

The integrated benchmark baseline is commit [`98aa3206`](https://github.com/KDTang001/IR2-JAX-Exploration-Benchmark/commit/98aa3206dd28af091e245ff9963759f8279f0221) on `main`. The [v1 release record](research/RELEASE_RECORD_V1.md) documents the wheel rebuilt from that commit and its CPU installation and CLI checks on the target Linux server.

Use a fresh Python 3.11 environment:

```bash
python -m pip install -r requirement.txt
python -m pip install --no-deps -e .
ir2-benchmark doctor --runtime-only --output artifacts/setup/doctor.json
```

`requirement.txt` includes the JAX benchmark, original PyTorch/Ray reference tools and development dependencies. It records observed runtime versions where available; a clean installation on every operating system has not been validated. For Linux with a compatible NVIDIA driver, additionally install the CUDA runtime requirements:

```bash
python -m pip install -r requirements-server.txt
ir2-benchmark doctor --runtime-only --require-gpu --output artifacts/setup/doctor-gpu.json
```

The original instructions below describe the upstream environment, not the new JAX environment. Generated training outputs and caches are ignored. Raw IR2 maps are in `DungeonMaps`; map generation and audit commands are documented in `docs/benchmark-running.md`.

The published snapshot also includes map manifests and 6,900 generated maps under `artifacts/maps/`. Their generation seeds and split records are included. Existing map pools must not be overwritten: run the generation script only when rebuilding absent pools. Validation scenario paths are project-relative; run commands from the repository root. No training checkpoints, training run logs or policy recordings are included; compact acceptance evidence is retained under `research/evidence`.

The independent communication-limited benchmark is documented in [docs/benchmark-running.md](docs/benchmark-running.md). Its entry points are `doctor`, `maps`, `rollout`, `validation-manifest`, `train-a`, `train-b`, `evaluate`, `report` and `throughput`; development smoke results are separate from formal experiment completion.

Current fixed baseline evaluation uses `configs/benchmark.json` and `configs/evaluation/benchmark.json`, separate from the retained A/B research configurations. Eight configurations combine two navigation methods, nearest-frontier or upstream pretrained IR2 adapted, with off/PLAN/MAP_DELTA/mixed communication. See the [Phase 4 acceptance report](research/BASELINE_EVALUATION_REPORT.md) for target-server evidence, reference-weight provenance and limits; this does not certify full formal benchmark experiments or a data-fair retrained IR2 model.

Graph capacity is explicit: `configs/benchmark.json` retains the C400 original-capacity control; `configs/benchmark_capacity1024.json` extends graph storage and node padding to 1024 while retaining K=30 and the existing graph rules. Frozen manifests keep their embedded configurations and identities. The separate [extended-capacity acceptance](research/EXTENDED_CAPACITY_REPORT.md) passed 20 complete H100 episodes and ten exact B1/B4 pairs on five validation maps (N4, Normal, mixed, 8KiB); it does not replace the frozen C400 240-task pilot or certify maximum parallelism.

Phase 5 has passed all 78 declared full-length validation episodes and clean Linux CPU/CUDA wheel installation, including real interruption/resume. The original CPU/H100 resource window retained 52 configurations; the authorized two-case CPU supplement ended on November 17. All 54 configurations now have execution records: 37 complete measurements, 6 GPU OOMs, and 11 CPU per-case/global deadline stops. The original 52 records and closed-window results are unchanged. Timings cover initial four-event segments. See the [full-episode report](research/FULL_EPISODE_VALIDATION_REPORT.md), [performance and reproduction report](research/PERFORMANCE_REPRODUCTION_REPORT.md), [pinned wheel reproduction](docs/phase5-reproduction.md), and [resource/scope proposals](research/EXPERIMENT_RESOURCE_PLAN.md). These are limited engineering results; all configurations having records does not imply all timings or release gates passed, and formal paper experiments remain pending.

Reward, message exchange, accounting and signal choices are described in [environment presets and custom methods](docs/environment-presets.md).

The [v2 C1024 resource pilot](research/RESOURCE_PILOT_V2_C1024.md) freezes 240 validation tasks. Its separately authorized 30-task H100 probe passed scoped execution acceptance (12 coverage successes, 18 normal timeouts; 52,567 active events). Burst had zero in-window transmitted bytes under 2KiB/mixed, so this is not evidence of robustness to exposed message loss. The paper and train-only SAC specifications use C1024 as a candidate pending broader validation. The original C400 protocol remains available.

Current accepted scope and remaining work are tracked in the [implementation roadmap](research/IMPLEMENTATION_ROADMAP.md). The earlier [method discussion](spec/spec-design-benchmark-status-and-discussion.md) is a historical design snapshot. On 2026-11-10, all experiment checkpoints, recordings and run outputs were removed at the user's request. The bundled reference weight was also removed; original inference scripts require a separately supplied checkpoint. The benchmark acceptance workflow has not resumed training.

[IROS 2024] Public code and model for [IR2: Implicit Rendezvous for Robotic Exploration Teams under Sparse Intermittent Connectivity](https://arxiv.org/abs/2409.04730).

Public Website: https://ir2-explore.github.io/

We present IR2, a deep reinforcement learning approach to information sharing for multi-robot exporation under communication constraints. Leveraging attention-based neural networks and hierarchical graph formulation, robots can effectively balance the longer-term trade-offs between disconnecting for solo exploration and reconnecting for information sharing. 

We show that our model trained in this python simulation can generalize well to complex large-scale environments, such as those in Gazebo simulation and real world experiments. We hope that our work will be a useful learning-based benchmark for future research. 


## GIF Demo

This demonstration showcases 4 robots exploring in an unknown `Complex` map under line-of-sight signal strength communication constraints. The top gif illustrates the global map and robot positions assuming no communication constraints. Conversely, the bottom 4 gifs illustrates the individual robots' map and position beliefs subjected to communication constraints. 

View the original demonstration [on GitHub](https://github.com/marmotlab/IR2-Multi-Robot-RL-Exploration/blob/master/gifs/combined_gif_demo.gif). The local GIF copy was removed during cleanup.





## Instructions


#### Dependencies
This repository was tested using the following dependencies. Newer version of these packages may work as well. 
* `python == 3.8`
* `pytorch == 1.10.0`
* `ray == 1.10.0`
* `scikit-image == 0.19.3`
* `scikit-learn == 1.2.1`
* `scipy == 1.10.0`
* `matplotlib == 3.6.3`
* `tensorboard == 2.8.0`


#### Training
1. Set training parameters in `parameters.py`.
2. Run python `driver.py`.


#### Evaluation
1. Set inference parameters in `test_parameters.py`.
2. Run `test_driver.py`.



## Code Structure

* `parameter.py` Training parameters.
* `driver.py` Driver of training program, maintain & update the global network.
* `runner.py` Wrapper of the local network.
* `multi_robot_worker.py` Interact with environment and collect episode experience.
* `model.py` Define attention-based network.
* `env.py` Autonomous exploration environment.
* `graph_generator.py` Generate and update the collision-free graph.
* `graph.py` Graph definition and utilities.
* `node.py` Initialize and update nodes in the coliision-free graph.
* `sensor.py` Simulate the sensor model of Lidar.
* `robot.py` Acts as a replay buffer.
* `ss_realistic_model.py` Realistic signal strength communication model.
* `/model` Trained model.
* `/DungeonMaps` Maps of training environments.



## References
If you intend to use our work in your research, please cite the following publication:
```bibtex
@inproceedings{tan2024ir,
  title={Ir 2: Implicit rendezvous for robotic exploration teams under sparse intermittent connectivity},
  author={Tan, Derek Ming Siang and Ma, Yixiao and Liang, Jingsong and Chng, Yi Cheng and Cao, Yuhong and Sartoretti, Guillaume},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={13245--13252},
  year={2024},
  organization={IEEE}
}
```


## Authors
[Derek Ming Siang Tan](https://github.com/derektan95)\
[Yixiao Ma](https://yixiao-m.github.io)\
[Jingsong Liang](https://github.com/JingsongLiang)\
[Yi Cheng Chng](https://github.com/chngyicheng)\
[Yuhong Cao](https://github.com/caoyuhong001)\
[Guillaume Sartoretti](https://github.com/gsartoretti)
