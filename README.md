# IR2 Multi-Robot RL Exploration

## JAX benchmark installation

This is an in-development derivative benchmark, not an official IR2 release or a validated trained model. Original upstream attribution appears below. Experiment weights and recordings are not included.

Use a fresh Python 3.11 environment:

```bash
python -m pip install -r requirement.txt
python -m pip install --no-deps -e .
ir2-benchmark doctor --output artifacts/setup/doctor.json
```

`requirement.txt` includes the JAX benchmark, original PyTorch/Ray reference tools and development dependencies. It records observed runtime versions where available; a clean installation on every operating system has not been validated. For Linux with a compatible NVIDIA driver, additionally install the CUDA runtime requirements:

```bash
python -m pip install -r requirements-server.txt
ir2-benchmark doctor --require-gpu --output artifacts/setup/doctor-gpu.json
```

The original instructions below describe the upstream environment, not the new JAX environment. Generated training outputs and caches are ignored. Raw IR2 maps are in `DungeonMaps`; map generation and audit commands are documented in `docs/benchmark-running.md`.

The published snapshot also includes map manifests and 6,900 generated maps under `artifacts/maps/`. Their generation seeds and split records are included. Existing map pools must not be overwritten: run the generation script only when rebuilding absent pools. Validation scenario paths are project-relative; run commands from the repository root. No training checkpoints, run logs or recordings are included.

The independent communication-limited benchmark is documented in [docs/benchmark-running.md](docs/benchmark-running.md). Its entry points are `doctor`, `maps`, `rollout`, `validation-manifest`, `train-a`, `train-b`, `evaluate`, `report` and `throughput`; development smoke results are separate from formal experiment completion.

Current fixed baseline evaluation uses `configs/benchmark.json` and `configs/evaluation/benchmark.json`, separate from the retained A/B research configurations. Eight methods combine nearest-frontier or upstream pretrained IR2 adapted navigation with off/PLAN/MAP_DELTA/mixed communication. See the [Phase 4 acceptance report](research/BASELINE_EVALUATION_REPORT.md) for target-server evidence, reference-weight provenance and limits; this does not certify full formal benchmark experiments or a data-fair retrained IR2 model.

Reward, message exchange, accounting and signal choices are described in [environment presets and custom methods](docs/environment-presets.md).

Current goals, implementation limits and proposed revisions are summarized in the [discussion document](spec/spec-design-benchmark-status-and-discussion.md). On 2026-11-10, all experiment checkpoints, recordings and run outputs were removed at the user's request. The bundled reference weight was also removed; original inference scripts require a separately supplied checkpoint. No training is running.

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
