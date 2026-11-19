"""Stop frozen trajectories at the first overflow; backend must match argv[3]."""
import json
import pathlib
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from ir2_benchmark.agents.baselines import baseline_action, spec_from_method
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.batching import ParameterCache, prepare_task
from ir2_benchmark.evaluation.episode import load_lane
from ir2_benchmark.evaluation.scenarios import read_task_manifest

root = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
out.mkdir(exist_ok=False)
assert jax.devices()[0].platform == sys.argv[3]
selected = [task for task in read_task_manifest(root / "nf_mixed.jsonl")
            if task.scenario.source == "procedural"]
config = selected[0].scenario.config
spec = spec_from_method(selected[0].method)

def single(parameters, data, key, spawns, fault_key, isolated):
    state = reset(data, config, key, spawns)._replace(key=fault_key, isolated_mask=isolated)
    def condition(state):
        return (~jnp.any(state.graph_memory.overflow) & ~jnp.any(state.graph_memory.path_failed)
                & ~state.terminated & ~state.truncated
                & (state.event < config.max_rounds * config.n_agents))
    def advance(state):
        observation = observe(state, state.current_agent, config)
        action = baseline_action(observation, config, spec, parameters)
        return step_agent(state, action, config, observation=observation).state
    return jax.lax.while_loop(condition, advance, state)

kernel = jax.jit(jax.vmap(single))
records = []
parameters = ParameterCache(1)
for task in selected:
    started = time.time()
    lane = load_lane(prepare_task(task), parameters)
    values = jax.tree.map(lambda value: jnp.stack([value]),
                         (lane.parameters, lane.map_data, lane.policy_key, lane.spawns,
                          lane.fault_key, lane.isolated_mask))
    result = kernel(*values)
    jax.block_until_ready(result)
    state = jax.tree.map(lambda value: value[0], result)
    memory = state.graph_memory
    record = {"result_id": task.result_id, "family": task.scenario.family,
              "backend": jax.devices()[0].platform, "event_at_first_failure_or_end": int(state.event),
              "overflow": np.asarray(memory.overflow).tolist(),
              "path_failed": np.asarray(memory.path_failed).tolist(),
              "local_node_counts": np.asarray(memory.valid).sum(axis=-1).tolist(),
              "global_node_counts": np.asarray(memory.global_valid).sum(axis=-1).tolist(),
              "max_outgoing_degrees": np.asarray(memory.edges).sum(axis=-1).max(axis=-1).tolist(),
              "history_counts": np.asarray(memory.history_count).tolist(),
              "graph_capacity": config.graph_capacity,
              "graph_padding_size": config.graph_padding_size,
              "max_candidates_plus_self": config.max_candidates + 1,
              "history_capacity": int(memory.history.shape[-2]),
              "wall_seconds_including_first_compile": time.time() - started}
    records.append(record)
    (out / "diagnostic.json").write_text(json.dumps({"records": records}, indent=2) + "\n")
    print(json.dumps(record), flush=True)
