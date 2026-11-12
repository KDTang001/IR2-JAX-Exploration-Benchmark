"""B ablation semantics and checkpoint completeness."""

import json
import shutil
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import serialization
from PIL import Image

from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.communication.faults import KeyPurpose, fold_key
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.env.reset import reset
from ir2_benchmark.env.step import step_agent
from ir2_benchmark.evaluation.scenarios import EvaluationMap, make_scenario, write_scenario_manifest
from ir2_benchmark.evaluation.validation import (
    ValidationEvaluator,
    ValidationScore,
    build_validation_callback,
)
from ir2_benchmark.maps.manifest import obstacle_hash, scenario_hash
from ir2_benchmark.training.b_state import (
    _decision_extras,
    _frame,
    _initialize,
    _update_predictor,
    plan_training_batch,
)
from ir2_benchmark.training.checkpoint import load_checkpoint, save_checkpoint
from ir2_benchmark.training.evaluate_b import run_evaluation
from ir2_benchmark.training.imagination import (
    ImaginedState,
    cognitive_from_environment,
    imagine_rollout,
    policy_decision,
)
from ir2_benchmark.training.replay import ReplayBuffer
from ir2_benchmark.training.train_b import BTrainingConfig, train_b
from ir2_benchmark.types import MapInput


def test_b_zoh_uses_dynamics_when_neighbor_prediction_disabled() -> None:
    # Given: a true event cutoff and ZOH actor inputs with no received neighbor intentions.
    config = HostConfig(sensor_radius=1, max_candidates=4, max_distance=2)
    grid = jnp.zeros((9, 9), jnp.uint8)
    state = reset(
        MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([4, 4])),
        config,
        jax.random.PRNGKey(7),
    )
    settings = BTrainingConfig(
        hidden=8, categories=2, classes=3, predicted_plans=False, development=True
    )
    actor, predictor, world, _, carry, latent = _initialize(
        config, settings, state, jax.random.PRNGKey(8)
    )
    initial = ImaginedState(cognitive_from_environment(state), latent, carry.actor_memory)
    # When: generating a modeled event with the predictor excluded from actor features.
    final, _, batch = imagine_rollout(
        initial,
        carry,
        actor,
        predictor,
        world,
        config,
        jax.random.PRNGKey(9),
        1,
        predicted_plans=False,
    )
    # Then: cold-start ZOH plans stay missing while the event world model actually advances.
    np.testing.assert_array_equal(batch.features.neighbor_features[0, :, :2], np.zeros((2, 2)))
    assert int(batch.valid.sum()) == 1
    assert not np.array_equal(final.latent.deterministic, initial.latent.deterministic)


def test_predictor_update_traces_compact_batch_when_replay_frames_remain_on_host() -> None:
    # Given: an actual real event establishes an intention label and a host replay cutoff.
    config = HostConfig(sensor_radius=1, max_candidates=4, max_distance=2)
    grid = jnp.zeros((9, 9), jnp.uint8)
    state = reset(
        MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([4, 4])),
        config,
        jax.random.PRNGKey(3),
    )
    observation = observe(state, state.current_agent, config)
    state = step_agent(
        state, nearest_frontier_action(observation, config), config, observation=observation
    ).state
    settings = BTrainingConfig(hidden=8, categories=2, classes=3, development=True)
    actor, predictor, _, _, carry, _ = _initialize(config, settings, state, jax.random.PRNGKey(4))
    observation = observe(state, state.current_agent, config)
    decision = policy_decision(actor, predictor, observation, carry, config, jax.random.PRNGKey(5))
    frame = _frame(state, _decision_extras(decision, observation, carry, state))
    batch = plan_training_batch((frame.fields,))
    # When: tracing a predictor update, only tensor inputs/memory/labels cross the jit boundary.
    updated, loss = jax.jit(_update_predictor)(predictor, batch, jax.random.PRNGKey(6))
    # Then: this real path executes, with no ReplayFrame or full map-history argument.
    assert int(updated.step) == 1
    assert np.isfinite(float(loss))
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(batch))


@pytest.mark.parametrize("legacy_checkpoint", [False, True])
def test_validation_selects_real_checkpoint_and_resumes_without_repeating_milestone(
    tmp_path: Path,
    legacy_checkpoint: bool,  # noqa: FBT001
) -> None:
    # Given: a frozen, actual validation scenario and an eight-event diagnostic training budget.
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=8)
    pixels = np.full((9, 9), 195, np.uint8)
    pixels[4, 4] = 208
    map_path = tmp_path / "validation.png"
    Image.fromarray(pixels).save(map_path)
    validation_map = EvaluationMap(
        str(map_path),
        obstacle_hash(np.zeros((9, 9), np.uint8)),
        "validation",
        "procedural",
        "rooms",
        64,
        scenario_hash(np.zeros((9, 9), np.uint8), (4, 4)),
    )
    manifest = tmp_path / "validation.jsonl"
    write_scenario_manifest(manifest, (make_scenario(validation_map, config, 31),))
    callback = build_validation_callback(manifest, tmp_path / "selection", training_seed=17)
    settings = BTrainingConfig(
        hidden=8,
        categories=2,
        classes=3,
        sequence_length=4,
        burn_in=1,
        logical_batch=2,
        update_every=8,
        warmup_events=0,
        validation_interval=8,
        development=True,
    )
    output = tmp_path / "training"
    # When: training evaluates a real checkpoint at the milestone, then restores that completed run.
    first = train_b(config, 17, 8, output, options=settings, validation=callback)
    if legacy_checkpoint:
        payload = serialization.msgpack_restore(
            load_checkpoint(output / "checkpoint", first["config_hash"])
        )
        del payload["update_pending"]
        save_checkpoint(
            output / "checkpoint", serialization.to_bytes(payload), first["config_hash"]
        )
    resumed = train_b(config, 17, 8, output, options=settings, validation=callback, resume=output)
    # Then: best is a real usable inference artifact, and selection is not repeated after recovery.
    records = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["evaluated"] == 1
    assert records[0]["real_events"] == 8  # noqa: PLR2004
    assert records[0]["selected"]
    assert first["best_validation_auc"] == resumed["best_validation_auc"]
    assert first["model_updates"] == resumed["model_updates"]
    assert first["actor_updates"] == resumed["actor_updates"]
    assert first["imagined_events"] >= 2  # noqa: PLR2004
    assert first["actor_parameter_delta"] > 0
    updates = [json.loads(line) for line in (output / "updates.jsonl").read_text().splitlines()]
    assert updates[0]["policy_trajectories"] == 2  # noqa: PLR2004
    assert first["last_validation_event"] == resumed["last_validation_event"] == 8  # noqa: PLR2004
    assert (output / "best" / "checkpoint").is_file()
    assert json.loads((output / "best" / "summary.json").read_text())["real_events"] == 8  # noqa: PLR2004


@pytest.mark.parametrize("cutoff", [8, 16])
def test_validation_checkpoint_resumes_pending_update_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cutoff: int
) -> None:
    # Given: real validation runs before the update at both intermediate and final boundaries.
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=8)
    pixels = np.full((9, 9), 195, np.uint8)
    pixels[4, 4] = 208
    map_path = tmp_path / "validation.png"
    Image.fromarray(pixels).save(map_path)
    validation_map = EvaluationMap(
        str(map_path), obstacle_hash(np.zeros((9, 9), np.uint8)), "validation",
        "procedural", "rooms", 64, scenario_hash(np.zeros((9, 9), np.uint8), (4, 4)),
    )
    manifest = tmp_path / "validation.jsonl"
    write_scenario_manifest(manifest, (make_scenario(validation_map, config, 31),))
    settings = BTrainingConfig(
        hidden=8, categories=2, classes=3, sequence_length=4, burn_in=1,
        logical_batch=2, update_every=8, warmup_events=0, validation_interval=8,
        development=True,
    )
    full, resumed = tmp_path / "full", tmp_path / "resumed"
    original_validate = ValidationEvaluator.__call__
    original_save = ReplayBuffer.save
    saved_paths: list[Path] = []

    def count_replay_save(buffer: ReplayBuffer, path: str | Path) -> None:
        original_save(buffer, path)
        saved_paths.append(Path(path))

    def capture_then_validate(
        evaluator: ValidationEvaluator, checkpoint: Path, real_events: int
    ) -> ValidationScore:
        if checkpoint.parent == full and real_events == cutoff:
            identity = sha256(
                json.dumps(json.loads((full / "config.json").read_text()), sort_keys=True).encode()
            ).hexdigest()
            before_update = serialization.msgpack_restore(load_checkpoint(checkpoint, identity))
            assert int(before_update["actor"]["step"]) == cutoff // 8 - 1
            assert int(before_update["world"]["step"]) == cutoff // 8 - 1
            shutil.copytree(full, resumed)
        return original_validate(evaluator, checkpoint, real_events)

    monkeypatch.setattr(ReplayBuffer, "save", count_replay_save)
    monkeypatch.setattr(ValidationEvaluator, "__call__", capture_then_validate)
    expected = train_b(
        config, 17, 16, full, options=settings,
        validation=build_validation_callback(
            manifest, tmp_path / "full-validation", training_seed=17
        ),
    )
    # When: recovering the complete pre-validation snapshot under the same seed and total quota.
    actual = train_b(
        config, 17, 16, resumed, options=settings, resume=resumed,
        validation=build_validation_callback(
            manifest, tmp_path / "resumed-validation", training_seed=17
        ),
    )
    # Then: the pending update happens once, including when no real-event quota remains.
    for directory in (full, resumed):
        updates = [
            json.loads(line) for line in (directory / "updates.jsonl").read_text().splitlines()
        ]
        assert [row["real_events"] for row in updates] == [8, 16]
    assert actual["real_events"] == expected["real_events"] == 16  # noqa: PLR2004
    assert actual["session_real_events"] == 16 - cutoff
    assert actual["model_updates"] == expected["model_updates"] == 2  # noqa: PLR2004
    assert actual["actor_updates"] == expected["actor_updates"] == 2  # noqa: PLR2004
    states = [
        serialization.msgpack_restore(load_checkpoint(path / "checkpoint", expected["config_hash"]))
        for path in (full, resumed)
    ]
    for observed, reference in zip(
        jax.tree_util.tree_leaves(states[1]), jax.tree_util.tree_leaves(states[0]), strict=True
    ):
        observed_array, reference_array = np.asarray(observed), np.asarray(reference)
        assert observed_array.shape == reference_array.shape
        assert observed_array.dtype == reference_array.dtype
        if np.issubdtype(reference_array.dtype, np.floating):
            np.testing.assert_allclose(observed_array, reference_array, rtol=1e-6, atol=1e-7)
        else:
            np.testing.assert_array_equal(observed_array, reference_array)
    replay = ReplayBuffer.load(resumed / "replay.npz")
    reference_replay = ReplayBuffer.load(full / "replay.npz")
    assert replay.size == reference_replay.size == 16  # noqa: PLR2004
    assert replay.sampling_state == reference_replay.sampling_state
    assert saved_paths.count(full / "replay.npz") == 3  # noqa: PLR2004
    assert saved_paths.count(resumed / "replay.npz") == (3 if cutoff == 8 else 2)  # noqa: PLR2004


def test_b_evaluation_samples_policy_when_uniform_actor_would_greedily_wait(tmp_path: Path) -> None:
    # Given: a zero-weight actor has five exactly equal legal waypoint logits.
    config = HostConfig(sensor_radius=1, max_distance=2, max_candidates=4, max_rounds=1)
    grid = jnp.zeros((9, 9), jnp.uint8)
    data = MapInput(grid, grid == 0, jnp.array([9, 9]), jnp.array([4, 4]))
    state = reset(data, config, jax.random.PRNGKey(1))
    settings = BTrainingConfig(hidden=8, categories=2, classes=3, development=True)
    actor, predictor, _, _, _, _ = _initialize(config, settings, state, jax.random.PRNGKey(2))
    configuration = {"environment": asdict(config), "training": asdict(settings)}
    (tmp_path / "config.json").write_text(json.dumps(configuration))
    identity = sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(
        checkpoint,
        serialization.to_bytes(
            {
                "actor": {"params": jax.tree_util.tree_map(jnp.zeros_like, actor.params)},
                "predictor": {"params": predictor.params},
            }
        ),
        identity,
    )
    keys = jax.random.split(jax.random.PRNGKey(18), 16)
    choices = jax.vmap(
        lambda key: jax.random.categorical(
            jax.random.split(fold_key(key, KeyPurpose.POLICY), 3)[1], jnp.zeros(5)
        )
    )(keys)
    key = keys[jnp.argmax(choices != 0)]
    spawns = jnp.array([[4, 4], [6, 6]])
    # When: a fixed policy stream independently known to sample a non-wait action is evaluated.
    result = run_evaluation(
        checkpoint, data, config, settings, key, spawns=spawns, fault_key=jax.random.PRNGKey(70)
    )
    # Then: a greedy override would always wait and fail this actual action check.
    assert not np.array_equal(result.trace.goals[0, 0], spawns[0])
