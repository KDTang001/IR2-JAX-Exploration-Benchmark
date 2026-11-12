"""Route-B settings, public-state replay adapters, and native compilation pool state."""

from dataclasses import dataclass
from typing import NamedTuple, TypeAlias, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from jax import Array

from ir2_benchmark.agents.actor import Actor, IndexAction, initial_memory
from ir2_benchmark.agents.heuristic import nearest_frontier_action
from ir2_benchmark.communication.channel import Inbox
from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.observe import observe
from ir2_benchmark.models.encoders import ActorFeatures, actor_features, selected_plan_features
from ir2_benchmark.models.plan_predictor import (
    PlanInput,
    PlanMemory,
    PlanPredictor,
    initial_plan_memory,
    plan_inputs,
)
from ir2_benchmark.models.world_model import (
    ImaginedCritic,
    RSSMState,
    WorldModel,
    WorldPrediction,
    initial_rssm,
    latent_features,
)
from ir2_benchmark.navigation.ir2_graph import GraphMemory, GraphObservation
from ir2_benchmark.training.dreamer import ImaginedBatch
from ir2_benchmark.training.imagination import (
    CognitiveState,
    PolicyCarry,
    PolicyDecision,
    cognitive_from_environment,
    encode_world_action,
)
from ir2_benchmark.training.plan_losses import PlanTarget, plan_loss, target_from_labels
from ir2_benchmark.training.replay import (
    Fields,
    ReplayBuffer,
    ReplayError,
    ReplayFrame,
    ReplaySequence,
    read_field,
)
from ir2_benchmark.types import EnvState, LocalObservation, PlanLabel, plan_labels

TensorRecord = TypeVar(
    "TensorRecord", PlanInput, PlanMemory, PlanTarget, ActorFeatures, IndexAction
)
RecordedTensors: TypeAlias = (
    PlanInput | PlanMemory | PlanTarget | ActorFeatures | IndexAction | PlanLabel
)


@dataclass(frozen=True, slots=True)
class BTrainingConfig:
    """Formal defaults; short development runs record their explicit reduced batch/warmup choices."""  # noqa: E501

    learned_communication: bool = True
    predicted_plans: bool = True
    use_neighbor_plan: bool = True
    real_updates: bool = False
    hidden: int = 256
    candidate_attention: bool = True
    categories: int = 16
    classes: int = 16
    sequence_length: int = 32
    burn_in: int = 8
    logical_batch: int = 32
    physical_env_count: int = 1
    update_every: int = 128
    warmup_events: int = 100000
    imagination_rounds: int = 5
    learning_rate: float = 3e-4
    snapshot_interval: int = 128
    replay_capacity: int = 4096
    entropy: float = 0.001
    development: bool = False
    validation_interval: int = 100000
    unchanged_sample_rate: float = 0.1

    def __post_init__(self) -> None:
        """Validate and freeze this value at its input boundary."""
        if (
            min(
                self.hidden,
                self.categories,
                self.classes,
                self.logical_batch,
                self.physical_env_count,
                self.update_every,
                self.imagination_rounds,
                self.snapshot_interval,
                self.replay_capacity,
                self.validation_interval,
            )
            < 1
        ):
            raise ReplayError("B_dimensions_and_capacities_must_be_positive")  # noqa: EM101
        if (
            not 0 <= self.burn_in < self.sequence_length
            or self.warmup_events < 0
            or self.learning_rate <= 0
            or not 0 <= self.unchanged_sample_rate <= 1
        ):
            raise ReplayError("invalid_B_sequence_warmup_or_learning_rate")  # noqa: EM101


class Counters(NamedTuple):
    """Keep actual environment events separate from modeled experience and optimizer calls."""

    real_events: int = 0
    imagined_events: int = 0
    model_updates: int = 0
    actor_updates: int = 0
    predictor_updates: int = 0


@dataclass(frozen=True, slots=True)
class TrainingPool:
    """Independent native-group lanes share bounded host replay and model weights."""

    config: HostConfig
    state: EnvState
    carry: PolicyCarry
    latent: RSSMState
    replay: ReplayBuffer
    episode: int
    sample_indices: tuple[int, ...]
    lane_episodes: tuple[int, ...]
    events: int = 0
    lane_cursor: int = 0


def _fields(cognition: CognitiveState) -> Fields:
    """Flatten small cognitive tensors for numeric host replay storage."""
    fields = []
    for name in CognitiveState._fields:
        if name not in ("local_maps", "observed_event", "inbox", "graph_memory", "peer_maps"):
            fields.append((name, np.asarray(getattr(cognition, name))))  # noqa: PERF401
    fields.extend(
        (f"inbox_{name}", np.asarray(getattr(cognition.inbox, name))) for name in Inbox._fields
    )
    fields.append(("graph_memory_present", np.asarray(cognition.graph_memory is not None)))
    if cognition.graph_memory is not None:
        for name in GraphMemory._fields:
            value = np.asarray(getattr(cognition.graph_memory, name))
            if name == "observable":
                fields.extend(
                    (
                        ("graph_memory_observable_shape", np.asarray(value.shape, dtype=np.int32)),
                        ("graph_memory_observable_indices", np.flatnonzero(value).astype(np.int32)),
                    )
                )
            else:
                fields.append((f"graph_memory_{name}", value))
    fields.append(("peer_maps_present", np.asarray(cognition.peer_maps is not None)))
    if cognition.peer_maps is not None:
        peer_maps = np.asarray(cognition.peer_maps)
        indices = np.flatnonzero(peer_maps != 2).astype(np.int32)  # noqa: PLR2004 - Unknown map cell.
        fields.extend(
            (
                ("peer_maps_shape", np.asarray(peer_maps.shape, dtype=np.int32)),
                ("peer_maps_indices", indices),
                ("peer_maps_values", peer_maps.reshape(-1)[indices]),
            )
        )
    return tuple(fields)


def _frame(state: EnvState, extras: Fields = ()) -> ReplayFrame:
    """Transfer one current map state to host; persistent replay retains only periodic snapshots/deltas."""  # noqa: E501
    cognition = cognitive_from_environment(state)
    return ReplayFrame(
        np.asarray(cognition.local_maps),
        np.asarray(cognition.observed_event),
        _fields(cognition) + extras,
    )


def _cognition(frame: ReplayFrame) -> CognitiveState:
    """Restore only the public modeled state from a replay cutoff."""
    inbox = Inbox(*(jnp.asarray(frame.field(f"inbox_{name}")) for name in Inbox._fields))
    grids = {
        "local_maps": jnp.asarray(frame.maps),
        "observed_event": jnp.asarray(frame.observed_event),
        "inbox": inbox,
        "graph_memory": None,
        "peer_maps": None,
    }
    fields = dict(frame.fields)
    if bool(fields.get("graph_memory_present", False)):
        observable = np.zeros(tuple(frame.field("graph_memory_observable_shape")), dtype=bool)
        observable.reshape(-1)[frame.field("graph_memory_observable_indices")] = True
        grids["graph_memory"] = GraphMemory(
            *(
                jnp.asarray(
                    observable if name == "observable" else frame.field(f"graph_memory_{name}")
                )
                for name in GraphMemory._fields
            )
        )
    if bool(fields.get("peer_maps_present", False)):
        peer_maps = np.full(tuple(frame.field("peer_maps_shape")), 2, dtype=np.uint8)
        peer_maps.reshape(-1)[frame.field("peer_maps_indices")] = frame.field("peer_maps_values")
        grids["peer_maps"] = jnp.asarray(peer_maps)
    return CognitiveState(
        *(
            grids[name] if name in grids else jnp.asarray(frame.field(name))
            for name in CognitiveState._fields
        )
    )


def _named_fields(prefix: str, value: RecordedTensors) -> Fields:
    """Store a small named PyTree's leaves with stable field names."""
    fields = []
    for name in value._fields:
        tensor = getattr(value, name)
        if name == "graph" and isinstance(value, ActorFeatures):
            fields.append((f"{prefix}_graph_present", np.asarray(tensor is not None)))
            if tensor is not None:
                fields.extend(
                    (f"{prefix}_graph_{key}", np.asarray(getattr(tensor, key)))
                    for key in GraphObservation._fields
                )
        else:
            fields.append((f"{prefix}_{name}", np.asarray(tensor)))
    return tuple(fields)


def _initialize(
    config: HostConfig, options: BTrainingConfig, state: EnvState, key: Array
) -> tuple[TrainState, TrainState, TrainState, TrainState, PolicyCarry, RSSMState]:
    """Build all train states from the same locally legal schema used during collection."""
    from ir2_benchmark.training.world_losses import _world_observation  # noqa: PLC0415

    # Match route A's shared actor/predictor initialization streams exactly.
    world_key, actor_key, critic_key, predictor_key = jax.random.split(key, 4)
    observation = observe(state, state.current_agent, config)
    actor_model = Actor(hidden=options.hidden, candidate_attention=options.candidate_attention)
    predictor_model = PlanPredictor(
        config.plan_horizon, options.hidden, options.categories, options.classes
    )
    plan_memory = initial_plan_memory(
        (config.n_slots, config.n_slots), options.hidden, options.categories, options.classes
    )
    receiver = jax.tree_util.tree_map(lambda value: value[0], plan_memory)
    predictor_variables = predictor_model.init(
        predictor_key, receiver, plan_inputs(observation), predictor_key
    )
    features = actor_features(
        observation,
        config.n_agents,
        initial_budget=config.initial_budget,
        max_rounds=config.max_rounds,
        patch_radius=int(config.sensor_radius),
        use_neighbor_plan=options.use_neighbor_plan,
        candidate_attention=options.candidate_attention,
    )
    plan = selected_plan_features(
        observation, observation.position, config.max_distance, config.plan_horizon
    )
    actor_variables = actor_model.init(
        actor_key, features, initial_memory(hidden=options.hidden), plan
    )
    radius = int(np.ceil(config.sensor_radius))
    world_model = WorldModel(
        config.n_slots,
        2 * radius + 1,
        config.max_distance,
        options.hidden,
        options.categories,
        options.classes,
    )
    latent = initial_rssm(
        deterministic=options.hidden, categories=options.categories, classes=options.classes
    )
    action = nearest_frontier_action(observation, config)
    world_variables = world_model.init(
        world_key,
        latent,
        encode_world_action(observation, action, config),
        _world_observation(state, state.current_agent, radius, config),
        world_key,
    )
    critic_model = ImaginedCritic(options.hidden)
    critic_variables = critic_model.init(critic_key, latent_features(latent))
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adam(options.learning_rate))
    actor = TrainState.create(
        apply_fn=actor_model.apply, params=actor_variables["params"], tx=optimizer
    )
    predictor = TrainState.create(
        apply_fn=predictor_model.apply, params=predictor_variables["params"], tx=optimizer
    )
    world = TrainState.create(
        apply_fn=world_model.apply, params=world_variables["params"], tx=optimizer
    )
    critic = TrainState.create(
        apply_fn=critic_model.apply, params=critic_variables["params"], tx=optimizer
    )
    return (
        actor,
        predictor,
        world,
        critic,
        PolicyCarry(initial_memory((config.n_slots,), options.hidden), plan_memory),
        latent,
    )


def _decision_extras(
    decision: PolicyDecision, observation: LocalObservation, carry: PolicyCarry, state: EnvState
) -> Fields:
    """Labels remain separate from the receiver-visible model forward inputs."""
    receiver = observation.agent_id
    plan_memory = jax.tree_util.tree_map(lambda value: value[receiver], carry.plan_memory)
    target = target_from_labels(
        plan_labels(state),
        state.valid_shape,
        state.active_mask & (jnp.arange(state.active_mask.shape[0]) != receiver),
    )
    return (
        _named_fields("actor", decision.features)
        + _named_fields("index", decision.index_action)
        + _named_fields("plan_input", plan_inputs(observation))
        + _named_fields("plan_memory", plan_memory)
        + _named_fields("plan_target", target)
        + (
            ("actor_old_memory", np.asarray(decision.old_memory)),
            ("actor_plan", np.asarray(decision.plan_features)),
            ("all_actor_memory", np.asarray(carry.actor_memory)),
        )
        + _named_fields("all_plan_memory", carry.plan_memory)
    )


def _restore_tensors(
    kind: type[TensorRecord], frames: tuple[Fields, ...], prefix: str
) -> TensorRecord:
    """Restore one typed compact record from numeric host replay metadata."""

    def stack(name: str) -> Array:
        return jnp.asarray(np.stack([read_field(frame, f"{prefix}_{name}") for frame in frames]))

    values = []
    for name in kind._fields:
        if kind is ActorFeatures and name == "graph":
            present = [bool(dict(frame).get(f"{prefix}_graph_present", False)) for frame in frames]
            if any(present) and not all(present):
                raise ReplayError("mixed_graph_and_legacy_actor_features")  # noqa: EM101
            values.append(
                GraphObservation(*(stack(f"graph_{key}") for key in GraphObservation._fields))
                if all(present)
                else None
            )
        else:
            values.append(stack(name))
    return kind(*values)


class PlanTrainingBatch(NamedTuple):
    """Compact predictor tensors prepared on host; replay maps never enter jit arguments."""

    inputs: PlanInput
    memory: PlanMemory
    target: PlanTarget


def plan_training_batch(frames: tuple[Fields, ...]) -> PlanTrainingBatch:
    """Extract the small logged local inputs and separate intention labels before device tracing."""
    return PlanTrainingBatch(
        _restore_tensors(PlanInput, frames, "plan_input"),
        _restore_tensors(PlanMemory, frames, "plan_memory"),
        _restore_tensors(PlanTarget, frames, "plan_target"),
    )


def _update_predictor(
    predictor: TrainState, batch: PlanTrainingBatch, key: Array
) -> tuple[TrainState, Array]:
    """Train on current logged intention labels; no target enters plan_inputs or actor_features."""

    def objective(params: FrozenDict) -> Array:
        prediction = predictor.apply_fn({"params": params}, batch.memory, batch.inputs, key)
        return plan_loss(prediction, batch.target).total

    loss, gradient = jax.value_and_grad(objective)(predictor.params)
    return predictor.apply_gradients(grads=gradient), loss


def _real_policy_batch(
    frames: tuple[Fields, ...], sequence: ReplaySequence, predictions: WorldPrediction
) -> ImaginedBatch:
    """Real-update ablation uses the same score-function optimizer with actual replay rewards."""
    valid_indices = np.flatnonzero(
        sequence.loss_mask[: len(frames)]
        & np.array([bool(read_field(frame, "policy_valid")) for frame in frames])
    )
    if not len(valid_indices):
        raise ReplayError("real_update_sequence_has_no_postwarmup_policy_events")  # noqa: EM101
    offset = int(valid_indices[0])
    count = len(frames)
    frames = frames[offset:]

    features = _restore_tensors(ActorFeatures, frames, "actor")
    actions = _restore_tensors(IndexAction, frames, "index")
    first = initial_rssm(
        deterministic=predictions.state.deterministic.shape[-1],
        categories=predictions.state.stochastic.shape[-2],
        classes=predictions.state.stochastic.shape[-1],
    )
    all_central = jnp.concatenate(
        (latent_features(first)[None], latent_features(predictions.state)[: count - 1]), 0
    )
    central_features = all_central[offset:count]
    rewards = jnp.asarray(
        [entry.reward for entry in sequence.transitions[offset:] if entry is not None]
    )
    continuation = jnp.asarray(
        [
            entry.continuation * (not entry.truncated)
            for entry in sequence.transitions[offset:]
            if entry is not None
        ]
    )
    return ImaginedBatch(
        features,
        jnp.asarray(np.stack([read_field(frame, "actor_old_memory") for frame in frames])),
        jnp.asarray(np.stack([read_field(frame, "actor_plan") for frame in frames])),
        actions,
        central_features,
        latent_features(jax.tree_util.tree_map(lambda value: value[count - 1], predictions.state)),
        rewards,
        continuation,
        jnp.ones(len(frames), bool),
    )
