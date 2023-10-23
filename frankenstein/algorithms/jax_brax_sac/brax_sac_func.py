import argparse
import functools
import time
from typing import TYPE_CHECKING, Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from brax import (
    base,  # brax primitives
    envs,  # brax environment
)
from brax.v1 import envs as envs_v1  # mujoco & basis environments
from tensorboardX import SummaryWriter
from tqdm import tqdm

from frankenstein.algorithms.jax_brax_sac.acting_in_env import actor_step
from frankenstein.algorithms.jax_brax_sac.evaluate import Evaluator
from frankenstein.algorithms.jax_brax_sac.losses_and_grad import gradient_update_fn, make_losses
from frankenstein.algorithms.jax_brax_sac.networks import Params, SACNetworks, make_inference_fn, make_sac_networks
from frankenstein.algorithms.jax_brax_sac.replay_buffers import UniformSamplingQueue
from frankenstein.algorithms.jax_brax_sac.running_statistics import (
    RunningStatisticsState,
    init_state,
    normalize,
    update,
)

# Local modules
from frankenstein.algorithms.jax_brax_sac.utils import (
    PMAP_AXIS_NAME,
    Array,
    Metrics,
    TrainingState,
    Transition,
    assert_is_replicated,
    handle_devices,
    save_params,
    synchronize_hosts,
    unpmap,
)


if TYPE_CHECKING:
    from jax.random import PRNGKey

    from frankenstein.algorithms.jax_brax_sac.replay_buffers import ReplayBufferState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="halfcheetah")
    parser.add_argument("--backend", type=str, default="generalized")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--episode_length", type=int, default=1_000)
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--num_evals", type=int, default=20)
    parser.add_argument("--deterministic_eval", type=bool, default=False)
    parser.add_argument("--action_repeat", type=int, default=1)
    parser.add_argument("--reward_scaling", type=int, default=1)
    parser.add_argument("--normalize_observations", type=bool, default=True)
    parser.add_argument("--discount_factor", type=float, default=0.99)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--grad_updates_per_step", type=int, default=128)
    parser.add_argument("--max_devices_per_host", type=int, default=1)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_start", type=int, default=25000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor_layers", type=Sequence[int], default=(256, 256))
    parser.add_argument("--critic_layers", type=Sequence[int], default=(256, 256))
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)  # If ent coef fixed - NOT USED
    parser.add_argument("--saved_model_freq", type=int, default=4)  # - NOT USED
    args = parser.parse_args()

    return args


# TO REMOVE OTHER PLACE
def init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
) -> TrainingState:
    """Inits the training state and replicates it over devices."""
    key_policy, key_q = jax.random.split(key)
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_optimizer_state = alpha_optimizer.init(log_alpha)

    policy_params = sac_network.policy_network.init(key_policy)
    policy_optimizer_state = policy_optimizer.init(policy_params)
    q_params = sac_network.q_network.init(key_q)
    q_optimizer_state = q_optimizer.init(q_params)

    normalizer_params = init_state(Array((obs_size,), jnp.dtype("float32")))

    training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=q_params,
        gradient_steps=jnp.zeros(()),
        env_steps=jnp.zeros(()),
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=log_alpha,
        normalizer_params=normalizer_params,
    )
    return jax.device_put_replicated(training_state, jax.local_devices()[:local_devices_to_use])


def train(
    environment: Union[envs_v1.Env, envs.Env],
    args,
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    checkpoint_logdir: Optional[str] = None,
    randomization_fn: Optional[Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]] = None,
):
    """SAC training."""

    # Devices handling
    process_id, local_devices_to_use, device_count = handle_devices(args.max_devices_per_host)

    # TO CONDENSATE

    # The number of environment steps executed for every `actor_step()` call.
    env_steps_per_actor_step = args.action_repeat * args.num_envs
    # equals to ceil(learning_start / env_steps_per_actor_step)
    num_prefill_actor_steps = -(-args.learning_start // args.num_envs)
    num_prefill_env_steps = num_prefill_actor_steps * env_steps_per_actor_step
    assert args.total_timesteps - num_prefill_env_steps >= 0
    num_evals_after_init = max(args.num_evals - 1, 1)

    # The number of run_one_sac_epoch calls per run_sac_training.
    # equals to
    # ceil(num_timesteps - num_prefill_env_steps /
    #      (num_evals_after_init * env_steps_per_actor_step))
    num_training_steps_per_epoch = -(
        -(args.total_timesteps - num_prefill_env_steps) // (num_evals_after_init * env_steps_per_actor_step)
    )

    assert args.num_envs % device_count == 0

    # Environment
    env = environment

    # Vmap Wrapper from Brax (inside: jax.vmap(env.reset and env.step())
    if isinstance(env, envs.Env):
        wrap_for_training = envs.training.wrap
    else:
        wrap_for_training = envs_v1.wrappers.wrap_for_training

    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)
    v_randomization_fn = None

    if randomization_fn is not None:
        v_randomization_fn = functools.partial(
            randomization_fn,
            rng=jax.random.split(key, args.num_envs // jax.process_count() // local_devices_to_use),
        )

    # Vectorization with Vmap from Brax
    env = wrap_for_training(
        env,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        randomization_fn=v_randomization_fn,
    )

    # Observation & action spaces dimensions
    obs_size = env.observation_size
    action_size = env.action_size

    normalize_fn = lambda x, y: x
    if args.normalize_observations:
        normalize_fn = normalize

    # Builds the SAC networks
    sac_network = make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
    )

    # Builds the FWD function of the SAC Policy
    make_policy = make_inference_fn(sac_network)

    # Optimizers
    alpha_optimizer = optax.adam(learning_rate=3e-4)
    policy_optimizer = optax.adam(learning_rate=args.learning_rate)
    q_optimizer = optax.adam(learning_rate=args.learning_rate)

    # Dummy transition (s,a,r,s') to initiate the replay buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        next_observation=dummy_obs,
        extras={"state_extras": {"truncation": 0.0}, "policy_extras": {}},
    )

    # Create Replay Buffer
    replay_buffer = UniformSamplingQueue(
        buffer_size=args.buffer_size // device_count,
        dummy_data_sample=dummy_transition,
        sample_batch_size=args.batch_size * args.grad_updates_per_step // device_count,
    )

    # Create Losses and Grad functions for SAC losses
    # Ent coef losses - could be fixed alpha
    alpha_loss, critic_loss, actor_loss = make_losses(
        sac_network=sac_network,
        reward_scaling=args.reward_scaling,
        discount_factor=args.discount_factor,
        action_size=action_size,
    )

    # Update gradients for all losses
    alpha_update = gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        alpha_loss,
        alpha_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )
    critic_update = gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        critic_loss,
        q_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )
    actor_update = gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
        actor_loss,
        policy_optimizer,
        pmap_axis_name=PMAP_AXIS_NAME,
    )

    # TO CONDENSTATE ALL FUNCTIONS BELOW

    # One step of stochastic gradient descend for all params (alpha, policy params, q params)
    def sgd_step(
        carry: Tuple[TrainingState, PRNGKey],
        transitions: Transition,
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry

        key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params)
        critic_loss, q_params, q_optimizer_state = critic_update(
            training_state.q_params,
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.target_q_params,
            alpha,
            transitions,
            key_critic,
            optimizer_state=training_state.q_optimizer_state,
        )
        actor_loss, policy_params, policy_optimizer_state = actor_update(
            training_state.policy_params,
            training_state.normalizer_params,
            training_state.q_params,
            alpha,
            transitions,
            key_actor,
            optimizer_state=training_state.policy_optimizer_state,
        )

        new_target_q_params = jax.tree_util.tree_map(
            lambda x, y: x * (1 - args.tau) + y * args.tau,
            training_state.target_q_params,
            q_params,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
        }

        new_training_state = TrainingState(
            policy_optimizer_state=policy_optimizer_state,
            policy_params=policy_params,
            q_optimizer_state=q_optimizer_state,
            q_params=q_params,
            target_q_params=new_target_q_params,
            gradient_steps=training_state.gradient_steps + 1,
            env_steps=training_state.env_steps,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
            normalizer_params=training_state.normalizer_params,
        )
        return (new_training_state, key), metrics

    # Collect rollout equivalent
    def get_experience(
        normalizer_params: RunningStatisticsState,
        policy_params: Params,
        env_state: Union[envs.State, envs_v1.State],
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[RunningStatisticsState, Union[envs.State, envs_v1.State], ReplayBufferState]:
        policy = make_policy((normalizer_params, policy_params))
        env_state, transitions = actor_step(env, env_state, policy, key, extra_fields=("truncation",))

        normalizer_params = update(  # Updates the running statistics with the given batch of data
            normalizer_params,
            transitions.observation,
            pmap_axis_name=PMAP_AXIS_NAME,
        )

        buffer_state = replay_buffer.insert(buffer_state, transitions)
        return normalizer_params, env_state, buffer_state

    # Training step --> One step collection (s,a,r,s') + one Sgd step
    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, Union[envs.State, envs_v1.State], ReplayBufferState, Metrics]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience(
            training_state.normalizer_params,
            training_state.policy_params,
            env_state,
            buffer_state,
            experience_key,
        )
        training_state = training_state.replace(
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_actor_step,
        )

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        # Change the front dimension of transitions so 'update_step' is called
        # grad_updates_per_step times by the scan.
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (args.grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), metrics = jax.lax.scan(sgd_step, (training_state, training_key), transitions)

        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    #
    def prefill_replay_buffer(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            new_normalizer_params, env_state, buffer_state = get_experience(
                training_state.normalizer_params,
                training_state.policy_params,
                env_state,
                buffer_state,
                key,
            )
            new_training_state = training_state.replace(
                normalizer_params=new_normalizer_params,
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )
            return (new_training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=num_prefill_actor_steps)[0]

    prefill_replay_buffer = jax.pmap(prefill_replay_buffer, axis_name=PMAP_AXIS_NAME)

    #
    def training_epoch(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        def f(carry, unused_t):
            ts, es, bs, k = carry
            k, new_key = jax.random.split(k)
            ts, es, bs, metrics = training_step(ts, es, bs, k)
            return (ts, es, bs, new_key), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(
            f,
            (training_state, env_state, buffer_state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return training_state, env_state, buffer_state, metrics

    # Compiled XLA for parallelism GPU cores
    training_epoch = jax.pmap(training_epoch, axis_name=PMAP_AXIS_NAME)

    # Note that this is NOT a pure jittable method --> Main training epoch function (in the loop)
    def training_epoch_with_timing(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
        nonlocal training_walltime
        t = time.time()
        (training_state, env_state, buffer_state, metrics) = training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        return training_state, env_state, buffer_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

    global_key, local_key = jax.random.split(rng)
    local_key = jax.random.fold_in(local_key, process_id)

    # Training state init
    training_state = init_training_state(
        key=global_key,
        obs_size=obs_size,
        local_devices_to_use=local_devices_to_use,
        sac_network=sac_network,
        alpha_optimizer=alpha_optimizer,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
    )
    del global_key

    local_key, rb_key, env_key, eval_key = jax.random.split(local_key, 4)

    # Env init
    env_keys = jax.random.split(env_key, args.num_envs // jax.process_count())
    env_keys = jnp.reshape(env_keys, (local_devices_to_use, -1) + env_keys.shape[1:])
    env_state = jax.pmap(env.reset)(env_keys)

    # Replay buffer init
    buffer_state = jax.pmap(replay_buffer.init)(jax.random.split(rb_key, local_devices_to_use))

    # Evaluator to evaluate policy sometimes
    evaluator = Evaluator(
        env,
        functools.partial(make_policy, deterministic=args.deterministic_eval),
        num_eval_envs=args.num_eval_envs,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        key=eval_key,
    )

    # Run initial eval
    metrics = {}
    if process_id == 0 and args.num_evals > 1:
        metrics = evaluator.run_evaluation(
            unpmap((training_state.normalizer_params, training_state.policy_params)),
            training_metrics={},
        )
        progress_fn(0, metrics)

    # Create and initialize the replay buffer.
    t = time.time()
    prefill_key, local_key = jax.random.split(local_key)
    prefill_keys = jax.random.split(prefill_key, local_devices_to_use)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state,
        env_state,
        buffer_state,
        prefill_keys,
    )

    replay_size = jnp.sum(jax.vmap(replay_buffer.size)(buffer_state)) * jax.process_count()
    print(f"replay size after prefill {replay_size}")
    assert replay_size >= args.learning_start
    training_walltime = time.time() - t

    # Main Training Loop
    current_step = 0
    for _ in tqdm(range(num_evals_after_init)):
        print(f"step {current_step}")

        # Optimization
        epoch_key, local_key = jax.random.split(local_key)
        epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
        (training_state, env_state, buffer_state, training_metrics) = training_epoch_with_timing(
            training_state,
            env_state,
            buffer_state,
            epoch_keys,
        )
        current_step = int(unpmap(training_state.env_steps))
        progress_fn(current_step, training_metrics)  # Progress function for training metrics

        # Eval and logging
        if process_id == 0:
            if checkpoint_logdir:
                # Save current policy.
                params = unpmap((training_state.normalizer_params, training_state.policy_params))
                path = f"{checkpoint_logdir}_sac_{current_step}.pkl"
                save_params(path, params)

            # Run evals.
            eval_metrics = evaluator.run_evaluation(
                unpmap((training_state.normalizer_params, training_state.policy_params)),
                training_metrics,
            )
            progress_fn(current_step, eval_metrics)

    total_steps = current_step
    assert total_steps >= args.total_timesteps

    params = unpmap((training_state.normalizer_params, training_state.policy_params))

    # If there was no mistakes the training_state should still be identical on all devices.
    assert_is_replicated(training_state)
    print(f"total steps: {total_steps}")
    synchronize_hosts()
    return (make_policy, params, metrics)


if __name__ == "__main__":
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    args_ = parse_args()
    env = envs.get_environment(env_name=args_.env_name, backend=args_.backend)

    # Metrics progression of training
    writer = SummaryWriter(f"runs/{args_.env_name}_sac_valentin_tu_pues_des_oreilles{int(time.time())}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args_).items()])),
    )

    def progress(num_steps, metrics):
        for key in metrics:
            print(f"{key}: {metrics[key]}")
            writer.add_scalar(key, metrics[key], num_steps)
        print()

    train(
        environment=env,
        args=args_,
        progress_fn=progress,
    )
