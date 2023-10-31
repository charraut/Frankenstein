import argparse
import dataclasses
import functools
import time
from collections import deque
from datetime import datetime
from typing import Sequence

import flax
import jax
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jax import numpy as jnp
from tensorboardX import SummaryWriter
from tqdm import tqdm
from waymax import config as _config
from waymax import dataloader, dynamics
from waymax import env as _env
from waymax.datatypes.action import Action
from waymax.datatypes.observation import observation_from_state
from waymax.env.wrappers.brax_wrapper import BraxWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--actor_layers", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--critic_layers", nargs="+", type=int, default=[256, 256])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--learning_start", type=int, default=25_000)
    parser.add_argument("--gradient_steps", type=int, default=1)
    parser.add_argument("--train_freq", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--capture_video", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


def custom_obs(state):
    observation = observation_from_state(state, coordinate_frame=_config.CoordinateFrame.OBJECT)
    print(observation)

    return observation


@functools.partial(jax.jit, static_argnums=0)
def actor_output(apply_fn, params, state, key):
    return apply_fn(params, state, key)


@functools.partial(jax.jit, static_argnums=0)
def critic_output(apply_fn, params, state, action):
    return apply_fn(params, state, action)


@functools.partial(jax.jit, static_argnums=(4, 5))
def critic_train_step(critic_train_state_1, critic_train_state_2, actor_train_state, batch, gamma, alpha, key):
    states, actions, rewards, next_states, flags = batch

    # Compute the target q-value
    next_state_actions, next_state_log_pi, key = actor_output(
        actor_train_state.apply_fn,
        actor_train_state.params,
        next_states,
        key,
    )
    critic_next_target_1 = critic_output(
        critic_train_state_1.apply_fn,
        critic_train_state_1.target_params,
        next_states,
        next_state_actions,
    )
    critic_next_target_2 = critic_output(
        critic_train_state_2.apply_fn,
        critic_train_state_2.target_params,
        next_states,
        next_state_actions,
    )
    min_qf_next_target = jnp.minimum(critic_next_target_1, critic_next_target_2) - alpha * next_state_log_pi

    td_target = rewards + gamma * flags * min_qf_next_target

    def loss_fn(params, apply_fn):
        td_predict = critic_output(apply_fn, params, states, actions)
        return jnp.mean((td_predict - td_target) ** 2)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(critic_train_state_1.params, critic_train_state_1.apply_fn)
    critic_train_state_1 = critic_train_state_1.apply_gradients(grads=grads)

    loss, grads = grad_fn(critic_train_state_2.params, critic_train_state_2.apply_fn)
    critic_train_state_2 = critic_train_state_2.apply_gradients(grads=grads)

    return critic_train_state_1, critic_train_state_2, loss, key


@functools.partial(jax.jit, static_argnums=4)
def actor_train_step(actor_train_state, critic_train_state_1, critic_train_state_2, batch, alpha, key):
    states, actions, _, _, _ = batch

    def loss_fn(params):
        pi, log_pi, _key = actor_output(actor_train_state.apply_fn, params, states, key)
        qf1_pi = critic_output(critic_train_state_1.apply_fn, critic_train_state_1.params, states, pi)
        qf2_pi = critic_output(critic_train_state_2.apply_fn, critic_train_state_2.params, states, pi)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)
        return jnp.mean(alpha * log_pi - min_qf_pi), _key

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _key), grads = grad_fn(actor_train_state.params)
    actor_train_state = actor_train_state.apply_gradients(grads=grads)

    return actor_train_state, loss, _key


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, observation_size, action_size, numpy_rng):
        self.states = np.zeros((buffer_size, observation_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_size), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.flags = np.zeros((buffer_size,), dtype=np.float32)

        self.batch_size = batch_size
        self.max_size = buffer_size
        self.idx = 0
        self.size = 0

        self.numpy_rng = numpy_rng

    def push(self, state, action, reward, flag):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.flags[self.idx] = flag

        self.idx = (self.idx + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idxs = self.numpy_rng.integers(0, self.size - 1, size=self.batch_size)

        return (
            self.states[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.states[idxs + 1],
            self.flags[idxs],
        )


class ActorNet(nn.Module):
    action_size: Sequence[int]
    action_scale: Sequence[int]
    action_bias: Sequence[int]

    @nn.compact
    def __call__(self, state, key):
        log_std_max = 2
        log_std_min = -5

        output = nn.Dense(256)(state)
        output = nn.relu(output)
        output = nn.Dense(256)(output)
        output = nn.relu(output)

        mean = nn.Dense(self.action_size)(output)
        log_std = nn.tanh(nn.Dense(self.action_size)(output))

        # Rescale log_std to ensure it is within range [log_std_min, log_std_max].
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = jnp.exp(log_std)

        x_t, key = self._sample(mean, std, key)
        y_t = nn.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias

        log_prob = self._log_prob(mean, std, x_t)

        log_prob -= jnp.log(self.action_scale * (1 - y_t**2) + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

        mean = nn.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob.squeeze(), key

    def _log_prob(self, mean, std, value):
        var = std**2
        log_scale = jnp.log(std)
        return -((value - mean) ** 2) / (2 * var) - log_scale - jnp.log(jnp.sqrt(2 * jnp.pi))

    def _sample(self, mean, std, key):
        key, subkey = jax.random.split(key)
        return jax.random.normal(subkey, shape=mean.shape) * std + mean, key


class CriticNet(nn.Module):
    @nn.compact
    def __call__(self, state, action):
        output = jnp.concatenate([state, action], axis=-1)
        output = nn.Dense(256)(output)
        output = nn.relu(output)
        output = nn.Dense(256)(output)
        output = nn.relu(output)
        output = nn.Dense(1)(output)

        return output.squeeze()


def train(args, run_name, run_dir):
    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    # Create vectorized environment
    max_num_objects = 4
    dynamics_model = dynamics.InvertibleBicycleModel(normalize_actions=True)
    env_config = _config.EnvironmentConfig(max_num_objects=max_num_objects)

    scenarios = dataloader.simulator_state_generator(
        dataclasses.replace(_config.WOD_1_1_0_TRAINING, max_num_objects=max_num_objects),
    )
    env = BraxWrapper(_env.PlanningAgentEnvironment(dynamics_model, env_config))
    env.observe = custom_obs

    init_scenario = next(scenarios)
    init_time_step = env.reset(init_scenario)

    # Metadata about the environment
    observation_size = init_time_step.observation.shape[-1]
    action_size = env.action_spec().data.shape[0]
    action_low = -1
    action_high = 1

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)

    key, actor_key, critic_key = jax.random.split(jax.random.PRNGKey(args.seed), 3)

    obs = init_time_step.observation

    # Create the networks and the optimizer
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    actor_net = ActorNet(action_size=action_size, action_scale=action_scale, action_bias=action_bias)

    actor_init_params = actor_net.init(actor_key, obs, key)

    critic_net = CriticNet()
    uniform_key, key = jax.random.split(key)
    random_action = jax.random.uniform(uniform_key, shape=(action_size,), minval=action_low, maxval=action_high)
    critic_init_params = critic_net.init(critic_key, obs, random_action)

    optimizer = optax.adam(learning_rate=args.learning_rate)

    actor_train_state = TrainState.create(
        apply_fn=actor_net.apply,
        params=actor_init_params,
        target_params=actor_init_params,
        tx=optimizer,
    )

    critic_train_state_1 = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_init_params,
        target_params=critic_init_params,
        tx=optimizer,
    )

    critic_train_state_2 = TrainState.create(
        apply_fn=critic_net.apply,
        params=critic_init_params,
        target_params=critic_init_params,
        tx=optimizer,
    )

    alpha = args.alpha

    # Create the replay buffer
    numpy_rng = np.random.default_rng(args.seed) if args.seed else np.random.default_rng()
    replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size, observation_size, action_size, numpy_rng)

    # Remove unnecessary variables
    del (
        observation_size,
        actor_key,
        critic_key,
        actor_net,
        critic_net,
        critic_init_params,
        actor_init_params,
        optimizer,
    )

    log_episodic_returns = deque(maxlen=5)
    log_episodic_lengths = deque(maxlen=5)
    start_time = time.process_time()

    reward_log = []

    state = init_time_step

    # Main loop
    for global_step in tqdm(range(args.total_timesteps)):
        if global_step < args.learning_start:
            uniform_key, key = jax.random.split(key)
            action = jax.random.uniform(uniform_key, shape=(action_size,), minval=action_low, maxval=action_high)
        else:
            print(state.observation)
            obs = state.observation
            # Normalize observation
            obs = obs.mean(axis=0) - obs.std(axis=0) / 2
            action, _, key = actor_output(actor_train_state.apply_fn, actor_train_state.params, state.observation, key)
            print(action)
            print()

        # Perform action
        action_waymax = Action(data=action, valid=jnp.bool_([True]))
        action_waymax.validate()
        state = jit_env_step(state, action_waymax)

        # Store transition in the replay buffer
        flag = 1.0 - state.done
        replay_buffer.push(state.observation, action, state.reward, flag)

        reward_log.append(state.reward)

        # Log episodic return and length
        if state.done:
            state = jit_env_reset(next(scenarios))

            log_episodic_returns.append(np.sum(reward_log))
            log_episodic_lengths.append(len(reward_log))

            print("Metrics:", state.metrics)

            writer.add_scalar("rollout/episodic_return", np.mean(log_episodic_returns), global_step)
            writer.add_scalar("rollout/episodic_length", np.mean(log_episodic_lengths), global_step)

            reward_log = []

        # Perform training step
        if global_step > args.learning_start and not (global_step % args.train_freq):
            for _ in range(args.gradient_steps):
                # Sample replay buffer
                batch = replay_buffer.sample()

                critic_train_state_1, critic_train_state_2, critic_loss, key = critic_train_step(
                    critic_train_state_1,
                    critic_train_state_2,
                    actor_train_state,
                    batch,
                    args.gamma,
                    alpha,
                    key,
                )

                critic_train_state_1 = critic_train_state_1.replace(
                    target_params=optax.incremental_update(
                        critic_train_state_1.params,
                        critic_train_state_1.target_params,
                        args.tau,
                    ),
                )
                critic_train_state_2 = critic_train_state_2.replace(
                    target_params=optax.incremental_update(
                        critic_train_state_2.params,
                        critic_train_state_2.target_params,
                        args.tau,
                    ),
                )

                # Update actor
                actor_train_state, actor_loss, key = actor_train_step(
                    actor_train_state,
                    critic_train_state_1,
                    critic_train_state_2,
                    batch,
                    alpha,
                    key,
                )

                writer.add_scalar("train/actor_loss", jax.device_get(actor_loss), global_step)

            # Log training metrics
            writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)
            writer.add_scalar("train/critic_loss", jax.device_get(critic_loss), global_step)

    # Close the environment
    writer.close()

    # Average of episodic returns (for the last 5% of the training)
    indexes = int(len(log_episodic_returns) * 0.05)
    mean_train_return = np.mean(log_episodic_returns[-indexes:])
    writer.add_scalar("rollout/mean_train_return", mean_train_return, global_step)

    return mean_train_return


if __name__ == "__main__":
    args_ = parse_args()

    # Create run directory
    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "SAC_Flax"
    run_dir = f"runs/waymax__{run_name}__{run_time}"

    print(f"Commencing training of {run_name} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")
