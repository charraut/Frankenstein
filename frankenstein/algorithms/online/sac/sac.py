# General
import argparse
import time
from collections import deque
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.distributions import Uniform
from torch.nn.functional import mse_loss
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from frankenstein.utils.architecture import ActorCriticNet

# Local
from frankenstein.utils.replay_buffer import ReplayBuffer
from frankenstein.utils.utils import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
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
    parser.add_argument("--saved_model_freq", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    return args


def train(args, run_name, run_dir):
    # Create tensorboard writer and save hyperparameters
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Create vectorized environment
    env = make_env(args.env_id)

    # Metadata about the environment
    observation_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    action_dim = np.prod(action_shape)
    action_low = torch.from_numpy(env.action_space.low).to(args.device)
    action_high = torch.from_numpy(env.action_space.high).to(args.device)

    # Set seed for reproducibility
    if args.seed:
        numpy_rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        state, _ = env.reset(seed=args.seed)
    else:
        numpy_rng = np.random.default_rng()
        state, _ = env.reset()

    # Create the networks and the optimizer
    policy = ActorCriticNet(
        observation_shape,
        action_dim,
        args.actor_layers,
        args.critic_layers,
        action_low,
        action_high,
        args.device,
    )
    target = ActorCriticNet(
        observation_shape,
        action_dim,
        args.actor_layers,
        args.critic_layers,
        action_low,
        action_high,
        args.device,
    )
    target.load_state_dict(policy.state_dict())

    optimizer_actor = optim.AdamW(policy.actor_net.parameters(), lr=args.learning_rate)
    optimizer_critic = optim.AdamW(
        list(policy.critic_net1.parameters()) + list(policy.critic_net2.parameters()),
        lr=args.learning_rate,
    )

    alpha = args.alpha

    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        args.buffer_size,
        args.batch_size,
        observation_shape,
        action_shape,
        numpy_rng,
        args.device,
    )

    # Remove unnecessary variables
    del observation_shape, action_shape, action_dim

    log_episodic_returns = deque(maxlen=5)
    log_episodic_lengths = deque(maxlen=5)
    start_time = time.process_time()

    # Main loop
    nb_save = 1
    for global_step in tqdm(range(args.total_timesteps)):
        if global_step < args.learning_start:
            action = Uniform(action_low, action_high).sample()
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(args.device).float().unsqueeze(0)
                action, _ = policy.actor(state_tensor)

        # Perform action
        action = action.cpu().numpy()
        next_state, reward, terminated, truncated, infos = env.step(action)

        # Store transition in the replay buffer
        flag = 1.0 - np.logical_or(terminated, truncated)
        replay_buffer.push(state, action, reward, flag)

        state = next_state

        # Log episodic return and length
        if terminated or truncated:
            state, _ = env.reset()
            log_episodic_returns.append(infos["episode"]["r"])
            log_episodic_lengths.append(infos["episode"]["l"])

            writer.add_scalar("rollout/episodic_return", np.mean(log_episodic_returns), global_step)
            writer.add_scalar("rollout/episodic_length", np.mean(log_episodic_lengths), global_step)

        # Perform training step
        if global_step > args.learning_start and not (global_step % args.train_freq):
            for _ in range(args.gradient_steps):
                # Sample a batch from the replay buffer
                states, actions, rewards, next_states, flags = replay_buffer.sample()

                # Update critic
                with torch.no_grad():
                    next_state_actions, next_state_log_pi = policy.actor(next_states)
                    critic1_next_target, critic2_next_target = target.critic(next_states, next_state_actions)
                    min_qf_next_target = torch.min(critic1_next_target, critic2_next_target) - alpha * next_state_log_pi
                    next_q_value = rewards + args.gamma * flags * min_qf_next_target

                qf1_a_values, qf2_a_values = policy.critic(states, actions)
                qf1_loss = mse_loss(qf1_a_values, next_q_value)
                qf2_loss = mse_loss(qf2_a_values, next_q_value)
                critic_loss = qf1_loss + qf2_loss

                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()

                # Update actor
                pi, log_pi = policy.actor(states)
                qf1_pi, qf2_pi = policy.critic(states, pi)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                actor_loss = (alpha * log_pi - min_qf_pi).mean()

                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()

                writer.add_scalar("train/actor_loss", actor_loss, global_step)

                # Update the target network (soft update)
                for param, target_param in zip(policy.critic_net1.parameters(), target.critic_net1.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(policy.critic_net2.parameters(), target.critic_net2.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            # Log training metrics
            writer.add_scalar("rollout/SPS", int(global_step / (time.process_time() - start_time)), global_step)
            writer.add_scalar("train/critic_loss", critic_loss, global_step)
            writer.add_scalar("train/qf1_a_values", qf1_a_values.mean(), global_step)
            writer.add_scalar("train/qf2_a_values", qf2_a_values.mean(), global_step)
            writer.add_scalar("train/critic1_next_target", critic1_next_target.mean(), global_step)
            writer.add_scalar("train/critic2_next_target", critic2_next_target.mean(), global_step)
            writer.add_scalar("train/qf1_loss", qf1_loss, global_step)
            writer.add_scalar("train/qf2_loss", qf2_loss, global_step)
            writer.add_scalar("train/min_qf_next_target", min_qf_next_target.mean(), global_step)
            writer.add_scalar("train/next_q_value", next_q_value.mean(), global_step)

        # Save final policy
        if global_step == int((nb_save / args.saved_model_freq) * args.total_timesteps):
            torch.save(policy.state_dict(), f"{run_dir}/policy_" + str(global_step) + ".pt")
            print(f"Saved policy to {run_dir}/policy_" + str(global_step) + ".pt")
            nb_save += 1

    # Close the environment
    env.close()
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
    run_name = "SAC_PyTorch_Base_RR1"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Starting training of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    print(f"Results will be saved to: {run_dir}")
    mean_train_return = train(args=args_, run_name=run_name, run_dir=run_dir)
    print(f"Training - Mean returns achieved: {mean_train_return}.")
