import argparse
from datetime import datetime

import gymnasium as gym
import minari
import numpy as np
import torch
from minari import DataCollectorV0

from frankenstein.utils.architecture import ActorCriticNet

# Local
from frankenstein.utils.utils import make_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_timestep", type=str, default="")
    parser.add_argument("--env_id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--number_episodes", type=int, default=100)
    parser.add_argument("--actor_layers", nargs="+", type=int, default=[256, 256, 256, 256])
    parser.add_argument("--critic_layers", nargs="+", type=int, default=[256, 256, 256, 256])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    return args


def evaluate(args, run_dir, act_randomly, collect_dataset):
    # Create environment
    if collect_dataset:
        env = gym.make(args.env_id)
        env = DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)
    else:
        env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=False, run_dir=run_dir)])

    if not act_randomly:
        # Metadata about the environment
        observation_shape = env.single_observation_space.shape
        action_shape = env.single_action_space.shape
        action_low = torch.from_numpy(env.single_action_space.low).to(args.device)
        action_high = torch.from_numpy(env.single_action_space.high).to(args.device)

        # Load policy
        policy = ActorCriticNet(
            observation_shape,
            action_shape,
            args.actor_layers,
            args.critic_layers,
            action_low,
            action_high,
            args.device,
        )
        policy.load_state_dict(torch.load(f"{run_dir}/actor_" + args.actor_timestep + ".pt"))
        policy.eval()

    count_episodes = 0
    list_returns = []

    # Run episodes
    while count_episodes < args.number_episodes:
        num_steps = 0
        while num_steps < env.spec.max_episode_steps:
            state, _ = env.reset()
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).to(args.device).float()
                if act_randomly:
                    action = env.action_space.sample()
                else:
                    action, _ = policy(state_tensor)
                    action = action.cpu().numpy()

            state, _, _, _, infos = env.step(action)
            num_steps += 1

        # End of an episode
        count_episodes += 1
        episode_return = infos["reward_run"]
        list_returns.append(0)
        print(f"-> Episode {count_episodes}: {episode_return} return")

    minari.create_dataset_from_collector_env(dataset_id=args.env_id + "random_policy_100ep_-v0", collector_env=env)

    env.close()
    return np.mean(list_returns)


if __name__ == "__main__":
    args_ = parse_args()

    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "SR_SAC_PyTorch_Base_RR4"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Starting evaluation of {run_name} on {args_.env_id} for {args_.number_episodes} episodes.")
    evaluate(args_, run_dir, act_randomly=True, collect_dataset=True)
