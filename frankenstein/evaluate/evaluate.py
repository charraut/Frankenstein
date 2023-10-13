import torch
import minari
import gymnasium as gym
from minari import DataCollectorV0
import numpy as np
import datetime
import argparse


# Local
from frankenstein.utils.utils import make_env
from frankenstein.utils.architecture import ActorCriticNet


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


def evaluate(args, run_dir, collect_dataset):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])

    if collect_dataset:
        env = DataCollectorV0(env, record_infos=True, max_buffer_steps=100000)

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

    #state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Run episodes
    while count_episodes < args.number_episodes:
        state, _ = env.reset()
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(args.device).float()
            action, _ = policy(state_tensor)

        action = action.cpu().numpy()
        state, _, _, _, infos = env.step(action)

        if "final_info" in infos:
            info = infos["final_info"][0]
            returns = info["episode"]["r"][0]
            count_episodes += 1
            list_returns.append(returns)
            print(f"-> Episode {count_episodes}: {returns} returns")

    dataset = minari.create_dataset_from_collector_env(dataset_id=args.env_id + "_", 
                                                       collector_env=env,
                                                       algorithm_name="Policy"
                                                       )

    env.close()
    return np.mean(list_returns)

if __name__ == '__main__':
    args_ = parse_args()

    run_time = str(datetime.now().strftime("%d-%m_%H:%M:%S"))
    run_name = "SR_SAC_PyTorch_Base_RR4"
    run_dir = f"runs/{args_.env_id}__{run_name}__{run_time}"

    print(f"Starting evaluation of {run_name} on {args_.env_id} for {args_.total_timesteps} timesteps.")
    evaluate(args_, run_dir, collect_dataset=False)