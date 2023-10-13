import torch
import gymnasium as gym
import numpy as np

# Local
from utils.utils import make_env
from utils.architecture import ActorCriticNet

def eval_and_render(args, run_dir):
    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(args.env_id, capture_video=True, run_dir=run_dir)])

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
    policy.load_state_dict(torch.load(f"{run_dir}/actor.pt"))
    policy.eval()

    count_episodes = 0
    list_rewards = []

    state, _ = env.reset(seed=args.seed) if args.seed else env.reset()

    # Run episodes
    while count_episodes < 30:
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).to(args.device).float()
            action, _ = policy(state_tensor)

        action = action.cpu().numpy()
        state, _, _, _, infos = env.step(action)

        if "final_info" in infos:
            info = infos["final_info"][0]
            returns = info["episode"]["r"][0]
            count_episodes += 1
            list_rewards.append(returns)
            print(f"-> Episode {count_episodes}: {returns} returns")

    env.close()

    return np.mean(list_rewards)

# ADD MINARI D4RL 

if __name__ == '__main__':
    eval_and_render()