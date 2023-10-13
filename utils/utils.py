# Utils methods - 
import torch
import numpy as np
import gymnasium as gym

# Method to build Gym Env
def make_env(env_id, capture_video=False, run_dir="."):
    def thunk():
        if capture_video:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(
                env=env,
                video_folder=f"{run_dir}/videos",
                episode_trigger=lambda x: x,
                disable_logger=True,
            )
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.FlattenObservation(env)

        return env

    return thunk

# Soft reset for weight & bias (perturbate)
def soft_reset_layer(layer, mu=0.0, std=0.01):
    layer.weight.add(torch.tensor(np.random.normal(
        mu, std, layer.weight.size()), dtype=torch.float).to(layer.weight.get_device()))
    layer.bias.add(torch.tensor(np.random.normal(
        mu, std, layer.bias.size()), dtype=torch.float).to(layer.bias.get_device()))

# Hard reset for weight & bias (reinit)
def hard_reset_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


# Method to reset layers of neural networks - SR_SAC
def reset_nn_layers(network):
    # Actor mean & logstd 
    if isinstance(network, torch.nn.Linear):
        soft_reset_layer(network)
    # Actor net, Critics & Targets 
    else:
        i=0
        for layer in network:
            # Avoid ReLU() layers
            if hasattr(layer, 'weight'):
                if i < len(network) // 4:
                    soft_reset_layer(layer)
                else:
                    hard_reset_layer(layer)
                i += 1