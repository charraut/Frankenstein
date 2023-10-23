# Utils methods -
import pathlib

import gymnasium as gym
import numpy as np
import torch


PATH_TO_MAIN_PROJECT = str(pathlib.Path(__file__).parent.parent.parent.absolute()) + "/"


# Method to build Gym Env
def make_env(env_id, capture_video=False, run_dir="."):
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


# Soft reset for weight & bias (perturbate)
def soft_reset_layer(layer, mu=0.0, std=0.01):
    layer.weight.add(
        torch.tensor(np.random.normal(mu, std, layer.weight.size()), dtype=torch.float).to(layer.weight.get_device()),
    )
    layer.bias.add(
        torch.tensor(np.random.normal(mu, std, layer.bias.size()), dtype=torch.float).to(layer.bias.get_device()),
    )


# Hard reset for weight & bias (reinit)
def hard_reset_layer(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)


# Method to reset layers of neural networks - SR_SAC
def reset_nn_layers(network, fully_hard=False, fully_soft=False):
    # Actor mean & logstd
    if isinstance(network, torch.nn.Linear):
        if fully_hard:
            hard_reset_layer(network)
        else:
            soft_reset_layer(network)
    # Actor net, Critics & Targets
    else:
        i = 0
        for layer in network:
            # Avoid ReLU() layers
            if hasattr(layer, "weight"):
                if i < len(network) // 2:
                    if fully_hard:
                        hard_reset_layer(layer)
                    else:
                        soft_reset_layer(layer)
                else:
                    if fully_soft:
                        soft_reset_layer(layer)
                    else:
                        hard_reset_layer(layer)
                i += 1
