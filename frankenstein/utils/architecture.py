import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCriticNet(nn.Module):
    def __init__(self, observation_shape, action_dim, actor_layers, critic_layers, action_low, action_high, device):
        super().__init__()

        observation_dim = np.prod(observation_shape)

        self.actor_net = self._build_net(observation_shape, actor_layers)
        self.actor_mean = self._build_linear(actor_layers[-1], action_dim)
        self.actor_logstd = self._build_linear(actor_layers[-1], action_dim)

        self.critic_net1 = self._build_net(observation_dim + action_dim, critic_layers)
        self.critic_net1.append(self._build_linear(critic_layers[-1], 1))

        self.critic_net2 = self._build_net(observation_dim + action_dim, critic_layers)
        self.critic_net2.append(self._build_linear(critic_layers[-1], 1))

        # Scale and bias the output of the network to match the action space
        self.register_buffer("action_scale", ((action_high - action_low) / 2.0))
        self.register_buffer("action_bias", ((action_high + action_low) / 2.0))

        if device.type == "cuda":
            self.cuda()

    def _build_linear(self, in_size, out_size, apply_init=True, std=np.sqrt(2), bias_const=0.0):
        layer = nn.Linear(in_size, out_size)

        if apply_init:
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)

        return layer

    def _build_net(self, observation_shape, hidden_layers):
        layers = nn.Sequential()
        in_size = np.prod(observation_shape)

        for out_size in hidden_layers:
            layers.append(self._build_linear(in_size, out_size))
            layers.append(nn.ReLU())
            in_size = out_size

        return layers

    def actor(self, state):
        log_std_max = 2
        log_std_min = -5

        output = self.actor_net(state)
        mean = self.actor_mean(output)
        log_std = torch.tanh(self.actor_logstd(output))

        # Rescale log_std to ensure it is within range [log_std_min, log_std_max].
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()

        # Sample action using reparameterization trick.
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        # Rescale and shift the action.
        action = y_t * self.action_scale + self.action_bias

        # Calculate the log probability of the sampled action.
        log_prob = normal.log_prob(x_t)

        # Enforce action bounds.
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        # Rescale mean and shift it to match the action range.
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob.squeeze()

    def critic(self, state, action):
        critic1 = self.critic_net1(torch.cat([state, action], 1)).squeeze()
        critic2 = self.critic_net2(torch.cat([state, action], 1)).squeeze()
        return critic1, critic2
