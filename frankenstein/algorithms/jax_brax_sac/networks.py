"""SAC networks."""
import dataclasses
from typing import Any, Callable, Mapping, Protocol, Sequence, Tuple

import flax
import jax
import jax.numpy as jnp
from flax import linen

from frankenstein.algorithms.jax_brax_sac.distributions import NormalTanhDistribution, ParametricDistribution


Params = Any
PolicyParams = Any
PreprocessorParams = Any
PolicyParams = Tuple[PreprocessorParams, Params]
PRNGKey = jnp.ndarray
Observation = jnp.ndarray
Action = jnp.ndarray
Extra = Mapping[str, Any]
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]


class PreprocessObservationFn(Protocol):
    def __call__(
        self,
        observation: Observation,
        preprocessor_params: PreprocessorParams,
    ) -> jnp.ndarray:
        pass


def identity_observation_preprocessor(observation: Observation, preprocessor_params: PreprocessorParams):
    del preprocessor_params
    return observation


class Policy(Protocol):
    def __call__(
        self,
        observation: Observation,
        key: PRNGKey,
    ) -> Tuple[Action, Extra]:
        pass


@dataclasses.dataclass
class FeedForwardNetwork:
    init: Callable[..., Any]
    apply: Callable[..., Any]


@flax.struct.dataclass
class SACNetworks:
    policy_network: FeedForwardNetwork
    q_network: FeedForwardNetwork
    parametric_action_distribution: ParametricDistribution


def make_inference_fn(sac_networks: SACNetworks):
    """Creates params and inference function for the SAC agent."""

    def make_policy(params: PolicyParams, deterministic: bool = False) -> Policy:
        def policy(observations: Observation, key_sample: PRNGKey) -> Tuple[Action, Extra]:
            logits = sac_networks.policy_network.apply(*params, observations)
            if deterministic:
                return sac_networks.parametric_action_distribution.mode(logits), {}
            return sac_networks.parametric_action_distribution.sample(logits, key_sample), {}

        return policy

    return make_policy


class MLP(linen.Module):
    """MLP module."""

    layer_sizes: Sequence[int]
    activation: ActivationFn = linen.relu
    kernel_init: Initializer = jax.nn.initializers.lecun_uniform()
    activate_final: bool = False
    bias: bool = True

    @linen.compact
    def __call__(self, data: jnp.ndarray):
        hidden = data
        for i, hidden_size in enumerate(self.layer_sizes):
            hidden = linen.Dense(hidden_size, name=f"hidden_{i}", kernel_init=self.kernel_init, use_bias=self.bias)(
                hidden,
            )
            if i != len(self.layer_sizes) - 1 or self.activate_final:
                hidden = self.activation(hidden)
        return hidden


# Builds the policy network
def make_policy_network(
    param_size: int,
    obs_size: int,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    actor_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> FeedForwardNetwork:
    """Creates a policy network."""
    policy_module = MLP(
        layer_sizes=list(actor_layers) + [param_size],
        activation=activation,
        kernel_init=jax.nn.initializers.lecun_uniform(),
    )

    def apply(processor_params, policy_params, obs):
        obs = preprocess_observations_fn(obs, processor_params)
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((1, obs_size))
    return FeedForwardNetwork(init=lambda key: policy_module.init(key, dummy_obs), apply=apply)


# Builds the critic Network
def make_q_network(
    obs_size: int,
    action_size: int,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    critic_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
    n_critics: int = 2,
) -> FeedForwardNetwork:
    class QModule(linen.Module):
        """Q Module."""

        n_critics: int

        @linen.compact
        def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray):
            hidden = jnp.concatenate([obs, actions], axis=-1)
            res = []
            for _ in range(self.n_critics):
                q = MLP(
                    layer_sizes=list(critic_layers) + [1],
                    activation=activation,
                    kernel_init=jax.nn.initializers.lecun_uniform(),
                )(hidden)
                res.append(q)
            return jnp.concatenate(res, axis=-1)

    q_module = QModule(n_critics=n_critics)

    def apply(processor_params, q_params, obs, actions):
        obs = preprocess_observations_fn(obs, processor_params)
        return q_module.apply(q_params, obs, actions)

    dummy_obs = jnp.zeros((1, obs_size))
    dummy_action = jnp.zeros((1, action_size))
    return FeedForwardNetwork(init=lambda key: q_module.init(key, dummy_obs, dummy_action), apply=apply)


# Builds the SAC network (action dist, pi network, q network)
def make_sac_networks(
    observation_size: int,
    action_size: int,
    preprocess_observations_fn: PreprocessObservationFn = identity_observation_preprocessor,
    actor_layers: Sequence[int] = (256, 256),
    critic_layers: Sequence[int] = (256, 256),
    activation: ActivationFn = linen.relu,
) -> SACNetworks:
    parametric_action_distribution = NormalTanhDistribution(event_size=action_size)

    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        preprocess_observations_fn=preprocess_observations_fn,
        actor_layers=actor_layers,
        activation=activation,
    )

    q_network = make_q_network(
        observation_size,
        action_size,
        preprocess_observations_fn=preprocess_observations_fn,
        critic_layers=critic_layers,
        activation=activation,
    )

    return SACNetworks(
        policy_network=policy_network,
        q_network=q_network,
        parametric_action_distribution=parametric_action_distribution,
    )
