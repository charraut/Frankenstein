from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax

from frankenstein.brax.networks import Params, SACNetworks
from frankenstein.brax.utils import PRNGKey, Transition


#############################
## Losses ##
#############################
def make_losses(sac_network: SACNetworks, reward_scaling: float, discount_factor: float, action_size: int):
    """Creates the SAC losses."""

    target_entropy = -0.5 * action_size
    policy_network = sac_network.policy_network
    q_network = sac_network.q_network
    parametric_action_distribution = sac_network.parametric_action_distribution

    def alpha_loss(
        log_alpha: jnp.ndarray,
        policy_params: Params,
        normalizer_params: Any,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
        dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        alpha = jnp.exp(log_alpha)
        alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
        return jnp.mean(alpha_loss)

    def critic_loss(
        q_params: Params,
        policy_params: Params,
        normalizer_params: Any,
        target_q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        q_old_action = q_network.apply(normalizer_params, q_params, transitions.observation, transitions.action)
        next_dist_params = policy_network.apply(normalizer_params, policy_params, transitions.next_observation)
        next_action = parametric_action_distribution.sample_no_postprocessing(next_dist_params, key)
        next_log_prob = parametric_action_distribution.log_prob(next_dist_params, next_action)
        next_action = parametric_action_distribution.postprocess(next_action)
        next_q = q_network.apply(normalizer_params, target_q_params, transitions.next_observation, next_action)
        next_v = jnp.min(next_q, axis=-1) - alpha * next_log_prob
        target_q = jax.lax.stop_gradient(
            transitions.reward * reward_scaling + transitions.discount * discount_factor * next_v,
        )
        q_error = q_old_action - jnp.expand_dims(target_q, -1)

        # Better bootstrapping for truncated episodes
        truncation = transitions.extras["state_extras"]["truncation"]
        q_error *= jnp.expand_dims(1 - truncation, -1)

        q_loss = 0.5 * jnp.mean(jnp.square(q_error))
        return q_loss

    def actor_loss(
        policy_params: Params,
        normalizer_params: Any,
        q_params: Params,
        alpha: jnp.ndarray,
        transitions: Transition,
        key: PRNGKey,
    ) -> jnp.ndarray:
        dist_params = policy_network.apply(normalizer_params, policy_params, transitions.observation)
        action = parametric_action_distribution.sample_no_postprocessing(dist_params, key)
        log_prob = parametric_action_distribution.log_prob(dist_params, action)
        action = parametric_action_distribution.postprocess(action)
        q_action = q_network.apply(normalizer_params, q_params, transitions.observation, action)
        min_q = jnp.min(q_action, axis=-1)
        actor_loss = alpha * log_prob - min_q
        return jnp.mean(actor_loss)

    return alpha_loss, critic_loss, actor_loss


######################
## Gradients & Loss ##
######################
def loss_and_pgrad(loss_fn: Callable[..., float], pmap_axis_name: Optional[str], has_aux: bool = False):
    g = jax.value_and_grad(loss_fn, has_aux=has_aux)

    def h(*args, **kwargs):
        value, grad = g(*args, **kwargs)
        return value, jax.lax.pmean(grad, axis_name=pmap_axis_name)

    return g if pmap_axis_name is None else h


def gradient_update_fn(
    loss_fn: Callable[..., float],
    optimizer: optax.GradientTransformation,
    pmap_axis_name: Optional[str],
    has_aux: bool = False,
):
    """Wrapper of the loss function that apply gradient updates.

    Args:
      loss_fn: The loss function.
      optimizer: The optimizer to apply gradients.
      pmap_axis_name: If relevant, the name of the pmap axis to synchronize
        gradients.
      has_aux: Whether the loss_fn has auxiliary data.

    Returns:
      A function that takes the same argument as the loss function plus the
      optimizer state. The output of this function is the loss, the new parameter,
      and the new optimizer state.
    """
    loss_and_pgrad_fn = loss_and_pgrad(loss_fn, pmap_axis_name=pmap_axis_name, has_aux=has_aux)

    def f(*args, optimizer_state):
        value, grads = loss_and_pgrad_fn(*args)
        params_update, optimizer_state = optimizer.update(grads, optimizer_state)
        params = optax.apply_updates(args[0], params_update)
        return value, params, optimizer_state

    return f
