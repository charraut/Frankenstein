from typing import Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from brax import envs
from brax.v1 import envs as envs_v1

from frankenstein.brax.networks import Policy
from frankenstein.brax.utils import PRNGKey, Transition
from waymax.datatypes import Action

# Functions to act in the environment (outputs the action from policy then generates an Env step())
def actor_step(
    env,
    env_state,
    policy: Policy,
    key: PRNGKey,
):
    """Collect data."""
    actions, _ = policy(env_state.observation, key)

    action_waymax = Action(
        data=actions, valid=jnp.ones_like(actions[..., 0:1], dtype=jnp.bool_)
    )
    action_waymax.validate()

    nstate = env.step(env_state, action_waymax)

    return nstate, Transition(
        observation=env_state.observation,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.observation,
    )


def generate_unroll(
    env,
    env_state,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
):
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(env, state, policy, current_key)
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(f, (env_state, key), (), length=unroll_length)
    return final_state, data
