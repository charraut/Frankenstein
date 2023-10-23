import functools

import jax
from brax import envs
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac
from tqdm import tqdm


def train_agent(env_name, backend):
    print("Start training")

    env = envs.get_environment(env_name=env_name, backend=backend)
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    num_timesteps = 10_000_000
    episode_length = 1000
    num_evals = 20
    step_interval = num_timesteps // num_evals

    train_fn_sac = functools.partial(
        sac.train,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_evals=num_evals,
    )

    train_fn_ppo = functools.partial(
        ppo.train,
        num_timesteps=num_timesteps,
        num_evals=num_evals,
        reward_scaling=1,
        episode_length=episode_length,
        normalize_observations=True,
        action_repeat=1,
        unroll_length=20,
        num_minibatches=32,
        num_updates_per_batch=8,
        discounting=0.95,
        learning_rate=3e-4,
        entropy_cost=0.001,
        num_envs=2048,
        batch_size=512,
        seed=3,
    )

    pbar = tqdm(total=num_timesteps)

    def progress(num_steps, metrics):
        reward = metrics["eval/episode_reward"]
        pbar.update(step_interval)
        pbar.set_description(f"reward: {reward}")

    make_inference_fn, params, _ = train_fn_ppo(environment=env, progress_fn=progress)
    inference_fn = make_inference_fn(params)

    print("End training")
    pbar.close()

    return inference_fn


def simulate(inference_fn, env_name, backend):
    print("Start simulation")
    env = envs.create(env_name=env_name, backend=backend)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)

    for _ in range(1000):
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
        print("Reward: ", state.reward)

    print("End simulation")


if __name__ == "__main__":
    env_name = "halfcheetah"
    backend = "generalized"

    inference_function = train_agent(env_name, backend)
    simulate(inference_function, env_name, backend)
