import functools

import jax
from brax import envs
from brax.training.agents.sac import train as sac
from tqdm import tqdm


def train_agent(env_name, backend):
    print("Start training")

    env = envs.get_environment(env_name=env_name, backend=backend)
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    num_timesteps = 100_000
    episode_length = 1000
    num_evals = 100
    step_interval = num_timesteps // num_evals

    train_fn = functools.partial(
        sac.train,
        num_timesteps=num_timesteps,
        episode_length=episode_length,
        num_evals=num_evals,
    )

    pbar = tqdm(total=num_timesteps)

    def progress(num_steps, metrics):
        reward = metrics["eval/episode_reward"]
        pbar.update(step_interval)
        pbar.set_description(f"reward: {reward}")

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
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
