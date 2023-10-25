import functools

import jax
from brax import envs
from brax.training.agents.sac import train as sac


if __name__ == "__main__":
    env_name = "halfcheetah"
    backend = "generalized"

    env = envs.get_environment(env_name=env_name, backend=backend)
    state = jax.jit(env.reset)(rng=jax.random.PRNGKey(seed=0))

    # We determined some reasonable hyperparameters offline and share them here.
    train_fn = {
        "halfcheetah": functools.partial(
            sac.train,
            num_timesteps=1_000_000,
            num_evals=20,
            reward_scaling=1,
            episode_length=1000,
            normalize_observations=True,
            action_repeat=1,
            discounting=0.997,
            learning_rate=6e-4,
            num_envs=128,
            batch_size=128,
            grad_updates_per_step=128,
            max_devices_per_host=8,
            max_replay_size=1048576,
            min_replay_size=8192,
            seed=1,
        ),
    }[env_name]

    def progress(num_steps, metrics):
        for key in metrics:
            print(f"{key}: {metrics[key]}")
        print()

    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)
    inference_fn = make_inference_fn(params)

    # create an env with auto-reset
    env = envs.create(env_name=env_name, backend=backend)

    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)

    rollout = []
    rng = jax.random.PRNGKey(seed=1)
    state = jit_env_reset(rng=rng)

    for _ in range(1000):
        rollout.append(state.pipeline_state)
        act_rng, rng = jax.random.split(rng)
        act, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_env_step(state, act)
