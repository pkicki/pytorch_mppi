import torch
import numpy as np
import jax
import jax.numpy as jnp
from time import perf_counter


class Brax:
    def __init__(self, env, n_envs=100) -> None:
        self.env = env
        rngs = jax.random.split(jax.random.PRNGKey(0), n_envs)
        self.init_state = jax.vmap(self.env.reset)(rngs)
        self.action_low = np.asarray(self.env.unwrapped.sys.actuator.ctrl_range[:, 0])
        self.action_high = np.asarray(self.env.unwrapped.sys.actuator.ctrl_range[:, 1])
        self.terminate_when_unhealthy = True
        nq = self.env.sys.nq
        nv = self.env.sys.nv

        def pipeline_init(q, qd):
            return self.env._pipeline.init(self.env.sys, q, qd)
        self.pipeline_init = pipeline_init

        @jax.jit
        def simulate(state, action):
            pipeline_state = jax.vmap(self.pipeline_init)(state[:, :nq], state[:, nq:nq+nv])
            states = self.init_state.replace(pipeline_state=pipeline_state)
            #state = jax.vmap(self.env.step)(self.init_state, np.random.rand(n_envs, self.env.action_size))
            state = jax.vmap(self.env.step)(states, action)
            state_jnp = jnp.concatenate([state.pipeline_state.q[:, :nq], state.pipeline_state.qd[:, :nv]], axis=-1)
            reward = state.reward
            if self.terminate_when_unhealthy:
                done = state.done
                reward = jnp.where(done, -1e5, reward)
            return state_jnp, reward

        self.convert_jnp_to_np = lambda x: np.asarray(x)

        for i in range(10):
            t0 = perf_counter()
            state_jnp = jnp.concatenate([self.init_state.pipeline_state.q[:, :nq],
                                         self.init_state.pipeline_state.qd[:, :nv]], axis=-1)
            simulate(state_jnp, np.random.rand(n_envs, self.env.action_size))
            #init_state = np.zeros((n_envs, self.env.sys.nq + self.env.sys.nv))
            #self.dynamics(torch.tensor(init_state), torch.zeros((n_envs, self.env.action_size)))
            t1 = perf_counter()
            print(t1 - t0)
        self.simulate = simulate

    def dynamics(self, state, perturbed_action):
        times = []
        #times.append(perf_counter())
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        nq = self.env.sys.nq
        nv = self.env.sys.nv
        #states = self.init_state.tree_replace({"pipeline_state.q": state_np[:, :nq],
        #                                       "pipeline_state.qd": state_np[:, nq:nq+nv]})
        #pipeline_state = self.env._pipeline.init(self.env.sys, state_np[:, :nq], state_np[:, nq:nq+nv], None, False)
        #states = self.init_state.replace(pipeline_state=pipeline_state)
        times.append(perf_counter())
        # transform actions to [-1,1] range required by brax environments
        perturbed_action_np = (perturbed_action_np - self.action_low) / (self.action_high - self.action_low) * 2 - 1
        state_jnp, reward = self.simulate(state_np, perturbed_action_np)
        state_jnp = state_jnp.block_until_ready()
        #state_jnp = self.simulate(states, perturbed_action_np).block_until_ready()
        times.append(perf_counter())
        state = torch.tensor(state_jnp._value, device=state.device, dtype=state.dtype)
        reward = torch.tensor(reward._value, device=state.device, dtype=state.dtype)
        times.append(perf_counter())
        #times = np.array(times)
        #print(times[1:] - times[:-1])
        return state, -reward

    @property
    def state_dim(self):
        return self.env.unwrapped.sys.nq + self.env.unwrapped.sys.nv
    
    @property
    def action_dim(self):
        return self.env.action_size

    @property
    def dt(self):
        return self.env.unwrapped.dt