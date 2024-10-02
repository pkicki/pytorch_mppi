import math
import torch
import numpy as np
import gymnasium as gym

from dynamics_models.gym_model import GymModel


class Humanoid(GymModel):
    def __init__(self, env):
        super(Humanoid, self).__init__(env)
        self.terminate_when_unhealthy = self.env.unwrapped._terminate_when_unhealthy

    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        x_velocity = state_np[:, 24]
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity

        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)

        z = state_np[:, 2]
        min_z, max_z = self.env.unwrapped._healthy_z_range
        is_healthy = np.logical_and(z > min_z, z < max_z)
        if self.terminate_when_unhealthy:
            healthy_reward = self.env.unwrapped._healthy_reward
        else:
            healthy_reward = is_healthy.astype(np.float64) * self.env.unwrapped._healthy_reward

        reward = forward_reward + healthy_reward - control_cost
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward

from brax import envs
import jax
import jax.numpy as jnp
from time import perf_counter


class HumanoidBrax:
    def __init__(self, n_envs=100) -> None:
        self.env = envs.create('humanoid', backend='mjx')
        #self.env = envs.create('humanoid', backend='spring')
        rngs = jax.random.split(jax.random.PRNGKey(0), n_envs)
        self.init_state = jax.vmap(self.env.reset)(rngs)
        self.action_low = np.asarray(self.env.unwrapped.sys.actuator.ctrl_range[:, 0])
        self.action_high = np.asarray(self.env.unwrapped.sys.actuator.ctrl_range[:, 1])
        pipeline_state = self.env.pipeline_init(self.env.sys.init_q, np.zeros(self.env.sys.nv))
        self.state = self.env.reset(jax.random.PRNGKey(0))
        self.terminate_when_unhealthy = True
        nq = self.env.sys.nq
        nv = self.env.sys.nv

        @jax.jit
        def simulate(state, action):
            #state = jax.vmap(self.env.step)(self.init_state, np.random.rand(n_envs, self.env.action_size))
            state = jax.vmap(self.env.step)(state, action)
            state_jnp = jnp.concatenate([state.pipeline_state.q[:, :nq], state.pipeline_state.qd[:, :nv]], axis=-1)
            return state_jnp

        @jax.jit
        def run_batch(state, actions):
            #rngs = jax.random.split(jax.random.PRNGKey(seed), batch_size)  # pytype: disable=wrong-arg-types  # jax-ndarray
            #init_state = jax.vmap(init_fn)(rngs)

            @jax.vmap
            def run(state, actions):
                #def step(state, i):
                def step(x, _):
                    state, i = x
                    state = self.env.step(state, actions[i])
                    return (state, i+1), ()

                return jax.lax.scan(step, (state, 0), (), length=30)
                #return jax.lax.scan(step, state, 0, length=30)
            result = run(state, actions)
            return result[0][0]

        self.convert_jnp_to_np = lambda x: np.asarray(x)

        for i in range(10):
            t0 = perf_counter()
            simulate(self.init_state, np.random.rand(n_envs, self.env.action_size))
            #init_state = np.zeros((n_envs, self.env.sys.nq + self.env.sys.nv))
            #self.dynamics(torch.tensor(init_state), torch.zeros((n_envs, self.env.action_size)))
            t1 = perf_counter()
            print(t1 - t0)
        self.simulate = simulate
        self.run_batch = run_batch
        a = 0

    def rollout(self, state, actions):
        nq = self.env.sys.nq
        nv = self.env.sys.nv
        state_np = jnp.array(state.detach().numpy()[0])
        actions = jnp.array(actions.detach().numpy())
        init_states = self.init_state.tree_replace({"pipeline_state.q": state_np[:, :nq],
                                                    "pipeline_state.qd": state_np[:, nq:nq+nv]})
        result = self.run_batch(init_states, actions)
        a = result.pipeline_state.q.block_until_ready()
        return a

    def dynamics(self, state, perturbed_action):
        times = []
        #times.append(perf_counter())
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        nq = self.env.sys.nq
        nv = self.env.sys.nv
        states = self.init_state.tree_replace({"pipeline_state.q": state_np[:, :nq],
                                               "pipeline_state.qd": state_np[:, nq:nq+nv]})
        times.append(perf_counter())
        #state_np = self.simulate(states, perturbed_action_np).block_until_ready()
        a = self.simulate(states, perturbed_action_np).block_until_ready()
        #state_np = np.concatenate([new_states.pipeline_state.q, new_states.pipeline_state.qd], axis=-1)
        #state_np = np.concatenate([np.asarray(new_states.pipeline_state.q), np.asarray(new_states.pipeline_state.qd)], axis=-1)
        times.append(perf_counter())
        ##state_np = np.concatenate([new_states.pipeline_state.q, new_states.pipeline_state.qd], axis=-1)
        #state_np = np.concatenate([np.asarray(new_states.pipeline_state.q), np.asarray(new_states.pipeline_state.qd)], axis=-1)

        #with jax.disable_jit():
        #    state = torch.tensor(state_np._value, device=state.device, dtype=state.dtype)
        #state__ = state_np._value
        #state__ = np.asarray(state_np)
        state = torch.tensor(state, device=state.device, dtype=state.dtype)
        #times.append(perf_counter())
        #print(t1 - t0, t2 - t1, t3 - t2, t4 - t3)
        times = np.array(times)
        print(times[1:] - times[:-1])
        return state

    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        x_velocity = state_np[:, 24]
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity

        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)

        z = state_np[:, 2]
        min_z, max_z = self.env.unwrapped._healthy_z_range
        is_healthy = np.logical_and(z > min_z, z < max_z)
        if self.terminate_when_unhealthy:
            healthy_reward = self.env.unwrapped._healthy_reward
        else:
            healthy_reward = is_healthy.astype(np.float64) * self.env.unwrapped._healthy_reward

        reward = forward_reward + healthy_reward - control_cost
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward

    @property
    def state_dim(self):
        return self.env.unwrapped.sys.nq + self.env.unwrapped.sys.nv
    
    @property
    def action_dim(self):
        return self.env.action_size

    @property
    def dt(self):
        return self.env.unwrapped.dt

    def train(self, new_data):
        pass
    