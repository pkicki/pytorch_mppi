from time import perf_counter
import torch
import dm_control.suite as suite
import shimmy
import numpy as np
import gymnasium as gym


class Acrobot:
    def __init__(self) -> None:
        self.env = gym.make("dm_control/acrobot-swingup-v0")
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.env.reset()

    def get_obs(self):
        return np.concat([self.env.unwrapped.physics.data.qpos, self.env.unwrapped.physics.data.qvel])

    def dynamics(self, state, perturbed_action):
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        nq = self.env.unwrapped.physics.model.nq
        nv = self.env.unwrapped.physics.model.nv
        #t0 = perf_counter()
        for i in range(state_np.shape[0]):
            with self.env.unwrapped.physics.reset_context():
                self.env.unwrapped.physics.data.qpos[:] = state_np[i, :nq]
                self.env.unwrapped.physics.data.qvel[:] = state_np[i, nq:nq+nv]
            _, _, _, _, _ = self.env.step(perturbed_action_np[i])
            state_np[i, :] = self.get_obs()
        #t1 = perf_counter()
        #print(t1 - t0)
        state = torch.tensor(state_np, device=state.device, dtype=state.dtype)
        return state

    def running_cost(self, state, action):
        q = state[:, :2]
        q_dot = state[:, 2:]
        return torch.sum(q ** 2, dim=-1) + 0.01 * torch.sum(q_dot ** 2, dim=-1)# + 0.01 * torch.sum(action ** 2, dim=-1)

    def train(self, new_data):
        pass

    @property
    def state_dim(self):
        return 4
    
    @property
    def action_dim(self):
        return 1
    
    @property
    def dt(self):
        return self.env.unwrapped.dt