import math
import torch
import numpy as np
import gymnasium as gym


class Humanoid:
    def __init__(self):
        #self.terminate_when_unhealthy = True
        self.terminate_when_unhealthy = False
        self.env = gym.make("Humanoid-v4", exclude_current_positions_from_observation=False,
                            terminate_when_unhealthy=self.terminate_when_unhealthy)
        self.env.reset()
        self.action_low = -0.4 * np.ones(self.env.action_space.shape)
        self.action_high = 0.4 * np.ones(self.env.action_space.shape)

    def dynamics(self, state, perturbed_action):
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        nq = self.env.unwrapped.model.nq
        nv = self.env.unwrapped.model.nv
        for i in range(state_np.shape[0]):
            self.env.set_state(state_np[i, :nq],
                               state_np[i, nq:nq+nv])
            state_np[i, :], _, _, _, _ = self.env.step(perturbed_action_np[i])
        state = torch.tensor(state_np, device=state.device, dtype=state.dtype)
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
        return -reward

    def train(self, new_data):
        pass
