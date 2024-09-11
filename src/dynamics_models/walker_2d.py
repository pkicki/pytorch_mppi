import math
import torch
import numpy as np
import gymnasium as gym


class Walker2D:
    def __init__(self):
        #self.terminate_when_unhealthy = True
        self.terminate_when_unhealthy = False
        self.env = gym.make("Walker2d-v4", exclude_current_positions_from_observation=False,
                            terminate_when_unhealthy=self.terminate_when_unhealthy)
        self.env.reset()
        self.action_low = -np.ones(self.env.action_space.shape)
        self.action_high = np.ones(self.env.action_space.shape)

    def dynamics(self, state, perturbed_action):
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        for i in range(state_np.shape[0]):
            self.env.set_state(state_np[i, :self.env.unwrapped.model.nq], state_np[i, self.env.unwrapped.model.nq:])
            state_np[i, :], _, _, _, _ = self.env.step(perturbed_action_np[i])
        state = torch.tensor(state_np, device=state.device, dtype=state.dtype)
        return state

    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        z, angle = state_np[:, 1], state_np[:, 2]

        min_z, max_z = self.env.unwrapped._healthy_z_range
        min_angle, max_angle = self.env.unwrapped._healthy_angle_range

        healthy_z = np.logical_and(z > min_z, z < max_z)
        healthy_angle = np.logical_and(angle > min_angle, angle < max_angle)
        is_healthy = np.logical_and(healthy_z, healthy_angle)

        if self.terminate_when_unhealthy:
            healthy_reward = self.env.unwrapped._healthy_reward
        else:
            healthy_reward = is_healthy.astype(np.float64) * self.env.unwrapped._healthy_reward

        x_velocity = state_np[:, 9]
        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity
        reward = forward_reward - control_cost + healthy_reward
        return -reward

    def train(self, new_data):
        pass