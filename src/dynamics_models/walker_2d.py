import math
import torch
import numpy as np
import gymnasium as gym

from dynamics_models.gym_model import GymModel


class Walker2D(GymModel):
    def __init__(self, env):
        super(Walker2D, self).__init__(env)
        self.terminate_when_unhealthy = self.env.unwrapped._terminate_when_unhealthy

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

        x_velocity = state_np[:, self.env.unwrapped.model.nq]
        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity
        reward = forward_reward - control_cost + healthy_reward
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward