import math
import torch
import numpy as np
import gymnasium as gym

from dynamics_models.mujoco import MuJoCo


class Humanoid(MuJoCo):
    def __init__(self):
        #self.terminate_when_unhealthy = True
        self.terminate_when_unhealthy = False
        self.env = gym.make("Humanoid-v4", exclude_current_positions_from_observation=False,
                            terminate_when_unhealthy=self.terminate_when_unhealthy)
        self.env.reset()
        self.action_low = -0.4 * np.ones(self.env.action_space.shape)
        self.action_high = 0.4 * np.ones(self.env.action_space.shape)

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