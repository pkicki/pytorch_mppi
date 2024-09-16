import math
from time import perf_counter
import torch
import numpy as np
import gymnasium as gym

from dynamics_models.mujoco import MuJoCo


class HalfCheetah(MuJoCo):
    def __init__(self):
        self.env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False)
        self.env.reset()
        self.action_low = -np.ones(self.env.action_space.shape)
        self.action_high = np.ones(self.env.action_space.shape)

    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        root_angle = state_np[:, 2]
        heading_penalty_factor = 10
        roll_penalty = (root_angle > math.pi/2) * heading_penalty_factor
        roll_penalty += (root_angle < -math.pi / 2) * heading_penalty_factor

        x_velocity = state_np[:, 9]
        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity
        reward = forward_reward - control_cost - roll_penalty
        return -reward