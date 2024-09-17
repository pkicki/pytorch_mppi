import math
import torch
import numpy as np

from dynamics_models.gym_model import GymModel


class HalfCheetah(GymModel):
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
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward