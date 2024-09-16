import numpy as np
import torch

from dynamics_models.gym_model import GymModel


class Swimmer(GymModel):
    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        x_velocity = state_np[:, self.env.unwrapped.model.nq]
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity

        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)

        reward = forward_reward - control_cost
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward