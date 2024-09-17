import numpy as np
import torch
from dynamics_models.gym_model import GymModel


class HumanoidStandup(GymModel):
    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        uph_cost = (state_np[:, 2] - 0) / self.env.unwrapped.model.opt.timestep

        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)

        reward = uph_cost - control_cost + 1
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward