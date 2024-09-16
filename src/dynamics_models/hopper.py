import numpy as np
import torch

from dynamics_models.gym_model import GymModel


class Hopper(GymModel):
    def __init__(self, env):
        super(Hopper, self).__init__(env)
        self.terminate_when_unhealthy = self.env.unwrapped._terminate_when_unhealthy

    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        x_velocity = state_np[:, self.env.unwrapped.model.nq]
        forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity

        control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)

        z, angle = state_np[:, 1], state_np[:, 2]
        state_ = state_np[:, 2:]

        min_state, max_state = self.env.unwrapped._healthy_state_range
        min_z, max_z = self.env.unwrapped._healthy_z_range
        min_angle, max_angle = self.env.unwrapped._healthy_angle_range


        healthy_state = np.all(np.logical_and(min_state < state_, state_ < max_state), axis=-1)
        healthy_z = np.logical_and(z > min_z, z < max_z)
        healthy_angle = np.logical_and(angle > min_angle, angle < max_angle)

        is_healthy = np.logical_and(healthy_z, healthy_angle)
        is_healthy = np.logical_and(is_healthy, healthy_state)

        if self.terminate_when_unhealthy:
            healthy_reward = self.env.unwrapped._healthy_reward
        else:
            healthy_reward = is_healthy.astype(np.float64) * self.env.unwrapped._healthy_reward

        reward = forward_reward + healthy_reward - control_cost
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward