from dynamics_models.gym_model import GymModel
import numpy as np


class Go1(GymModel):
    def __init__(self, env, terminate_when_unhealthy=False):
        super(Go1, self).__init__(env)
        self.terminate_when_unhealthy = self.env.unwrapped._terminate_when_unhealthy

    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        x_velocity = state_np[:, self.env.unwrapped.model.nq]
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
