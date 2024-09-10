import math
import torch
import numpy as np
import gymnasium as gym


class HalfCheetah:
    def __init__(self):
        self.env = gym.make("HalfCheetah-v4", exclude_current_positions_from_observation=False)
        self.env.reset()
        self.action_low = -np.ones(self.env.action_space.shape)
        self.action_high = np.ones(self.env.action_space.shape)

    def dynamics(self, state, perturbed_action):
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        for i in range(state_np.shape[0]):
            self.env.set_state(state_np[i, :self.env.unwrapped.model.nq], state_np[i, self.env.unwrapped.model.nq:])
            state_np[i, :], _, _, _, _ = self.env.step(perturbed_action_np[i])
        #self.env.set_state(state[:self.env.unwrapped.model.nq], state[self.env.unwrapped.model.nq:])
        #state, _, _, _ = self.env.step(perturbed_action)
        state = torch.tensor(state_np, device=state.device, dtype=state.dtype)
        return state

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

    def train(self, new_data):
        pass
