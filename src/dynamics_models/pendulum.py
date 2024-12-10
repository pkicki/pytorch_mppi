import math
import torch
import numpy as np

# TODO move to utils
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


class Pendulum:
    action_low = np.array([-2.0], dtype=np.float32)
    action_high = np.array([2.0], dtype=np.float32)
    dt = 0.05

    def dynamics(self, state, perturbed_action):
        # true dynamics from gym
        cos_th = state[:, 0].view(-1, 1)
        sin_th = state[:, 1].view(-1, 1)
        thdot = state[:, 2].view(-1, 1)
        th = np.arctan2(sin_th, cos_th)

        cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (perturbed_action**2)
        cost = cost.squeeze(1)

        g = 10
        m = 1
        l = 1

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * Pendulum.dt
        newthdot = np.clip(newthdot, -8, 8)
        newth = th + newthdot * self.dt


        #state = torch.cat((newth, newthdot), dim=1)
        state = torch.cat((np.cos(newth), np.sin(newth), newthdot), dim=1)
        return state, cost

    def train(self, new_data):
        pass

    @property
    def state_dim(self):
        return 3
    
    @property
    def action_dim(self):
        return 1

