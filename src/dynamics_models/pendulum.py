import math
import torch
import numpy as np

# TODO move to utils
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


class Pendulum:
    action_low = -2.0
    action_high = 2.0

    def dynamics(self, state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -8, 8)
        newth = th + newthdot * dt

        state = torch.cat((newth, newthdot), dim=1)
        return state

    def running_cost(self, state, action):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        action = action[:, 0]
        #cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2 + 0.001 * (action**2)
        return cost

    def train(self, new_data):
        pass
