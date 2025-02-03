import math
import torch
import numpy as np

# TODO move to utils
def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)


class Pendulum:
    def __init__(self, g=10., l=1., m=1., max_torque=2.0, dt=0.05):
        self.action_low = np.array([-max_torque], dtype=np.float32)
        self.action_high = np.array([max_torque], dtype=np.float32)
        self.dt = dt
        t = 0
        self.l = l
        self.m = m
        self.g = g
        self.max_torque = max_torque

    def dynamics(self, state, perturbed_action):
        # true dynamics from gym
        cos_th = state[:, 0].view(-1, 1)
        sin_th = state[:, 1].view(-1, 1)
        thdot = state[:, 2].view(-1, 1)
        t = state[:, 3].view(-1, 1)
        th = np.arctan2(sin_th, cos_th)

        #cost = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (perturbed_action**2)
        #cost = abs(angle_normalize(th)) + 0.1 * abs(thdot) + 0.001 * abs(perturbed_action)
        A = 0.3
        f = 2.0
        th_desired = A * np.sin(2 * np.pi * f * t) + np.pi
        thdot_desired = A * 2 * np.pi * f * np.cos(2 * np.pi * f * t)
        #cost = np.square(angle_normalize(th - th_desired)) + 0.1 * np.square(thdot - thdot_desired) + 0.01 * np.square(perturbed_action)
        #cost = np.square(angle_normalize(th - th_desired)) + 0.1 * np.square(thdot - thdot_desired) + 0.001 * np.square(perturbed_action)
        #cost = 1. * np.maximum(0., np.abs(angle_normalize(th - th_desired)) - 0.01) + 0.001 * np.square(perturbed_action)
        #cost = cost.squeeze(1)

        q = angle_normalize(th)
        #qt = torch.tensor([[A, 0.125], [-A, 0.375], [A, 0.625], [-A, 0.875]]) + np.array([[0., 0.]])
        qt = torch.tensor([[A, 0.125], [-A, 0.375], [A, 0.625], [-A, 0.875], [A, 1.125], [-A, 1.375], [A, 1.625], [-A, 1.875]]) + np.array([[0., 0.]])
        #qt = torch.tensor([[A, 0.125], [-A, 0.375], [A, 0.625], [-A, 0.875]]) + np.array([[np.pi, 0.]])
        diff_t = t - qt[:, 1]
        diff_q = angle_normalize(q - qt[:, 0])
        dists = torch.sqrt(diff_t ** 2 + diff_q ** 2)
        min_dist = torch.min(dists, axis=-1)[0]
        cost = torch.where(min_dist < 0.05, -1., 0.)
        cost = cost + 0.001 * np.square(perturbed_action[:, 0])

        #distrubance = 1.0 * np.sin(Pendulum.t * 2 * np.pi)
        #perturbed_action = perturbed_action + distrubance
        #Pendulum.t += Pendulum.dt
        
        u = perturbed_action
        u = torch.clamp(u, -self.max_torque, self.max_torque)

        newthdot = thdot + (3 * self.g / (2 * self.l) * np.sin(th) + 3.0 / (self.m * self.l ** 2) * u) * self.dt
        #newthdot = np.clip(newthdot, -8, 8)
        newth = th + newthdot * self.dt


        #state = torch.cat((newth, newthdot), dim=1)
        state = torch.cat((np.cos(newth), np.sin(newth), newthdot, t + self.dt), dim=1)
        return state, cost

    def train(self, new_data):
        pass

    @property
    def state_dim(self):
        return 4
    
    @property
    def action_dim(self):
        return 1

