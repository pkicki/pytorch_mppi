import torch
import numpy as np
from time import perf_counter


class SingleTrack:
    def __init__(self, env):
        self.env = env
        self.env.unwrapped.render_mode = None
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.env.reset()

    def dynamics(self, state, perturbed_action):
        t0 = perf_counter()
        self.env.unwrapped.simulator.set_state(state.to(torch.float32))
        #state, rewards, _, _ = self.env.step(perturbed_action.to(torch.float32))
        state, rewards, _, _ = self.env.unwrapped.simulator.forward(perturbed_action.to(torch.float32)) 
        t1 = perf_counter()
        #print(t1 - t0)
        return state.to(torch.float64), -rewards.to(torch.float64)

    @property
    def state_dim(self):
        return self.env.unwrapped.state.shape[-1]
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]

    @property
    def dt(self):
        return self.env.unwrapped.dt

    def train(self, new_data):
        pass

    def running_cost(self, state, action):
        raise NotImplementedError