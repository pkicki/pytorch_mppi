from dynamics_models.mujoco import MuJoCo
import numpy as np


class GymModel(MuJoCo):
    def __init__(self, env):
        self.env = env
        self.env.unwrapped.render_mode = None
        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high
        self.env.reset()

    def running_cost(self, state, action):
        raise NotImplementedError