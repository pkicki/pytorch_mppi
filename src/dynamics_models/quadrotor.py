import numpy as np
import torch

from dynamics_models.gym_model import GymModel

def hover_reward(observation, action, weights={'x': 1, 'v': 0.1, 'w': 0, 'u': 1e-5}):
    """
    Rewards hovering at (0, 0, 0). It is a combination of position error, velocity error, body rates, and 
    action reward.
    """

    # Compute the distance to goal
    dist_reward = -weights['x']*np.linalg.norm(observation[..., 0:3], axis=-1)

    # Compute the velocity reward
    vel_reward = -weights['v']*np.linalg.norm(observation[..., 3:6], axis=-1)

    # Compute the angular rate reward
    ang_rate_reward = -weights['w']*np.linalg.norm(observation[..., 10:13], axis=-1)

    # Compute the action reward
    action_reward = -weights['u']*np.linalg.norm(action, axis=-1)

    return dist_reward + vel_reward + action_reward + ang_rate_reward

class Quadrotor(GymModel):
    def dynamics(self, state, perturbed_action):
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        #t0 = perf_counter()
        for i in range(state_np.shape[0]):
            self.env.unwrapped.vehicle_state['x'] = state_np[i, :3]
            self.env.unwrapped.vehicle_state['v'] = state_np[i, 3:6]
            self.env.unwrapped.vehicle_state['q'] = state_np[i, 6:10]
            self.env.unwrapped.vehicle_state['wind'] = state_np[i, 10:13]
            self.env.unwrapped.vehicle_state['rotor_speeds'] = state_np[i, 13:17]
            #state_np[i, :], _, _, _, _ = self.env.step(perturbed_action_np[i])
            self.env.step(perturbed_action_np[i])
            state_np[i, :3] = self.env.unwrapped.vehicle_state['x']
            state_np[i, 3:6] = self.env.unwrapped.vehicle_state['v']
            state_np[i, 6:10] = self.env.unwrapped.vehicle_state['q']
            state_np[i, 10:13] = self.env.unwrapped.vehicle_state['wind']
            state_np[i, 13:17] = self.env.unwrapped.vehicle_state['rotor_speeds']

        #t1 = perf_counter()
        #print(t1 - t0)
        state = torch.tensor(state_np, device=state.device, dtype=state.dtype)
        return state, None


    def running_cost(self, state, action):
        state_np = state.detach().numpy()
        action_np = action.detach().numpy()

        # TODO implement this using the quadrotor model
        reward = hover_reward(state_np, action_np)

        #x_velocity = state_np[:, self.env.unwrapped.model.nq]
        #forward_reward = self.env.unwrapped._forward_reward_weight * x_velocity

        #control_cost = self.env.unwrapped._ctrl_cost_weight * np.sum(np.square(action_np), axis=-1)

        #reward = forward_reward - control_cost
        reward = torch.tensor(reward, device=state.device, dtype=state.dtype)
        return -reward

    @property
    def state_dim(self):
        dims = [len(v) for v in self.env.unwrapped.vehicle_state.values()]
        return np.sum(dims)
    
#    @property
#    def action_dim(self):
#        return self.env.action_space.shape[0]
#
    @property
    def dt(self):
        return self.env.unwrapped.t_step