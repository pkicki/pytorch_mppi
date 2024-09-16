import torch
from time import perf_counter


class MuJoCo:
    def dynamics(self, state, perturbed_action):
        state_np = state.detach().numpy()
        perturbed_action_np = perturbed_action.detach().numpy()
        nq = self.env.unwrapped.model.nq
        nv = self.env.unwrapped.model.nv
        #t0 = perf_counter()
        for i in range(state_np.shape[0]):
            self.env.unwrapped.set_state(state_np[i, :nq], state_np[i, nq:nq+nv])
            state_np[i, :], _, _, _, _ = self.env.step(perturbed_action_np[i])
        #t1 = perf_counter()
        #print(t1 - t0)
        state = torch.tensor(state_np, device=state.device, dtype=state.dtype)
        return state

    @property
    def state_dim(self):
        return self.env.unwrapped.model.nq + self.env.unwrapped.model.nv
    
    @property
    def action_dim(self):
        return self.env.action_space.shape[0]

    @property
    def dt(self):
        return self.env.unwrapped.dt

    def train(self, new_data):
        pass