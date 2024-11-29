from time import perf_counter
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from pytorch_mppi.noises.colored import ColoredMultivariateNormal
from pytorch_mppi.noises.filtered import FilteredMultivariateNormal

class MPPI:
    def __init__(self, env, horizon, num_samples, control_dim, state_dim, noise_sigma,
                 noise_beta=None, noise_cutoff_freq=None, sampling_freq=None, lambda_=1.0, device='cpu'):
        self.env = env  # Environment
        self.dynamics = env.dynamics  # System dynamics function
        self.horizon = horizon  # Prediction horizon
        self.num_samples = num_samples  # Number of sampled control sequences
        self.lambda_ = lambda_  # Temperature parameter
        self.device = device
        self.control_dim = control_dim
        self.state_dim = state_dim
        self.U = torch.zeros((self.horizon, self.control_dim), device=self.device)
        self.costs = torch.zeros(self.num_samples, device=self.device)
        self.states = torch.zeros((self.num_samples, self.horizon + 1, self.state_dim), device=self.device)

        self.noise_mu = torch.zeros(self.control_dim)
        self.noise_sigma = noise_sigma
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        if noise_beta is not None:
            self.noise_dist = ColoredMultivariateNormal(self.noise_mu, noise_beta, covariance_matrix=self.noise_sigma)
        elif noise_cutoff_freq is not None:
            assert sampling_freq is not None
            self.noise_dist = FilteredMultivariateNormal(self.noise_mu, sampling_freq, noise_cutoff_freq, covariance_matrix=self.noise_sigma)
        #elif noise_interpolate_nodes is not None and self.interpolation_type == "cubic_noise":
        #    assert noise_interpolate_nodes > 1 # TODO check what is the minimum number of nodes
        #    assert noise_interpolate_nodes < horizon
        #    self.noise_dist = InterpolatedMultivariateNormal(self.noise_mu, horizon, covariance_matrix=self.noise_sigma)

    def reset(self, initial_controls=None):
        if initial_controls is not None:
            self.U = torch.tensor(initial_controls, device=self.device)
        else:
            self.U = torch.zeros((self.horizon, self.control_dim), device=self.device)

    def command(self, state, s=None):
        """
        Compute the optimal control sequence using MPPI.
        """
        # Sample perturbations
        perturbations = self.noise_dist.sample((self.num_samples, self.horizon))
        
        # Generate sampled control sequences
        control_samples = self.U.unsqueeze(0) + perturbations  # Shape: (num_samples, horizon, control_dim)
        control_samples = torch.clamp(control_samples, torch.tensor(self.env.action_low[None, None]),
                                      torch.tensor(self.env.action_high[None, None]))
        clamped_perturbations = control_samples - self.U.unsqueeze(0)
        
        # Simulate trajectories and compute costs
        self.costs = torch.zeros(self.num_samples, device=self.device)
        current_states = torch.tile(state[None], (self.num_samples, 1))
        self.states = torch.zeros((self.num_samples, self.horizon + 1, self.state_dim), device=self.device)
        self.states[:, 0] = current_states
        # hotfix for setting s to good value
        if s is not None:
            self.env.env.unwrapped.simulator.set_s(s)
        for t in range(self.horizon):
            current_control = control_samples[:, t]
            next_states, cost = self.dynamics(current_states, current_control)
            self.costs += cost
            current_states = next_states
            self.states[:, t + 1] = current_states

        
        # Compute weights
        min_cost = self.costs.min()
        weights = torch.exp(-1 / self.lambda_ * (self.costs - min_cost))
        weights /= weights.sum()  # Normalize weights
        
        # Update control sequence
        #weighted_control_update = (weights.unsqueeze(1).unsqueeze(2) * perturbations).sum(dim=0)
        weighted_control_update = (weights.unsqueeze(1).unsqueeze(2) * clamped_perturbations).sum(dim=0)
        self.U = self.U + weighted_control_update

        current_control = self.U[0]
        self.U = torch.cat([self.U[1:], torch.zeros((1, self.U.shape[1]), device=self.device)], dim=0)
        
        return current_control

def run_mppi(mppi, env, iter=1000, render=True):
    dataset = torch.zeros((iter, mppi.state_dim + mppi.control_dim), device=mppi.device)
    total_reward = 0
    mppi.reset(env.default_controls(mppi.horizon)[0])
    for i in range(iter):
        #print("Time step", i)
        state = env.get_state()
        if hasattr(env.unwrapped.simulator, "last_s"):
            action = mppi.command(state, s=env.unwrapped.simulator.last_s)
        else:
            action = mppi.command(state)
        res = env.step(action.cpu().numpy())

        s, r = res[0], res[1]
        total_reward += r
        if render:
            env.render(trajectories=mppi.states[:, :, :2], costs=mppi.costs)

        dataset[i, :mppi.state_dim] = torch.tensor(state, dtype=mppi.U.dtype)
        dataset[i, mppi.state_dim:] = action

    return total_reward, dataset