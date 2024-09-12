from time import perf_counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import wandb

class RolloutDataset:
    def __init__(self) -> None:
        self.input_data = []
        self.output_data = []

    def append_data(self, states, actions):
        self.input_data.append(torch.concat([states[:-1], actions[:-1]], dim=-1))
        self.output_data.append(states[1:])

    def save_data(self, filename):
        input_data = torch.concat(self.input_data, dim=0)
        output_data = torch.concat(self.output_data, dim=0)
        dataset = TensorDataset(input_data, output_data)
        torch.save(dataset, filename)

    def load_data(self, filename):
        dataset = torch.load(filename)
        self.input_data = [dataset.tensors[0]]
        self.output_data = [dataset.tensors[1]]

    def __call__(self):
        input_data = torch.concat(self.input_data, dim=0)
        output_data = torch.concat(self.output_data, dim=0)
        return TensorDataset(input_data, output_data)
    


class NeuralModel:
    def __init__(self, mujoco_model, state_dim, action_dim, device, dtype):
        self.mujoco_model = mujoco_model
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, state_dim)
        ).to(device).to(dtype)

        self.optimizer = torch.optim.Adam(self.nn.parameters(), lr=5e-4)
        self.n_epochs = 1000
        self.n_trains = 0


    def dynamics(self, state, perturbed_action):
        state = state[..., :self.state_dim]
        #t0 = perf_counter()
        d_state = 0.1 * self.nn(torch.concat([state, perturbed_action], dim=-1))
        next_state = state + d_state
        #t1 = perf_counter()
        #print(t1 - t0)
        return next_state

    def running_cost(self, state, action):
        return self.mujoco_model.running_cost(state, action)

    def train(self, dataset):
        loader = DataLoader(dataset, batch_size=4)

        for e in range(self.n_epochs):
            epoch_loss = []
            for sample in loader:
                self.optimizer.zero_grad()
                input_data, output_data = sample
                prediction = self.nn(input_data)
                model_loss = torch.sum(torch.square(prediction - output_data), dim=-1)
                loss = model_loss.mean()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(model_loss)
            epoch_loss = torch.concatenate(epoch_loss, dim=0).mean()
            print(f"Epoch {e} Loss {epoch_loss}")
            wandb.log({"Epoch_loss": epoch_loss}, step=self.n_trains * self.n_epochs + e)
        self.n_trains += 1
