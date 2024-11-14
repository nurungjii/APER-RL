import torch
import numpy as np

class ReplayBuffer:
    def __init__(self,state_size, buffer_size, window_length, device):
        self.state = torch.empty((buffer_size, *state_size), dtype=torch.int8).to(device)
        self.action = torch.empty(buffer_size, dtype=torch.long).to(device)
        self.reward = torch.empty(buffer_size, dtype=torch.float).to(device)
        self.done = torch.empty(buffer_size, dtype=torch.int).to(device)

        self.count = 0
        self.real_size = 0
        self.buff_size = buffer_size
        self.window_length = window_length
        self.device = device

    def add(self, state, action, reward, done):
        state = np.array(state, dtype=np.int8)

        self.state[self.count] = torch.from_numpy(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.buff_size
        self.real_size = min(self.buff_size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].unsqueeze(1).to(self.device),
            self.reward[sample_idxs].unsqueeze(1).to(self.device),
            self.state[[((x+1) % self.buff_size) for x in sample_idxs]].to(self.device),
            self.done[sample_idxs].unsqueeze(1).to(self.device),
        )
        return batch
