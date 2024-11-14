import torch
import numpy as np

class ReplayBuffer:
    def __init__(self,state_size, buffer_size, window_length, device):
        self.state = torch.empty((buffer_size, *state_size), dtype=torch.int8).to(device)
        self.action = torch.empty(buffer_size, dtype=torch.long).to(device)
        self.reward = torch.empty(buffer_size, dtype=torch.float).to(device)
        self.next_state = torch.empty((buffer_size, *state_size), dtype=torch.int8).to(device)
        self.done = torch.empty(buffer_size, dtype=torch.int).to(device)

        self.count = 0
        self.real_size = 0
        self.buff_size = buffer_size
        self.window_length = window_length
        self.device = device

    def add(self, state, action, reward, next_state, done):
        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.buff_size
        self.real_size = min(self.buff_size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size

        sample_idxs = np.random.choice(self.real_size, batch_size, replace=False)
        states, next_states = [], []

        for idx in sample_idxs:
            state_frames = []
            next_state_frames = []

            for offset in range(self.window_length):
                frame_idx = (idx - offset) % self.real_size

                if self.state[frame_idx] is None or (offset > 0 and self.done[frame_idx]):
                    state_frames.insert(0, torch.zeros_like(self.state[idx]))
                else:
                    state_frames.insert(0, self.state[frame_idx])

                if self.next_state[frame_idx] is None or (offset > 0 and self.done[frame_idx]):
                    next_state_frames.insert(0, torch.zeros_like(self.next_state[idx]))
                else:
                    next_state_frames.insert(0, self.next_state[frame_idx])

            states.append(torch.stack(state_frames, dim=0))
            next_states.append(torch.stack(next_state_frames, dim=0))

        batch = (
            torch.stack(states).to(self.device),
            self.action[sample_idxs].unsqueeze(1).to(self.device),
            self.reward[sample_idxs].unsqueeze(1).to(self.device),
            torch.stack(next_states).to(self.device),
            self.done[sample_idxs].unsqueeze(1).to(self.device),
        )
        return batch
