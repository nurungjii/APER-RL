import torch
import random

from PERs.SumTree import SumTree

class PrioritizedReplayBuffer():
    def __init__(self, state_size, buffer_size, window_length, device, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)

        # Initialize PER params

        # Hard boundary for minimal priority -> don't let priority ever be 0
        self.eps = eps
        # Determine prioritization: a -> 0, samnpling -> uniform
        self.alpha = alpha
        # IS correction
        self.beta = beta
        # Max priority, also priority for new samples
        self.max_priority = eps

        # A single experience constitutes (state, action, reward, next_state, reward)
        self.state = torch.empty((buffer_size, *state_size), dtype=torch.int8).to(device)
        self.action = torch.empty(buffer_size, dtype=torch.long).to(device)
        self.reward = torch.empty(buffer_size, dtype=torch.float).to(device)
        self.next_state = torch.empty((buffer_size, *state_size), dtype=torch.int8).to(device)
        self.done = torch.empty(buffer_size, dtype=torch.int).to(device)

        self.count: int = 0
        self.real_size: int = 0
        self.buff_size: int = buffer_size
        self.window_length = window_length
        self.device = device

    def add(self, state, action, reward, next_state, done):

        self.tree.add(self.max_priority, self.count)

        self.state[self.count] = torch.as_tensor(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.next_state[self.count] = torch.as_tensor(next_state)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.buff_size
        self.real_size = min(self.buff_size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "Buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        states, next_states = [], []

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a,b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        probs = priorities / self.tree.total

        weights = (self.real_size * probs) ** -self.beta

        weights = weights / weights.max()

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
            self.done[sample_idxs].unsqueeze(1).to(self.device)
        )

        return batch, weights.to(self.device), tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = float(priority[0])
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
