import torch
import random
import numpy as np

from PERs.SumTree import SumTree

class PrioritizedReplayBuffer():
    def __init__(self, state_size, buffer_size, window_length, device, end_ep, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)

        # Initialize PER params

        # Hard boundary for minimal priority -> don't let priority ever be 0
        self.eps = eps
        # Determine prioritization: a -> 0, samnpling -> uniform
        self.alpha = alpha
        # IS correction
        self.beta = beta
        self.initial_beta = beta
        # Max priority, also priority for new samples
        self.max_priority = eps
        self.end_ep = end_ep

        # A single experience constitutes (state, action, reward, next_state, reward)
        self.state = torch.empty((buffer_size, *state_size), dtype=torch.int8).to(device)
        self.action = torch.empty(buffer_size, dtype=torch.long).to(device)
        self.reward = torch.empty(buffer_size, dtype=torch.float).to(device)
        self.done = torch.empty(buffer_size, dtype=torch.int).to(device)

        self.count: int = 0
        self.real_size: int = 0
        self.buff_size: int = buffer_size
        self.window_length = window_length
        self.device = device

    def add(self, state, action, reward, done):

        self.tree.add(self.max_priority, self.count)

        state = np.array(state, dtype=np.int8)

        self.state[self.count] = torch.from_numpy(state)
        self.action[self.count] = torch.as_tensor(action)
        self.reward[self.count] = torch.as_tensor(reward)
        self.done[self.count] = torch.as_tensor(done)

        self.count = (self.count + 1) % self.buff_size
        self.real_size = min(self.buff_size, self.real_size + 1)

    def sample(self, batch_size):
        assert self.real_size >= batch_size, "Buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

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

        batch = (
            self.state[sample_idxs].to(self.device),
            self.action[sample_idxs].unsqueeze(1).to(self.device),
            self.reward[sample_idxs].unsqueeze(1).to(self.device),
            self.state[[((x+1) % self.buff_size) for x in sample_idxs]].to(self.device),
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

    def anneal_beta(self, curr_ep):
        self.beta = self.initial_beta + (1-self.initial_beta) * (curr_ep/self.end_ep)
