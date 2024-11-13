import gymnasium as gym
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch

from copy import deepcopy
from PERs.PER import PrioritizedReplayBuffer
from PERs.ReplayBuffer import ReplayBuffer

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )

        fc_input = self.feature_size(input_shape)

        self.fc_net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def feature_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv_net(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_net(x)
        return self.fc_net(x)
