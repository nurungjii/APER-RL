import os
import gymnasium as gym
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing 
from gymnasium.wrappers.frame_stack import FrameStack
# import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import argparse
import wandb

from copy import deepcopy
from PERs.PER import PrioritizedReplayBuffer
from PERs.ReplayBuffer import ReplayBuffer
from agent import DQNAgent

device = torch.device('cuda')

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

# Define Linear Decay Greedy Policy
class LinearDecayGreedyEpsilonPolicy():
    def __init__(self, start_value, end_value, num_steps):
        self.start_value = start_value
        self.end_value = end_value
        self.num_steps = num_steps
        self.epsilon = start_value
        self.step_count = 0

    def select_action(self, **kwargs):
        if kwargs["is_training"]:
            if self.step_count < self.num_steps:
                decay_rate = (self.start_value - self.end_value) / self.num_steps
                self.epsilon = self.start_value - decay_rate * self.step_count
                self.step_count += 1
            else:
                self.epsilon = self.end_value

        if np.random.randn() < self.epsilon:
            return np.random.randint(len(kwargs["q_values"]))
        else:
            return torch.argmax(kwargs["q_values"]).item()

    def reset(self):
        self.epsilon = self.start_value
        self.step_count = 0

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Run DQN on Atari Env')
    parser.add_argument('--env', default='CartPole-v0', help='Atari env name')
    parser.add_argument('--buffer', default='ER', help='Atari env name')

    args = parser.parse_args()

    # Define env and wrap
    env = gym.make(args.env)
    env = AtariPreprocessing(env)
    env = FrameStack(env, num_stack=4)

    # Define observation and action sizes
    observation_size = env.observation_space.shape
    action_size = env.action_space.n

    dqn = DQN(observation_size, action_size).to(device) # DQN Network
    if args.buffer == 'PER': # Initalize Replay Buffer
        memory = PrioritizedReplayBuffer(observation_size[1:], 500000, 4, device, 100000)
    elif args.buffer == 'ER':
        memory = ReplayBuffer(observation_size[1:], 500000, 4, device)
    else:
        raise RuntimeError("Unkown buffer")
    policy = LinearDecayGreedyEpsilonPolicy(1.0, 0.1, 1000000) # Define policy
    gamma = 0.99 # gamma
    target_update_freq = 10000
    num_burn_in = 50000
    train_freq = 4
    batch_size = 32

    optimizer = optim.Adam(dqn.parameters(), 3e-4)

    num_updates = 0
    last_ep = 0

    # Load Model if it exists and resume training from last point
    if os.path.exists(f'models/ddqn_{type(memory)}.pth'):
        print("Loading existing model...")
        checkpoint = torch.load(f'models/ddqn_{type(memory)}.pth', map_location=device)
        dqn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        num_updates = checkpoint['epoch']
        last_ep = checkpoint['stop_ep_num']
        print(f"Resuming Training at episode {last_ep}")

    # Define learning agent
    dqn_agent = DQNAgent(dqn, memory, policy, gamma, target_update_freq, num_burn_in, train_freq, batch_size)
    dqn_agent.compile(optimizer, device, last_ep, num_updates)

    # Log results
    wandb.init(
        project=f"ddqn-ER",
        config = {
            "env": f"{args.env}"
        }
    )

    # Fit
    dqn_agent.fit(env, 10000000, 100000, 10000)
    torch.save(dqn.state_dict(), f"models/ddqn_{type(memory)}.path")

    dqn_agent.evaluate(env, 100, 10000)

    env.close()

if __name__ == '__main__':
    main()
