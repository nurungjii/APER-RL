import numpy as np
import torch
import copy
import wandb

from PERs.PER import PrioritizedReplayBuffer
from PERs.ReplayBuffer import ReplayBuffer
from PERs.APER import AnnealedPrioritizedReplayBuffer

class DQNAgent:
    def __init__(self,
                 q_network,
                 memory,
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size):
        
        self.q_network = q_network
        self.memory = memory
        self.policy = policy
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.num_burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size

    def compile(self, optimizer, device, env_name, run_num, start_ep=0, num_updates=0):

        # Set cuda device
        self.device = device

        # Deep copy the q_network to create the target network
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.to(self.device)
        self.optimizer = optimizer
        self.update_count = 0
        self.start_ep = start_ep
        self.num_updates = num_updates
        self.env_name = env_name
        self.run_num = run_num

    def calc_q_values(self, state):

        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(state)

        return q_values

    def select_action(self, state, **kwargs):
        # convert numpy array to tensor and of type float32
        state = np.array(state, dtype=np.float32)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        # state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        q_values = self.calc_q_values(state)

        return self.policy.select_action(q_values=q_values, is_training=True)

    def update_policy(self, batch, weights=None):
        # Sample experience
        states, actions, rewards, next_states, done = batch

        # Convert tensors to type float32
        states = states.type(torch.float32)
        next_states = next_states.type(torch.float32)
        done = done.type(torch.float32)

        rewards = torch.clamp(rewards, min=-1, max=1)

        # Calculate the q values and then pick the q values corresponding to each action
        q_expected = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Use the main q network to select the best actions
            best_actions_next = self.q_network(next_states).max(1)[1].unsqueeze(1)

            # Evaluate each action chosen by the q network using the target network
            q_targets_next = self.target_network(next_states).gather(1, best_actions_next)

            # Calculate target values
            target_q_values = rewards + (1 - done) * self.gamma * q_targets_next


        if weights is None:
            weights = torch.ones_like(q_expected)

        td_error = torch.abs(q_expected - target_q_values).detach()
        td_error = torch.clamp(td_error, min=-1.0, max=1.0)
        loss = torch.mean((q_expected - target_q_values)**2 * weights)

        # Perform gradient descent with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10)
        self.optimizer.step()

        # Maybe update target network
        self.hard_target_update()

        self.update_count += 1

        return loss.item(), td_error


    def hard_target_update(self):
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def fit(self, env, num_episodes, max_steps, max_episode_length=None, eval_interval=5000):
        total_steps = 0
        episode_rewards = []
        episode_losses = []

        # Reset the environment to grab new state and stack most recent frames
        state, _ = env.reset()

        # Perform burn in if necessary
        print("Burning in memory")
        while self.memory.real_size < self.num_burn_in:
            total_steps += 1
            if self.start_ep > 0:
                # If trained model already exists, use it to select corresponding action
                action = self.select_action(state)
            else:
                # If not, randomly pick action
                action = env.action_space.sample()

            next_state, reward, done, _, _ = env.step(action)
            
            # append the latest observed state and next state
            self.memory.add(state, action, reward, done)

            state = next_state

        print("Finished burning in")

        if self.start_ep != 0:
            print(f"starting from {self.start_ep}")
        for i in range(self.start_ep, num_episodes):

            episode_reward = 0
            episode_loss = 0
            steps_in_episode = 0
            done = False

            if total_steps > max_steps:
                break

            state, _ = env.reset()

            while not done:
                # select action and take step into env
                action = self.select_action(state)

                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward

                # append the latest observed state and next state
                self.memory.add(state, action, reward, done)

                # Update policy every 4 steps
                if total_steps % self.train_freq == 0 and total_steps > 0:
                    if isinstance(self.memory, ReplayBuffer):
                        batch = self.memory.sample(self.batch_size)
                        loss, td_error = self.update_policy(batch)
                    elif isinstance(self.memory, PrioritizedReplayBuffer):
                        batch, weights, tree_idxs = self.memory.sample(self.batch_size)
                        loss, td_error = self.update_policy(batch, weights=weights)
                        td_error = td_error.cpu().numpy()

                        self.memory.update_priorities(tree_idxs, td_error)
                        # self.memory.anneal_beta(i)
                    elif isinstance(self.memory, AnnealedPrioritizedReplayBuffer):
                        batch, weights, tree_idxs = self.memory.sample(self.batch_size, total_steps)
                        loss, td_error = self.update_policy(batch, weights=weights)
                        td_error = td_error.cpu().numpy()

                        self.memory.update_priorities(tree_idxs, td_error)
                        # self.memory.anneal_beta(i)
                    else:
                        raise RuntimeError("Unknown buffer")

                    episode_loss += loss if loss is not None else 0

                state = next_state
                steps_in_episode += 1
                total_steps += 1

                # break if we reach maximum length for episode
                if max_episode_length and steps_in_episode >= max_episode_length:
                    break

                if total_steps > max_steps:
                    break

            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss)

            wandb.log({
                "avg_loss": np.mean(episode_losses),
                "avg_reward": np.mean(episode_rewards),
                "ep_reward": episode_reward,
                "ep_loss": episode_loss,
                "curr steps": total_steps
            })

            # Evaluate every so often during training process
            if i % eval_interval == 0 and eval_interval > 0 and i > 0:
                print("Evaluating...")
                mean_reward, std_reward, _ = self.evaluate(env, num_episodes=100, max_episode_length=10000)

                wandb.log({
                    "eval_avg_reward": mean_reward
                })

                print(f"Epoch {i}, Avg Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")
                print("Finished evaluating.")

            # Observe Average Reward and Loss for Debugging
            # if i % 1000== 0 and i > 0 and self.memory.current_size >= self.num_burn_in:
            #     print(f"Avg Reward: {np.mean(episode_rewards):.2f}, Avg Loss: {np.mean(episode_losses):.2f}")

            # Checkpoint model
            if i%5000 == 0 and i != 0:
                print("Checkpointing model...")
                torch.save({
                    'epoch': self.num_updates,
                    'stop_ep_num': i,
                    'model_state_dict': self.q_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, f'models/ddqn_{type(self.memory)}_{self.env_name}_{self.run_num}.pth')

    def evaluate(self, env, num_episodes, max_episode_length=None, is_training=True):

        self.q_network.eval()

        all_rewards = []
        episode_lengths = []

        with torch.no_grad():
            for _ in range(num_episodes):
                # Grab states, preprocess, and initialize variables
                state, _ = env.reset()
                done = False
                total_reward = 0
                steps = 0

                while not done:
                    # User greedy policy to pick next action
                    state = np.array(state, dtype=np.float32)
                    state = torch.from_numpy(state).unsqueeze(0).to(self.device)
                    q_values = self.calc_q_values(state)
                    action = torch.argmax(q_values).item()

                    # Step in the environment using action
                    next_state, reward, done, _, _ = env.step(action)

                    total_reward += reward
                    state = next_state
                    steps += 1

                    if max_episode_length and steps >= max_episode_length:
                        break

                # print(f'Total Rewards for ep: {total_reward}')
                all_rewards.append(total_reward)
                episode_lengths.append(steps)

        avg_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        avg_episode_length = np.mean(episode_lengths)

        # Log results
        print(f"Num of Episodes: {num_episodes}\nAvg Reward: {avg_reward:.2f}\nAvg Episode Length: {avg_episode_length:.2f}\nStd Reward: {std_reward:.2f}")

        return avg_reward, std_reward, avg_episode_length
