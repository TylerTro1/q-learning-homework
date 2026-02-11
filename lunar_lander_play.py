import gymnasium as gym
import argparse
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


def simple_run(teleop=False):

    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="human")
    obs, info = env.reset()

    done = False
    rewards = []
    while not done:
        # Control Branch for human teleop (Teleop=True, Heuristic=False)
        action = 0  # Default action = do nothing
        if teleop:
            pygame.event.pump()
            keys = pygame.key.get_pressed()

            if keys[pygame.K_LEFT]:
                action = 1  # Fire left orientation engine
            elif keys[pygame.K_UP]:
                action = 2  # Fire main engine
            elif keys[pygame.K_RIGHT]:
                action = 3  # Fire right orientation engine
            else:
                action = 0  # Default to doing nothing

        # Control Branch for Random Actions (Teleop=False, Heuristic=False)
        else:
            # Randomly Sample an Action
            action = env.action_space.sample()

        obs, rew, done, truncated, info = env.step(action)
        rewards.append(rew)
        env.render()

    print("Cumulative Reward:", sum(rewards))

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--teleop", action="store_true", default=False, help="Use the keyboard keys to control the movement")
    #If you don't specify --teleop it will default to a random policy

    args = params.parse_args()

    simple_run(args.teleop)



class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim))

        def forward(self, x):
            return self.net(x)
        
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # This "unpacks" the batch into separate tensors
    batch = list(zip(*transitions))
    
    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat(batch[3])
    done_batch = torch.cat(batch[4])

    # Compute Q(s_t, a) - the model's current guess
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
    
    # Compute the expected Q values (The Bellman Equation)
    expected_state_action_values = reward_batch + (gamma * next_state_values * (1 - done_batch))

    # Compute Huber loss (more robust to outliers than MSE)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

