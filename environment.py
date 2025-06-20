#Practice Environment
#This will serve as a practice kitchen for cooking up custom environments
#The goal is create an environment that is modeled as a logarithmic variable resistor
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
import random
from gym.utils import seeding


class prac_env_v0(gym.Env):    
    def __init__(self):
        super(prac_env_v0, self).__init__()
        self.action_list = [-50, -25, -10, -5, -2, -1, -0.5, -0.1, -0.01, 0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50]
        self.action_space = spaces.Discrete(len(self.action_list))
        # Expand observation space to include both state and target
        self.observation_space = spaces.Box(
            low=np.array([-375.0, -275.0]), 
            high=np.array([375.0, 275.0]),
            shape=(2,), dtype=np.float32
        )
        self.state = 0.0
        self.target = 0.0
        self.current = 0.0  # For compatibility with evaluation
        self.max_steps = 200  # Prevent infinite episodes
        self.step_count = 0

    def seed(self, seed=None):
        #Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        #Reset the environment to an initial state
        if seed is not None:
            self.seed(seed)

        # Start closer to center for easier learning initially
        self.state = self.np_random.uniform(-100.0, 100.0)
        self.target = self.np_random.uniform(-200.0, 200.0)
        self.current = self.target  # For compatibility with evaluation
        self.step_count = 0
        
        info = {
            'target': self.target,
            'initial_state': self.state
        }
        
        # Return both state and target as observation
        return np.array([self.state, self.target], dtype=np.float32), info
    
    def step(self, action):
        self.step_count += 1
        next_state = self.state + self.action_list[action]

        info = {
            'previous_state': self.state,
            'action_value': self.action_list[action],
            'target': self.target,
            'step_count': self.step_count
        }

        # Update state
        self.state = next_state

        # Calculate reward with better shaping
        distance = abs(self.state - self.target)
        
        # Calculate previous distance for reward shaping
        prev_distance = abs(info['previous_state'] - self.target) if 'previous_state' in info else distance
        
        # Reward shaping: positive reward for getting closer, negative for moving away
        distance_reward = prev_distance - distance  # Positive when getting closer
        
        # Scale the distance reward to make it more significant
        shaped_reward = distance_reward * 10
        
        terminated = False
        truncated = False
        
        if distance < 0.5:  # Success condition
            reward = 100 + shaped_reward  # Success bonus plus distance reward
            terminated = True
        elif abs(self.state) > 375.0:  # Out of bounds
            reward = -10 + shaped_reward  # Penalty but still consider progress
            terminated = True
        elif self.step_count >= self.max_steps:  # Episode timeout
            reward = -1 + shaped_reward  # Small penalty plus distance reward
            truncated = True
        else:
            # Give continuous feedback based on distance improvement
            reward = shaped_reward - 0.1  # Small step penalty to encourage efficiency

        return np.array([self.state, self.target], dtype=np.float32), reward, terminated, truncated, info

    def close(self):
        pass
