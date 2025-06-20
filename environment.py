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
        self.action_list = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box(
            low=np.array([-375.0, -275.0]), 
            high=np.array([375.0, 275.0]),
            shape=(2,), dtype=np.float32
        )
        self.state = 0.0
        self.target = 0.0
        self.current = 0.0
        self.max_steps = 200
        self.step_count = 0

    def seed(self, seed=None):
        #Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        #Reset the environment to an initial state
        if seed is not None:
            self.seed(seed)        # Start closer to center for easier learning initially
        self.state = self.np_random.uniform(-100.0, 100.0)
        self.target = self.np_random.uniform(-200.0, 200.0)
        
        self.current = self.target  # For compatibility with evaluation
        self.step_count = 0
        
        info = {
            'target': self.target,
            'initial_state': self.state,
            'target_bounds': [self.target_min, self.target_max]
        }
        
        # Return both state and target as observation
        return np.array([self.state, self.target], dtype=np.float32), info
    
    def step(self, action):
        self.step_count += 1

        # Calculate error before and after action
        prev_error = abs(self.state - self.target)
        self.state = self.state + self.action_list[action]
        current_error = abs(self.state - self.target)

        info = {
            'previous_state': self.state,
            'action_value': self.action_list[action],
            'target': self.target,
            'step_count': self.step_count
        }

        if current_error < prev_error:
            reward = 1.0 + (current_error * 1e-3)
        else:
            reward = -1.0 + (current_error * 1e-3)

        # Check if we're close enough to target
        threshold = 2.0
        done = False
        if current_error <= threshold:
            done = True
            reward += 10.0
        
        # Check if we've exceeded maximum steps
        truncated = self.step_count >= self.max_steps
        
        return np.array([self.state, self.target], dtype=np.float32), reward, done, truncated, info

    def close(self):
        pass