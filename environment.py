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
        # Smaller, more controlled action list for stable learning
        self.action_list = [-50, -20, -10, -5, -2, -1, -0.5, -0.1, 0.1, 0.5, 1, 2, 5, 10, 20, 50]
        self.action_space = spaces.Discrete(len(self.action_list))
        
        # State bounds
        self.state_min = -400.0
        self.state_max = 400.0
        self.target_min = -250.0
        self.target_max = 250.0

        # Bounds to encourage staying in reasonable range
        self.observation_space = spaces.Box(
            low=np.array([self.state_min, self.target_min]), 
            high=np.array([self.state_max, self.target_max]),
            shape=(2,), dtype=np.float32
        )
        
        self.current = 0.0
        self.target = 0.0
        self.step_count = 0
        self.max_steps = 200

    def seed(self, seed=None):
        #Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, seed=None, options=None):
        #Reset the environment to an initial state
        if seed is not None:
            self.seed(seed)
       
        self.target = self.np_random.uniform(self.target_min, self.target_max)      # Start closer to target for easier learning
        self.current = self.np_random.uniform(self.state_min, self.state_max)
        self.step_count = 0
        self.best_error = abs(self.current - self.target)
        
        info = {
            'target': self.target,
            'initial_state': self.current,
            'target_bounds': [self.target_min, self.target_max],
            'initial_error': self.best_error
        }

        # Return both state and target as observation
        return np.array([self.current, self.target], dtype=np.float32), info
    
    def reward_function(self, x, x_goal, done):
        ε = 0.5
        distance = abs(x - x_goal)
        if done and distance < ε:
            return 100  # Strong goal bonus
        return -1 - np.log(1 + distance)
    
    def step(self, action):
        self.step_count += 1

        # Perform action, getting errors
        # prev_error = abs(self.current - self.target)
        self.current += self.action_list[action]
        self.current = np.clip(self.current, self.state_min, self.state_max)    # Clip state to bounds
        # current_error = abs(self.current - self.target)

        done = self.step_count >= self.max_steps
        truncated = False
        reward = self.reward_function(self.current, self.target, done)

        info = {
            'target': self.target,
            'step_count': self.step_count,
            'max_steps': self.max_steps,
            'current_error': abs(self.current - self.target),
            'best_error': self.best_error
        }

        return np.array([self.current, self.target], dtype=np.float32), reward, done, truncated, info

    def close(self):
        pass