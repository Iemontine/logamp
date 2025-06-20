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
        self.action_list = [0.01, 0.1, 0.5, 1, 2, 5, 10, 25, 50, -0.01, -0.1, -0.5, -1, -2, -5, -10, -25, -50]
        self.action_space = spaces.Discrete(len(self.action_list))
        self.observation_space = spaces.Box( low = -375.0, high = 375.0,
                                             shape = (1,), dtype = float) #randomly chose the bounds of sys
        self.state = 0.0
        self.target = 0.0
        

    def seed(self, seed=None):
        #Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def reset(self, seed=None, options=None):
        #Reset the environment to an initial state
        if seed is not None:
            self.seed(seed)

        self.state = self.np_random.uniform(-375.0,375.0)
        self.target = self.np_random.uniform(-275,275)
        
        info = {
            'target': self.target,
            'initial_state': self.state
        }
        
        return np.array([self.state], dtype=np.float32), info
      def step(self,action):
        next_state = self.state + self.action_list[action]

        info = {
            'previous_state': self.state,
            'action_value': self.action_list[action],
            'target': self.target
        }
        done = False
        reward = 0

        # Update state
        self.state = next_state

        # Calculate reward based on distance to target
        distance = abs(self.state - self.target)
        reward = -distance  # Negative reward for being far from target

        # Check if episode should end
        if distance < 1.0:  # Close enough to target
            reward += 100  # Bonus for reaching target
            done = True
        elif abs(self.state) > 375.0:  # Out of bounds
            reward = -1000  # Large penalty for going out of bounds
            done = True

        terminated = done
        truncated = False  # Add truncated flag for new gymnasium format

        return np.array([next_state], dtype=np.float32), reward, terminated, truncated, info


    def render(self, mode = 'human'):
        print(f'State: {self.state}')
    

    def close(self):
        pass