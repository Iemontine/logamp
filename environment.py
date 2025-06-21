#Practice Environment
#This will serve as a practice kitchen for cooking up custom environments
#The goal is create an environment that is modeled as a logarithmic variable resistor
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from gym.utils import seeding


class LogAmpEnvironment(gym.Env):
    def __init__(self):
        super(LogAmpEnvironment, self).__init__()

        # Define the possible actions and the action space
        self.action_list = [-100, -10, -1, -0.1, 0.1, 1, 10, 100]
        self.action_space = spaces.Discrete(len(self.action_list))

        # Initialize parameters
        self.max_steps = 200
        self.state_min = -400.0
        self.state_max = 400.0
        self.target_min = -250.0
        self.target_max = 250.0

        # Shape of the observation space, 2-dimensional: [current state, target state]
        self.observation_space = spaces.Box(
            low=np.array([self.state_min, self.target_min]),
            high=np.array([self.state_max, self.target_max]),
            shape=(2,), dtype=np.float64
        )

        # Initialize variables
        self.current = 0.0
        self.target = 0.0
        self.step_count = 0

    def seed(self, seed=None):
        # Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state
        self.seed(seed)

        # Randomly select a target and starting position within the defined bounds
        self.target = self.np_random.uniform(self.target_min, self.target_max)
        self.current = self.np_random.uniform(self.state_min, self.state_max)

        # Reset the step count
        self.step_count = 0

        # print(f"Target: {self.target:.2f}, Initial State: {self.current:.2f}")

        # Create the observation, which is a 2D array with current state and target
        obs = np.array([self.current, self.target], dtype=np.float32)
        info = {
            'target': self.target,
            'target_bounds': [self.target_min, self.target_max],
            'initial_state': self.current,
            'state_bounds': [self.state_min, self.state_max],
        }

        return obs, info

    def reward_function(self, x, x_goal, done):
        distance = abs(x - x_goal)

        threshold = 0.5

        if distance <= threshold:
            return 10.0
        else:
            return -1 if done else 0

    def step(self, action):
        # Perform action, clipping it to stay within the state bounds
        self.current += self.action_list[action]
        self.current = np.clip(self.current, self.state_min, self.state_max)
        self.step_count += 1

        # Check if target is reached
        distance = abs(self.current - self.target)
        target_reached = distance <= 0.5

        # Episode termination conditions
        done = self.step_count >= self.max_steps or target_reached
        truncated = False

        # Calculate reward
        reward = self.reward_function(self.current, self.target, done)

        # Create the observation, which is a 2D array with current state and target
        obs = np.array([self.current, self.target], dtype=np.float32)
        info = {
            'target': self.target,
        }

        # if done:
        #     if target_reached:
        #         print(f"Target reached in {self.step_count} steps! Final State: {self.current:.2f}, Target: {self.target:.2f}")
        #     else:
        #         print(f"Episode ended after {self.step_count} steps. Final State: {self.current:.2f}, Target: {self.target:.2f}")

        return obs, reward, done, truncated, info

    def close(self):
        pass