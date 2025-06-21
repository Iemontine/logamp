#Practice Environment
#This will serve as a practice kitchen for cooking up custom environments
#The goal is create an environment that is modeled as a logarithmic variable resistor
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum


class LogAmpEnvironment(gym.Env):
    def __init__(self, max_steps=200, target_min=-1000.0, target_max=1000.0, start_min=-500.0, start_max=500.0):
        super(LogAmpEnvironment, self).__init__()

        # Define the possible actions and size of action space
        self.action_list = [-100, -10, -1, -0.1, 0.1, 1, 10, 100]
        self.action_space = spaces.Discrete(len(self.action_list))

        # Initialize parameters
        self.max_steps = max_steps
        self.target_min = target_min
        self.target_max = target_max
        self.start_min = start_min
        self.start_max = start_max

        print(f"Environment initialized with parameters:")
        print(f"   Max steps: {self.max_steps}")
        print(f"   Target bounds: [{self.target_min}, {self.target_max}]")
        print(f"   Start bounds: [{self.start_min}, {self.start_max}]")

        # Shape of the observation space, 2-dimensional: [current state, target state]
        self.observation_space = spaces.Box(
            low=np.array([self.start_min, self.target_min]),
            high=np.array([self.start_max, self.target_max]),
            shape=(2,), dtype=np.float64
        )

        # Initialize variables
        self.current = 0.0
        self.target = 0.0
        self.step_count = 0

    def reset(self, seed=None):
        # Reset the environment to an initial state
        super().reset(seed=seed)

        # Randomly select a target and starting position within the defined bounds
        self.target = self.np_random.uniform(self.target_min, self.target_max)
        self.current = self.np_random.uniform(self.start_min, self.start_max)

        # Reset the step count
        self.step_count = 0

        # print(f"Target: {self.target:.2f}, Initial State: {self.current:.2f}")

        # Create the observation, which is a 2D array with current state and target
        obs = np.array([self.current, self.target], dtype=np.float32)
        info = {
            'target': self.target,
            'target_bounds': [self.target_min, self.target_max],
            'state_bounds': [self.start_min, self.start_max],
            'initial_state': self.current,
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
        self.current = np.clip(self.current, self.start_min, self.start_max)
        self.step_count += 1

        # Check if target is reached
        distance = abs(self.current - self.target)
        target_reached = distance <= 0.5

        # Episode termination conditions
        terminated = target_reached
        truncated = self.step_count >= self.max_steps

        # Calculate reward
        reward = self.reward_function(self.current, self.target, terminated or truncated)

        # Create the observation, which is a 2D array with current state and target
        obs = np.array([self.current, self.target], dtype=np.float32)
        info = {
            'target': self.target,
        }

        # if terminated:
        #     if target_reached:
        #         print(f"Target reached in {self.step_count} steps! Final State: {self.current:.2f}, Target: {self.target:.2f}")
        #     else:
        #         print(f"Episode ended after {self.step_count} steps. Final State: {self.current:.2f}, Target: {self.target:.2f}")

        return obs, reward, terminated, truncated, info

    def close(self):
        pass