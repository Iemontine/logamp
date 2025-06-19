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
        # Renderer state
        self._render_mode = False
        self.episode_history = []
        self.current_episode = []

        #How many possible actions
        self.action_space = spaces.Discrete(6)

        #Shape of observation space
        self.observation_space = spaces.Box( low = -375.0, high = 375.0,
                                             shape = (1,), dtype = float) #randomly chose the bounds of sys
        
        #Possible actions
        self.action_list = [1, 10, 100, -1, -10, -100]

        #Action space, how the environment will interpret the actions
        self.action_space = spaces.Discrete(len(self.action_list))
        print(f"Debug: {self.action_space}")
        


    def seed(self, seed=None):
        # Seed is used to generate random numbers, used for reproducibility
        self.np_random, seed = seeding.np_random(seed)
        # Every time we run algorithm, we generate 1 new seed. This seed generates a random current,
        # which is not changed for the running of the code. It does change every time we run the code.
        self.current = self.np_random.uniform(-275, 275)
        return [seed]    # reset environment (Gymnasium API)
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        # Resetting system will give the initial state of the system a random state
        self.state = self.np_random.uniform(-375.0, 375.0)
        # Randomize self.current for each episode
        self.current = self.np_random.uniform(-275, 275)
        # Reset step counter
        self.step_count = 0
        # Reseting reward
        self.cumulative_reward = 0
        if self._render_mode:
            if self.current_episode:
                self.episode_history.append(self.current_episode)
            self.current_episode = []
        print(f"=================RESET!! Target: {self.current:.2f} ==================")
        return np.array([self.state], dtype=float), {}
      # time step (Gymnasium API)
    def step(self, action):
        info = {}
        terminated = False
        truncated = False
        
        # current and initial state of system
        current = self.current
        current_state = self.state
        
        # Where algorithm actually makes its decision
        direction = self.action_list[action]
        
        # Resulting state due to action
        next_state = current_state + direction
        
        # Improved reward system - goal is to get close to current target
        distance_before = abs(current_state - current)
        distance_after = abs(next_state - current)
        
        # Base reward for getting closer to target
        if distance_after < distance_before:
            reward = 1.0  # Good! Moving closer
        elif distance_after > distance_before:
            reward = -0.5  # Bad! Moving away
        else:
            reward = -0.1  # Neutral, small penalty for time
            
        # Bonus for being very close to target
        if distance_after < 10:
            reward += 10.0
            terminated = True  # Success!
        elif distance_after < 50:
            reward += 1.0  # Good progress
            
        # Penalty for going out of bounds, but don't terminate immediately
        if abs(next_state) > 374:
            reward = -5.0
            terminated = True
            
        # Add episode length limit
        self.step_count = getattr(self, 'step_count', 0) + 1
        if self.step_count >= 200:  # Max episode length
            truncated = True
            
        if not terminated and not truncated:
            self.state = next_state
            
        # Don't accumulate rewards, just return step reward
        reward = reward        # If rendering, record the step
        if self._render_mode:
            self.current_episode.append({
                'state': current_state,
                'action': action,
                'direction': direction,
                'next_state': next_state,
                'distance_to_target': abs(next_state - current),
                'reward': reward,
                'terminated': terminated,
                'truncated': truncated
            })
        if terminated:
            if distance_after < 10:
                print(f"SUCCESS! Reached target {current:.2f} from {next_state:.2f}")
            else:
                print(f"OUT OF BOUNDS: {next_state:.2f}")
            print("---------------------DONE------------------------")
        return np.array([next_state], dtype=float), reward, terminated, truncated, info

    #Inteface output 
    def render(self, mode = 'human'):
        """
        If called during evaluation, prints a summary of the episode so far.
        Set env._render_mode = True before evaluation to enable recording.
        """
        if not self._render_mode:
            print(f'State: {self.state}')
            return        if self.current_episode:
            print(f"\n--- Episode Summary (Target: {self.current:.2f}) ---")
            for i, step in enumerate(self.current_episode):
                action_desc = f"{step['direction']:+d}" if 'direction' in step else f"Action {step['action']}"
                dist = step.get('distance_to_target', 'N/A')
                print(f"Step {i}: {step['state']:.1f} → {action_desc} → {step['next_state']:.1f} | Dist: {dist:.1f} | Reward: {step['reward']:.1f}")
            print(f"Final State: {self.state:.2f} | Target: {self.current:.2f}")
            print("----------------------\n")
    
    #close any open resources used by environment
    def close(self):
        pass
