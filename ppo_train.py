import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from environment import prac_env_v0
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import time
import os
from stable_baselines3.common.callbacks import CheckpointCallback

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./tensorboard_logs", exist_ok=True)

# Use vectorized environment for better, stable performance
vec_env = make_vec_env(prac_env_v0, n_envs=1)
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)# , clip_reward=10.0)

print("🚀 Starting PPO Training for Target Reaching Task")
print(f"   Environment: Variable resistor simulation, 4 parallel environments")
print("=" * 60)

# Simplified PPO hyperparameters for this simple problem
model = PPO(
    'MlpPolicy', 
    vec_env, 
    verbose=1,  # Enable verbose output to see training progress
    device="cpu",
    learning_rate=lambda remaining_progress: 3e-5 * remaining_progress,
    vf_coef=0.25,  # reduce if value loss dominates
    n_epochs=10,
    clip_range_vf=0.2,
    normalize_advantage=True,
    # learning_rate=3e-4,   # Standard learning rate
    # n_steps=1024,         # Shorter rollouts for simple problem
    # batch_size=64,        # Good batch size
    # n_epochs=4,           # Fewer epochs for simpler training
    # gamma=0.99,           # Reward discount factor
    # gae_lambda=0.95,      # GAE smoothing
    # clip_range=0.2,       # PPO clipping parameter
    # ent_coef=0.01,        # Entropy for exploration
    # vf_coef=0.5,          # Value function coefficient
    # max_grad_norm=0.5,    # Gradient clipping for stability
    # policy_kwargs=dict(
    #     net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),
    #     activation_fn=nn.ReLU
    # ),
    tensorboard_log="./tensorboard_logs/"
)

# Simplified training - no callbacks needed for this simple problem
total_timesteps = 1e5

# Create a checkpoint callback to save the model every 1e5 timesteps
checkpoint_callback = CheckpointCallback(
    save_freq=1e4,
    save_path="./models/",
    name_prefix="ppo_checkpoint"
)

print("🏃 Starting training...")
print(f"🔧 Model device: {model.device}")
start_time = time.time()

model.learn(
    total_timesteps=total_timesteps,
    progress_bar=True,
    callback=checkpoint_callback,
)

training_time = time.time() - start_time
print(f"\n✅ Training completed! Time taken: {training_time:.2f} seconds")