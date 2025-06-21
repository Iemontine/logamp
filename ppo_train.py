import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import set_random_seed
from environment import LogAmpEnvironment
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import time
import os
from stable_baselines3.common.callbacks import CheckpointCallback

# Set random seeds for reproducibility
SEED = 42
set_random_seed(SEED)

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./tensorboard_logs", exist_ok=True)

vec_env = make_vec_env(LogAmpEnvironment, n_envs=4, seed=SEED)

model = PPO(
    'MlpPolicy', 
    vec_env, 
    verbose=1,
    device="cpu",
    tensorboard_log="./tensorboard_logs/",
    seed=SEED
)

total_timesteps = 250_000

checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./models/",
    name_prefix="ppo_checkpoint"
)

print("Starting Training")
print(f"   Total timesteps: {int(total_timesteps):,}")
print(f"   Random seed: {SEED}")
print("=" * 60)
start_time = time.time()

model.learn(
    total_timesteps=total_timesteps,
    progress_bar=True,
    callback=checkpoint_callback,
)

training_time = time.time() - start_time
print(f"Training completed in {training_time:.2f} seconds")