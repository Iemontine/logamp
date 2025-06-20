import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from environment import prac_env_v0
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import deque
import time
import os

# Use the environment class directly with StableBaselines3
device = "cpu"
env = prac_env_v0()
eval_env = prac_env_v0()

# Initialize curriculum learning
env.target_min = -50.0
env.target_max = 50.0

print("🚀 Starting PPO Training for Target Reaching Task")
print(f"   Environment: Variable resistor simulation")
print(f"   Action space: {len(env.action_list)} discrete actions")
print(f"   Action values: {env.action_list}")
print("=" * 60)

# Improved PPO hyperparameters for better convergence
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=1,
    device=device,
    learning_rate=3e-4,       # Slightly higher learning rate for faster convergence
    n_steps=4096,             # Increased steps for more experience per update
    batch_size=128,           # Larger batch size for more stable gradients  
    tensorboard_log="./tensorboard_logs/"
)

print("🏃 Starting training...")
print(f"🔧 Model device: {model.device}")
start_time = time.time()

model.learn(
    total_timesteps=2500000,  # Increased for better convergence with curriculum learning
    progress_bar=True,
)

training_time = time.time() - start_time
print(f"\n✅ Training completed! Time taken: {training_time:.2f} seconds")

# Save the trained model for evaluation
model.save("ppo_model.zip")
print("💾 Model saved as ppo_model.zip")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"🖥️  Using device: {device}")
# if torch.cuda.is_available():
#     print(f"   GPU: {torch.cuda.get_device_name(0)}")
#     print(f"   CUDA Version: {torch.version.cuda}")
#     print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
#     print("=" * 60)