import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from environment import prac_env_v0

# Use the environment class directly with StableBaselines3
env = prac_env_v0()
eval_env = prac_env_v0()

import torch.nn as nn

# Improved PPO hyperparameters for precise targeting
model = PPO(
    'MlpPolicy', 
    env, 
    verbose=0,
#     learning_rate=1e-4,   # Lower learning rate for more stable learning
#     n_steps=1024,         # Smaller steps for more frequent updates
#     batch_size=32,        # Smaller batch for better gradient estimates
#     n_epochs=20,          # More epochs to learn from each batch
#     gamma=0.995,          # Higher gamma to care more about long-term rewards
#     gae_lambda=0.98,      # Higher GAE lambda for better credit assignment
#     clip_range=0.1,       # Smaller clip range for more conservative updates
#     ent_coef=0.001,       # Lower entropy to focus on exploitation
#     vf_coef=1.0,          # Higher value function coefficient
#     max_grad_norm=0.3,    # Stricter gradient clipping
)

# # Train with less frequent evaluation to focus on learning
# eval_callback = EvalCallback(eval_env, best_model_save_path="./best_model/",
#                              log_path="./logs/", eval_freq=5000,
#                              deterministic=True, render=False, n_eval_episodes=10)

model.learn(total_timesteps=10000) #, callback=eval_callback)

# Save the trained model for evaluation
model.save("ppo_model.zip")

print("Training completed! Model saved as ppo_model.zip")
