import numpy as np
from stable_baselines3 import PPO
from environment import prac_env_v0

# Load the trained model
model = PPO.load("ppo_model.zip")  # Make sure to save your model as 'ppo_model.zip' after training

env = prac_env_v0()
env._render_mode = True  # Enable renderer for evaluation

episodes = 5
for ep in range(episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    total_reward = 0
    step_count = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
    print(f"Episode {ep+1} finished in {step_count} steps, total reward: {total_reward:.2f}")
    env.render()

env.close()
