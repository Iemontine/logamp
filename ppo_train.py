import gymnasium as gym
from stable_baselines3 import PPO
from environment import prac_env_v0

# Use the environment class directly with StableBaselines3
env = prac_env_v0()

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

# Save the trained model for evaluation
model.save("ppo_model.zip")

# Optional: quick test run (not for evaluation)
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, _ = env.reset()
