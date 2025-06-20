import numpy as np
from stable_baselines3 import PPO
from environment import prac_env_v0

# Load the trained model
model = PPO.load("ppo_model.zip")

env = prac_env_v0()
env._render_mode = True  # Enable renderer for evaluation

episodes = 10
success_count = 0
exact_count = 0
total_steps = 0
total_rewards = []

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
        step_count += 1    # Check success and precision
    final_value = env.state  # The final position we reached
    target = env.current     # The target we were trying to reach
    error = abs(final_value - target)
    
    if error <= 5.0:  # Success if within 5 units
        success_count += 1
        if error <= 0.1:  # Exact if within 0.1 units
            exact_count += 1
    
    total_steps += step_count
    total_rewards.append(total_reward)
    
    print(f"Episode {ep+1}: {step_count} steps, reward: {total_reward:.2f}, "
          f"target: {target:.2f}, final: {final_value:.2f}, error: {error:.2f}")

print("\n=== EVALUATION SUMMARY ===")
print(f"Episodes: {episodes}")
print(f"Success rate: {success_count}/{episodes} ({100*success_count/episodes:.1f}%)")
print(f"Exact rate: {exact_count}/{episodes} ({100*exact_count/episodes:.1f}%)")
print(f"Average steps: {total_steps/episodes:.1f}")
print(f"Average reward: {np.mean(total_rewards):.2f}")
print(f"Best reward: {np.max(total_rewards):.2f}")

env.close()
