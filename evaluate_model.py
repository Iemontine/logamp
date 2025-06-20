import numpy as np
from stable_baselines3 import PPO
from environment import prac_env_v0

# Load the trained model (try improved version first)
try:
    model = PPO.load("ppo_model_improved.zip")
    print("📊 Loaded improved model")
except:
    try:
        model = PPO.load("./best_model/best_model.zip")
        print("📊 Loaded best model from training")
    except:
        model = PPO.load("ppo_model.zip")
        print("📊 Loaded original model")

env = prac_env_v0()

episodes = 20
success_count = 0
exact_count = 0
total_steps = 0
total_rewards = []
distances = []

print("🎯 Evaluating model performance...")
print("=" * 50)

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
        
        # Safety check to prevent infinite loops
        if step_count > 300:
            break
    
    # Check success and precision
    final_value = env.state  # The final position we reached
    target = env.target     # The target we were trying to reach
    error = abs(final_value - target)
    distances.append(error)
    
    if error <= 5.0:  # Success if within 5 units
        success_count += 1
        if error <= 0.5:  # Exact if within 0.5 units
            exact_count += 1
    
    total_steps += step_count
    total_rewards.append(total_reward)
    
    status = "✅" if error <= 5.0 else "❌"
    precision = "🎯" if error <= 0.5 else "📍" if error <= 2.0 else "📌"
    
    print(f"{status} Episode {ep+1:2d}: {step_count:3d} steps, reward: {total_reward:6.1f}, "
          f"target: {target:6.2f}, final: {final_value:6.2f}, error: {error:5.3f} {precision}")

print("\n" + "=" * 60)
print("=== EVALUATION SUMMARY ===")
print(f"Episodes: {episodes}")
print(f"Success rate (≤5.0): {success_count}/{episodes} ({100*success_count/episodes:.1f}%)")
print(f"Precision rate (≤0.5): {exact_count}/{episodes} ({100*exact_count/episodes:.1f}%)")
print(f"Average steps: {total_steps/episodes:.1f}")
print(f"Average reward: {np.mean(total_rewards):.2f}")
print(f"Best reward: {np.max(total_rewards):.2f}")
print(f"Average distance error: {np.mean(distances):.3f}")
print(f"Median distance error: {np.median(distances):.3f}")
print(f"Best distance error: {np.min(distances):.3f}")

# Performance categories
excellent = sum(1 for d in distances if d <= 0.5)
good = sum(1 for d in distances if 0.5 < d <= 2.0)
fair = sum(1 for d in distances if 2.0 < d <= 5.0)
poor = sum(1 for d in distances if d > 5.0)

print(f"\n📈 Performance Distribution:")
print(f"   Excellent (≤0.5): {excellent}/{episodes} ({100*excellent/episodes:.1f}%)")
print(f"   Good (0.5-2.0):   {good}/{episodes} ({100*good/episodes:.1f}%)")
print(f"   Fair (2.0-5.0):   {fair}/{episodes} ({100*fair/episodes:.1f}%)")
print(f"   Poor (>5.0):      {poor}/{episodes} ({100*poor/episodes:.1f}%)")

env.close()
