import os
import glob
import numpy as np
from stable_baselines3 import PPO
from environment import LogAmpEnvironment


def evaluate_model(model_path: str, n_episodes: int = 200) -> dict:
    """Evaluate a single model and return performance metrics."""
    env = LogAmpEnvironment()
    model = PPO.load(model_path, env=env)
    
    rewards = []
    successes = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            done = done or truncated
        
        rewards.append(episode_reward)
        
        # Check if target was reached (distance <= 0.5)
        if abs(obs[0] - obs[1]) <= 0.5:
            successes += 1
    
    return {
        'name': os.path.basename(model_path),        'success_rate': successes / n_episodes,
        'avg_reward': np.mean(rewards)
    }


def main():
    """Evaluate all models and print results."""
    models_dir = "./models/"
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))
    
    if not model_files:
        print("No models found in ./models/")
        return
    
    print(f"Evaluating {len(model_files)} models...\n")
    
    results = []
    for model_file in model_files:
        try:
            result = evaluate_model(model_file)
            results.append(result)
            print(f"✓ {result['name']}")
        except Exception as e:
            print(f"✗ {os.path.basename(model_file)} - Error: {e}")
    
    if not results:
        print("No models could be evaluated.")
        return
    
    # Sort by success rate, then by average reward
    results.sort(key=lambda x: (x['success_rate'], x['avg_reward']), reverse=True)
    
    print(f"\n{'='*50}")
    print("RESULTS (ranked by performance)")
    print(f"{'='*50}")
    print(f"{'Rank':<4} {'Model':<25} {'Success%':<9} {'Avg Reward':<10}")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        name = result['name'][:24] if len(result['name']) > 24 else result['name']
        print(f"{i:<4} {name:<25} {result['success_rate']:.1%}     {result['avg_reward']:>6.1f}")
    
    print(f"\nBest model: {results[0]['name']} ({results[0]['success_rate']:.1%} success rate)")


if __name__ == "__main__":
    main()
