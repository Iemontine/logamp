from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from environment import LogAmpEnvironment
import time
import os
from stable_baselines3.common.callbacks import CheckpointCallback

# Create output directories
os.makedirs("./models", exist_ok=True)
os.makedirs("./tensorboard_logs", exist_ok=True)

# Set random seeds for reproducibility
SEED = 113350
VERBOSE = 1
set_random_seed(SEED)

# env = LogAmpEnvironment(max_steps=200, target_min=-250.0, target_max=250.0, start_min=-400.0, start_max=400.0)
env = LogAmpEnvironment(max_steps=200, target_min=-500.0, target_max=500.0, start_min=-200.0, start_max=200.0)
env.reset(seed=SEED)  # Seed the environment directly

model = PPO(
    policy='MlpPolicy',
    env=env,
    tensorboard_log="./tensorboard_logs/",
    device="cpu",
    verbose=VERBOSE,
    seed=SEED,
    learning_rate=lambda progress: 0.0003 * progress
)

total_timesteps = 500_000

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