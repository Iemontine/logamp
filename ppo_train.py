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
SEED = 42
VERBOSE = 1
set_random_seed(SEED)

vec_env = make_vec_env(LogAmpEnvironment, n_envs=1, seed=SEED, env_kwargs={
    'max_steps': 200,
    'target_min': -1000.0,
    'target_max': 1000.0,
    'start_min': -500.0,
    'start_max': 500.0,
})
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO(
    policy='MlpPolicy',
    env=vec_env,
    tensorboard_log="./tensorboard_logs/",
    device="cpu",
    verbose=VERBOSE,
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