"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

PPO Training Pipeline
-----------------------------------------------------------------
Trains a Proximal Policy Optimisation (PPO) agent through a multi-stage
curriculum using the environment defined in R1_Training_Environment.py.

"""

import argparse
from pathlib import Path
from datetime import datetime  # NEW: For timestamping model filenames
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# Import PPO and torch only if using manual config
from stable_baselines3 import PPO
import torch as th

# Use custom environment
from R1_Training_Environment import BasicObstacleEnv
from R1_PPO_Config import make_model  # Swap this with Optuna if needed

# ───────────────────────────────────────────────────────────────────────────────
# Curriculum Configuration
# ───────────────────────────────────────────────────────────────────────────────

# Same curriculum as DQN and A2C
# Each stage progressively increases complexity
# - size: Environment size (width/height)
# - num_obstacles: Number of static obstacles  
# - obstacle_size_range: Range of obstacle sizes
# - moving: Whether obstacles are moving or static
# - sensor_noise_std: Standard deviation of sensor noise
# - timesteps: Total training steps for this stage 

CURRICULUM = [
    dict(size=5.0, num_obstacles=3,  obstacle_size_range=(0.30, 0.50), moving=False, sensor_noise=0.0, timesteps=600_000),
    dict(size=7.0, num_obstacles=8,  obstacle_size_range=(0.30, 0.60), moving=False, sensor_noise=0.0, timesteps=600_000),
    dict(size=9.0, num_obstacles=15, obstacle_size_range=(0.30, 0.70), moving=False, sensor_noise=0.01, timesteps=600_000),
    dict(size=9.0, num_obstacles=20, obstacle_size_range=(0.30, 0.70), moving=True,  sensor_noise=0.02, timesteps=600_000),
]


# ──────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────

def create_env(stage: int):
    """Return a fresh monitored environment instance for the given curriculum stage."""
    cfg = CURRICULUM[stage].copy()

    # Extract and remove items not meant for BasicObstacleEnv
    moving_flag = cfg.pop("moving", False)
    sensor_noise = cfg.pop("sensor_noise", 0.0)
    cfg.pop("timesteps", None)  #  Fix: Remove unsupported key

    cfg["moving"] = moving_flag
    cfg["sensor_noise_std"] = sensor_noise  # Rename for the environment

    env = BasicObstacleEnv(**cfg)
    return Monitor(env)

# ─────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────

def train_curriculum(final_stage: int):
    model = None

    for stage in range(final_stage):
        print(f"\n[Stage {stage + 1}/{final_stage}] {CURRICULUM[stage]}")

        env = create_env(stage)
        stage_timesteps = CURRICULUM[stage].get("timesteps", 400_000)  # Use default if missing

        # Save stage checkpoints for safety
        ckpt_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./ppo_checkpoints/R1_PPO_stage_{stage+1}/",
            name_prefix=f"R1_PPO_stage_{stage+1}",
        )

        if model is None:
            model = make_model(env)
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=stage_timesteps,
            callback=ckpt_cb,
            reset_num_timesteps=False
        )

        env.close()

        # NEW: Add timestamp to filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/R1_PPO_Stage{stage+1}_{timestamp}.zip"
        model.save(model_path)
        print(f"  Saved model: {model_path}\n")

    print(f" Final model: {model_path}")

# ─────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Curriculum Trainer for PPO")
    parser.add_argument("--until", type=int, default=len(CURRICULUM),
                        help="Final stage to train (1 to 4). Default = 4")
    args = parser.parse_args()

    Path("./models").mkdir(exist_ok=True)
    Path("./ppo_checkpoints").mkdir(exist_ok=True)

    train_curriculum(final_stage=args.until)
    print("Run complete. Check ./models for the final model.")
    print("Checkpoints saved in ./ppo_checkpoints.")
    