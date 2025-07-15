"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

A2C Curriculum Trainer
-----------------------------------------------------------------
Trains an A2C agent across a staged curriculum in BasicObstacleEnv.
Model and hyperparameters are imported from R1_A2C_Config.py.

Supports:
- Per-stage timesteps embedded in curriculum
- Stage-wise checkpoints
- Final model saving with timestamp
"""

import argparse
from pathlib import Path
from datetime import datetime
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from R1_Training_Environment import BasicObstacleEnv
from R1_A2C_Config import make_model


# ───────────────────────────────────────────────────────────────────────────────
# Curriculum Configuration
# ───────────────────────────────────────────────────────────────────────────────

# same as DQN and PPO 

CURRICULUM = [
    dict(size=5.0, num_obstacles=3,  obstacle_size_range=(0.30, 0.50), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=7.0, num_obstacles=8,  obstacle_size_range=(0.30, 0.60), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=9.0, num_obstacles=15, obstacle_size_range=(0.30, 0.70), moving=False, sensor_noise_std=0.01, timesteps=600_000),
    dict(size=9.0, num_obstacles=20, obstacle_size_range=(0.30, 0.70), moving=True,  sensor_noise_std=0.02, timesteps=600_000),
]



def create_env(stage_idx: int):

    # Builds environment for a given curriculum stage.

    cfg = CURRICULUM[stage_idx].copy()
    return Monitor(BasicObstacleEnv(**cfg))

# ─────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────

def train_curriculum(final_stage: int):
    model = None

    for stage in range(final_stage):
        print(f"\n[Stage {stage + 1}/{final_stage}] {CURRICULUM[stage]}")

        stage_cfg = CURRICULUM[stage].copy()
        total_steps = stage_cfg.pop("timesteps")
        env = create_env(stage)

        ckpt_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./a2c_checkpoints/R1_A2C_stage_{stage+1}/",
            name_prefix=f"R1_A2C_stage_{stage+1}"
        )

        if model is None:
            model = make_model(env)
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=total_steps,
            callback=ckpt_cb,
            reset_num_timesteps=False
        )

        env.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/R1_A2C_Stage{stage+1}_{timestamp}.zip"
        model.save(model_path)
        print(f"Saved model: {model_path}\n")

    print(f"Final model: {model_path}")

# ─────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Curriculum Trainer for A2C")
    parser.add_argument("--until", type=int, default=len(CURRICULUM),
                        help="Final stage to train (1 to 4). Default = 4")
    args = parser.parse_args()

    Path("./models").mkdir(exist_ok=True)
    Path("./a2c_checkpoints").mkdir(exist_ok=True)

    train_curriculum(final_stage=args.until)
    print("Run complete. Check ./models for the final model.")
