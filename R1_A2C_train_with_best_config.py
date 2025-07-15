"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

A2C Curriculum Trainer Using Optuna-Tuned Hyperparameters
------------------------------------------------------------
Loads the best parameters from Optuna tuning (best_config.json)
and runs full curriculum training using BasicObstacleEnv.
Saves trained models per stage with timestamps.

Outputs:
- Trained models in ./models/
- Checkpoints in ./a2c_checkpoints/
------------------------------------------------------------
"""

import json
from pathlib import Path
from datetime import datetime
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from R1_Training_Environment import BasicObstacleEnv
import torch as th


# --- Loads best hyperparameters found by Optuna tuning ---
with open("Optuna/A2C/best_config.json", "r") as f:
    config = json.load(f)

# Converts the architecture from string to list if needed
# This is important when reading from JSON since tuples/lists may be stringified
if isinstance(config["net_arch"], str):
    config["net_arch"] = eval(config["net_arch"])

# Policy configuration passed to A2C
# I’m using Tanh activation which tends to work better than ReLU for this model
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=config["net_arch"]
)


# --- Define curriculum stages ---
# Each stage progressively increases complexity
# I designed this to allow the agent to gradually adapt to harder environments
CURRICULUM = [
    dict(size=5.0, num_obstacles=3,  obstacle_size_range=(0.30, 0.50), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=7.0, num_obstacles=8,  obstacle_size_range=(0.30, 0.60), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=9.0, num_obstacles=15, obstacle_size_range=(0.30, 0.70), moving=False, sensor_noise_std=0.01, timesteps=800_000),
    dict(size=9.0, num_obstacles=20, obstacle_size_range=(0.30, 0.70), moving=True,  sensor_noise_std=0.02, timesteps=800_000),
]


# --- Ensure output folders exist ---
Path("./models").mkdir(exist_ok=True)
Path("./a2c_checkpoints").mkdir(exist_ok=True)


# --- Helper to create monitored environments per stage ---
def create_env(stage_cfg):
    env_cfg = stage_cfg.copy()  # Copy to avoid modifying original
    return Monitor(BasicObstacleEnv(**env_cfg))  # Wrap with Monitor for logging


# --- Main training loop across curriculum stages ---
def train_with_best_config():
    model = None  # Will hold the agent instance and be reused across stages

    for stage_idx, stage_cfg in enumerate(CURRICULUM):
        print(f"\n[Stage {stage_idx + 1}] Config: {stage_cfg}")

        # Extract and separate training timesteps from other config
        stage = stage_cfg.copy()
        total_steps = stage.pop("timesteps")

        # Create environment for the current stage
        env = create_env(stage)

        # Setup checkpoint saving for this stage
        ckpt_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./a2c_checkpoints/R1_A2C_Stage{stage_idx + 1}/",
            name_prefix=f"R1_A2C_Stage{stage_idx + 1}"
        )

        # If it's the first stage, initialise a new model
        # Otherwise, continue training from previous model using new environment
        if model is None:
            model = A2C(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                n_steps=config["n_steps"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                ent_coef=config["ent_coef"],
                vf_coef=config["vf_coef"],
                policy_kwargs=policy_kwargs,
                verbose=1
            )
        else:
            # Reuse the same model but switch to new environment
            model.set_env(env)

        # Train for defined timesteps with checkpointing
        model.learn(
            total_timesteps=total_steps,
            callback=ckpt_cb,
            reset_num_timesteps=False  # Continue from previous timestep count
        )

        env.close()

        # Save the model after this stage with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/R1_A2C_Stage{stage_idx + 1}_{timestamp}.zip"
        model.save(model_path)
        print(f"  Saved model: {model_path}\n")

    print("\nAll stages trained using best A2C config. Final model(s) saved.")


if __name__ == "__main__":
    train_with_best_config()
