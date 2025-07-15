"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

DQN Curriculum Trainer Using Optuna-Tuned Hyperparameters
------------------------------------------------------------
Loads the best parameters from Optuna tuning (best_config.json)
and runs full curriculum training using BasicObstacleEnv.
Saves trained models per stage with timestamps.

Outputs:
- Trained models in ./models/
- Checkpoints in ./dqn_checkpoints/
------------------------------------------------------------
"""

import json
from pathlib import Path
from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from R1_Training_Environment import BasicObstacleEnv
import torch as th


# --- Load best hyperparameters from Optuna tuning ---
with open("Optuna/DQN/best_config.json", "r") as f:
    config = json.load(f)

# If Optuna saved net_arch as a string or tuple, we convert it to a proper list
# This ensures compatibility when passing into policy_kwargs
if isinstance(config["net_arch"], str):
    config["net_arch"] = eval(config["net_arch"])

# DQN commonly uses ReLU for non-linearity in its Q-network
policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=config["net_arch"]
)


# --- Define the curriculum training plan ---
# Each stage increases complexity in environment design and difficulty
CURRICULUM = [
    dict(size=5.0, num_obstacles=3,  obstacle_size_range=(0.30, 0.50), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=7.0, num_obstacles=8,  obstacle_size_range=(0.30, 0.60), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=9.0, num_obstacles=15, obstacle_size_range=(0.30, 0.70), moving=False, sensor_noise_std=0.01, timesteps=800_000),
    dict(size=9.0, num_obstacles=20, obstacle_size_range=(0.30, 0.70), moving=True,  sensor_noise_std=0.02, timesteps=800_000),
]


# --- Create folders to store model outputs and checkpoints ---
Path("./models").mkdir(exist_ok=True)
Path("./dqn_checkpoints").mkdir(exist_ok=True)


# --- Wraps the environment for logging and monitoring ---
def create_env(stage_cfg):
    env_cfg = stage_cfg.copy()  # Copy to avoid modifying original
    return Monitor(BasicObstacleEnv(**env_cfg))


# --- Train the agent across all curriculum stages ---
def train_with_best_config():
    model = None  # Will hold our agent; reused across stages

    for stage_idx, stage_cfg in enumerate(CURRICULUM):
        print(f"\n[Stage {stage_idx + 1}] Config: {stage_cfg}")
        
        stage = stage_cfg.copy()
        total_steps = stage.pop("timesteps")  # Separate out the training duration
        env = create_env(stage)

        # Setup checkpoint saving during training for this stage
        ckpt_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./dqn_checkpoints/R1_DQN_Stage{stage_idx + 1}/",
            name_prefix=f"R1_DQN_Stage{stage_idx + 1}",
        )

        # First stage — we initialise the model with Optuna-tuned params
        if model is None:
            model = DQN(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                buffer_size=config["buffer_size"],
                batch_size=config["batch_size"],
                train_freq=config["train_freq"],
                exploration_fraction=config["exploration_fraction"],
                exploration_final_eps=config["exploration_final_eps"],
                policy_kwargs=policy_kwargs,
                verbose=1
            )
        else:
            # For later stages, we keep the same model but swap out the environment
            model.set_env(env)

        # Train on the current stage
        model.learn(
            total_timesteps=total_steps,
            callback=ckpt_cb,
            reset_num_timesteps=False  # Continue timestep count across stages
        )

        env.close()

        # Save model checkpoint with timestamp for clarity and versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/R1_DQN_Stage{stage_idx + 1}_{timestamp}.zip"
        model.save(model_path)
        print(f"  Saved model: {model_path}\n")

    print("\nAll stages trained using best DQN config. Final model(s) saved.")


if __name__ == "__main__":
    train_with_best_config()
