"""
PPO Curriculum Trainer Using Optuna-Tuned Hyperparameters
------------------------------------------------------------
Loads the best parameters from Optuna tuning (best_config.json)
and runs full curriculum training using BasicObstacleEnv.
Saves trained models per stage with timestamps.

Outputs:
- Trained models in ./models/
- Checkpoints in ./ppo_checkpoints/
------------------------------------------------------------
"""

import json
from pathlib import Path
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from R1_Training_Environment import BasicObstacleEnv
import torch as th

# Load best hyperparameters found by Optuna
with open("Optuna/PPO/best_config.json", "r") as f:
    config = json.load(f)

policy_kwargs = dict(
    activation_fn=th.nn.Tanh,  # Default: Tanh used during tuning
    net_arch=config["net_arch"],
)

# Define full curriculum with timesteps per stage
CURRICULUM = [
    dict(size=5.0, num_obstacles=3,  obstacle_size_range=(0.30, 0.50), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=7.0, num_obstacles=8,  obstacle_size_range=(0.30, 0.60), moving=False, sensor_noise_std=0.0, timesteps=600_000),
    dict(size=9.0, num_obstacles=15, obstacle_size_range=(0.30, 0.70), moving=False, sensor_noise_std=0.01, timesteps=800_000),
    dict(size=9.0, num_obstacles=20, obstacle_size_range=(0.30, 0.70), moving=True,  sensor_noise_std=0.02, timesteps=800_000),
]

# Create save directories
Path("./models").mkdir(exist_ok=True)
Path("./ppo_checkpoints").mkdir(exist_ok=True)

def create_env(stage_cfg):
    """Wrap BasicObstacleEnv with monitoring and prepare config."""
    env_cfg = stage_cfg.copy()
    return Monitor(BasicObstacleEnv(**env_cfg))

def train_with_best_config():
    model = None

    for stage_idx, stage_cfg in enumerate(CURRICULUM):
        print(f"\n[Stage {stage_idx + 1}] Config: {stage_cfg}")
        stage = stage_cfg.copy()
        total_steps = stage.pop("timesteps")
        env = create_env(stage)

        ckpt_cb = CheckpointCallback(
            save_freq=100_000,
            save_path=f"./ppo_checkpoints/PPO_Optuna_Stage{stage_idx + 1}/",
            name_prefix=f"PPO_Optuna_Stage{stage_idx + 1}",
        )

        if model is None:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                n_steps=config["n_steps"],
                gamma=config["gamma"],
                ent_coef=config["ent_coef"],
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=None,
            )
        else:
            model.set_env(env)

        model.learn(
            total_timesteps=total_steps,
            callback=ckpt_cb,
            reset_num_timesteps=False
        )

        env.close()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/PPO_Optuna_Stage{stage_idx + 1}_{timestamp}.zip"
        model.save(model_path)
        print(f"  Saved model: {model_path}\n")

    print("\nAll stages trained using the best config. Final model(s) saved.")

if __name__ == "__main__":
    train_with_best_config()
    