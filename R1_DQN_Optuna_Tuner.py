"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

DQN Hyperparameter Tuner using Optuna
-------------------------------------------------------------------------
Uses Optuna to search for optimal DQN hyperparameters on a simple environment
stage. This is useful before full curriculum training.

Outputs:
- Best trial parameters
- Saves best model to ./Optuna/DQN/models/R1_DQN_Optuna_Best.zip
- Writes all trial results to ./Optuna/DQN/optuna_results.csv
- Saves hyperparameter importance plot (importance.png)
- Saves best config as best_config.json
-------------------------------------------------------------------------
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import json
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from R1_Training_Environment import BasicObstacleEnv
from pathlib import Path
import torch as th


# --- Setup Output Folders ---
ROOT_DIR = Path("Optuna/DQN")
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper function to generate consistent environment for each trial ---
def make_env():
    """
    Returns a simple Stage 1 environment for consistent and fair comparison
    across trials. Keeps things deterministic (no moving obstacles, no noise).
    """
    cfg = dict(
        size=5.0,
        num_obstacles=3,
        obstacle_size_range=(0.30, 0.50),
        moving=False,
        sensor_noise_std=0.0
    )
    return Monitor(BasicObstacleEnv(**cfg))


# --- Stores all Optuna trial results for later analysis ---
trial_data = []


# --- Main objective function that Optuna will optimise ---
def objective(trial):
    env = make_env()  # Fresh environment per trial to avoid contamination

    # --- Suggest DQN hyperparameters for this trial ---
    # These cover memory size, learning dynamics, and exploration strategy
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    buffer_size = trial.suggest_categorical("buffer_size", [50_000, 100_000, 200_000])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8])
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)
    net_arch = trial.suggest_categorical("net_arch", [(64, 64), (128, 128), (256, 256)])

    # Policy architecture configuration
    # ReLU is commonly used with DQNs due to its simplicity and performance
    policy_kwargs = dict(
        net_arch=list(net_arch),
        activation_fn=th.nn.ReLU
    )

    # Create the DQN model using trial-specific parameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        train_freq=train_freq,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=0
    )

    # Train the model — I've kept it short here to allow fast tuning
    model.learn(total_timesteps=80_000)

    # Evaluate how well the trained model performs
    # I use deterministic policy to get reproducible results
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    # Store result and parameters for this trial
    trial_data.append({
        'trial': trial.number,
        'mean_reward': mean_reward,
        'learning_rate': learning_rate,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'train_freq': train_freq,
        'exploration_fraction': exploration_fraction,
        'exploration_final_eps': exploration_final_eps,
        'net_arch': str(net_arch)
    })

    # Save the best model so far to disk
    if not hasattr(objective, "best_reward") or mean_reward > objective.best_reward:
        objective.best_reward = mean_reward
        model.save(MODEL_DIR / "R1_DQN_Optuna_Best")

    env.close()
    return mean_reward  # This value is what Optuna will try to maximise


if __name__ == "__main__":
    # Start tuning — aim to maximise the agent's average reward
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\nBest trial:")
    print(study.best_trial)

    # Save best config as JSON for reuse in full training later
    with open(ROOT_DIR / "best_config.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print("Saved best trial config to best_config.json")

    # Save all trial results to CSV for later review
    df = pd.DataFrame(trial_data)
    df.to_csv(ROOT_DIR / "optuna_results.csv", index=False)
    print("Saved trial results to optuna_results.csv")

    # Save plot showing which hyperparameters mattered most
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "importance.png")
    print("Saved hyperparameter importance plot to importance.png")

    print("Run complete! Check ./Optuna/DQN for models and tuning results.")
