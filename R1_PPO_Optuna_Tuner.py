"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

PPO Hyperparameter Tuner using Optuna
-------------------------------------------------------------------------
Uses Optuna to search for optimal PPO hyperparameters on a single stage
of the curriculum. This should be used before full training to identify
strong configurations.

Outputs:
- Best trial parameters
- Saves the best model to ./Optuna/PPO/models/R1_PPO_Optuna_Best.zip
- Writes all results to ./Optuna/PPO/optuna_results.csv
- Generates a hyperparameter importance plot (importance.png)
- Saves best config to best_config.json for reuse in final training
-------------------------------------------------------------------------
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import json
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from R1_Training_Environment import BasicObstacleEnv
from pathlib import Path
import torch as th 


# --- Set up folders for storing tuning results ---
ROOT_DIR = Path("Optuna/PPO")
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --- Helper to create a fixed environment for all trials ---
def make_env():
    """
    Returns a controlled and reproducible environment setup (Stage 1-like).
    I avoid moving obstacles or noise here so learning performance is solely
    influenced by the hyperparameters.
    """
    cfg = dict(
        size=5.0,
        num_obstacles=3,
        obstacle_size_range=(0.30, 0.50),
        moving=False,
        sensor_noise_std=0.0
    )
    return Monitor(BasicObstacleEnv(**cfg))


# Stores each trial's performance and parameters
trial_data = []


# --- Objective function: defines one Optuna trial ---
def objective(trial):
    env = make_env()  # Create fresh env per trial

    # --- Sample a set of PPO hyperparameters from Optuna's search space ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048])  # update interval
    gamma = trial.suggest_float("gamma", 0.95, 0.999)  # discount factor
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.01, log=True)  # entropy bonus (exploration)
    net_arch = trial.suggest_categorical("net_arch", [[64, 64], [128, 128], [256, 256]])

    # Define policy architecture
    # I use Tanh activation here, which typically helps stabilise PPO updates
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,  
        net_arch=net_arch
    )

    # Initialise the PPO model with the sampled hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=None  # Logging disabled during trials to avoid unnecessary overhead
    )

    # Train the agent briefly — enough to compare configs but not overfit
    model.learn(total_timesteps=80_000)

    # Evaluate model performance — I use deterministic to keep it consistent
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    # Log results of this trial
    trial_data.append({
        'trial': trial.number,
        'mean_reward': mean_reward,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'n_steps': n_steps,
        'gamma': gamma,
        'ent_coef': ent_coef,
        'net_arch': str(net_arch),
    })

    # Save the model if it’s the best one seen so far
    if not hasattr(objective, "best_reward") or mean_reward > objective.best_reward:
        objective.best_reward = mean_reward
        model.save(MODEL_DIR / "R1_PPO_Optuna_Best")

    env.close()
    return mean_reward  # This is what Optuna will attempt to maximise


if __name__ == "__main__":
    # Start Optuna study to maximise average reward
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    print("\nBest trial:")
    print(study.best_trial)

    # Save best configuration for reuse in final training
    with open(ROOT_DIR / "best_config.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print("Saved best trial config to best_config.json")

    # Save all trials to CSV for later reference
    df = pd.DataFrame(trial_data)
    df.to_csv(ROOT_DIR / "optuna_results.csv", index=False)
    print("Saved trial results to optuna_results.csv")

    # Generate a plot showing which hyperparameters influenced performance the most
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "importance.png")
    print("Saved hyperparameter importance plot to importance.png")

    print("Run complete. Check ./Optuna/PPO for models and tuning results.")
