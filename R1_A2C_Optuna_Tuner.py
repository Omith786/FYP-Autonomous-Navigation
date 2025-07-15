"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

A2C Hyperparameter Tuner using Optuna
-------------------------------------------------------------------------
Uses Optuna to search for optimal A2C hyperparameters on a simple stage
of the curriculum. Best config can be reused in full curriculum training.

It runs through 20 trials.

Outputs:
- Best trial parameters
- Saves best model to ./Optuna/A2C/models/R1_A2C_Optuna_Best.zip
- Logs all trials to optuna_results.csv
- Saves hyperparameter importance plot to importance.png
- Stores best config as best_config.json
-------------------------------------------------------------------------
"""

import optuna
import pandas as pd
import matplotlib.pyplot as plt
import json

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


from R1_Training_Environment import BasicObstacleEnv
from pathlib import Path

import torch as th


# --- Setup Output Folders ---
# Defines where Optuna-related A2C results will be stored
ROOT_DIR = Path("Optuna/A2C")
MODEL_DIR = ROOT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)  # Creates the folder structure if it doesn't exist


# --- Helper function to create the environment ---
def make_env():
    # I use a fixed simple stage of the environment here to keep tuning consistent
    cfg = dict(
        size=5.0,  # Size of the map
        num_obstacles=3,  # Keep obstacle count moderate to avoid random variation in difficulty
        obstacle_size_range=(0.30, 0.50),  # Obstacle sizes have slight variation
        moving=False,  # No moving obstacles during tuning to keep trials deterministic
        sensor_noise_std=0.0  # Sensor noise disabled here for controlled training
    )
    # Wrap the environment with a Monitor to track episode-level metrics
    return Monitor(BasicObstacleEnv(**cfg))


# --- Global list to store results of all Optuna trials ---
trial_data = []


# --- This is the main Optuna objective function ---
def objective(trial):
    env = make_env()  # We create a new environment for each trial to avoid memory issues

    # --- Suggest hyperparameters using Optuna's search space sampling ---
    # These are the A2C-specific hyperparameters I am tuning
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical("n_steps", [5, 10, 20, 64, 128])  # number of steps before update
    gamma = trial.suggest_float("gamma", 0.9, 0.9999)  # discount factor
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)  # GAE smoothing parameter
    ent_coef = trial.suggest_float("ent_coef", 1e-5, 0.01, log=True)  # entropy loss coefficient (for exploration)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 1.0)  # value function loss weight
    net_arch = trial.suggest_categorical("net_arch", [(64, 64), (128, 128), (256, 256)])  # policy network size

    # I set the policy architecture using a dictionary
    # I use Tanh activation because it's more stable with A2C compared to ReLU in my environment
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=list(net_arch)
    )

    # Instantiate the A2C model with suggested hyperparameters
    model = A2C(
        "MlpPolicy",       # Multi-layer perceptron policy
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,         # Keep output clean during tuning
        tensorboard_log=None  # No tensorboard logging during tuning
    )

    # Train the model for a fixed number of timesteps for fair trial comparison
    model.learn(total_timesteps=80_000)

    # Evaluate the model’s mean reward across 5 episodes
    # I use deterministic=True to get consistent evaluation
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    # Store trial information for later analysis/export
    trial_data.append({
        "trial": trial.number,
        "mean_reward": mean_reward,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "net_arch": str(net_arch)
    })

    # Saves the best-performing model so far
    if not hasattr(objective, "best_reward") or mean_reward > objective.best_reward:
        objective.best_reward = mean_reward
        model.save(MODEL_DIR / "R1_A2C_Optuna_Best")

    env.close()  # Clean up environment
    return mean_reward  # Optuna uses this to decide which trial is "best"


# --- Entry point ---
if __name__ == "__main__":
    # Start the hyperparameter tuning process
    study = optuna.create_study(direction="maximize")  # We want to maximise the mean reward
    study.optimize(objective, n_trials=20)  # Run 20 tuning trials

    print("\nBest trial:")
    print(study.best_trial)

    # Save best configuration to JSON for use in future curriculum training
    with open(ROOT_DIR / "best_config.json", "w") as f:
        json.dump(study.best_trial.params, f, indent=4)
    print("Saved best trial config to best_config.json")

    # Save all trial results to CSV for analysis or plotting later
    df = pd.DataFrame(trial_data)
    df.to_csv(ROOT_DIR / "optuna_results.csv", index=False)
    print("Saved trial results to optuna_results.csv")

    # Generate and save a bar plot showing which hyperparameters were most important
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.tight_layout()
    fig.savefig(ROOT_DIR / "importance.png")
    print("Saved hyperparameter importance plot to importance.png")

    print("Run complete! Check ./Optuna/A2C for models and tuning results.")
