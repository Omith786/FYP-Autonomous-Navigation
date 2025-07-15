"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

PPO Model Configuration (Round 1)
------------------------------------------------------
Creates a PPO agent using Stable-Baselines3 with support for curriculum training.
This file can be used manually or plugged into Optuna for hyperparameter tuning.

In the currently saved file, I have kept the parameters at their default values but during training I used this file
to manually modify parameters as I saw fit.
------------------------------------------------------

"""

from stable_baselines3 import PPO
import torch as th


def make_model(env):
    """
    Builds and return a PPO model for the given environment.
    This function is designed to be imported by R1_Curriculum_Trainer_PPO.py.
    """

    # ▸ Learning rate: how quickly the policy updates. Lower = slower but more stable.
    learning_rate = 3e-4  # Default: 3e-4 (can use a schedule function)

    # ▸ Batch size: how many samples per gradient update. Must divide n_steps evenly.
    batch_size = 64  # Default: 64

    # ▸ Number of steps to run in each environment per policy update
    n_steps = 2048  # Default: 2048

    # ▸ Discount factor for future rewards
    gamma = 0.99  # Default: 0.99

    # ▸ Policy network architecture and activation
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,        # Default: Tanh (can try ReLU)
        net_arch=[64, 64],               # Default: two hidden layers with 64 units each
    )

    # ▸ Instantiate the PPO agent
    model = PPO(
        "MlpPolicy",          # Multilayer perceptron policy (dense NN)
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        gae_lambda=0.95,       # How much advantage estimation uses bootstrapping
        clip_range=0.2,        # PPO clipping range (stability)
        ent_coef=0.0,          # Entropy regularisation (encourages exploration)
        vf_coef=0.5,           # Value function loss weight
        max_grad_norm=0.5,     # Gradient clipping
        target_kl=0.03,        # Stop early if KL divergence too high
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
        # tensorboard_log="./logs/tensorboard/R1_PPO/", this is usefull when trying to get more info on how the training is going.
        # but I deactivated it here because it was causing issues in the later stages when training in the curriculum.
    )

    return model