"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

DQN Model Configuration
------------------------------------------------------
Creates a Deep Q-Network (DQN) agent using Stable-Baselines3 for curriculum training.
This config is compatible with the curriculum trainer script and can be tuned manually
or extended using Optuna.

In the currently saved file, I have kept the parameters at their default values but during training I used this file
to manually modify parameters as I saw fit.
------------------------------------------------------

"""

from stable_baselines3 import DQN
import torch as th


def make_model(env):
    """
    Build and return a DQN model for the given environment.
    Designed to be used by R1_Curriculum_Trainer_DQN.py.
    """

    # ▸ Learning rate – how fast the network learns
    learning_rate = 1e-4  # Default = 1e-4

    # ▸ Buffer size – number of past transitions to store
    buffer_size = 100_000

    # ▸ Exploration – how often to take random actions
    exploration_fraction = 0.1  # fraction of total training to linearly anneal epsilon
    exploration_final_eps = 0.05  # minimum epsilon value

    # ▸ Target network update frequency
    target_update_interval = 1000

    # ▸ Network architecture
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,         # Typically ReLU is preferred for DQN
        net_arch=[128, 128],              # Two hidden layers
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,                           # Soft update factor
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
        #tensorboard_log="./logs/tensorboard/R1_DQN/" # Deactivated because it was crashing the terminal in the final stages.
    )

    return model
