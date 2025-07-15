"""

R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

A2C Model Configuration
------------------------------------------------------
This file creates an Advantage Actor-Critic (A2C) agent using Stable-Baselines3
for curriculum-based training. Here in R1_A2C_Config.py, we define the model 
architecture and training parameters. 

In the currently saved file, I have kept the parameters at their default values but during training I used this file
to manually modify parameters as I saw fit.

"""

from stable_baselines3 import A2C
import torch as th


def make_model(env):
    """
    Builds and return an A2C model for the given environment.
    This function is imported by the curriculum trainer , R1_Curriculum_Trainer_A2C.py.
    """

    # ▸ Learning rate – controls the step size during gradient descent
    learning_rate = 7e-4  # SB3 default is 7e-4

    # ▸ Gamma – discount factor for future rewards
    gamma = 0.99  # Default is 0.99

    # ▸ n_steps – number of steps to run for each environment per update
    n_steps = 5  # Default for on-policy algorithms like A2C

    # ▸ Entropy coefficient – encourages exploration by penalising certainty
    ent_coef = 0.0  #  I could increase it to 0.01 for more exploration

    # ▸ Value function coefficient – weight of the critic loss
    vf_coef = 0.5  # Default

    # ▸ Max gradient norm – clips gradients to avoid exploding updates
    max_grad_norm = 0.5

    # ▸ Neural network architecture
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,         # Tanh activation
        net_arch=[64, 64],                # Two hidden layers of 64 units each
    )

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=None,
        #tensorboard_log="./logs/tensorboard/R1_A2C/", # Deactivated because it was crashing the terminal in the final stages.
    )

    return model
