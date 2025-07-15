"""
Initial Prototype 1 - Grid

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

Deep Q Network model training

"""

# Since this is a initial prototype , I'll be using the default DQN implementation from Stable Baselines3.
# This will allow us to focus on the environment and the model training without getting into the complexities of custom DQN implementations.

import os
from Environment import RobotNavEnv
from stable_baselines3 import DQN

def train_dqn_model(steps=10000, save_path="models/dqn_robot"):
    env = RobotNavEnv(grid_size=10, obstacle_count=10, max_steps=200)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()

if __name__ == "__main__":
    train_dqn_model(steps=200000, save_path="models/dqn_robot")
