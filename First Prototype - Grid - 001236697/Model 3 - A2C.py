"""

Initial Prototype 1 - Grid

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

Actor-Critic (A2C) model training

"""

import os
from Environment import RobotNavEnv
from stable_baselines3 import A2C

def train_a2c_model(steps=10000, save_path="models/a2c_robot"):
    env = RobotNavEnv(grid_size=10, obstacle_count=10, max_steps=200)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()

if __name__ == "__main__":
    train_a2c_model(steps=200000, save_path="models/a2c_robot")
