"""
Initial Prototype 1 - Grid

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

Proximal Policy Optimization (PPO) model training

"""

import os
from Environment import RobotNavEnv
from stable_baselines3 import PPO

def train_ppo_model(steps=10000, save_path="models/ppo_robot"):
    env = RobotNavEnv(grid_size=10, obstacle_count=10, max_steps=200)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=steps)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    env.close()

if __name__ == "__main__":
    train_ppo_model(steps=200000, save_path="models/ppo_robot")
