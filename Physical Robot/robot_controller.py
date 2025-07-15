"""
FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

This script deploys a trained PPO reinforcement learning model
to control a Raspberry Pi-powered robot using three HC-SR04
ultrasonic sensors. 

The robot will move based on if there are obstacles detected by
the sensors the decision will be made by the PPO model.

The goal is for the robot to navigate through an environment
avoiding obstacles using its intelligence and go as further as possible
without crashing. No human needed to control and make it avoid obstacles.

I've set it up to run forward and during testing irl I'll put obstacles in front
and see how it reacts.

Once activated it starts moving and making decisions
based on the trained model.

"""

import time
import numpy as np
from stable_baselines3 import PPO

from robot_sensors import read_sonars
from robot_motors import move_forward, turn_left, turn_right

# Load the trained PPO model from local directory
model = PPO.load("PPO_Optuna_Stage4_pruned.zip")

def get_obs():
    """
    Creates an observation vector to match the format used during training:
    - 3 normalised sonar readings (front, left, right)
    - fixed bearing to goal (0.0 degrees, as we're assuming a virtual goal ahead)
    - fixed distance to goal (1.0, normalised)
    """
    sonar_front, sonar_left, sonar_right = read_sonars()

    bearing_to_goal = 0.0      # Fixed (as we assume goal is straight ahead)
    distance_to_goal = 1.0     # Normalised max distance

    return np.array([sonar_front, sonar_left, sonar_right, bearing_to_goal, distance_to_goal], dtype=np.float32)

# Main loop to continuously run the policy and move the robot
while True:
    obs = get_obs()
    action, _ = model.predict(obs, deterministic=True)

    if action == 0:
        move_forward()
    elif action == 1:
        turn_left()
    elif action == 2:
        turn_right()

    time.sleep(0.1)
