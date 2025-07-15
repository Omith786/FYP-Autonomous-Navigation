"""
First Prototype - Grid

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

"""


import gym
from gym import spaces
import numpy as np
import random

class RobotNavEnv(gym.Env):
    """
    A 20x20 grid environment:
      - Robot starts in the center (grid_size//2, grid_size//2)
      - Randomly placed obstacles
      - Random target (not on obstacle or robot)
      - Discrete actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
      - Reward shaping:
         * -0.1 base cost per step
         * +0.2 * distance_improvement (Manhattan distance) each step
         * additional -0.1 if action=Stay , this is so it just doenst stay still
         * +10.0 when target is reached
    """
    def __init__(self, grid_size=20, obstacle_count=25, max_steps=200):
        super(RobotNavEnv, self).__init__()

        self.grid_size = grid_size
        self.obstacle_count = obstacle_count
        self.max_steps = max_steps
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right, 4=Stay
        self.action_space = spaces.Discrete(5)

        # Observations: [robot_x, robot_y, target_x, target_y,
        #                dist_up, dist_down, dist_left, dist_right] (normalized in [0..1])
        low_obs = np.array([0.0]*8, dtype=np.float32)
        high_obs = np.array([1.0]*8, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        self.reset()

    def reset(self):
        self.steps = 0
        self.done = False

        # Randomly generate obstacles
        self.obstacles = set()
        while len(self.obstacles) < self.obstacle_count:
            ox = random.randint(0, self.grid_size - 1)
            oy = random.randint(0, self.grid_size - 1)
            self.obstacles.add((ox, oy))

        # Robot in the center
        self.robot_pos = (self.grid_size // 2, self.grid_size // 2)

        # Random target (not on obstacles & not on robot)
        while True:
            tx = random.randint(0, self.grid_size - 1)
            ty = random.randint(0, self.grid_size - 1)
            if (tx, ty) not in self.obstacles and (tx, ty) != self.robot_pos:
                self.target_pos = (tx, ty)
                break

        return self._get_obs()

    def step(self, action):
        self.steps += 1
        x, y = self.robot_pos

        # Compute old distance (Manhattan distance to target)
        # Manhattan distance measures the distance in a grid-like path
        # It is the sum of the absolute differences in the x and y coordinates
        # This is used to determine how much closer we got to the target
        
        old_dist = abs(x - self.target_pos[0]) + abs(y - self.target_pos[1])

        # Determine next position
        if action == 0:   # Up
            new_pos = (x, y - 1)
        elif action == 1: # Down
            new_pos = (x, y + 1)
        elif action == 2: # Left
            new_pos = (x - 1, y)
        elif action == 3: # Right
            new_pos = (x + 1, y)
        else:             # Stay
            new_pos = (x, y)

        # Checking boundaries and obstacles
        if (0 <= new_pos[0] < self.grid_size) and (0 <= new_pos[1] < self.grid_size):
            if new_pos not in self.obstacles:
                self.robot_pos = new_pos

        # Computing the new distance
        new_x, new_y = self.robot_pos
        new_dist = abs(new_x - self.target_pos[0]) + abs(new_y - self.target_pos[1])

        # Distance improvement (positive if we moved closer to the target)
        distance_improvement = old_dist - new_dist

        # Base step penalty
        reward = -0.1

        # Reward/penalize distance change
        reward += 0.2 * distance_improvement

        # Extra penalty for choosing 'Stay', added this because the first trained models were just staying still
        if action == 4:  # Stay
            reward -= 0.1

        # Check if reached the target
        if self.robot_pos == self.target_pos:
            reward += 10.0
            self.done = True

        # If we run out of steps
        if self.steps >= self.max_steps:
            self.done = True

        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        rx, ry = self.robot_pos
        tx, ty = self.target_pos

        # Distances to obstacle/boundary in each direction
        dist_up = 0
        for i in range(1, ry + 1):
            if (rx, ry - i) in self.obstacles:
                break
            dist_up += 1

        dist_down = 0
        for i in range(1, self.grid_size - ry):
            if (rx, ry + i) in self.obstacles:
                break
            dist_down += 1

        dist_left = 0
        for i in range(1, rx + 1):
            if (rx - i, ry) in self.obstacles:
                break
            dist_left += 1

        dist_right = 0
        for i in range(1, self.grid_size - rx):
            if (rx + i, ry) in self.obstacles:
                break
            dist_right += 1

        # Normalize positions and distances by grid_size
        obs = np.array([
            rx / self.grid_size,
            ry / self.grid_size,
            tx / self.grid_size,
            ty / self.grid_size,
            dist_up / self.grid_size,
            dist_down / self.grid_size,
            dist_left / self.grid_size,
            dist_right / self.grid_size
        ], dtype=np.float32)

        return obs

    def render(self, mode='human'):
        pass
