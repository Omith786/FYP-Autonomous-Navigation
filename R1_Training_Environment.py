"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

R1 RL Environment Definition
---------------------------------------------------------------
Defines a Gymnasium-compatible navigation environment for a point robot
with 3 simulated ultrasonic sensors. The sensors are positioned around the different sides of the robot so
it can have a wide visualization of the objects and obstacles around it. 
The robot must reach a goal while avoiding
rectangular obstacles. Supports curriculum learning by normalising observations
and configuring map difficulty.

Key features:
- Normalised sonar and goal distance readings (for consistent input).
- Axis-aligned rectangular static and optional moving obstacles.
- Reward shaping based on goal progress, sparse collision/success signals.
- Realistic sensor noise simulation.
---------------------------------------------------------------
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Tuple, Dict, Any, Optional


class BasicObstacleEnv(gym.Env):
    """
    A Gym-compatible environment where a point robot navigates toward a goal
    using ultrasonic sensors to avoid various types of obstacles.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        size: float = 6.0,
        num_obstacles: int = 10,
        obstacle_size_range: Tuple[float, float] = (0.3, 0.8),
        moving: bool = False,
        sensor_noise_std: float = 0.0,  # Gaussian noise to apply to sonar readings
        max_episode_steps: int = 400,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Set environment parameters
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacle_size_range = obstacle_size_range
        self.moving_flag = moving
        self.sensor_noise_std = sensor_noise_std  # Save noise level for observation injection
        self.max_episode_steps = max_episode_steps
        self.rng = np.random.default_rng(seed)

        # Observation space: 3 sonar readings, goal bearing (degrees), goal distance
        low = np.array([0.0, 0.0, 0.0, -180.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 180.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: 0 = forward, 1 = turn left, 2 = turn right
        self.action_space = spaces.Discrete(3)

        # Placeholder internal states
        self.robot_pos: np.ndarray
        self.robot_angle: float
        self.goal_pos: np.ndarray
        self.obstacles: List[Dict[str, Any]]
        self.steps_taken: int
        self.prev_goal_dist: float

        self.reset()

    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Try placing robot and goal safely, retrying if any collision occurs after obstacle placement
        for attempt in range(50):
            self.steps_taken = 0
            self.robot_pos = self._sample_point(0.6)
            self.goal_pos = self._sample_point(0.6)

            if np.linalg.norm(self.goal_pos - self.robot_pos) <= self.size * 0.4:
                continue  # Too close

            # Building static/moving obstacles
            self.obstacles = []
            success = True
            for i in range(self.num_obstacles):
                try:
                    self._add_obstacle(movable=(self.moving_flag and i == 0))
                except RuntimeError:
                    success = False
                    break

            if not success:
                continue

            if self._in_collision(self.robot_pos) or self._in_collision(self.goal_pos):
                continue

            # Success
            break
        else:
            raise RuntimeError("Failed to place robot, goal, and obstacles after 50 attempts")

        self.robot_angle = self.rng.uniform(0, 360)
        self.prev_goal_dist = np.linalg.norm(self.goal_pos - self.robot_pos)
        return self._get_obs(), {}

    def step(self, action: int):
        self.steps_taken += 1
        assert self.action_space.contains(action)

        fwd = 0.12
        turn = 18.0

        # Apply action
        if action == 0:  # Move forward
            delta = fwd * np.array([
                math.cos(math.radians(self.robot_angle)),
                math.sin(math.radians(self.robot_angle)),
            ])
            candidate = self.robot_pos + delta
            if not self._in_collision(candidate):
                self.robot_pos = candidate

        elif action == 1:  # Turn left
            self.robot_angle = (self.robot_angle - turn) % 360

        elif action == 2:  # Turn right
            self.robot_angle = (self.robot_angle + turn) % 360

        # Update obstacle (if any moving)
        if self.moving_flag:
            self._update_moving_obstacle()

        # Calculate reward based on progress toward goal
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos)
        reward = -0.005 + 0.1 * (self.prev_goal_dist - dist_to_goal)
        self.prev_goal_dist = dist_to_goal

        # Termination logic
        terminated = False
        truncated = False

        if self._in_collision(self.robot_pos):
            reward -= 1.0
            terminated = True
        if dist_to_goal < 0.35:
            reward += 2.0
            terminated = True
        if np.any(self.robot_pos < 0.0) or np.any(self.robot_pos > self.size):
            reward -= 1.0
            terminated = True
        if self.steps_taken >= self.max_episode_steps:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        arena_diag = math.sqrt(2) * self.size
        sonars = [
            self._ultra(self.robot_angle),
            self._ultra(self.robot_angle - 45),
            self._ultra(self.robot_angle + 45),
        ]
        sonars = np.clip(np.array(sonars) / arena_diag, 0.0, 1.0)

        # Injecting Gaussian noise to sonar values to make it more realistic
        if self.sensor_noise_std > 0:
            noise = self.rng.normal(0, self.sensor_noise_std, size=3)
            sonars += noise
            sonars = np.clip(sonars, 0.0, 1.0)

        vec = self.goal_pos - self.robot_pos
        dist = np.linalg.norm(vec)
        bearing = (math.degrees(math.atan2(vec[1], vec[0])) - self.robot_angle + 540) % 360 - 180
        dist_norm = dist / arena_diag

        return np.array([sonars[0], sonars[1], sonars[2], bearing, dist_norm], dtype=np.float32)

    def _sample_point(self, margin: float) -> np.ndarray:
        return self.rng.uniform(margin, self.size - margin, size=2)

    def _add_obstacle(self, movable: bool):
        for _ in range(200):
            shape_choice = self.rng.uniform(0, 1)
            w = self.rng.uniform(*self.obstacle_size_range)
            h = self.rng.uniform(*self.obstacle_size_range)
            c = self._sample_point(max(w, h))

            if shape_choice < 0.1:
                # 10% chance: Circle
                radius = min(w, h) / 2
                shape = {"c": c, "r": radius, "shape": "circle", "movable": movable, "vel": np.array([0.02, 0.0])}
            elif shape_choice < 0.2:
                # 10% chance: pillar (narrow rectangle) , was trying to make it seem like a wall or door
                w_pillar = self.rng.uniform(0.1, 0.2)
                h_pillar = self.rng.uniform(0.5, 1.0)
                shape = {"c": c, "w": w_pillar, "h": h_pillar, "angle": 0.0, "shape": "rect", "movable": movable, "vel": np.array([0.02, 0.0])}
            elif shape_choice < 0.5:
                # 30% chance: Rotated rectangle
                angle = self.rng.uniform(-15, 15)
                shape = {"c": c, "w": w, "h": h, "angle": angle, "shape": "rect", "movable": movable, "vel": np.array([0.02, 0.0])}
            else:
                # 50% chance: Axis-aligned rectangle
                shape = {"c": c, "w": w, "h": h, "angle": 0.0, "shape": "rect", "movable": movable, "vel": np.array([0.02, 0.0])}

            if not self._overlaps(shape):
                self.obstacles.append(shape)
                return

        raise RuntimeError("Could not place obstacle without overlap")

    def _overlaps(self, r):
        cx, cy = r["c"]
        for o in self.obstacles:
            dx = abs(cx - o["c"][0])
            dy = abs(cy - o["c"][1])
            if dx < 1.0 and dy < 1.0:
                return True
        return False
    
# This function checks if a point is in collision with any obstacle
    # It checks both circular and rectangular obstacles

    def _in_collision(self, p: np.ndarray) -> bool:
        px, py = p
        for o in self.obstacles:
            if o.get("shape") == "circle":
                if np.linalg.norm(p - o["c"]) < o["r"] + 0.15:
                    return True
            else:
                ox, oy = o["c"]
                w, h = o["w"], o["h"]
                angle = o.get("angle", 0.0)
                if angle == 0.0:
                    if abs(px - ox) < (w / 2 + 0.15) and abs(py - oy) < (h / 2 + 0.15):
                        return True
                else:
                    # Rotate point in reverse to align axis
                    theta = math.radians(-angle)
                    dx, dy = px - ox, py - oy
                    rx = dx * math.cos(theta) - dy * math.sin(theta)
                    ry = dx * math.sin(theta) + dy * math.cos(theta)
                    if abs(rx) < (w / 2 + 0.15) and abs(ry) < (h / 2 + 0.15):
                        return True
        return False
    
# Ultra-sonar simulation: returns distance until collision or max range

    def _ultra(self, angle: float) -> float:
        max_len = math.sqrt(2) * self.size
        step = 0.05
        d = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        pos = np.copy(self.robot_pos)
        travelled = 0.0
        while travelled < max_len:
            pos += step * d
            travelled += step
            if self._in_collision(pos):
                return travelled
        return max_len

 # Move obstacle and reverse direction on bounds

    def _update_moving_obstacle(self):
        o = self.obstacles[0]
        o["c"] += o["vel"]
        if o["c"][0] < o.get("w", o.get("r", 0.2)) / 2 or o["c"][0] > self.size - o.get("w", o.get("r", 0.2)) / 2:
            o["vel"][0] *= -1
        if o["c"][1] < o.get("h", o.get("r", 0.2)) / 2 or o["c"][1] > self.size - o.get("h", o.get("r", 0.2)) / 2:
            o["vel"][1] *= -1

    def render(self, mode: str = "human"):
        pass  # Placeholder for rendering logic, if needed
