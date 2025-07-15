"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

Bug2 Evaluation Script
------------------------------------------------------------
Simulates a classical Bug2 navigation strategy on BasicObstacleEnv
for multiple episodes. Outputs trajectory plots and evaluation metrics
for comparison with RL agents.
------------------------------------------------------------
Parts of this script, including the plotting and evaluation
logic, are adapted from R1_Evaluate_Model.py. Couldnt use R1_Evaluate_Model.py
as it expects a trained model from Stable-Baselines3, but Bug2 uses a hardcoded policy.

Run it with this: python R1_Bug2_Classical.py --episodes 30 --out eval_runs/Bug2
"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from R1_Training_Environment import BasicObstacleEnv

# --- Helper: plot trajectory of a single episode ---
# Reused from R1_Evaluate_Model.py 

def _plot_episode(ax, env: BasicObstacleEnv, path: List[np.ndarray], wire: bool):
    face_static = "none" if wire else "#5c5c5c"
    face_moving = "none" if wire else "#ff8c00"
    edge_col = "#202020"

    for o in env.obstacles:
        if o.get("shape") == "circle":
            cx, cy = o["c"]
            r = o["r"]
            face = face_moving if o.get("movable", False) else face_static
            ax.add_patch(plt.Circle((cx, cy), r, edgecolor=edge_col, facecolor=face, linewidth=1.0))
        elif ("c" in o) and {"w", "h"}.issubset(o):
            cx, cy = o["c"]
            w, h = o["w"], o["h"]
            angle = o.get("angle", 0.0)
            face = face_moving if o.get("movable", False) else face_static
            rect = plt.Rectangle((cx - w/2, cy - h/2), w, h,
                                 edgecolor=edge_col, facecolor=face,
                                 linewidth=1.0)
            if angle != 0.0:
                t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(cx, cy, angle) + ax.transData
                rect.set_transform(t)
            ax.add_patch(rect)

    pts = np.asarray(path)
    ax.plot(pts[:, 0], pts[:, 1], "b-", lw=1.4)
    ax.scatter(*pts[0], c="lime", s=90, edgecolors=edge_col, zorder=3)
    ax.scatter(*env.goal_pos, c="red", marker="*", s=90, edgecolors=edge_col, zorder=3)
    ax.set_xlim(0, env.size); ax.set_ylim(0, env.size)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

# --- Bug2 policy state ---

class Bug2Agent:
    def __init__(self, env: BasicObstacleEnv):
        self.env = env
        self.goal_line = self._compute_goal_line()
        self.following_obstacle = False

    def _compute_goal_line(self):
        # Line equation: ax + by + c = 0 for the direct line from start to goal
        start, goal = self.env.robot_pos, self.env.goal_pos
        dx, dy = goal - start
        a = dy
        b = -dx
        c = dx * start[1] - dy * start[0]
        return a, b, c

    def _on_goal_line(self):
        a, b, c = self.goal_line
        x, y = self.env.robot_pos
        dist = abs(a * x + b * y + c) / np.hypot(a, b)
        return dist < 0.2

    def decide_action(self):
        sonar_center = self.env._ultra(self.env.robot_angle)
        vec = self.env.goal_pos - self.env.robot_pos
        angle_to_goal = (np.degrees(np.arctan2(vec[1], vec[0])) - self.env.robot_angle + 360) % 360

        if angle_to_goal > 180:
            angle_to_goal -= 360

        if not self.following_obstacle:
            if sonar_center > 0.3:
                return 0  # Move forward
            else:
                self.following_obstacle = True
                return 2  # Turn right
        else:
            if self._on_goal_line() and sonar_center > 0.3:
                self.following_obstacle = False
                return 0
            elif sonar_center > 0.3:
                return 0
            else:
                return 2

# --- Evaluation logic ---
# Based on structure from R1_Evaluate_Model.py, adapted for hardcoded agent ---

def run_bug_eval(episodes: int, env_kwargs: dict, out_dir: Path, wire: bool):
    successes, steps_ok, log_rows = 0, [], []
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, episodes + 1):
        env = BasicObstacleEnv(**env_kwargs)
        obs, _ = env.reset()
        path = [env.robot_pos.copy()]
        done = trunc = False
        steps = 0
        agent = Bug2Agent(env)

        while not (done or trunc):
            act = agent.decide_action()
            obs, _, done, trunc, _ = env.step(act)
            path.append(env.robot_pos.copy())
            steps += 1

        reached = done and np.linalg.norm(env.robot_pos - env.goal_pos) < 0.35
        if reached:
            successes += 1
            steps_ok.append(steps)

        fig, ax = plt.subplots(figsize=(4, 4))
        _plot_episode(ax, env, path, wire)
        status = "SUCCESS" if reached else "FAIL"
        ax.set_title(f"Ep {ep} – {status} in {steps} steps", fontsize=9)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_dir / f"episode_{ep:03}.png", dpi=160)
        plt.close(fig)

        log_rows.append({"episode": ep, "status": status, "steps": steps})
        env.close()

    df = pd.DataFrame(log_rows)
    df.to_csv(out_dir / "eval_log.csv", index=False)

    print("\nBug2 Evaluation complete")
    print(f"Episodes : {episodes}")
    print(f"Successes: {successes} ({(successes/episodes)*100:.1f}%)")
    if steps_ok:
        m, s = np.mean(steps_ok), np.std(steps_ok)
        print(f"Avg steps (successes): {m:.1f} ± {s:.1f}")
    else:
        print("Avg steps: –")

    counts = df["status"].value_counts()
    plt.figure(figsize=(4, 3))
    counts.plot(kind="bar", color=["lime" if x == "SUCCESS" else "tomato" for x in counts.index])
    plt.title("Bug2 Episode Outcomes")
    plt.ylabel("Count"); plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_plot.png")
    plt.close()

# --- Main entry point ---
# Like R1_Evaluate_Model.py, this defines CLI usage ---

def main():
    parser = argparse.ArgumentParser("Bug2 navigation evaluation on BasicObstacleEnv")
    parser.add_argument("--episodes", type=int, default=20, help="How many episodes to run")
    parser.add_argument("--size", type=float, default=9.0, help="Arena size")
    parser.add_argument("--obstacles", type=int, default=20, help="Number of obstacles")
    parser.add_argument("--moving", action="store_true", help="Include a moving obstacle")
    parser.add_argument("--wireframe", action="store_true", help="Use wireframe rendering")
    parser.add_argument("--out", type=Path, default="eval_runs/Bug2", help="Output directory")
    parser.add_argument("--sensor-noise", type=float, default=0.01, help="Optional: sensor noise")
    args = parser.parse_args()

    env_kwargs = dict(
        size=args.size,
        num_obstacles=args.obstacles,
        obstacle_size_range=(0.3, 0.7),
        moving=args.moving,
        sensor_noise_std=args.sensor_noise
    )

    run_bug_eval(args.episodes, env_kwargs, args.out, args.wireframe)


if __name__ == "__main__":
    main()
