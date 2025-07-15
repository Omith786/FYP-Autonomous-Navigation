"""
R1 - FYP

COMP 1682 - Final Year Project
Author: Omith Chowdhury
ID: 001236697

Agent Evaluation Script
------------------------------------------------------------
Evaluates a trained model on the BasicObstacleEnv and visualises each episode.
Also logs performance metrics to CSV and displays summary statistics.

Outputs:
- PNG trajectory plot for each episode (eval_runs/...)
- Evaluation metrics printed to console
- CSV log with per-episode performance
- Mean ± Std of successful episode lengths
- Success rate
- Summary bar plot from eval_log.csv
------------------------------------------------------------

This one needs to be run on the terminal where in 1 line you can specify the model to evaluate and environment parameters.
For example:  
python R1_Evaluate_Model.py --model models/R1_PPO_Stage4_20250407_133512.zip --episodes 30 --size 9.0 --obstacles 20 --moving --out eval_runs/PPO_stage4 


python R1_Evaluate_Model.py --model models/R1_PPO_Stage2_20250407_073835.zip --episodes 50 --size 9.0 --obstacles 20 --moving --out eval_runs/A2C_stage4_test

"""

import argparse
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN, A2C

from R1_Training_Environment import BasicObstacleEnv


# --- Helper: draw a visual summary of a single episode ---
def _plot_episode(ax, env: BasicObstacleEnv, path: List[np.ndarray], wire: bool):
    """
    Renders the trajectory taken by the agent in a matplotlib subplot.
    Obstacles are shaded based on type (static or moving).
    """
    face_static = "none" if wire else "#5c5c5c"     # Static = grey
    face_moving = "none" if wire else "#ff8c00"     # Moving = orange
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

    # Plot trajectory path
    pts = np.asarray(path)
    ax.plot(pts[:, 0], pts[:, 1], "b-", lw=1.4)
    ax.scatter(*pts[0], c="lime", s=90, edgecolors=edge_col, zorder=3)         # Start
    ax.scatter(*env.goal_pos, c="red", marker="*", s=90, edgecolors=edge_col, zorder=3)  # Goal
    ax.set_xlim(0, env.size); ax.set_ylim(0, env.size)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


# --- Main evaluation loop ---
def run_eval(model_path: Path, episodes: int, env_kwargs: dict, out_dir: Path, wire: bool):
    # Automatically load model based on filename (PPO by default)
    model_name = model_path.name.lower()
    if "dqn" in model_name:
        model = DQN.load(model_path, device="cpu")
    elif "a2c" in model_name:
        model = A2C.load(model_path, device="cpu")
    else:
        model = PPO.load(model_path, device="cpu")

    successes, steps_ok, log_rows = 0, [], []
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, episodes + 1):
        env = BasicObstacleEnv(**env_kwargs)
        obs, _ = env.reset()
        path = [env.robot_pos.copy()]
        done = trunc = False
        steps = 0

        while not (done or trunc):
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, trunc, _ = env.step(act)
            path.append(env.robot_pos.copy())
            steps += 1

        # Success is defined by distance to goal being within threshold
        reached = done and np.linalg.norm(env.robot_pos - env.goal_pos) < 0.35
        if reached:
            successes += 1
            steps_ok.append(steps)

        # Save trajectory plot for this episode
        fig, ax = plt.subplots(figsize=(4, 4))
        _plot_episode(ax, env, path, wire)
        status = "SUCCESS" if reached else "FAIL"
        ax.set_title(f"Ep {ep} – {status} in {steps} steps", fontsize=9)
        fig.tight_layout(pad=0.2)
        fig.savefig(out_dir / f"episode_{ep:03}.png", dpi=160)
        plt.close(fig)

        log_rows.append({"episode": ep, "status": status, "steps": steps})
        env.close()

    # --- Save evaluation log ---
    df = pd.DataFrame(log_rows)
    df.to_csv(out_dir / "eval_log.csv", index=False)

    # --- Console summary ---
    print("\nEvaluation complete")
    print(f"Episodes : {episodes}")
    print(f"Successes: {successes} ({(successes/episodes)*100:.1f}%)")
    if steps_ok:
        m, s = np.mean(steps_ok), np.std(steps_ok)
        print(f"Avg steps (successes): {m:.1f} ± {s:.1f}")
    else:
        print("Avg steps: –")

    # --- Summary bar plot: visual overview of success/fail rate ---
    counts = df["status"].value_counts()
    plt.figure(figsize=(4, 3))
    counts.plot(kind="bar", color=["lime" if x == "SUCCESS" else "tomato" for x in counts.index])
    plt.title("Episode Outcomes")
    plt.ylabel("Count"); plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_plot.png")
    plt.close()


# --- Helper: fetches most recent model if one isn’t manually specified ---
def latest_model(models_dir: Path = Path("./models")) -> Path | None:
    zips = sorted(models_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


# --- Argument parser for terminal usage ---
def main():
    parser = argparse.ArgumentParser("Evaluate RL agent on BasicObstacleEnv")
    parser.add_argument("--model", type=Path, default=latest_model(),
                        help="Path to model .zip (default: newest in ./models/)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="How many episodes to run")
    parser.add_argument("--size", type=float, default=9.0,
                        help="Arena size")
    parser.add_argument("--obstacles", type=int, default=20,
                        help="Number of obstacles")
    parser.add_argument("--moving", action="store_true",
                        help="Include a moving obstacle")
    parser.add_argument("--wireframe", action="store_true",
                        help="Use wireframe rendering for plots")
    parser.add_argument("--out", type=Path, default="eval_runs",
                        help="Output directory")
    parser.add_argument("--sensor-noise", type=float, default=0.01,
                        help="Optional: add sensor noise during evaluation")
    args = parser.parse_args()

    if args.model is None or not args.model.exists():
        parser.error("Model file not found – use --model path.zip")

    # Construct env config based on CLI args
    env_kwargs = dict(
        size=args.size,
        num_obstacles=args.obstacles,
        obstacle_size_range=(0.3, 0.7),  # I match the upper-stage training config here
        moving=args.moving,
        sensor_noise_std=args.sensor_noise
    )

    run_eval(args.model, args.episodes, env_kwargs, args.out, args.wireframe)


if __name__ == "__main__":
    main()
# This script is intended to be run from the terminal