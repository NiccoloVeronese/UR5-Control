"""
evaluate_ur5e.py
----------------

Run:
    python evaluate_ur5e.py
    python evaluate_ur5e.py --traj figure_eight
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from stable_baselines3 import SAC
from envs.ur5e_env import UR5eTrackerEnv


def run_episode(model, env):
    obs, _ = env.reset()
    done = False
    ee_positions, target_positions, errors, rewards = [], [], [], []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ee_positions.append(info["ee_pos"].copy())
        target_positions.append(info["target"].copy())
        errors.append(info["tracking_error"])
        rewards.append(reward)

    return (np.array(ee_positions), np.array(target_positions),
            np.array(errors), np.array(rewards))


def plot_results(ee_pos, target_pos, errors, rewards, save_path, traj_name):
    t = np.arange(len(errors))
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"UR5e End-Effector Tracking — {traj_name.replace('_',' ').title()}",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 3D overlay
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax3d.plot(*target_pos.T, "b--", lw=1.5, alpha=0.7, label="Target")
    ax3d.plot(*ee_pos.T, "r-", lw=1.5, alpha=0.9, label="End-effector")
    ax3d.scatter(*ee_pos[0], color="green", s=50, zorder=5, label="Start")
    ax3d.set_xlabel("X (m)"); ax3d.set_ylabel("Y (m)"); ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D trajectory overlay"); ax3d.legend(fontsize=8)

    # Error Over Time
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.plot(t, errors, color="#E85D24", lw=1.2)
    ax_err.axhline(np.mean(errors), color="gray", ls="--", lw=1,
                   label=f"Mean = {np.mean(errors):.4f} m")
    ax_err.fill_between(t, errors, alpha=0.15, color="#E85D24")
    ax_err.set_xlabel("Step"); ax_err.set_ylabel("L2 error (m)")
    ax_err.set_title("Tracking error over time"); ax_err.legend(); ax_err.grid(alpha=0.3)

    # XY Projection
    ax_xy = fig.add_subplot(gs[1, 0])
    ax_xy.plot(target_pos[:,0], target_pos[:,1], "b--", alpha=0.7, label="Target")
    ax_xy.plot(ee_pos[:,0], ee_pos[:,1], "r-", alpha=0.85, label="End-effector")
    ax_xy.set_xlabel("X (m)"); ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY projection"); ax_xy.legend()
    ax_xy.set_aspect("equal"); ax_xy.grid(alpha=0.3)

    # Reward
    ax_rew = fig.add_subplot(gs[1, 1])
    ax_rew.plot(t, rewards, color="#1D9E75", lw=1.0, alpha=0.8)
    w = min(20, len(rewards))
    smoothed = np.convolve(rewards, np.ones(w)/w, mode="valid")
    ax_rew.plot(np.arange(w-1, len(rewards)), smoothed,
                color="#085041", lw=2, label=f"{w}-step avg")
    ax_rew.set_xlabel("Step"); ax_rew.set_ylabel("Reward")
    ax_rew.set_title("Reward per step"); ax_rew.legend(); ax_rew.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Plot salvato → {save_path}")
    plt.close()

    print(f"\n--- Tracking summary UR5e ---")
    print(f"  Mean error  : {np.mean(errors):.4f} m")
    print(f"  Max error   : {np.max(errors):.4f} m")
    print(f"  Final error : {errors[-1]:.4f} m")
    print(f"  Mean reward : {np.mean(rewards):.4f}")
    print(f"  Total steps : {len(errors)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj",  default="circle", choices=["circle", "circle_fast", "figure_eight"])
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results_ur5e")
    model_path  = args.model or os.path.join(results_dir, "best_model.zip")

    if not os.path.exists(model_path):
        print(f"Modello non trovato: {model_path}")
        print("Esegui prima: python train_ur5e.py")
        sys.exit(1)

    env   = UR5eTrackerEnv(trajectory=args.traj)
    model = SAC.load(model_path, env=env)

    print("Running evaluation episode...")
    ee_pos, target_pos, errors, rewards = run_episode(model, env)

    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, f"tracking_ur5e_{args.traj}.png")
    plot_results(ee_pos, target_pos, errors, rewards, plot_path, args.traj)
    env.close()


if __name__ == "__main__":
    main()
