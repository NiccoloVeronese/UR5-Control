"""
train_ur5e.py
-------------
SAC training for 3D trajectory tracking with UR5e.

Run:
    python train_ur5e.py                         # circle, 300k steps
    python train_ur5e.py --traj figure_eight
    python train_ur5e.py --steps 500000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from envs.ur5e_env import UR5eTrackerEnv


def make_env(trajectory, noise_std):
    def _init():
        env = UR5eTrackerEnv(trajectory=trajectory, noise_std=noise_std)
        return Monitor(env)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", default="circle", choices=["circle", "circle_fast", "figure_eight"])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results_ur5e")
    tb_dir = os.path.join(results_dir, "tb_logs")
    ckpt_dir = os.path.join(results_dir, "checkpoints")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'=' * 55}")
    print(f"  Simulator  : UR5e (MuJoCo Menagerie)")
    print(f"  Trajectory : {args.traj}")
    print(f"  Steps      : {args.steps:,}")
    print(f"  Noise σ    : {args.noise}")
    print(f"  TensorBoard: tensorboard --logdir {tb_dir}")
    print(f"{'=' * 55}\n")

    train_env = make_vec_env(
        make_env(args.traj, args.noise),
        n_envs=args.n_envs,
        seed=args.seed
    )

    eval_env = make_vec_env(
        make_env(args.traj, 0.0),
        n_envs=1,
        seed=args.seed + 99
    )

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=3e-4,
        buffer_size=500_000,
        learning_starts=1,
        batch_size=512,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log=tb_dir,
        verbose=1,
        seed=args.seed,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=results_dir,
        log_path=results_dir,
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=ckpt_dir,
        name_prefix=f"sac_ur5e_{args.traj}",
    )

    model.learn(
        total_timesteps=args.steps,
        callback=[eval_cb, ckpt_cb],
        tb_log_name=f"SAC_UR5e_{args.traj}",
        progress_bar=True,
    )

    model_path = os.path.join(results_dir, f"sac_ur5e_{args.traj}_final")
    model.save(model_path)

    print(f"\nModel saved → {model_path}.zip")
    print("Run: python evaluate_ur5e.py\n")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()