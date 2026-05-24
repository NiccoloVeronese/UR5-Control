# UR5e 3D End-Effector Trajectory Tracking

> Three  approaches to smooth trajectory tracking on a UR5e 6-DOF arm in MuJoCo, from classical control to residual reinforcement learning.

---

## The Core Problem

Trajectory tracking is harder than point reaching. The end-effector must not just *arrive* at a target, it must continuously follow a moving reference with minimal lag, overshoot, and jitter. This project explores three approaches of increasing sophistication, each building on the limitations of the last.

---

## Three Approaches
### Approach 1 — Full RL (SAC) [`envs/ur5e_env.py`](envs/ur5e_env.py) + [`train_ur5e.py`](train_ur5e.py) + [`evaluate_ur5e.py`](evaluate_ur5e.py)

Environment where SAC learns the entire control policy from scratch to track the circular trajectory. Implemented using MuJoCo and Gymnasium.

**State space (35-dim):** end-effector position, current target, error vector, target Cartesian velocity, lookahead error, joint positions and velocities, phase encoding, previous action. The target velocity and lookahead are the key additions that allow the agent to anticipate motion rather than just react to error.

**Action space:** normalised joint velocity commands `[-1, 1]` integrated as position increments (`Δq = a × 0.05 rad/step`), matching how real UR5e controllers operate via `servoj`.

**Reward:**
```python
r = exp(-10 · dist)              # dense exponential tracking signal
  - λ_v · ‖v_ee - v_target‖    # penalise velocity mismatch
  - λ_s · ‖Δaction‖            # penalise jitter
  - λ_e · ‖action‖             # penalise effort
```



Run it:
```bash
python train_ur5e.py --traj circle --steps 300000
tensorboard --logdir results_ur5e/tb_logs    # monitor at localhost:6006
python evaluate_ur5e.py
```

---
### Approach 2 — PID + Jacobian IK [`ur5_circular_trajectory.py`](ur5_circular_trajectory.py)

A classical control baseline. A Cartesian-space PID outputs a desired velocity, which is mapped to joint velocities via a Damped Least Squares Jacobian pseudo-inverse:

```
v_cart = v_desired + Kp·e + Ki·∫e dt + Kd·ė
dq     = J†(q) · v_cart          (DLS pseudo-inverse)
```

**Strengths:** No training required, interpretable, numerically stable near singularities (DLS handles them gracefully), real-time capable.

**Limitations:** Fixed gains — performance degrades at higher frequencies. Integral windup even with clamping. Cannot adapt to model mismatch or unmodelled dynamics.

Run it:
```bash
python ur5_circular_trajectory.py
# Interactive menu: select XY / XZ / YZ plane
```

---

### Approach 3 — PID + Residual SAC [`ur5_residual_rl.py`](ur5_residual_rl.py)

The key insight motivating this approach: **pure RL from scratch on a 6-DOF arm is a hard exploration problem**. Instead of replacing the PID, a SAC agent learns to *correct* the residual errors the PID cannot eliminate:

```
v_cart = v_pid(t)  +  α · v_rl(observation)
```

where `α = 0.3` controls how much authority the RL agent has. This architecture:
- Provides a warm start — the PID already gets the arm near the circle
- Narrows the learning problem to residual correction (small action space)
- Guarantees graceful degradation: if RL fails, the PID still tracks

The agent observes `[pos_error(3), vel_error(3), sin(ωt), cos(ωt)]` — an 8-dimensional state focused on what matters for correction. A warmup of 800 steps per episode lets the PID stabilise before RL starts collecting data, preventing large transient errors from corrupting the replay buffer.

Run it:
```bash
python ur5_residual_rl.py
# Mode [1] Train  — 300 episodes, no viewer, fast
# Mode [2] Demo   — load saved policy, open viewer
# Mode [3] Baseline — PID only with viewer, for comparison
```

---



## Approach Comparison

| | Full RL (SAC) | PID + IK | PID + Residual RL |
|---|---|---|---|
| **Training required** | 500k+ steps (~hours) | None | ~300 episodes (~5 min) |
| **Singularity handling** | Learned implicitly | DLS (explicit) | DLS + learned correction |
| **Adapts to model mismatch** | Yes | No | Partially |
| **Safe by default** | Requires careful tuning | Yes | Yes (PID fallback) |
| **Generalises to new trajectories** | Limited | Yes (analytical) | Limited |


**The residual approach offers the best practical trade-off**: near-instant training, interpretable failure modes, and measurable improvement over the PID baseline on the same trajectory.

---

## Installation

```bash
git clone https://github.com/NiccoloVeronese/UR5-Control.git
cd UR5-Control
pip install mujoco numpy matplotlib torch stable-baselines3 gymnasium tensorboard
```

The UR5e model downloads automatically from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on first run for Approaches 1 and 2. For Approach 3, place the model at `ur5e/scene.xml`.

---

## Repository Structure

```
UR5-Control/
├── ur5_circular_trajectory.py   # Approach 2: PID + Jacobian IK
├── ur5_residual_rl.py           # Approach 3: PID + Residual SAC (self-contained)
├── envs/
│   └── ur5e_env.py              # Approach 1: Gymnasium environment
├── train_ur5e.py                # Approach 1: SAC training script
├── evaluate_ur5e.py             # Evaluation + plot generation
├── ur5e/
│   └── scene.xml                # MuJoCo Menagerie UR5e model (Approach 1)
└── results_ur5e/                # Saved models, checkpoints, TensorBoard logs
```

---

## Trajectories

All trajectories are analytical functions `p(t) → ℝ³`, not pre-recorded waypoints. This gives exact target velocity via finite differences and trivial lookahead — both critical for feedforward control.

| Name | Shape | Frequency | Notes |
|---|---|---|---|
| `circle` | Horizontal circle, r = 0.5m | 0.12 Hz | Primary benchmark |
| `circle_fast` | Smaller circle, r = 0.2m | 0.5 Hz | Tests high-speed tracking |
| `figure_eight` | 3D lemniscate | 0.2 Hz | Tests non-planar tracking |

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| Mean L2 error (mm) | Average Euclidean distance between end-effector and target |
| Steady-state error (mm) | Mean error after the first half of the episode (transient removed) |
| Max error (mm) | Worst-case deviation |
| Smoothness | Mean joint jerk (rad/s²) — lower is better |

---

## Design Choices — Short Note

**Why residual RL over pure RL?** Pure RL on a 6-DOF arm from scratch is a notoriously hard exploration problem, the agent must discover that circular motion is required without any prior. Residual RL sidesteps this by letting the PID solve the easy part, leaving only the correction task for the agent. This reduces sample complexity by roughly an order of magnitude.

**Why SAC?** Continuous control with a smooth action requirement. SAC's entropy regularisation naturally encourages smooth, exploratory policies and its off-policy replay buffer is sample-efficient. PPO (on-policy) would require far more environment steps.

**Why include velocity in the state?** A position-only error signal allows the agent to learn a policy that minimises instantaneous distance without caring about direction of motion. Including `v_target` in the observation forces the agent to match not just where to be, but how fast to move there — the difference between hovering near the circle and actually tracking it.

**Why not use `progress = prev_dist - dist` in the reward?** This term oscillates in sign every step whenever the agent is near the trajectory, creating a reward signal that actively destabilises learning. The agent discovers it can collect repeated positive progress signals by oscillating in place near the closest point on the circle — a degenerate strategy that scores well but tracks nothing.

---

## License

MIT
