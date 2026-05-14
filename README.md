# UR5e Circular Trajectory Control — MuJoCo

Control the **Universal Robots UR5e** robot arm in [MuJoCo](https://mujoco.org/) so that its end-effector tracks a **circular trajectory** in the plane of your choice (XY, XZ, or YZ).

The controller is a **Cartesian-space PID** coupled with a **Jacobian-based IK** (Damped Least Squares pseudo-inverse). At the end of every simulation run a plot is generated showing the real vs. target trajectory and the tracking error over time.

---

## Demo

```
==================================================
  UR5e — Seleziona il piano della traiettoria
==================================================
  [1]  Piano XY  (movimento orizzontale)
  [2]  Piano XZ  (movimento verticale frontale)
  [3]  Piano YZ  (movimento verticale laterale)
==================================================
  Scelta [1/2/3]: 2
```

The MuJoCo viewer opens in real-time. Close it to generate the plot.

---

## Features

- Real-time 3D viewer via `mujoco.viewer`
- Interactive menu to select the trajectory plane at startup
- PID controller in Cartesian space with anti-windup
- Damped Least Squares (DLS) pseudo-inverse to handle kinematic singularities
- Joint velocity saturation
- End-of-simulation plot: 3D trajectory + tracking error [mm] over time
- Plot saved automatically as `traiettoria_ur5.png`

---

## Requirements

| Dependency   | Tested version |
|--------------|---------------|
| Python       | 3.11          |
| mujoco       | ≥ 3.0         |
| numpy        | ≥ 1.23        |
| matplotlib   | ≥ 3.7         |

The UR5e model is downloaded automatically from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) on first run — no manual download needed.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<YOUR_USERNAME>/ur5-circular-trajectory.git
cd ur5-circular-trajectory

# 2. (Optional) create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install mujoco numpy matplotlib
```

---

## Usage

```bash
python ur5_circular_trajectory.py
```

1. Select the trajectory plane from the menu (1 = XY, 2 = XZ, 3 = YZ).
2. The MuJoCo viewer opens — watch the robot track the circle in real time.
3. Close the viewer window to stop the simulation and generate the plot.
4. The plot is displayed and saved as `traiettoria_ur5.png` in the project folder.

---

## Configuration

All parameters are at the top of `ur5_circular_trajectory.py`:

| Parameter        | Default              | Description                              |
|------------------|----------------------|------------------------------------------|
| `CIRCLE_RADIUS`  | `0.15` m             | Radius of the circular trajectory        |
| `CIRCLE_CENTER`  | `[0.4, 0.0, 0.5]` m  | Center of the circle in world frame      |
| `CIRCLE_FREQUENCY` | `0.2` Hz           | How fast the end-effector travels        |
| `KP`             | `50.0`               | Proportional gain                        |
| `KI`             | `5.0`                | Integral gain (removes steady-state error)|
| `KD`             | `2.0`                | Derivative gain (damping)                |
| `LAMBDA_DLS`     | `5e-3`               | DLS damping factor (singularity handling)|
| `MAX_JOINT_VEL`  | `2.0` rad/s          | Joint velocity saturation limit          |
| `I_CLAMP`        | `0.05` m             | Anti-windup clamp on the integral term   |

---

## How It Works

### Circular trajectory

The desired position and velocity are computed analytically:

```
p(t) = center + R · [cos(ωt),  0,  sin(ωt)]   (XZ plane)
v(t) = dp/dt  = R·ω · [−sin(ωt), 0,  cos(ωt)]
```

### Cartesian PID

```
v_cart = v_d + Kp·e + Ki·∫e dt + Kd·ė
```

### Jacobian IK (Damped Least Squares)

```
J† = Jᵀ (J Jᵀ + λ²I)⁻¹
dq = J† · v_cart
```

Joint positions are updated by integrating `dq` at each timestep.

---

## Project Structure

```
ur5-circular-trajectory/
├── ur5_circular_trajectory.py   # Main script
├── README.md                    # This file
├── .gitignore
└── mujoco_menagerie/            # Auto-downloaded on first run (git-ignored)
```

---

## License

MIT License — feel free to use, modify, and distribute.
