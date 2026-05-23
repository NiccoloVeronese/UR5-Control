"""
ur5e_env.py
-----------
Gymnasium environment per il tracking di traiettorie 3D con il braccio UR5e.
Usa MuJoCo direttamente (senza wrapper Gymnasium) per il controllo completo.

Caratteristiche:
  - Braccio UR5e a 6 DOF — workspace sferico ampio (~0.85m di raggio)
  - Traiettorie: cerchio e figura a otto in 3D
  - Reward: accuratezza + smoothness
  - Incertezza: rumore gaussiano sulle azioni
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Trajectory generators — calibrate sul workspace reale del UR5e
# ---------------------------------------------------------------------------

def circle_trajectory(t: float, radius: float = 0.50, freq: float = 0.02) -> np.ndarray:
    angle = 2 * np.pi * freq * t
    x = -0.005 + radius * np.cos(angle)
    y =  0.003 + radius * np.sin(angle)
    z =  0.35
    return np.array([x, y, z], dtype=np.float32)


def circle_fast_trajectory(t: float, radius: float = 0.2, freq: float = 0.5) -> np.ndarray:
    angle = 2 * np.pi * freq * t
    x = -0.005 + radius * np.cos(angle)
    y =  0.003 + radius * np.sin(angle)
    z =  0.35
    return np.array([x, y, z], dtype=np.float32)


def figure_eight_trajectory(t: float, scale: float = 0.20, freq: float = 0.2) -> np.ndarray:
    angle = 2 * np.pi * freq * t
    x = -0.005 + scale * np.sin(angle)
    y =  0.003 + scale * np.sin(2 * angle) * 0.5
    z =  0.35 + scale * np.cos(angle) * 0.3
    return np.array([x, y, z], dtype=np.float32)

TRAJECTORIES = {
    "circle": circle_trajectory,
    "circle_fast": circle_fast_trajectory,
    "figure_eight": figure_eight_trajectory,
}


def finite_difference_velocity(traj_fn, t: float, dt: float) -> np.ndarray:
    """Velocita cartesiana target stimata con differenza centrale."""
    t0 = max(0.0, t - dt)
    t1 = t + dt
    return ((traj_fn(t1) - traj_fn(t0)) / (t1 - t0)).astype(np.float32)


# ---------------------------------------------------------------------------
# UR5e Environment
# ---------------------------------------------------------------------------

class UR5eTrackerEnv(gym.Env):
    """
    Environment per il tracking di traiettorie 3D con UR5e in MuJoCo.

    Observation space:
        [0:3]   posizione end-effector (x, y, z)
        [3:6]   posizione target corrente
        [6:9]   errore vettoriale (target - ee_pos)
        [9:12]  velocità target cartesiana
        [12:15] errore al prossimo target
        [15:21] angoli giunti (6 DOF)
        [21:27] velocità giunti (6 DOF)
        [27:29] fase traiettoria sin/cos
        [29:35] azione precedente

    Action space (6-dim):
        Velocità articolari normalizzate [-1, 1] per i 6 giunti.
        Vengono scalate e integrate come posizioni target.

    Uncertainty:
        Rumore gaussiano N(0, noise_std²) aggiunto alle azioni.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        trajectory: str = "circle",
        render_mode: str = None,
        noise_std: float = 0,
        dt: float = 0.02,
        max_steps: int = 15000,
        smoothness_weight: float = 0.1,
        action_weight: float = 0.005,
        velocity_weight: float = 0.05,
        lookahead: float = 0.12,
        model_path: str = None,
    ):
        super().__init__()

        import mujoco

        # --- trova il file XML del UR5e ---
        if model_path is None:
            # cerca nella cartella del file corrente, poi nella cwd
            candidates = [
                os.path.join(os.path.dirname(__file__), "..", "ur5e", "scene.xml"),
                os.path.join(os.getcwd(), "ur5e", "scene.xml"),
                os.path.join(os.path.dirname(__file__), "ur5e", "scene.xml"),
            ]
            model_path = None
            for c in candidates:
                if os.path.exists(c):
                    model_path = os.path.abspath(c)
                    break
            if model_path is None:
                raise FileNotFoundError(
                    "Non trovo ur5e/scene.xml. Assicurati che la cartella ur5e/ "
                    "sia nella stessa directory del file Python."
                )

        print(f"Caricando modello: {model_path}")
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data  = mujoco.MjData(self._model)

        # --- trova l'indice del sito end-effector ---
        # nel modello UR5e di MuJoCo Menagerie il sito si chiama "attachment_site"
        try:
            self._ee_site_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
            )
        except Exception:
            # fallback: usa il body "wrist_3_link"
            self._ee_site_id = None
            self._ee_body_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3_link"
            )

        # --- numero di giunti attuati ---
        self._n_joints = self._model.nu   # dovrebbe essere 6 per UR5e

        # --- traiettoria ---
        if trajectory not in TRAJECTORIES:
            raise ValueError(f"trajectory deve essere uno di {list(TRAJECTORIES.keys())}")
        self._traj_fn = TRAJECTORIES[trajectory]
        self.trajectory_name = trajectory

        # --- config ---
        self.noise_std        = noise_std
        self.dt               = dt
        self.max_steps        = max_steps
        self.smoothness_weight = smoothness_weight
        self.action_weight     = action_weight
        self.velocity_weight   = velocity_weight
        self.lookahead         = lookahead
        self._render_mode     = render_mode

        # --- spaces ---
        obs_dim = 3 + 3 + 3 + 3 + 3 + self._n_joints + self._n_joints + 2 + self._n_joints
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self._n_joints,), dtype=np.float32
        )

        # --- renderer opzionale ---
        self._renderer = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self._model, height=480, width=640)

        # --- stato interno ---
        self._step_count  = 0
        self._t           = 0.0
        self._prev_action = np.zeros(self._n_joints, dtype=np.float32)
        self._prev_ee_pos = None
        self._prev_dist   = None
        self._ctrl_qpos   = np.zeros(self._n_joints, dtype=np.float32)

        # logging
        self.ep_errors  = []
        self.ep_rewards = []

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_ee_pos(self) -> np.ndarray:
        """Posizione 3D dell'end-effector."""
        import mujoco
        mujoco.mj_forward(self._model, self._data)
        if self._ee_site_id is not None:
            return self._data.site_xpos[self._ee_site_id].copy().astype(np.float32)
        else:
            return self._data.xpos[self._ee_body_id].copy().astype(np.float32)

    def _get_obs(self) -> np.ndarray:
        ee_pos = self._get_ee_pos()
        target = self._traj_fn(self._t)
        error  = target - ee_pos
        target_vel = finite_difference_velocity(self._traj_fn, self._t, self.dt)
        next_target = self._traj_fn(self._t + self.lookahead)
        next_error = next_target - ee_pos
        qpos   = self._data.qpos[:self._n_joints].astype(np.float32)
        qvel   = self._data.qvel[:self._n_joints].astype(np.float32)
        phase = np.array(
            [np.sin(2 * np.pi * self._t / max(self.max_steps * self.dt, self.dt)),
             np.cos(2 * np.pi * self._t / max(self.max_steps * self.dt, self.dt))],
            dtype=np.float32,
        )
        return np.concatenate([
            ee_pos,
            target,
            error,
            target_vel,
            next_error,
            qpos,
            qvel,
            phase,
            self._prev_action,
        ]).astype(np.float32)
    # def _compute_reward(self, ee_pos, target, target_vel, action) -> float:
    #     dist = np.linalg.norm(ee_pos - target)
    #     return -dist
    def _compute_reward(self, ee_pos, target, target_vel, action):
        dist = float(np.linalg.norm(target - ee_pos))
        jitter = float(np.linalg.norm(action - self._prev_action))
        effort = float(np.linalg.norm(action))

        tracking = np.exp(-10.0 * dist)          # single smooth tracking term

        if self._prev_ee_pos is not None:
            ee_vel = (ee_pos - self._prev_ee_pos) / self.dt
            vel_error = float(np.linalg.norm(target_vel - ee_vel))
        else:
            vel_error = 0.0

        return (
            tracking
            - self.velocity_weight * vel_error
            - self.smoothness_weight * jitter
            - self.action_weight * effort
        )
    # def _compute_reward(self, ee_pos, target, target_vel, action) -> float:
    #     dist   = float(np.linalg.norm(target - ee_pos))
    #     jitter = float(np.linalg.norm(action - self._prev_action))
    #     effort = float(np.linalg.norm(action))

    #     if self._prev_ee_pos is not None:
    #         ee_vel = (ee_pos - self._prev_ee_pos) / self.dt
    #         velocity_error = float(np.linalg.norm(target_vel - ee_vel))
    #     else:
    #         velocity_error = 0.0

    #     if self._prev_dist is not None:
    #         progress = self._prev_dist - dist
    #     else:
    #         progress = 0.0
    #     self._prev_dist = dist

    #     tracking_reward = 2.0 * np.exp(-12.0 * dist)
    #     precision_bonus = 0.5 if dist < 0.03 else 0.0
    #     return (
    #         tracking_reward
    #         - 4.0 * dist
    #         - self.velocity_weight * velocity_error
    #         - self.smoothness_weight * jitter
    #         - self.action_weight * effort
    #         + 3.0 * progress
    #         + precision_bonus
    #     )
    # # -----------------------------------------------------------------------
    # Gymnasium API
    # -----------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        import mujoco

        # reset MuJoCo
        mujoco.mj_resetData(self._model, self._data)

        # posizione iniziale neutra (tutti i giunti a 0 tranne il secondo
        # che mettiamo a -π/2 per avere il braccio orizzontale davanti)
        self._data.qpos[:self._n_joints] = [0.8392, -1.5621, -2.8631, 0.8532, -2.2047, 2.5318]
        self._data.ctrl[:self._n_joints] = self._data.qpos[:self._n_joints]
        mujoco.mj_forward(self._model, self._data)

        self._step_count  = 0
        self._t           = 0.0
        self._prev_action = np.zeros(self._n_joints, dtype=np.float32)
        self._prev_ee_pos = self._get_ee_pos()
        self._prev_dist   = None
        self._ctrl_qpos   = self._data.qpos[:self._n_joints].copy().astype(np.float32)
        self.ep_errors    = []
        self.ep_rewards   = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        import mujoco

        # --- rumore gaussiano (fonte di incertezza) ---
        noise = self.np_random.normal(0, self.noise_std, size=action.shape).astype(np.float32)
        noisy_action = np.clip(action + noise, -1.0, 1.0)

        # --- applica azione come incremento di posizione articolare ---
        # scala da [-1,1] a ±0.05 radianti per step (movimento lento e controllato)
        delta_scale = 0.03
        new_qpos = self._ctrl_qpos + noisy_action * delta_scale

        # clamp ai limiti degli attuatori definiti nel modello MuJoCo
        ctrl_range = self._model.actuator_ctrlrange[:self._n_joints]
        new_qpos = np.clip(new_qpos, ctrl_range[:, 0], ctrl_range[:, 1])
        self._data.ctrl[:self._n_joints] = new_qpos
        self._ctrl_qpos = new_qpos.astype(np.float32)

        # --- step simulazione (più substep per stabilità) ---
        n_substeps = max(1, int(self.dt / self._model.opt.timestep))
        for _ in range(n_substeps):
            mujoco.mj_step(self._model, self._data)

        # --- reward e obs ---
        self._t += self.dt
        target  = self._traj_fn(self._t)
        target_vel = finite_difference_velocity(self._traj_fn, self._t, self.dt)
        ee_pos  = self._get_ee_pos()
        reward  = self._compute_reward(ee_pos, target, target_vel, action)

        self._step_count  += 1
        self._prev_ee_pos  = ee_pos.copy()
        self._prev_action  = action.copy()

        error = float(np.linalg.norm(target - ee_pos))
        self.ep_errors.append(error)
        self.ep_rewards.append(reward)

        obs       = self._get_obs()
        terminated = False
        truncated  = self._step_count >= self.max_steps
        info = {
            "tracking_error": error,
            "ee_pos": ee_pos,
            "target": target,
            "t": self._t,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._renderer is None:
            return None
        import mujoco
        self._renderer.update_scene(self._data, camera="track_ee"
                                    if "track_ee" in [self._model.cam(i).name
                                                      for i in range(self._model.ncam)]
                                    else 0)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
