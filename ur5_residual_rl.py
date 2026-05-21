"""
UR5e Residual RL — PID + SAC
==============================
- Training: NESSUN viewer, massima velocità
- Demo finale: viewer aperto per vedere il risultato
- Baseline: solo PID, per confronto diretto

Requisiti:
    pip install mujoco numpy matplotlib torch

Utilizzo:
    python ur5_residual_rl.py
"""

import numpy as np
import mujoco
import mujoco.viewer
import os, zipfile, urllib.request, collections, random, time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ─────────────────────────────────────────────────────────────
# 1. Download modello UR5e
# ─────────────────────────────────────────────────────────────

MODEL_DIR = "mujoco_menagerie"
UR5_XML   = os.path.join(MODEL_DIR, "mujoco_menagerie-main",
                         "universal_robots_ur5e", "ur5e.xml")

def download_ur5_model():
    if os.path.exists(UR5_XML):
        print("Modello UR5e già presente.")
        return
    url = ("https://github.com/google-deepmind/mujoco_menagerie/"
           "archive/refs/heads/main.zip")
    print("Scaricamento modello UR5e...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(MODEL_DIR, "menagerie.zip")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(MODEL_DIR)
    os.remove(zip_path)
    print("Modello pronto.\n")

# ─────────────────────────────────────────────────────────────
# 2. Parametri
# ─────────────────────────────────────────────────────────────

CIRCLE_RADIUS    = 0.15
CIRCLE_CENTER    = np.array([0.4, 0.0, 0.5])
CIRCLE_FREQUENCY = 0.2

# PID (invariati)
KP            = 50.0
KI            = 5.0
KD            = 2.0
LAMBDA_DLS    = 5e-3
MAX_JOINT_VEL = 2.0
I_CLAMP       = 0.05

# Residual RL
RL_ALPHA      = 0.3    # peso correzione RL
RL_ACTION_MAX = 0.2    # [m/s] max correzione cartesiana

# Dimensioni
OBS_DIM = 8
ACT_DIM = 3

# Training (senza viewer → molto più veloce)
# dt del modello UR5e Menagerie ≈ 0.002 s
# → 1 giro completo = 1/0.2 Hz = 5 s = ~2500 step
# → aggiungiamo ~1000 step di transiente iniziale
MAX_EPISODE_STEPS = 5000   # ~10 s di simulazione (2+ giri completi)
WARMUP_STEPS      = 800    # step iniziali: solo PID avvicina l'EE al cerchio
                           # (RL non agisce e non impara durante il warmup)
TOTAL_EPISODES    = 300    # episodi totali di training
UPDATE_EVERY      = 10     # gradient update ogni N step (velocizza)
PRINT_EVERY       = 10     # stampa progress ogni N episodi

# SAC
LR_ACTOR     = 3e-4
LR_CRITIC    = 3e-4
LR_ALPHA_ENT = 3e-4
GAMMA        = 0.99
TAU          = 0.005
BUFFER_SIZE  = 100_000
BATCH_SIZE   = 256
LEARN_START  = 1_000

# ─────────────────────────────────────────────────────────────
# 3. Funzioni MuJoCo
# ─────────────────────────────────────────────────────────────

def circle_target(t: float, plane: str):
    omega = 2.0 * np.pi * CIRCLE_FREQUENCY
    a     = omega * t
    if plane == "xy":
        p = CIRCLE_CENTER + np.array([ CIRCLE_RADIUS*np.cos(a),  CIRCLE_RADIUS*np.sin(a), 0.0])
        v = np.array([-CIRCLE_RADIUS*omega*np.sin(a),  CIRCLE_RADIUS*omega*np.cos(a), 0.0])
    elif plane == "xz":
        p = CIRCLE_CENTER + np.array([ CIRCLE_RADIUS*np.cos(a), 0.0,  CIRCLE_RADIUS*np.sin(a)])
        v = np.array([-CIRCLE_RADIUS*omega*np.sin(a), 0.0,  CIRCLE_RADIUS*omega*np.cos(a)])
    else:
        p = CIRCLE_CENTER + np.array([0.0,  CIRCLE_RADIUS*np.cos(a),  CIRCLE_RADIUS*np.sin(a)])
        v = np.array([0.0, -CIRCLE_RADIUS*omega*np.sin(a),  CIRCLE_RADIUS*omega*np.cos(a)])
    return p, v

def get_jacobian(model, data, site_id, nq):
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, site_id)
    return jacp[:, :nq]

def dls_pinv(J, lam=LAMBDA_DLS):
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(m))

def make_obs(err, prev_err, dt, t):
    omega = 2.0 * np.pi * CIRCLE_FREQUENCY
    d_err = (err - prev_err) / dt
    return np.array([
        *np.clip(err,   -0.5, 0.5),
        *np.clip(d_err, -0.5, 0.5),
        np.sin(omega * t),
        np.cos(omega * t),
    ], dtype=np.float32)

# ─────────────────────────────────────────────────────────────
# 4. SAC — reti neurali
# ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x): return self.net(x)

class Actor(nn.Module):
    LOG_STD_MIN, LOG_STD_MAX = -5, 2
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h       = self.shared(obs)
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std  = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t  = dist.rsample()
        y_t  = torch.tanh(x_t)
        act  = y_t * RL_ACTION_MAX
        log_prob = dist.log_prob(x_t) \
                 - torch.log(RL_ACTION_MAX * (1 - y_t.pow(2)) + 1e-6)
        return act, log_prob.sum(-1, keepdim=True)

class TwinCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden)
    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

# ─────────────────────────────────────────────────────────────
# 5. Replay Buffer
# ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buf = collections.deque(maxlen=capacity)

    def push(self, obs, act, rew, next_obs, done):
        self.buf.append((obs.astype(np.float32), act.astype(np.float32),
                         np.float32(rew), next_obs.astype(np.float32), np.float32(done)))

    def sample(self):
        batch = random.sample(self.buf, BATCH_SIZE)
        obs, act, rew, nobs, done = map(np.array, zip(*batch))
        t = lambda x: torch.FloatTensor(x)
        return t(obs), t(act), t(rew).unsqueeze(1), t(nobs), t(done).unsqueeze(1)

    def __len__(self): return len(self.buf)

# ─────────────────────────────────────────────────────────────
# 6. Agente SAC
# ─────────────────────────────────────────────────────────────

class SACAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}\n")

        self.actor         = Actor(OBS_DIM, ACT_DIM).to(self.device)
        self.critic        = TwinCritic(OBS_DIM, ACT_DIM).to(self.device)
        self.critic_target = TwinCritic(OBS_DIM, ACT_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.opt_actor  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.target_entropy = -float(ACT_DIM)
        self.log_alpha_ent  = torch.zeros(1, requires_grad=True, device=self.device)
        self.opt_alpha      = optim.Adam([self.log_alpha_ent], lr=LR_ALPHA_ENT)

        self.buffer      = ReplayBuffer()
        self.total_steps = 0

    @property
    def alpha_ent(self): return self.log_alpha_ent.exp()

    def select_action(self, obs, explore=True):
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if explore:
                act, _ = self.actor.sample(obs_t)
            else:
                mu, _ = self.actor(obs_t)
                act = torch.tanh(mu) * RL_ACTION_MAX
        return act.cpu().numpy()[0]

    def store(self, obs, act, rew, next_obs, done):
        self.buffer.push(obs, act, rew, next_obs, done)
        self.total_steps += 1

    def update(self):
        if len(self.buffer) < LEARN_START:
            return

        obs, act, rew, nobs, done = self.buffer.sample()
        obs  = obs.to(self.device);  act  = act.to(self.device)
        rew  = rew.to(self.device);  nobs = nobs.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_act, next_log_pi = self.actor.sample(nobs)
            q1_t, q2_t = self.critic_target(nobs, next_act)
            q_target = rew + GAMMA * (1 - done) * (
                torch.min(q1_t, q2_t) - self.alpha_ent * next_log_pi)

        q1, q2      = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
        self.opt_critic.zero_grad(); critic_loss.backward(); self.opt_critic.step()

        new_act, log_pi = self.actor.sample(obs)
        q1_new, q2_new  = self.critic(obs, new_act)
        actor_loss = (self.alpha_ent * log_pi - torch.min(q1_new, q2_new)).mean()
        self.opt_actor.zero_grad(); actor_loss.backward(); self.opt_actor.step()

        alpha_loss = -(self.log_alpha_ent * (log_pi + self.target_entropy).detach()).mean()
        self.opt_alpha.zero_grad(); alpha_loss.backward(); self.opt_alpha.step()

        for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_t.data.copy_(TAU * p.data + (1 - TAU) * p_t.data)

    def save(self, path="ur5_sac_residual.pt"):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)
        print(f"  Checkpoint salvato: {path}")

    def load(self, path="ur5_sac_residual.pt"):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        print(f"Modello caricato: {path}")

# ─────────────────────────────────────────────────────────────
# 7. Episodio di simulazione (senza viewer = veloce)
# ─────────────────────────────────────────────────────────────

def run_episode(model_mj, data, ee_id, nq, dt, plane,
                agent, explore=True, viewer=None):
    mujoco.mj_resetData(model_mj, data)
    q_home = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    data.qpos[:nq] = q_home
    mujoco.mj_forward(model_mj, data)

    integral = np.zeros(3)
    prev_err = np.zeros(3)
    log      = {"t": [], "ee": [], "target": [], "error": []}
    total_reward = 0.0
    errors       = []

    for step in range(MAX_EPISODE_STEPS):
        t      = data.time
        mujoco.mj_forward(model_mj, data)
        ee_pos = data.site_xpos[ee_id].copy()

        # PID
        pos_d, vel_d = circle_target(t, plane)
        err           = pos_d - ee_pos
        integral     += err * dt
        integral      = np.clip(integral, -I_CLAMP, I_CLAMP)
        d_err_pid     = (err - prev_err) / dt
        v_pid         = vel_d + KP * err + KI * integral + KD * d_err_pid

        # Warmup: i primi WARMUP_STEPS il PID porta l EE vicino al cerchio.
        # L agente RL non agisce e non impara (evita dati spuri dal transiente).
        in_warmup = (step < WARMUP_STEPS)
        obs       = make_obs(err, prev_err, dt, t)
        if in_warmup:
            action_rl = np.zeros(ACT_DIM, dtype=np.float32)
        else:
            action_rl = agent.select_action(obs, explore=explore)
        v_cart = v_pid + (0.0 if in_warmup else RL_ALPHA) * action_rl

        # IK
        J     = get_jacobian(model_mj, data, ee_id, nq)
        J_inv = dls_pinv(J)
        dq    = J_inv @ v_cart
        norm  = np.linalg.norm(dq)
        if norm > MAX_JOINT_VEL:
            dq *= MAX_JOINT_VEL / norm

        # Step simulazione
        data.qpos[:nq] += dq * dt
        data.qvel[:nq]  = dq
        mujoco.mj_step(model_mj, data)
        if viewer is not None:
            viewer.sync()

        # Transizione RL (solo fuori dal warmup)
        pos_error = np.linalg.norm(err)
        if not in_warmup:
            ee_new       = data.site_xpos[ee_id].copy()
            pos_d_new, _ = circle_target(data.time, plane)
            err_new      = pos_d_new - ee_new
            next_obs     = make_obs(err_new, err, dt, data.time)

            reward = (- pos_error
                      - 0.01 * float(np.dot(action_rl, action_rl))
                      + (0.5 if pos_error < 0.005 else 0.0))

            done = float(step == MAX_EPISODE_STEPS - 1)
            agent.store(obs, action_rl, reward, next_obs, done)

            if step % UPDATE_EVERY == 0:
                agent.update()

            total_reward += reward

        prev_err = err.copy()
        errors.append(pos_error)

        log["t"].append(t)
        log["ee"].append(ee_pos.copy())
        log["target"].append(pos_d.copy())
        log["error"].append(pos_error)

    return total_reward, np.mean(errors), log

# ─────────────────────────────────────────────────────────────
# 8. Demo con viewer (dopo il training)
# ─────────────────────────────────────────────────────────────

def run_demo(model_mj, data, ee_id, nq, dt, plane, agent):
    """Apre il viewer e fa girare la policy allenata in loop."""
    print("\nViewer aperto — premi Ctrl+C o chiudi la finestra per fermare.\n")
    with mujoco.viewer.launch_passive(model_mj, data) as viewer:
        viewer.cam.lookat[:]  = [0.3, 0.0, 0.4]
        viewer.cam.distance   = 1.8
        viewer.cam.azimuth    = 120
        viewer.cam.elevation  = -20

        ep = 0
        while viewer.is_running():
            ep += 1
            print(f"[Demo ep {ep}]")
            tot_r, mean_err, log = run_episode(
                model_mj, data, ee_id, nq, dt, plane,
                agent, explore=False, viewer=viewer
            )
            print(f"  Reward={tot_r:.1f}  Errore medio={mean_err*1000:.2f} mm")

    return log

# ─────────────────────────────────────────────────────────────
# 9. Plot
# ─────────────────────────────────────────────────────────────

def plot_comparison(log_pid, log_rl, plane, reward_hist, error_hist):
    """Mostra PID baseline vs PID+RL affiancati + curva apprendimento."""

    fig = plt.figure(figsize=(20, 5))
    fig.suptitle(f"UR5e — Confronto PID vs Residual RL  (piano {plane.upper()})",
                 fontsize=13, fontweight="bold")

    # ── Traiettoria PID ───────────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1, projection="3d")
    ee_pid  = np.array(log_pid["ee"])
    tgt_pid = np.array(log_pid["target"])
    ax1.plot(tgt_pid[:,0], tgt_pid[:,1], tgt_pid[:,2], "b--", lw=1.5, label="Target")
    ax1.plot(ee_pid[:,0],  ee_pid[:,1],  ee_pid[:,2],  "r-",  lw=1.5, label="EE")
    ax1.set_title("Solo PID"); ax1.legend(fontsize=8)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    _set_equal_axes(ax1, ee_pid, tgt_pid)

    # ── Traiettoria PID+RL ────────────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    ee_rl  = np.array(log_rl["ee"])
    tgt_rl = np.array(log_rl["target"])
    ax2.plot(tgt_rl[:,0], tgt_rl[:,1], tgt_rl[:,2], "b--", lw=1.5, label="Target")
    ax2.plot(ee_rl[:,0],  ee_rl[:,1],  ee_rl[:,2],  "g-",  lw=1.5, label="EE")
    ax2.set_title("PID + RL"); ax2.legend(fontsize=8)
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    _set_equal_axes(ax2, ee_rl, tgt_rl)

    # ── Errore nel tempo: confronto ───────────────────────────
    ax3 = fig.add_subplot(1, 4, 3)
    t_pid = np.array(log_pid["t"])
    t_rl  = np.array(log_rl["t"])
    e_pid = np.array(log_pid["error"]) * 1000
    e_rl  = np.array(log_rl["error"])  * 1000
    ax3.plot(t_pid, e_pid, color="tomato",   lw=1.2, label=f"Solo PID  (media: {e_pid.mean():.1f} mm)")
    ax3.plot(t_rl,  e_rl,  color="seagreen", lw=1.2, label=f"PID + RL  (media: {e_rl.mean():.1f} mm)")
    ax3.set_xlabel("Tempo [s]"); ax3.set_ylabel("Errore [mm]")
    ax3.set_title("Errore tracking — confronto")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.4)

    # ── Curva di apprendimento ────────────────────────────────
    ax4  = fig.add_subplot(1, 4, 4)
    ax4b = ax4.twinx()
    eps  = range(1, len(reward_hist) + 1)
    ax4.plot(eps,  reward_hist, color="steelblue", lw=1.2, label="Reward")
    ax4b.plot(eps, error_hist,  color="coral",     lw=1.2, ls="--", label="Errore [mm]")
    ax4.set_xlabel("Episodio")
    ax4.set_ylabel("Reward cumulativo", color="steelblue")
    ax4b.set_ylabel("Errore medio [mm]", color="coral")
    ax4.set_title("Curva di apprendimento")
    l1, lb1 = ax4.get_legend_handles_labels()
    l2, lb2 = ax4b.get_legend_handles_labels()
    ax4.legend(l1+l2, lb1+lb2, fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "confronto_rl_pid.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {out}")
    plt.show()

def _set_equal_axes(ax, *point_arrays):
    pts = np.vstack(point_arrays)
    mid = pts.mean(axis=0)
    rng = (pts.max(axis=0) - pts.min(axis=0)).max() / 2 + 0.02
    ax.set_xlim(mid[0]-rng, mid[0]+rng)
    ax.set_ylim(mid[1]-rng, mid[1]+rng)
    ax.set_zlim(mid[2]-rng, mid[2]+rng)

# ─────────────────────────────────────────────────────────────
# 10. Menu
# ─────────────────────────────────────────────────────────────

def ask_plane():
    print("\n" + "="*50)
    print("  UR5e — Piano della traiettoria")
    print("="*50)
    print("  [1]  XY   [2]  XZ   [3]  YZ")
    print("="*50)
    while True:
        c = input("  Scelta [1/2/3]: ").strip()
        if c == "1": return "xy"
        if c == "2": return "xz"
        if c == "3": return "yz"

def ask_mode():
    print("\n" + "="*50)
    print("  Modalità")
    print("="*50)
    print("  [1]  Train  — allena SAC (senza viewer, veloce)")
    print("  [2]  Demo   — carica modello e apre il viewer")
    print("  [3]  Baseline — solo PID con viewer")
    print("="*50)
    while True:
        c = input("  Scelta [1/2/3]: ").strip()
        if c in ("1","2","3"): return c

# ─────────────────────────────────────────────────────────────
# 11. Main
# ─────────────────────────────────────────────────────────────

def main():
    download_ur5_model()
    plane = ask_plane()
    mode  = ask_mode()

    model_mj = mujoco.MjModel.from_xml_path(UR5_XML)
    data     = mujoco.MjData(model_mj)
    dt       = model_mj.opt.timestep
    nq       = model_mj.nu

    try:
        ee_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    except KeyError:
        ee_id = model_mj.nsite - 1

    SAVE_PATH = "ur5_sac_residual.pt"
    agent     = SACAgent()

    # ── 1. TRAINING (nessun viewer) ──────────────────────────
    if mode == "1":
        print(f"\n{'='*50}")
        print(f"  TRAINING — {TOTAL_EPISODES} episodi  (nessun viewer)")
        print(f"  Update ogni {UPDATE_EVERY} step  |  Reti: 128 neuroni")
        print(f"{'='*50}\n")

        reward_hist, error_hist = [], []
        t0 = time.time()

        for ep in range(1, TOTAL_EPISODES + 1):
            tot_r, mean_err, log = run_episode(
                model_mj, data, ee_id, nq, dt, plane,
                agent, explore=True, viewer=None        # <── nessun viewer
            )
            reward_hist.append(tot_r)
            error_hist.append(mean_err * 1000)

            if ep % PRINT_EVERY == 0:
                elapsed = time.time() - t0
                eta     = elapsed / ep * (TOTAL_EPISODES - ep)
                print(f"[Ep {ep:3d}/{TOTAL_EPISODES}]  "
                      f"Reward={tot_r:8.1f}  "
                      f"Errore={mean_err*1000:5.2f} mm  "
                      f"Buffer={len(agent.buffer):6d}  "
                      f"ETA={eta/60:.1f} min")

            if ep % 50 == 0:
                agent.save(SAVE_PATH)

        agent.save(SAVE_PATH)
        print(f"\nTraining completato in {(time.time()-t0)/60:.1f} min.")

        # Baseline PID (agente muto, nessun viewer)
        print("\nCalcolo baseline PID per confronto...")
        class DummyAgent:
            total_steps = 0
            def select_action(self, obs, explore=False): return np.zeros(ACT_DIM, np.float32)
            def store(self, *a): pass
            def update(self): pass

        _, _, log_pid = run_episode(model_mj, data, ee_id, nq, dt, plane,
                                    DummyAgent(), explore=False, viewer=None)

        # Demo con viewer — risultato finale
        print("\nOra apro il viewer per mostrare il risultato allenato...")
        log_rl = run_demo(model_mj, data, ee_id, nq, dt, plane, agent)

        plot_comparison(log_pid, log_rl, plane, reward_hist, error_hist)

    # ── 2. DEMO (carica modello, apre viewer) ────────────────
    elif mode == "2":
        if not os.path.exists(SAVE_PATH):
            print(f"Errore: '{SAVE_PATH}' non trovato. Esegui prima il training.")
            return
        agent.load(SAVE_PATH)
        run_demo(model_mj, data, ee_id, nq, dt, plane, agent)

    # ── 3. BASELINE PID (viewer + plot) ─────────────────────
    else:
        print(f"\nBaseline PID puro su piano {plane.upper()}...")

        class DummyAgent:
            total_steps = 0
            def select_action(self, obs, explore=False): return np.zeros(ACT_DIM, np.float32)
            def store(self, *a): pass
            def update(self): pass

        with mujoco.viewer.launch_passive(model_mj, data) as viewer:
            viewer.cam.lookat[:]  = [0.3, 0.0, 0.4]
            viewer.cam.distance   = 1.8
            viewer.cam.azimuth    = 120
            viewer.cam.elevation  = -20
            _, mean_err, log = run_episode(
                model_mj, data, ee_id, nq, dt, plane,
                DummyAgent(), explore=False, viewer=viewer
            )

        print(f"\nPID baseline — Errore medio: {mean_err*1000:.2f} mm")

        # Plot semplice solo PID
        times     = np.array(log["t"])
        ee_pos    = np.array(log["ee"])
        tgt_pos   = np.array(log["target"])
        errors_mm = np.array(log["error"]) * 1000

        fig = plt.figure(figsize=(12, 5))
        fig.suptitle(f"UR5e — Baseline PID  (piano {plane.upper()})", fontsize=13)
        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax3d.plot(tgt_pos[:,0], tgt_pos[:,1], tgt_pos[:,2], "b--", lw=1.5, label="Target")
        ax3d.plot(ee_pos[:,0],  ee_pos[:,1],  ee_pos[:,2],  "r-",  lw=1.5, label="EE")
        ax3d.set_title("Traiettoria 3D"); ax3d.legend()
        _set_equal_axes(ax3d, ee_pos, tgt_pos)
        ax_e = fig.add_subplot(1, 2, 2)
        ax_e.plot(times, errors_mm, color="tomato")
        ax_e.axhline(errors_mm[len(errors_mm)//2:].mean(), color="navy",
                     ls="--", label=f"Media: {errors_mm[len(errors_mm)//2:].mean():.1f} mm")
        ax_e.set_xlabel("Tempo [s]"); ax_e.set_ylabel("Errore [mm]")
        ax_e.set_title("Errore tracking"); ax_e.legend(); ax_e.grid(alpha=0.4)
        plt.tight_layout()
        plt.savefig("baseline_pid.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    main()
