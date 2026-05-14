"""
UR5 Circular Trajectory Control in MuJoCo — PID + Plot
=======================================================
Controllore PID in spazio cartesiano con pseudo-inversa smorzata dello Jacobiano.
Al termine mostra un grafico 3D + errore nel tempo.

Requisiti:
    pip install mujoco numpy matplotlib
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import urllib.request
import os
import zipfile
import matplotlib
matplotlib.use("TkAgg")          # backend interattivo su Windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ─────────────────────────────────────────────
# 1. Scarica il modello UR5e da MuJoCo Menagerie
# ─────────────────────────────────────────────

MODEL_DIR = "mujoco_menagerie"
UR5_XML   = os.path.join(MODEL_DIR, "mujoco_menagerie-main",
                         "universal_robots_ur5e", "ur5e.xml")

def download_ur5_model():
    if os.path.exists(UR5_XML):
        print("Modello UR5e gia' presente.")
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

# ─────────────────────────────────────────────
# 2. Parametri traiettoria
# ─────────────────────────────────────────────

CIRCLE_RADIUS    = 0.15                       # [m]
CIRCLE_CENTER    = np.array([0.4, 0.0, 0.5]) # [m]
CIRCLE_FREQUENCY = 0.2                        # [Hz]

VALID_PLANES = ["xy", "xz", "yz"]

def ask_plane() -> str:
    """Menu interattivo da terminale per scegliere il piano della traiettoria."""
    print("\n" + "="*50)
    print("  UR5e — Seleziona il piano della traiettoria")
    print("="*50)
    print("  [1]  Piano XY  (movimento orizzontale)")
    print("  [2]  Piano XZ  (movimento verticale frontale)")
    print("  [3]  Piano YZ  (movimento verticale laterale)")
    print("="*50)
    while True:
        choice = input("  Scelta [1/2/3]: ").strip()
        if choice == "1":
            print("  -> Piano XY selezionato.\n")
            return "xy"
        elif choice == "2":
            print("  -> Piano XZ selezionato.\n")
            return "xz"
        elif choice == "3":
            print("  -> Piano YZ selezionato.\n")
            return "yz"
        else:
            print("  Inserisci 1, 2 oppure 3.")

# ─────────────────────────────────────────────
# 3. Guadagni PID cartesiano
# ─────────────────────────────────────────────

KP            = 50.0    # proporzionale
KI            = 5.0     # integrale  (elimina offset stazionario)
KD            = 2.0     # derivativo (smorzamento)
LAMBDA_DLS    = 5e-3    # smorzamento pseudo-inversa (Damped Least Squares)
MAX_JOINT_VEL = 2.0     # [rad/s]
I_CLAMP       = 0.05    # anti-windup: satura l'integrale [m]

# ─────────────────────────────────────────────
# 4. Funzioni di supporto
# ─────────────────────────────────────────────

def circle_target(t: float, plane: str):
    omega = 2.0 * np.pi * CIRCLE_FREQUENCY
    a = omega * t
    if plane == "xy":
        p = CIRCLE_CENTER + np.array([ CIRCLE_RADIUS*np.cos(a),  CIRCLE_RADIUS*np.sin(a),  0.0])
        v = np.array([-CIRCLE_RADIUS*omega*np.sin(a),  CIRCLE_RADIUS*omega*np.cos(a),  0.0])
    elif plane == "xz":
        p = CIRCLE_CENTER + np.array([ CIRCLE_RADIUS*np.cos(a),  0.0,  CIRCLE_RADIUS*np.sin(a)])
        v = np.array([-CIRCLE_RADIUS*omega*np.sin(a),  0.0,  CIRCLE_RADIUS*omega*np.cos(a)])
    else:  # yz
        p = CIRCLE_CENTER + np.array([ 0.0,  CIRCLE_RADIUS*np.cos(a),  CIRCLE_RADIUS*np.sin(a)])
        v = np.array([ 0.0, -CIRCLE_RADIUS*omega*np.sin(a),  CIRCLE_RADIUS*omega*np.cos(a)])
    return p, v


def get_jacobian(model, data, site_id):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return jacp


def dls_pinv(J, lam=LAMBDA_DLS):
    """Pseudo-inversa Damped Least Squares: J+ = J^T (J J^T + lam^2 I)^-1"""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(m))

# ─────────────────────────────────────────────
# 5. Plot finale
# ─────────────────────────────────────────────

def plot_results(log: dict, plane: str):
    times      = np.array(log["t"])
    ee_pos     = np.array(log["ee"])      # (N,3)
    target_pos = np.array(log["target"])  # (N,3)
    errors_mm  = np.array(log["error"]) * 1000.0

    fig = plt.figure(figsize=(14, 5))
    fig.suptitle(f"UR5e — Traiettoria circolare piano {plane.upper()} (MuJoCo)", fontsize=13, fontweight="bold")

    # ── Traiettoria 3D ─────────────────────────────────────
    ax3d = fig.add_subplot(1, 3, (1, 2), projection="3d")
    ax3d.plot(target_pos[:, 0], target_pos[:, 1], target_pos[:, 2],
              "b--", linewidth=1.5, label="Target")
    ax3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2],
              "r-", linewidth=1.5, label="End-effector reale")
    ax3d.scatter(*ee_pos[0], color="green", s=60, zorder=5, label="Start")
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_title("Traiettoria 3D")
    ax3d.legend(fontsize=9)

    # Aspect ratio uniforme
    all_pts = np.vstack([ee_pos, target_pos])
    mid = all_pts.mean(axis=0)
    rng = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2 + 0.02
    ax3d.set_xlim(mid[0]-rng, mid[0]+rng)
    ax3d.set_ylim(mid[1]-rng, mid[1]+rng)
    ax3d.set_zlim(mid[2]-rng, mid[2]+rng)

    # ── Errore nel tempo ────────────────────────────────────
    ax_err = fig.add_subplot(1, 3, 3)
    ax_err.plot(times, errors_mm, color="tomato", linewidth=1.2)
    half = len(errors_mm) // 2
    mean_ss = errors_mm[half:].mean()
    ax_err.axhline(mean_ss, color="navy", linestyle="--", linewidth=1,
                   label=f"Media regime: {mean_ss:.1f} mm")
    ax_err.set_xlabel("Tempo [s]")
    ax_err.set_ylabel("Errore posizione [mm]")
    ax_err.set_title("Errore tracking nel tempo")
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.4)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traiettoria_ur5.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Grafico salvato: {out_path}")
    plt.show()

# ─────────────────────────────────────────────
# 6. Simulazione principale
# ─────────────────────────────────────────────

def run_simulation():
    download_ur5_model()

    # Selezione interattiva del piano
    plane = ask_plane()

    model = mujoco.MjModel.from_xml_path(UR5_XML)
    data  = mujoco.MjData(model)

    # Trova sito end-effector
    ee_name = "attachment_site"
    try:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
    except KeyError:
        ee_id   = model.nsite - 1
        ee_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, ee_id)
    print(f"End-effector: '{ee_name}' (id={ee_id})")
    print(f"Piano: {plane.upper()}  |  Raggio: {CIRCLE_RADIUS} m  |  "
          f"Centro: {CIRCLE_CENTER}  |  Freq: {CIRCLE_FREQUENCY} Hz\n")

    # Configurazione iniziale (home)
    q_home = np.array([0.0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0.0])
    data.qpos[:model.nu] = q_home
    mujoco.mj_forward(model, data)

    dt = model.opt.timestep
    nq = model.nu

    # Stato PID
    integral = np.zeros(3)
    prev_err = np.zeros(3)

    # Log per il plot
    log = {"t": [], "ee": [], "target": [], "error": []}

    print("Simulazione avviata. Chiudi il viewer per generare il grafico.\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:]  = [0.3, 0.0, 0.4]
            viewer.cam.distance   = 1.8
            viewer.cam.azimuth    = 120
            viewer.cam.elevation  = -20

            while viewer.is_running():
                step_start = time.time()
                t = data.time

                mujoco.mj_forward(model, data)
                ee_pos = data.site_xpos[ee_id].copy()

                # Target circolare
                pos_d, vel_d = circle_target(t, plane)
                err = pos_d - ee_pos

                # Integrale con anti-windup (clamping)
                integral += err * dt
                integral  = np.clip(integral, -I_CLAMP, I_CLAMP)

                # Derivativo
                d_err    = (err - prev_err) / dt
                prev_err = err.copy()

                # Legge PID cartesiana
                v_cart = vel_d + KP * err + KI * integral + KD * d_err

                # Cinematica inversa con Jacobiano
                J     = get_jacobian(model, data, ee_id)[:, :nq]
                J_inv = dls_pinv(J)
                dq    = J_inv @ v_cart

                # Saturazione velocita' giunti
                norm = np.linalg.norm(dq)
                if norm > MAX_JOINT_VEL:
                    dq *= MAX_JOINT_VEL / norm

                # Integrazione in posizione
                data.qpos[:nq] += dq * dt
                data.qvel[:nq]  = dq
                mujoco.mj_step(model, data)
                viewer.sync()

                # Salva dati per il plot
                log["t"].append(t)
                log["ee"].append(ee_pos.copy())
                log["target"].append(pos_d.copy())
                log["error"].append(np.linalg.norm(err))

                # Console ogni 0.5 s
                if int(t * 2) != int((t - dt) * 2):
                    print(f"t={t:6.2f}s | EE={np.round(ee_pos,4)} | "
                          f"Target={np.round(pos_d,4)} | "
                          f"Errore={np.linalg.norm(err)*1000:.1f} mm")

                # Real-time pacing
                elapsed = time.time() - step_start
                if dt - elapsed > 0:
                    time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nSimulazione interrotta (Ctrl+C).")

    # Genera sempre il plot alla fine
    if len(log["t"]) > 10:
        print("\nGenerazione grafico in corso...")
        plot_results(log, plane)
    else:
        print("Dati insufficienti per generare il grafico.")


if __name__ == "__main__":
    run_simulation()
