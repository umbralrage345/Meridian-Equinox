"""Visualization and stress testing utilities for Lunar Lander."""

from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon

from meridian.act4_lander.env import VecLander
from meridian.act4_lander.train import evaluate


def record_episode(policy, conditions: dict | None = None, seed: int = 42) -> str:
    """Record a single episode and return it as HTML FuncAnimation."""
    conditions = conditions or {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0}
    env = VecLander(
        n_envs=1,
        seed=seed,
        gravity=conditions.get("gravity", 7.0),
        wind=conditions.get("wind", 2.0),
        fuel_pct=conditions.get("fuel_pct", 100.0),
        sensor_noise=conditions.get("sensor_noise", 0.0),
    )

    obs = env.reset()
    trajectory = []

    for _ in range(1000):
        if env.done[0]:
            break
        if hasattr(policy, "act"):
            act = policy.act(obs)
        else:
            act = policy(obs)
        if isinstance(act, np.ndarray):
            act = int(act[0])

        trajectory.append({
            "x": float(env.x[0]),
            "y": float(env.y[0]),
            "ang": float(env.ang[0]),
            "act": act,
            "vy": float(env.vy[0]),
            "vx": float(env.vx[0]),
        })
        obs, _, _, _ = env.step(act)

    if not trajectory:
        trajectory.append({"x": 0.0, "y": 1.0, "ang": 0.0, "act": 0, "vy": 0.0, "vx": 0.0})

    # Subsample to at most 120 frames
    total_frames = len(trajectory)
    if total_frames > 120:
        indices = np.linspace(0, total_frames - 1, 120, dtype=int)
        frames = [trajectory[i] for i in indices]
    else:
        frames = trajectory

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.1, 1.5)
    ax.set_aspect("equal")
    ax.set_title("Meridian Lander Descent")
    ax.set_xlabel("Lateral Position (m)")
    ax.set_ylabel("Altitude (m)")

    # Draw terrain and landing pad
    ax.axhline(0, color="gray", linewidth=2)
    ax.plot([-0.2, 0.2], [0, 0], color="lime", linewidth=4, label="Pad [-0.2, 0.2]")
    ax.legend(loc="upper right")

    # Base lander triangle vertices centered at (0, 0)
    base_verts = np.array([
        [0.0, 0.06],    # nose
        [-0.04, -0.04], # bottom-left
        [0.04, -0.04],  # bottom-right
    ])
    ship = Polygon(base_verts, closed=True, fc="cyan", ec="blue")
    ax.add_patch(ship)

    flame = Polygon(np.zeros((3, 2)), closed=True, fc="orange", ec="red", visible=False)
    ax.add_patch(flame)

    def update(frame_idx):
        f = frames[frame_idx]
        x, y, ang, act = f["x"], f["y"], f["ang"], f["act"]

        cos_t = np.cos(ang)
        sin_t = np.sin(ang)
        rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Rotate and translate ship
        rotated_verts = (base_verts @ rot_matrix.T) + np.array([x, y])
        ship.set_xy(rotated_verts)

        # Draw engine flame
        if act == 2:
            base_flame = np.array([
                [-0.02, -0.04],
                [0.02, -0.04],
                [0.0, -0.09],
            ])
            flame_verts = (base_flame @ rot_matrix.T) + np.array([x, y])
            flame.set_xy(flame_verts)
            flame.set_visible(True)
        else:
            flame.set_visible(False)

        return ship, flame

    anim = FuncAnimation(fig, update, frames=len(frames), interval=50, blit=False)
    html_output = anim.to_jshtml()
    plt.close(fig)
    return html_output


def plot_learning_curve(history: list[dict]):
    """Plot training history: iterations vs mean reward and landing rate."""
    iterations = [h["iteration"] for h in history]
    rewards = [h["mean_reward"] for h in history]
    rates = [h["landing_rate"] * 100.0 for h in history]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color = "tab:blue"
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mean Reward", color=color)
    ax1.plot(iterations, rewards, color=color, marker="o", label="Mean Reward")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Landing Rate (%)", color=color)
    ax2.plot(iterations, rates, color=color, marker="s", linestyle="--", label="Landing Rate (%)")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.tight_layout()
    plt.title("Lander ES Optimization Learning Curve")
    return fig


def stress_test(policy):
    """Stress test policy under diverse planetary conditions."""
    scenarios = {
        "Default (g=7, w=2)": {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0},
        "Mars-like (g=3.7)": {"gravity": 3.7, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0},
        "Strong Wind (w=6)": {"gravity": 7.0, "wind": 6.0, "fuel_pct": 100.0, "sensor_noise": 0.0},
        "Heavy Planet (g=12)": {"gravity": 12.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0},
    }

    rates = []
    rewards = []
    names = list(scenarios.keys())

    for name, cond in scenarios.items():
        res = evaluate(policy, cond, n_episodes=30, seed=42)
        rates.append(res["landing_rate"] * 100.0)
        rewards.append(res["mean_reward"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bars = ax1.bar(names, rates, color=["steelblue", "coral", "mediumseagreen", "purple"])
    ax1.set_ylabel("Landing Rate (%)")
    ax1.set_title("Landing Rate Across Planetary Conditions")
    ax1.set_ylim(0, 105)
    ax1.tick_params(axis="x", rotation=15)
    for b in bars:
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 2, f"{b.get_height():.1f}%", ha="center")

    ax2.bar(names, rewards, color=["steelblue", "coral", "mediumseagreen", "purple"])
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Official Reward Across Conditions")
    ax2.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    return fig


def human_vs_ai(conditions: dict | None = None) -> dict[str, Any]:
    """Compare baseline heuristic PD controller against unguided descent."""
    cond = conditions or {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0}
    from meridian.act4_lander.policy import pd_controller
    pd_eval = evaluate(pd_controller, cond, n_episodes=20, seed=42)

    class Unguided:
        def act(self, obs: np.ndarray) -> np.ndarray:
            return np.zeros(obs.shape[0], dtype=int)

    unguided_eval = evaluate(Unguided(), cond, n_episodes=20, seed=42)

    print("\n--- HUMAN VS AI: DESCENT GUIDANCE BENCHMARK ---")
    print(f"Planetary Conditions: Gravity={cond.get('gravity', 7.0):.1f} m/s², Wind={cond.get('wind', 2.0):.1f} m/s²")
    print(f"Unguided Free-Fall : Landing Rate = {unguided_eval['landing_rate'] * 100.0:.1f}%, Mean Reward = {unguided_eval['mean_reward']:.1f}")
    print(f"Heuristic PD Guidance: Landing Rate = {pd_eval['landing_rate'] * 100.0:.1f}%, Mean Reward = {pd_eval['mean_reward']:.1f}")
    print("=" * 60 + "\n")
    return {
        "unguided_landing_rate": unguided_eval["landing_rate"],
        "pd_landing_rate": pd_eval["landing_rate"],
    }

