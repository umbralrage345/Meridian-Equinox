"""Visualization and stress testing utilities for Lunar Lander."""

from __future__ import annotations
import base64
import io
import time
from typing import Any
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Polygon, Rectangle
from PIL import Image

from meridian.act4_lander.env import VecLander
from meridian.act4_lander.train import evaluate


class _RandomPolicy:
    """Random action policy for visual benchmark comparison."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def act(self, obs: np.ndarray) -> np.ndarray:
        return self.rng.randint(0, 4, size=obs.shape[0])


def _simulate_flight(policy, conditions: dict, seed: int, max_steps: int = 250) -> list[dict]:
    """Simulate a single lander episode and return step state dictionaries."""
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

    for _ in range(max_steps):
        if env.done[0]:
            break
        if hasattr(policy, "act"):
            act = policy.act(obs)
        else:
            act = policy(obs)
        if isinstance(act, (np.ndarray, list)):
            act = int(act[0])
        else:
            act = int(act)

        trajectory.append({
            "x": float(env.x[0]),
            "y": float(env.y[0]),
            "vx": float(env.vx[0]),
            "vy": float(env.vy[0]),
            "ang": float(env.ang[0]),
            "omega": float(env.omega[0]),
            "fuel": float(env.fuel[0]),
            "wind_val": float(env.wind_val[0]),
            "act": act,
            "done": False,
            "landed": False,
            "crashed": False,
            "reward": 0.0,
        })
        obs, _, _, _ = env.step(act)

    landed = bool(env.landed[0])
    crashed = bool(env.crashed[0])
    main_burn = float(env.main_burn_s[0])
    side_burn = float(env.side_burn_s[0])
    dist = float(env.distance_at_end[0])
    speed_penalty = max(0.0, float(env.landing_speed[0]) - 0.5)

    reward_official = (
        100.0 * (1.0 if landed else 0.0)
        - 100.0 * (1.0 if crashed else 0.0)
        - 0.3 * main_burn
        - 0.05 * side_burn
        - 20.0 * dist
        - 20.0 * speed_penalty
    )

    if not trajectory:
        trajectory.append({
            "x": 0.0, "y": 1.0, "vx": 0.0, "vy": 0.0, "ang": 0.0,
            "omega": 0.0, "fuel": 1.0, "wind_val": 0.0, "act": 0,
            "done": False, "landed": False, "crashed": False, "reward": 0.0,
        })

    # Subsample every 2nd step to meet <= 100 frame budget
    subsampled = trajectory[::2]
    if len(subsampled) > 65:
        subsampled = subsampled[:65]

    last = dict(trajectory[-1])
    last["done"] = True
    last["landed"] = landed
    last["crashed"] = crashed
    last["reward"] = reward_official
    last["act"] = 0

    frames = list(subsampled)
    for _ in range(15):
        frames.append(last)

    return frames


def _build_panel(ax, seed: int, conditions: dict):
    """Initialize static elements (starfield, terrain, pad, HUD) on a flight axis."""
    ax.set_facecolor("#05070f")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.1, 1.4)
    ax.set_xticks([])
    ax.set_yticks([])

    # Seeded starfield (~80 stars, a few brighter)
    star_rng = np.random.RandomState(seed + 9999)
    star_x = star_rng.uniform(-1.18, 1.18, 80)
    star_y = star_rng.uniform(0.12, 1.38, 80)
    star_s = star_rng.uniform(1.5, 4.0, 80)
    star_a = star_rng.uniform(0.3, 0.7, 80)
    star_s[:8] = star_rng.uniform(10.0, 18.0, 8)
    star_a[:8] = 0.95
    ax.scatter(star_x, star_y, s=star_s, c="#ffffff", alpha=star_a, edgecolors="none", zorder=1)

    # Lunar ground with gentle undulations and crater arcs
    gx = np.linspace(-1.2, 1.2, 80)
    gy = np.zeros_like(gx)
    for i, x_val in enumerate(gx):
        if abs(x_val) > 0.22:
            gy[i] = 0.012 * np.sin(5.0 * x_val) + 0.008 * np.cos(11.0 * x_val)
        else:
            gy[i] = 0.0
    ground_verts = [(-1.2, -0.1)] + [(gx[i], gy[i]) for i in range(len(gx))] + [(1.2, -0.1)]
    ground = Polygon(ground_verts, closed=True, facecolor="#1e293b", edgecolor="#334155", lw=1.2, zorder=2)
    ax.add_patch(ground)

    craters = [(-0.8, -0.02, 0.2, 0.04), (-0.4, -0.03, 0.15, 0.03), (0.5, -0.02, 0.25, 0.05), (0.9, -0.03, 0.18, 0.04)]
    for cx, cy, cw, ch in craters:
        arc = Arc((cx, cy), cw, ch, theta1=0, theta2=180, color="#0f172a", lw=1.5, zorder=2)
        ax.add_patch(arc)

    # Landing pad and end lights
    pad = Rectangle((-0.2, -0.012), 0.4, 0.016, facecolor="#94a3b8", edgecolor="#e2e8f0", lw=1.5, zorder=3)
    ax.add_patch(pad)
    ax.scatter([-0.2, 0.2], [0.008, 0.008], s=30, color="#22c55e", edgecolors="#86efac", lw=1, zorder=4)

    # Lander geometry (~0.09 world units tall)
    body_patch = Polygon(np.zeros((5, 2)), closed=True, fc="#e2e8f0", ec="#1e3a8a", lw=1.2, zorder=6)
    left_leg, = ax.plot([], [], color="#94a3b8", lw=1.5, zorder=5)
    right_leg, = ax.plot([], [], color="#94a3b8", lw=1.5, zorder=5)
    left_foot, = ax.plot([], [], color="#cbd5e1", lw=2.0, zorder=5)
    right_foot, = ax.plot([], [], color="#cbd5e1", lw=2.0, zorder=5)
    flame_patch = Polygon(np.zeros((3, 2)), closed=True, fc="#f97316", ec="#ef4444", lw=1.0, zorder=5, visible=False)
    left_puff = Polygon(np.zeros((3, 2)), closed=True, fc="#38bdf8", ec="none", alpha=0.8, zorder=5, visible=False)
    right_puff = Polygon(np.zeros((3, 2)), closed=True, fc="#38bdf8", ec="none", alpha=0.8, zorder=5, visible=False)
    trail_line, = ax.plot([], [], color="#38bdf8", lw=1.2, alpha=0.45, linestyle=":", zorder=3)

    ax.add_patch(body_patch)
    ax.add_patch(flame_patch)
    ax.add_patch(left_puff)
    ax.add_patch(right_puff)

    # HUD Elements
    fuel_bg = Rectangle((0.04, 0.92), 0.22, 0.035, transform=ax.transAxes, facecolor="#1e293b", edgecolor="#475569", lw=0.8, zorder=10)
    fuel_fill = Rectangle((0.04, 0.92), 0.22, 0.035, transform=ax.transAxes, facecolor="#22c55e", zorder=11)
    ax.add_patch(fuel_bg)
    ax.add_patch(fuel_fill)

    fuel_text = ax.text(0.04, 0.86, "FUEL: 100%", transform=ax.transAxes, color="#e2e8f0", fontsize=7.5, family="monospace", zorder=12)
    telem_text = ax.text(0.04, 0.79, "vx: +0.00 | vy: -0.00 | ang: 0.0°", transform=ax.transAxes, color="#94a3b8", fontsize=7.0, family="monospace", zorder=12)

    g_val = conditions.get("gravity", 7.0)
    w_val = conditions.get("wind", 2.0)
    ax.text(0.96, 0.92, f"g={g_val:.1f} | w_max={w_val:.1f}", transform=ax.transAxes, color="#94a3b8", fontsize=7.5, family="monospace", ha="right", zorder=12)
    wind_label = ax.text(0.96, 0.85, "WIND: 0.0", transform=ax.transAxes, color="#38bdf8", fontsize=7.0, family="monospace", ha="right", zorder=12)

    # End banner (held 15 frames)
    banner_box = ax.text(
        0.0, 0.65, "",
        color="white", fontsize=15, fontweight="bold", ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#064e3b", edgecolor="#22c55e", alpha=0.92),
        zorder=20, visible=False,
    )
    sub_banner = ax.text(
        0.0, 0.52, "",
        color="#e2e8f0", fontsize=10, fontweight="bold", ha="center", va="center",
        zorder=20, visible=False,
    )

    return {
        "body_patch": body_patch,
        "left_leg": left_leg,
        "right_leg": right_leg,
        "left_foot": left_foot,
        "right_foot": right_foot,
        "flame_patch": flame_patch,
        "left_puff": left_puff,
        "right_puff": right_puff,
        "trail_line": trail_line,
        "fuel_fill": fuel_fill,
        "fuel_text": fuel_text,
        "telem_text": telem_text,
        "wind_label": wind_label,
        "banner_box": banner_box,
        "sub_banner": sub_banner,
    }


def _update_panel(p, frame: dict, history_pos: list[tuple[float, float]], step_idx: int):
    """Update dynamic patches and HUD telemetry for a single animation frame."""
    x, y, vx, vy, ang, act, fuel = (
        frame["x"], frame["y"], frame["vx"], frame["vy"],
        frame["ang"], frame["act"], frame["fuel"]
    )
    history_pos.append((x, y))
    recent = history_pos[-40:]
    if len(recent) > 1:
        rx, ry = zip(*recent)
        p["trail_line"].set_data(rx, ry)

    cos_t = np.cos(ang)
    sin_t = np.sin(ang)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    # Body
    base_body = np.array([
        [0.0, 0.045], [0.025, 0.030], [0.035, -0.015],
        [-0.035, -0.015], [-0.025, 0.030]
    ])
    p["body_patch"].set_xy((base_body @ rot.T) + np.array([x, y]))

    # Legs & feet
    l_leg = (np.array([[-0.025, -0.015], [-0.045, -0.045]]) @ rot.T) + np.array([x, y])
    r_leg = (np.array([[0.025, -0.015], [0.045, -0.045]]) @ rot.T) + np.array([x, y])
    l_foot = (np.array([[-0.055, -0.045], [-0.035, -0.045]]) @ rot.T) + np.array([x, y])
    r_foot = (np.array([[0.035, -0.045], [0.055, -0.045]]) @ rot.T) + np.array([x, y])

    p["left_leg"].set_data(l_leg[:, 0], l_leg[:, 1])
    p["right_leg"].set_data(r_leg[:, 0], r_leg[:, 1])
    p["left_foot"].set_data(l_foot[:, 0], l_foot[:, 1])
    p["right_foot"].set_data(r_foot[:, 0], r_foot[:, 1])

    # Flame
    if act == 2:
        flame_len = 0.035 + 0.025 * ((step_idx % 3) / 2.0)
        base_flame = np.array([[-0.018, -0.015], [0.018, -0.015], [0.0, -0.015 - flame_len]])
        p["flame_patch"].set_xy((base_flame @ rot.T) + np.array([x, y]))
        p["flame_patch"].set_visible(True)
    else:
        p["flame_patch"].set_visible(False)

    # Side thrusters
    if act == 1:
        base_lpuff = np.array([[-0.025, 0.020], [-0.055, 0.030], [-0.055, 0.010]])
        p["left_puff"].set_xy((base_lpuff @ rot.T) + np.array([x, y]))
        p["left_puff"].set_visible(True)
    else:
        p["left_puff"].set_visible(False)

    if act == 3:
        base_rpuff = np.array([[0.025, 0.020], [0.055, 0.030], [0.055, 0.010]])
        p["right_puff"].set_xy((base_rpuff @ rot.T) + np.array([x, y]))
        p["right_puff"].set_visible(True)
    else:
        p["right_puff"].set_visible(False)

    # HUD updates
    fuel_pct = max(0.0, min(100.0, fuel * 100.0))
    p["fuel_fill"].set_width(0.22 * (fuel_pct / 100.0))
    if fuel_pct > 30.0:
        p["fuel_fill"].set_facecolor("#22c55e")
    elif fuel_pct > 15.0:
        p["fuel_fill"].set_facecolor("#f59e0b")
    else:
        p["fuel_fill"].set_facecolor("#ef4444")
    p["fuel_text"].set_text(f"FUEL: {fuel_pct:4.1f}%")
    p["telem_text"].set_text(f"vx:{vx:+4.2f} vy:{vy:+4.2f} ang:{np.degrees(ang):+4.1f}°")

    w_val = frame["wind_val"]
    w_arrow = ">>>" if w_val > 0.5 else ("<<<" if w_val < -0.5 else "--")
    p["wind_label"].set_text(f"WIND: {w_val:+3.1f} {w_arrow}")

    if frame.get("done", False):
        p["banner_box"].set_visible(True)
        p["sub_banner"].set_visible(True)
        if frame.get("landed", False):
            p["banner_box"].set_text("TOUCHDOWN")
            p["banner_box"].set_color("#22c55e")
            p["banner_box"].set_bbox(dict(boxstyle="round,pad=0.5", facecolor="#064e3b", edgecolor="#22c55e", alpha=0.92))
            p["sub_banner"].set_text(f"Reward: {frame['reward']:+.1f}")
            p["sub_banner"].set_color("#a7f3d0")
        else:
            p["banner_box"].set_text("CRASH")
            p["banner_box"].set_color("#ef4444")
            p["banner_box"].set_bbox(dict(boxstyle="round,pad=0.5", facecolor="#450a0a", edgecolor="#ef4444", alpha=0.92))
            p["sub_banner"].set_text(f"Reward: {frame['reward']:+.1f}")
            p["sub_banner"].set_color("#fca5a5")


def record_episode(
    policy,
    conditions: dict | None = None,
    seed: int = 42,
    compare_with_random: bool = False,
    **kwargs: Any,
) -> str:
    """Record a single episode or comparison and return as HTML string with base64 GIF."""
    cond = conditions or {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0}

    frames_pol = _simulate_flight(policy, cond, seed=seed)

    if compare_with_random:
        frames_rnd = _simulate_flight(_RandomPolicy(seed=seed), cond, seed=seed)
        total_frames = max(len(frames_pol), len(frames_rnd))
        while len(frames_pol) < total_frames:
            frames_pol.append(frames_pol[-1])
        while len(frames_rnd) < total_frames:
            frames_rnd.append(frames_rnd[-1])

        fig, (ax_rnd, ax_pol) = plt.subplots(1, 2, figsize=(8.0, 3.4), dpi=100)
        fig.patch.set_facecolor("#05070f")
        ax_rnd.set_title("Random Policy (Uncontrolled)", color="#e2e8f0", fontsize=9.0)
        ax_pol.set_title("Trained Lander Policy", color="#e2e8f0", fontsize=9.0)

        p_rnd = _build_panel(ax_rnd, seed=seed + 1, conditions=cond)
        p_pol = _build_panel(ax_pol, seed=seed + 2, conditions=cond)
        plt.tight_layout()

        hist_rnd: list[tuple[float, float]] = []
        hist_pol: list[tuple[float, float]] = []
        pil_images = []

        for i in range(total_frames):
            _update_panel(p_rnd, frames_rnd[i], hist_rnd, step_idx=i)
            _update_panel(p_pol, frames_pol[i], hist_pol, step_idx=i)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            im = Image.fromarray(rgba).convert("RGB").quantize(colors=64, dither=Image.Dither.NONE)
            pil_images.append(im)

        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=100)
        fig.patch.set_facecolor("#05070f")
        ax.set_title("Meridian Lander Flight Guidance", color="#e2e8f0", fontsize=10)
        panel = _build_panel(ax, seed=seed, conditions=cond)
        plt.tight_layout()

        hist_pol = []
        pil_images = []
        for i in range(len(frames_pol)):
            _update_panel(panel, frames_pol[i], hist_pol, step_idx=i)
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            im = Image.fromarray(rgba).convert("RGB").quantize(colors=64, dither=Image.Dither.NONE)
            pil_images.append(im)

        plt.close(fig)

    bio = io.BytesIO()
    pil_images[0].save(
        bio,
        format="GIF",
        save_all=True,
        append_images=pil_images[1:],
        duration=40,
        loop=0,
        optimize=True,
    )
    gif_bytes = bio.getvalue()
    b64_str = base64.b64encode(gif_bytes).decode("ascii")

    return f'<div style="display:inline-block; border-radius:6px; overflow:hidden;"><img src="data:image/gif;base64,{b64_str}" style="max-width:100%; height:auto;" alt="Act 4 Lander Flight Animation"/></div>'


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
    """Compare baseline heuristic PD controller against unguided descent and offer interactive prediction challenge."""
    cond = conditions or {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0}
    from meridian.act4_lander.policy import pd_controller
    pd_eval = evaluate(pd_controller, cond, n_episodes=20, seed=42)

    class Unguided:
        def act(self, obs: np.ndarray) -> np.ndarray:
            return np.zeros(obs.shape[0], dtype=int)

    unguided_eval = evaluate(Unguided(), cond, n_episodes=20, seed=42)

    # 3 benchmark test episode outcomes for interactive participant prediction
    test_seeds = [101, 202, 303]
    test_outcomes = []
    for s in test_seeds:
        res = evaluate(pd_controller, cond, n_episodes=1, seed=s)
        test_outcomes.append(1 if res["landing_rate"] > 0.5 else 0)

    print("\n--- HUMAN VS AI: DESCENT GUIDANCE BENCHMARK ---")
    print(f"Planetary Conditions : Gravity={cond.get('gravity', 7.0):.1f} m/s², Wind={cond.get('wind', 2.0):.1f} m/s²")
    print(f"Unguided Free-Fall   : Landing Rate = {unguided_eval['landing_rate'] * 100.0:.1f}%, Mean Reward = {unguided_eval['mean_reward']:.1f}")
    print(f"Heuristic PD Guidance: Landing Rate = {pd_eval['landing_rate'] * 100.0:.1f}%, Mean Reward = {pd_eval['mean_reward']:.1f}")
    print("-" * 60)
    print("Challenge: Predict whether heuristic PD guidance lands (1) or crashes (0) across 3 test wind gusts.")
    print("Edit guesses = [None, None, None] or call reveal(guesses, challenge).\n")

    def _reveal_fn(guesses: list[int | None] | np.ndarray) -> dict[str, Any]:
        return reveal(guesses=guesses, challenge={"test_seeds": test_seeds, "test_outcomes": test_outcomes, "conditions": cond})

    return {
        "unguided_landing_rate": unguided_eval["landing_rate"],
        "pd_landing_rate": pd_eval["landing_rate"],
        "test_seeds": test_seeds,
        "test_outcomes": test_outcomes,
        "reveal": _reveal_fn,
    }


def reveal(
    guesses: list[int | None] | np.ndarray,
    challenge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate participant descent outcome predictions against simulated flights."""
    if challenge is None or "test_outcomes" not in challenge:
        raise ValueError("Must provide challenge dict from human_vs_ai(conditions).")

    truths = challenge["test_outcomes"]
    seeds = challenge.get("test_seeds", [1, 2, 3])
    guesses_list = list(guesses[:len(truths)]) if hasattr(guesses, "__iter__") else [guesses]

    attempted = 0
    correct = 0
    records = []

    print("\n" + "=" * 60)
    print("DESCENT GUIDANCE PREDICTION RESULTS")
    print("-" * 60)
    for i, (g, t) in enumerate(zip(guesses_list, truths)):
        t_label = "SUCCESS" if t == 1 else "CRASH"
        if g is not None:
            attempted += 1
            g_int = int(g)
            g_label = "SUCCESS" if g_int == 1 else "CRASH"
            is_corr = (g_int == t)
            if is_corr:
                correct += 1
            status = "CORRECT" if is_corr else "MISSED"
            print(f"Flight #{i+1} (Seed {seeds[i]}): Guess={g_label:<7} | Actual={t_label:<7} [{status}]")
            records.append({"flight": i + 1, "guess": g_label, "actual": t_label, "status": status})
        else:
            print(f"Flight #{i+1} (Seed {seeds[i]}): Guess=SKIP    | Actual={t_label:<7} [SKIPPED]")
            records.append({"flight": i + 1, "guess": "SKIP", "actual": t_label, "status": "SKIPPED"})

    acc = float(correct / attempted) if attempted > 0 else 0.0
    print("=" * 60)
    if attempted > 0:
        print(f"Human Accuracy : {correct}/{attempted} correct ({acc * 100:.1f}%) [Attempted {attempted}/{len(truths)}]")
    else:
        print(f"Human Accuracy : SKIPPED (0/{len(truths)} attempted).")
        print(">> Instruction : Edit guesses = [1, 0, 1] (1: success, 0: crash) to test your descent predictions!")
    print("=" * 60 + "\n")

    return {
        "human_accuracy": acc,
        "attempted": attempted,
        "correct": correct,
        "total": len(truths),
        "records": records,
    }

