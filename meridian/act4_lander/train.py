"""Evolution Strategy training and evaluation for Lunar Lander."""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np

from meridian.config import is_fast
from meridian.act4_lander.env import VecLander
from meridian.act4_lander.policy import MLPPolicy


TRAIN_KNOBS = {
    "w_land": (0.0, 200.0, 100.0),
    "w_crash": (0.0, 200.0, 100.0),
    "w_fuel": (0.0, 2.0, 0.3),
    "w_dist": (0.0, 50.0, 20.0),
    "w_speed": (0.0, 50.0, 20.0),
    "w_leg": (0.0, 20.0, 5.0),
    "iterations": (5, 100, 40),
    "hidden": (8, 32, 16),
    "warm_start": (False, True, True),
}


def build_conditions(state) -> dict[str, float]:
    """Extract lander flight conditions from the mission state."""
    gravity = 7.0
    wind = 2.0
    fuel_pct = 100.0
    sensor_noise = 0.0

    if hasattr(state, "act1") and hasattr(state.act1, "target"):
        target = state.act1.target
        gravity = getattr(target, "gravity", 7.0)
        wind = getattr(target, "wind", 2.0)

    if hasattr(state, "resources"):
        res = state.resources
        fuel_pct = getattr(res, "fuel_pct", 100.0)
        sensor_noise = getattr(res, "sensor_noise", 0.0)

    return {
        "gravity": float(gravity),
        "wind": float(wind),
        "fuel_pct": float(fuel_pct),
        "sensor_noise": float(sensor_noise),
    }


def validate_knobs(knobs: dict | None = None, fast: bool | None = None) -> dict:
    """Validate and clip training hyperparameters, warning on adjustments."""
    fast_mode = is_fast() if fast is None else bool(fast)
    cfg = {}
    knobs = knobs or {}

    for k, (low, high, default) in TRAIN_KNOBS.items():
        if k == "iterations" and fast_mode and "iterations" not in knobs:
            val = 20
        else:
            val = knobs.get(k, default)

        if isinstance(val, bool):
            cfg[k] = val
            continue

        if val < low:
            warnings.warn(f"Knob {k}={val} below min {low}; clipping to {low}.")
            val = low
        elif val > high:
            warnings.warn(f"Knob {k}={val} above max {high}; clipping to {high}.")
            val = high

        cfg[k] = int(val) if isinstance(default, int) else float(val)

    return cfg


def evaluate(
    policy: MLPPolicy,
    conditions: dict[str, float],
    n_episodes: int = 30,
    seed: int = 42,
) -> dict:
    """Run official fixed evaluation on the given policy."""
    env = VecLander(
        n_envs=n_episodes,
        seed=seed,
        gravity=conditions.get("gravity", 7.0),
        wind=conditions.get("wind", 2.0),
        fuel_pct=conditions.get("fuel_pct", 100.0),
        sensor_noise=conditions.get("sensor_noise", 0.0),
    )

    obs = env.reset()
    for _ in range(1000):
        if np.all(env.done):
            break
        if hasattr(policy, "act"):
            actions = policy.act(obs)
        else:
            actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    landed = env.landed.astype(np.float32)
    crashed = env.crashed.astype(np.float32)
    main_burn = env.main_burn_s
    side_burn = env.side_burn_s
    dist = env.distance_at_end
    speed_penalty = np.maximum(0.0, env.landing_speed - 0.5)

    reward_official = (
        100.0 * landed
        - 100.0 * crashed
        - 0.3 * main_burn
        - 0.05 * side_burn
        - 20.0 * dist
        - 20.0 * speed_penalty
    )

    return {
        "landing_rate": float(np.mean(landed)),
        "mean_reward": float(np.mean(reward_official)),
        "conditions": conditions,
    }


def train(
    conditions: dict[str, float] | None = None,
    fast: bool | None = None,
    seed: int = 42,
    **knobs,
) -> tuple[MLPPolicy, list[dict]]:
    """Train MLPPolicy using vectorized OpenAI-ES evolution strategy."""
    cfg = validate_knobs(knobs, fast=fast)
    fast_mode = is_fast() if fast is None else bool(fast)
    conditions = conditions or {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0}

    hidden = int(cfg["hidden"])
    pop_size = 32 if fast_mode else 64
    episodes_per_cand = 8
    n_envs = pop_size * episodes_per_cand
    sigma = 0.1
    lr = 0.05
    n_iters = int(cfg["iterations"])

    policy = MLPPolicy(hidden=hidden)
    default_ckpt = Path("checkpoints/act4_default.npz")
    if not default_ckpt.exists():
        default_ckpt = Path(__file__).resolve().parent.parent.parent / "checkpoints" / "act4_default.npz"

    if cfg["warm_start"] and default_ckpt.exists():
        loaded = MLPPolicy.load(default_ckpt)
        if loaded.hidden == hidden:
            policy.set_params(loaded.get_params())

    theta = policy.get_params()
    n_params = len(theta)
    rng = np.random.RandomState(seed)

    half_pop = pop_size // 2
    history = []

    for it in range(n_iters):
        eps_half = rng.randn(half_pop, n_params).astype(np.float32)
        eps = np.concatenate([eps_half, -eps_half], axis=0)  # Antithetic pairs (pop_size, n_params)
        candidates = theta + sigma * eps

        # Build candidate weights for all n_envs: each candidate repeated episodes_per_cand times
        # Shape: (pop_size, episodes_per_cand, n_params) -> (n_envs, n_params)
        cand_expanded = np.repeat(candidates, episodes_per_cand, axis=0)

        # Vectorized candidate forward pass unpacked into weights
        h = hidden
        idx0 = 0
        idx1 = 8 * h
        idx2 = idx1 + h
        idx3 = idx2 + h * 4
        idx4 = idx3 + 4

        W1_batch = cand_expanded[:, idx0:idx1].reshape(n_envs, 8, h)
        b1_batch = cand_expanded[:, idx1:idx2]
        W2_batch = cand_expanded[:, idx2:idx3].reshape(n_envs, h, 4)
        b2_batch = cand_expanded[:, idx3:idx4]

        env = VecLander(
            n_envs=n_envs,
            seed=seed + it * 1000,
            gravity=conditions.get("gravity", 7.0),
            wind=conditions.get("wind", 2.0),
            fuel_pct=conditions.get("fuel_pct", 100.0),
            sensor_noise=conditions.get("sensor_noise", 0.0),
        )

        obs = env.reset()
        for _ in range(1000):
            if np.all(env.done):
                break
            # Batched forward pass: obs (n_envs, 8), W1_batch (n_envs, 8, h)
            # einsum for batch matrix multiplication: (n_envs, 8) x (n_envs, 8, h) -> (n_envs, h)
            z1 = np.tanh(np.einsum("bi,bih->bh", obs, W1_batch) + b1_batch)
            logits = np.einsum("bh,bho->bo", z1, W2_batch) + b2_batch
            actions = np.argmax(logits, axis=-1)
            obs, _, _, _ = env.step(actions)

        # Participant-weighted training reward
        w_land = cfg["w_land"]
        w_crash = cfg["w_crash"]
        w_fuel = cfg["w_fuel"]
        w_dist = cfg["w_dist"]
        w_speed = cfg["w_speed"]
        w_leg = cfg["w_leg"]

        landed = env.landed.astype(np.float32)
        crashed = env.crashed.astype(np.float32)
        fuel_s = env.main_burn_s
        dist = env.distance_at_end
        speed = env.landing_speed
        legs = env.legs_down_on_pad

        rewards = (
            w_land * landed
            - w_crash * crashed
            - w_fuel * fuel_s
            - w_dist * dist
            - w_speed * speed
            + w_leg * legs
        )

        # Average reward per candidate over its 8 episodes
        cand_rewards = rewards.reshape(pop_size, episodes_per_cand).mean(axis=1)
        cand_landing_rates = landed.reshape(pop_size, episodes_per_cand).mean(axis=1)

        # Rank-normalization of fitness
        order = np.argsort(cand_rewards)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(pop_size)
        utilities = (ranks / (pop_size - 1.0)) - 0.5

        # ES parameter update
        grad = np.dot(utilities, eps) / (pop_size * sigma)
        theta = theta + lr * grad

        history.append({
            "iteration": it + 1,
            "mean_reward": float(np.mean(cand_rewards)),
            "landing_rate": float(np.mean(cand_landing_rates)),
        })

    policy.set_params(theta)
    return policy, history


def apply_to_state(state_or_results: Any, results_or_state: Any) -> float:
    """Compute Act 4 score from evaluation results and update state."""
    from meridian.state import MissionState
    if isinstance(state_or_results, MissionState):
        state = state_or_results
        results = results_or_state
    else:
        results = state_or_results
        state = results_or_state

    landing_rate = float(results["landing_rate"])
    mean_reward = float(results["mean_reward"])

    normalized_reward = float(np.clip((mean_reward + 100.0) / 400.0, 0.0, 1.0))
    act4_score = 400.0 * (0.6 * landing_rate + 0.4 * normalized_reward)
    act4_score = round(act4_score, 1)
    if isinstance(results, dict):
        results["score"] = act4_score

    state.act4.done = True
    state.act4.score = act4_score
    state.act4.landing_rate = landing_rate
    state.act4.mean_reward = mean_reward
    state.act4.conditions = results.get("conditions", {})
    state.save()

    return act4_score
