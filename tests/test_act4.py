"""Unit and acceptance tests for Act 4 (Lunar Lander simulation & training)."""

import time
from pathlib import Path
import pytest
import numpy as np

from meridian.state import MissionState
from meridian.act4_lander.env import VecLander
from meridian.act4_lander.policy import MLPPolicy, pd_controller
from meridian.act4_lander.train import (
    TRAIN_KNOBS,
    validate_knobs,
    build_conditions,
    evaluate,
    train,
    apply_to_state,
)
from meridian.act4_lander.play import record_episode, plot_learning_curve, stress_test


class SingleEnvRef:
    """Minimal single-environment reference implementation for test verification."""

    def __init__(
        self,
        seed: int = 123,
        gravity: float = 7.0,
        wind: float = 2.0,
        fuel_pct: float = 100.0,
        sensor_noise: float = 0.0,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.gravity = float(gravity)
        self.wind = float(wind)
        self.fuel = float(fuel_pct) / 100.0
        self.sensor_noise = float(sensor_noise)

        self.dt = 0.02
        self.F_main = 1.8 * 7.0
        self.alpha_side = 14.0
        self.F_side = 0.1
        self.drag = 0.2
        self.ang_drag = 0.8
        self.reset()

    def reset(self) -> None:
        self.x = float(self.rng.uniform(-0.5, 0.5))
        self.y = 1.0
        self.vx = float(self.rng.uniform(-0.05, 0.05))
        self.vy = float(self.rng.uniform(-0.05, 0.05))
        self.ang = float(self.rng.uniform(-0.05, 0.05))
        self.omega = float(self.rng.uniform(-0.05, 0.05))
        self.wind_val = 0.0
        self.hold_count = 0
        self.step_count = 0
        self.done = False
        self.landed = False
        self.crashed = False

    def get_obs(self) -> np.ndarray:
        sin_t = np.sin(self.ang)
        cos_t = np.cos(self.ang)
        y_l = self.y - 0.05 * sin_t - 0.05 * cos_t
        y_r = self.y + 0.05 * sin_t - 0.05 * cos_t
        l_c = 1.0 if y_l <= 0.01 else 0.0
        r_c = 1.0 if y_r <= 0.01 else 0.0
        return np.array([self.x, self.y, self.vx, self.vy, self.ang, self.omega, l_c, r_c], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, bool]:
        if self.done:
            return self.get_obs(), self.done

        self.step_count += 1
        self.wind_val = 0.99 * self.wind_val + 0.3 * self.wind * np.sqrt(1.0 - 0.99**2) * self.rng.randn()

        sin_t = np.sin(self.ang)
        cos_t = np.cos(self.ang)
        y_l = self.y - 0.05 * sin_t - 0.05 * cos_t
        y_r = self.y + 0.05 * sin_t - 0.05 * cos_t
        l_contact = y_l <= 0.01
        r_contact = y_r <= 0.01

        touching = l_contact or r_contact or (self.y <= 0.05)
        if touching:
            safe = (
                (abs(self.vy) < 0.5)
                and (abs(self.vx) < 0.4)
                and (abs(self.ang) < 0.25)
                and (abs(self.x) <= 0.2)
            )
            if not safe:
                self.crashed = True
                self.done = True
            else:
                if y_l < 0.0:
                    self.y -= y_l
                if y_r < 0.0:
                    self.y -= y_r
                self.vy = max(self.vy, 0.0)
                self.vx *= 0.1
                self.omega *= 0.1
                if l_contact and r_contact:
                    self.hold_count += 1
                    if self.hold_count >= 25:
                        self.landed = True
                        self.done = True
                else:
                    self.hold_count = 0

        if not self.done:
            ax = self.wind_val - self.drag * self.vx
            ay = -self.gravity - self.drag * self.vy
            tau = -self.ang_drag * self.omega

            engine_allowed = (self.fuel > 0) and (self.hold_count == 0)
            if engine_allowed:
                if action == 2:
                    self.fuel -= self.dt / 6.0
                    ax += self.F_main * sin_t
                    ay += self.F_main * cos_t
                elif action == 1:
                    self.fuel -= self.dt / 24.0
                    tau += self.alpha_side
                    ax += self.F_side
                elif action == 3:
                    self.fuel -= self.dt / 24.0
                    tau -= self.alpha_side
                    ax -= self.F_side

            self.fuel = max(self.fuel, 0.0)

            if touching and self.hold_count > 0:
                ax = 0.0

            self.vx += ax * self.dt
            self.vy += ay * self.dt
            self.omega += tau * self.dt

            self.x += self.vx * self.dt
            self.y += self.vy * self.dt
            self.ang += self.omega * self.dt

            if abs(self.x) > 1.5 or self.y > 2.0 or self.y < -0.1 or self.step_count >= 1000:
                self.crashed = True
                self.done = True

        return self.get_obs(), self.done


def test_acceptance_1_determinism():
    """Acceptance 1: Identical seeds produce identical trajectories and results."""
    env1 = VecLander(n_envs=3, seed=777)
    env2 = VecLander(n_envs=3, seed=777)

    obs1 = env1.reset()
    obs2 = env2.reset()
    assert np.allclose(obs1, obs2)

    for step in range(40):
        actions = np.array([step % 4, (step + 1) % 4, (step + 2) % 4])
        o1, _, d1, info1 = env1.step(actions)
        o2, _, d2, info2 = env2.step(actions)

        assert np.allclose(o1, o2)
        assert np.array_equal(d1, d2)
        assert np.array_equal(info1["landed"], info2["landed"])
        assert np.array_equal(info1["crashed"], info2["crashed"])


def test_acceptance_2_vectorized_matches_single_env():
    """Acceptance 2: Vectorized env matches single-env reference logic over 5 seeds."""
    seeds = [11, 22, 33, 44, 55]
    for s in seeds:
        ref = SingleEnvRef(seed=s)
        vec = VecLander(n_envs=1, seed=s)
        assert np.allclose(ref.get_obs(), vec._get_obs()[0])

        for step in range(40):
            act = (step * 3) % 4
            ref_obs, ref_done = ref.step(act)
            vec_obs, _, vec_done, _ = vec.step(act)

            assert np.allclose(ref_obs, vec_obs[0], atol=1e-5), f"Mismatch seed {s} step {step}"
            assert ref_done == vec_done[0], f"Done mismatch seed {s} step {step}"


def test_acceptance_3_pd_controller_and_random():
    """Acceptance 3: PD controller >= 90% and random policy < 2% in default conditions."""
    # 1. PD Controller on 200 episodes
    env_pd = VecLander(n_envs=200, seed=42)
    obs = env_pd.reset()
    for _ in range(1000):
        if np.all(env_pd.done):
            break
        acts = pd_controller(obs)
        obs, _, _, _ = env_pd.step(acts)

    pd_rate = float(np.mean(env_pd.landed))
    print(f"\nMeasured PD Controller Landing Rate: {pd_rate * 100:.1f}%")
    assert pd_rate >= 0.90, f"PD controller landing rate {pd_rate*100:.1f}% below 90%"

    # 2. Uniform Random Policy on 200 episodes
    rng = np.random.RandomState(42)
    env_rnd = VecLander(n_envs=200, seed=42)
    obs = env_rnd.reset()
    for _ in range(1000):
        if np.all(env_rnd.done):
            break
        acts = rng.randint(0, 4, size=200)
        obs, _, _, _ = env_rnd.step(acts)

    rnd_rate = float(np.mean(env_rnd.landed))
    print(f"Measured Random Policy Landing Rate: {rnd_rate * 100:.1f}%")
    assert rnd_rate < 0.02, f"Random policy landing rate {rnd_rate*100:.1f}% >= 2%"


def test_acceptance_4_default_checkpoint():
    """Acceptance 4: Default checkpoint achieves >= 80% landing rate in default conditions."""
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "act4_default.npz"
    assert ckpt_path.exists(), f"Default checkpoint missing at {ckpt_path}"

    policy = MLPPolicy.load(ckpt_path)
    res = evaluate(
        policy,
        conditions={"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0},
        n_episodes=100,
        seed=42,
    )

    landing_rate = res["landing_rate"]
    print(f"\nMeasured Default Checkpoint Landing Rate: {landing_rate * 100:.1f}%")
    assert landing_rate >= 0.80, f"Checkpoint landing rate {landing_rate*100:.1f}% below 80%"


def test_acceptance_5_fuel_and_noise_lower_rate():
    """Acceptance 5: fuel_pct=40 and sensor_noise=0.5 lower the default policy landing rate."""
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "act4_default.npz"
    policy = MLPPolicy.load(ckpt_path)

    r_def = evaluate(policy, {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.0}, n_episodes=50, seed=4)
    r_fuel = evaluate(policy, {"gravity": 7.0, "wind": 2.0, "fuel_pct": 40.0, "sensor_noise": 0.0}, n_episodes=50, seed=4)
    r_noise = evaluate(policy, {"gravity": 7.0, "wind": 2.0, "fuel_pct": 100.0, "sensor_noise": 0.5}, n_episodes=50, seed=4)

    def_rate = r_def["landing_rate"]
    fuel_rate = r_fuel["landing_rate"]
    noise_rate = r_noise["landing_rate"]

    print(f"\nAcceptance 5 Measured Rates (Seed 4, 50 eps):")
    print(f"  Default conditions:   {def_rate * 100:.1f}%")
    print(f"  fuel_pct=40:          {fuel_rate * 100:.1f}%")
    print(f"  sensor_noise=0.5:     {noise_rate * 100:.1f}%")

    assert fuel_rate < def_rate, f"fuel_pct=40 ({fuel_rate*100:.1f}%) not lower than default ({def_rate*100:.1f}%)"
    assert noise_rate < def_rate, f"sensor_noise=0.5 ({noise_rate*100:.1f}%) not lower than default ({def_rate*100:.1f}%)"


def test_acceptance_6_training_runtime():
    """Acceptance 6: train() with warm start in FAST <= 120s; from-scratch <= 300s."""
    # Warm start
    t0 = time.time()
    _, hist_warm = train(fast=True, warm_start=True, iterations=20)
    t_warm = time.time() - t0
    print(f"\nMeasured Warm-start FAST training time: {t_warm:.2f}s")
    assert t_warm <= 120.0, f"Warm-start train took {t_warm:.2f}s > 120s"
    assert len(hist_warm) == 20

    # From scratch
    t1 = time.time()
    _, hist_scratch = train(fast=True, warm_start=False, iterations=20)
    t_scratch = time.time() - t1
    print(f"Measured From-scratch FAST training time: {t_scratch:.2f}s")
    assert t_scratch <= 300.0, f"From-scratch train took {t_scratch:.2f}s > 300s"
    assert len(hist_scratch) == 20


def test_acceptance_7_animation_renders():
    """Acceptance 7: Animation renders without error and saves to artifacts/act4_demo.html."""
    ckpt_path = Path(__file__).resolve().parent.parent / "checkpoints" / "act4_default.npz"
    policy = MLPPolicy.load(ckpt_path)

    # 1. Single flight GIF render
    t0 = time.time()
    html = record_episode(policy, seed=42, compare_with_random=False)
    t_single = time.time() - t0
    assert isinstance(html, str)
    assert "data:image/gif;base64," in html
    assert t_single <= 20.0, f"Single render took {t_single:.2f}s > 20s"

    # Extract base64 payload and verify size <= 600 KB
    import base64
    b64_part = html.split("data:image/gif;base64,")[1].split('"')[0]
    raw_gif = base64.b64decode(b64_part)
    gif_size_kb = len(raw_gif) / 1024.0
    print(f"\nSingle GIF render: {gif_size_kb:.1f} KB in {t_single:.2f}s")
    assert gif_size_kb <= 600.0, f"Single GIF {gif_size_kb:.1f} KB > 600 KB"

    # Save artifact demo
    out_path = Path(__file__).resolve().parent.parent / "artifacts" / "act4_demo.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    assert out_path.exists()
    assert out_path.stat().st_size > 1000

    # 2. Comparison flight GIF render (Random vs Trained)
    t1 = time.time()
    html_comp = record_episode(policy, seed=42, compare_with_random=True)
    t_comp = time.time() - t1
    assert isinstance(html_comp, str)
    assert "data:image/gif;base64," in html_comp
    assert t_comp <= 20.0, f"Comparison render took {t_comp:.2f}s > 20s"

    b64_comp_part = html_comp.split("data:image/gif;base64,")[1].split('"')[0]
    raw_comp_gif = base64.b64decode(b64_comp_part)
    comp_size_kb = len(raw_comp_gif) / 1024.0
    print(f"Comparison GIF render: {comp_size_kb:.1f} KB in {t_comp:.2f}s")
    assert comp_size_kb <= 700.0, f"Comparison GIF {comp_size_kb:.1f} KB > 700 KB"


def test_state_integration_and_knobs():
    """Test state coupling and knob validation for Act 4."""
    state = MissionState(team_id="team_test_act4")
    cond = build_conditions(state)
    assert cond["gravity"] == 7.0
    assert cond["wind"] == 2.0
    assert cond["fuel_pct"] == 100.0
    assert cond["sensor_noise"] == 0.0

    results = {"landing_rate": 0.85, "mean_reward": 80.0, "conditions": cond}
    score = apply_to_state(state, results)
    assert state.act4.done is True
    assert 0.0 <= score <= 400.0
    assert state.act4.score == score

    # Knobs clipping & warnings
    with pytest.warns(UserWarning):
        clipped = validate_knobs({"w_land": 999.0, "iterations": 1})
    assert clipped["w_land"] == 200.0
    assert clipped["iterations"] == 5
