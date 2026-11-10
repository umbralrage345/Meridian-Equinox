"""VecLander vectorized 2D Lunar Lander simulation."""

from __future__ import annotations
import numpy as np


class VecLander:
    """Vectorized 2D Lunar Lander environment using pure NumPy."""

    def __init__(
        self,
        n_envs: int = 1,
        seed: int | None = None,
        gravity: float = 7.0,
        wind: float = 2.0,
        fuel_pct: float = 100.0,
        sensor_noise: float = 0.0,
    ) -> None:
        self.n_envs = int(n_envs)
        self.gravity = float(gravity)
        self.wind = float(wind)
        self.initial_fuel = float(fuel_pct) / 100.0
        self.sensor_noise = float(sensor_noise)

        # Physics constants
        self.dt = 0.02  # 50 Hz
        self.F_main = 1.8 * 7.0  # 12.6 N (thrust-to-weight 1.8 at default g=7.0)
        self.alpha_side = 14.0
        self.F_side = 0.1
        self.drag = 0.2
        self.ang_drag = 0.8
        self.pad_half_width = 0.2

        self.rng = np.random.RandomState(seed if seed is not None else 42)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all environments to initial conditions."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        n = self.n_envs
        self.x = self.rng.uniform(-0.5, 0.5, n).astype(np.float32)
        self.y = np.ones(n, dtype=np.float32)
        self.vx = self.rng.uniform(-0.05, 0.05, n).astype(np.float32)
        self.vy = self.rng.uniform(-0.05, 0.05, n).astype(np.float32)
        self.ang = self.rng.uniform(-0.05, 0.05, n).astype(np.float32)
        self.omega = self.rng.uniform(-0.05, 0.05, n).astype(np.float32)

        self.fuel = np.full(n, self.initial_fuel, dtype=np.float32)
        self.wind_val = np.zeros(n, dtype=np.float32)
        self.hold_count = np.zeros(n, dtype=np.int32)
        self.step_count = np.zeros(n, dtype=np.int32)

        self.done = np.zeros(n, dtype=bool)
        self.landed = np.zeros(n, dtype=bool)
        self.crashed = np.zeros(n, dtype=bool)

        self.main_burn_s = np.zeros(n, dtype=np.float32)
        self.side_burn_s = np.zeros(n, dtype=np.float32)
        self.distance_at_end = np.zeros(n, dtype=np.float32)
        self.landing_speed = np.zeros(n, dtype=np.float32)
        self.legs_down_on_pad = np.zeros(n, dtype=np.float32)

        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        """Compute the 8-float observation vector for all environments."""
        sin_t = np.sin(self.ang)
        cos_t = np.cos(self.ang)

        y_l = self.y - 0.05 * sin_t - 0.05 * cos_t
        y_r = self.y + 0.05 * sin_t - 0.05 * cos_t

        l_contact = (y_l <= 0.01).astype(np.float32)
        r_contact = (y_r <= 0.01).astype(np.float32)

        obs = np.stack(
            [self.x, self.y, self.vx, self.vy, self.ang, self.omega, l_contact, r_contact],
            axis=1,
        ).astype(np.float32)

        if self.sensor_noise > 0.0:
            noise_std = 0.1 * self.sensor_noise
            noise = self.rng.normal(0.0, noise_std, size=(self.n_envs, 6)).astype(np.float32)
            obs[:, :6] += noise

        return obs

    def step(self, actions: np.ndarray | int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """Execute one simulation step across all active environments."""
        actions = np.asarray(actions, dtype=np.int32)
        if actions.ndim == 0:
            actions = np.full(self.n_envs, actions, dtype=np.int32)

        active = ~self.done
        if np.any(active):
            self.step_count[active] += 1
            n_active = int(np.sum(active))

            # Ornstein-Uhlenbeck lateral wind acceleration
            self.wind_val[active] = (
                0.99 * self.wind_val[active]
                + 0.3 * self.wind * np.sqrt(1.0 - 0.99**2) * self.rng.randn(n_active)
            )

            # Leg contact checks
            sin_t = np.sin(self.ang)
            cos_t = np.cos(self.ang)
            y_l = self.y - 0.05 * sin_t - 0.05 * cos_t
            y_r = self.y + 0.05 * sin_t - 0.05 * cos_t
            l_contact = y_l <= 0.01
            r_contact = y_r <= 0.01

            touching = active & (l_contact | r_contact | (self.y <= 0.05))
            if np.any(touching):
                t_idx = np.where(touching)[0]
                for i in t_idx:
                    safe = (
                        (abs(self.vy[i]) < 0.5)
                        and (abs(self.vx[i]) < 0.4)
                        and (abs(self.ang[i]) < 0.25)
                        and (abs(self.x[i]) <= self.pad_half_width)
                    )
                    if not safe:
                        self.crashed[i] = True
                        self.done[i] = True
                    else:
                        # Ground contact reaction & normal support
                        if y_l[i] < 0.0:
                            self.y[i] -= y_l[i]
                        if y_r[i] < 0.0:
                            self.y[i] -= y_r[i]
                        self.vy[i] = max(self.vy[i], 0.0)
                        self.vx[i] *= 0.1
                        self.omega[i] *= 0.1

                        if l_contact[i] and r_contact[i]:
                            self.hold_count[i] += 1
                            if self.hold_count[i] >= 25:  # 0.5 s held
                                self.landed[i] = True
                                self.done[i] = True
                        else:
                            self.hold_count[i] = 0

                active = ~self.done

            if np.any(active):
                ax = self.wind_val[active] - self.drag * self.vx[active]
                ay = -self.gravity - self.drag * self.vy[active]
                tau = -self.ang_drag * self.omega[active]

                act_active = actions[active]
                fuel_active = self.fuel[active] > 0
                engine_allowed = fuel_active & (self.hold_count[active] == 0)

                # Action 2: Main engine
                main_mask = engine_allowed & (act_active == 2)
                if np.any(main_mask):
                    active_indices = np.where(active)[0]
                    main_idx = active_indices[main_mask]
                    self.fuel[main_idx] -= self.dt / 6.0
                    self.main_burn_s[main_idx] += self.dt
                    ax[main_mask] += self.F_main * sin_t[main_idx]
                    ay[main_mask] += self.F_main * cos_t[main_idx]

                # Action 1: Left orientation thruster
                left_mask = engine_allowed & (act_active == 1)
                if np.any(left_mask):
                    active_indices = np.where(active)[0]
                    left_idx = active_indices[left_mask]
                    self.fuel[left_idx] -= self.dt / 24.0
                    self.side_burn_s[left_idx] += self.dt
                    tau[left_mask] += self.alpha_side
                    ax[left_mask] += self.F_side

                # Action 3: Right orientation thruster
                right_mask = engine_allowed & (act_active == 3)
                if np.any(right_mask):
                    active_indices = np.where(active)[0]
                    right_idx = active_indices[right_mask]
                    self.fuel[right_idx] -= self.dt / 24.0
                    self.side_burn_s[right_idx] += self.dt
                    tau[right_mask] -= self.alpha_side
                    ax[right_mask] -= self.F_side

                self.fuel[active] = np.maximum(self.fuel[active], 0.0)

                # Ground contact cancellation of lateral acceleration when holding
                on_ground = touching[active] & (self.hold_count[active] > 0)
                ax[on_ground] = 0.0

                # Semi-implicit Euler integration
                self.vx[active] += ax * self.dt
                self.vy[active] += ay * self.dt
                self.omega[active] += tau * self.dt

                self.x[active] += self.vx[active] * self.dt
                self.y[active] += self.vy[active] * self.dt
                self.ang[active] += self.omega[active] * self.dt

                # Bounds and timeout check
                oob = (
                    (np.abs(self.x[active]) > 1.5)
                    | (self.y[active] > 2.0)
                    | (self.y[active] < -0.1)
                    | (self.step_count[active] >= 1000)
                )
                if np.any(oob):
                    active_indices = np.where(active)[0]
                    oob_idx = active_indices[oob]
                    self.crashed[oob_idx] = True
                    self.done[oob_idx] = True

        # End-of-episode stat recording
        self.distance_at_end = np.abs(self.x)
        self.landing_speed = np.abs(self.vy)
        sin_t = np.sin(self.ang)
        cos_t = np.cos(self.ang)
        y_l = self.y - 0.05 * sin_t - 0.05 * cos_t
        y_r = self.y + 0.05 * sin_t - 0.05 * cos_t
        on_pad = np.abs(self.x) <= self.pad_half_width
        both_down = (y_l <= 0.01) & (y_r <= 0.01) & on_pad
        self.legs_down_on_pad = both_down.astype(np.float32)

        obs = self._get_obs()
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        info = {
            "landed": self.landed.copy(),
            "crashed": self.crashed.copy(),
            "main_burn_s": self.main_burn_s.copy(),
            "side_burn_s": self.side_burn_s.copy(),
            "distance_at_end": self.distance_at_end.copy(),
            "landing_speed": self.landing_speed.copy(),
            "legs_down_on_pad": self.legs_down_on_pad.copy(),
        }
        return obs, rewards, self.done.copy(), info
