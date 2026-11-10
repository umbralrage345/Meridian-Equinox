"""MLPPolicy and PD controller for Lunar Lander."""

from __future__ import annotations
from pathlib import Path
import numpy as np


class MLPPolicy:
    """Multi-layer perceptron policy: 8 -> hidden -> 4 with tanh activation."""

    def __init__(self, hidden: int = 16, weights: np.ndarray | None = None) -> None:
        self.hidden = int(hidden)
        self.n_params = 8 * self.hidden + self.hidden + self.hidden * 4 + 4

        if weights is not None:
            self.set_params(weights)
        else:
            # Initialize with small random weights
            rng = np.random.RandomState(42)
            init_w = rng.randn(self.n_params).astype(np.float32) * 0.1
            self.set_params(init_w)

    def get_params(self) -> np.ndarray:
        """Return the flat parameter vector."""
        return self.theta.copy()

    def set_params(self, params: np.ndarray) -> None:
        """Set weights from a flat parameter vector."""
        self.theta = np.asarray(params, dtype=np.float32).flatten()
        h = self.hidden
        idx0 = 0
        idx1 = 8 * h
        idx2 = idx1 + h
        idx3 = idx2 + h * 4
        idx4 = idx3 + 4

        self.W1 = self.theta[idx0:idx1].reshape(8, h)
        self.b1 = self.theta[idx1:idx2]
        self.W2 = self.theta[idx2:idx3].reshape(h, 4)
        self.b2 = self.theta[idx3:idx4]

    def act(self, obs: np.ndarray) -> int | np.ndarray:
        """Forward pass computing argmax action for single or batch observation."""
        obs = np.asarray(obs, dtype=np.float32)
        is_single = obs.ndim == 1
        if is_single:
            obs = obs[None, :]

        h = np.tanh(obs @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        actions = np.argmax(logits, axis=-1)

        return int(actions[0]) if is_single else actions

    def save(self, path: str | Path) -> None:
        """Save policy weights to an .npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.theta, hidden=self.hidden)

    @classmethod
    def load(cls, path: str | Path) -> MLPPolicy:
        """Load policy weights from an .npz file."""
        data = np.load(path)
        hidden = int(data["hidden"]) if "hidden" in data else 16
        weights = data["weights"]
        return cls(hidden=hidden, weights=weights)


def pd_controller(obs: np.ndarray) -> int | np.ndarray:
    """Hand-coded PD flight controller for Lunar Lander.

    Observation vector: [x, y, vx, vy, angle, omega, left_contact, right_contact].
    Actions: 0 noop, 1 left thruster, 2 main engine, 3 right thruster.
    """
    obs = np.asarray(obs, dtype=np.float32)
    is_single = obs.ndim == 1
    if is_single:
        obs = obs[None, :]

    x = obs[:, 0]
    y = obs[:, 1]
    vx = obs[:, 2]
    vy = obs[:, 3]
    ang = obs[:, 4]
    omega = obs[:, 5]
    l_contact = obs[:, 6]
    r_contact = obs[:, 7]

    angle_targ = np.clip(-x * 0.7 - vx * 1.3, -0.3, 0.3)
    hover_targ = np.clip(0.45 * np.abs(x), 0.0, 0.6)

    angle_todo = (ang - angle_targ) * 6.0 + omega * 2.5
    hover_todo = -(y - hover_targ) * 10.0 - vy * 8.0

    touch_mask = (l_contact > 0.5) | (r_contact > 0.5)
    angle_todo[touch_mask] = ang[touch_mask] * 6.0 + omega[touch_mask] * 2.5
    hover_todo[touch_mask] = -vy[touch_mask] * 12.0

    n = len(obs)
    actions = np.zeros(n, dtype=np.int32)
    main_mask = (hover_todo > np.abs(angle_todo)) & (hover_todo > 0.05)
    left_mask = ~main_mask & (angle_todo < -0.05)
    right_mask = ~main_mask & (angle_todo > 0.05)

    actions[main_mask] = 2
    actions[left_mask] = 1
    actions[right_mask] = 3

    both_touch = (l_contact > 0.5) & (r_contact > 0.5)
    actions[both_touch] = 0

    return int(actions[0]) if is_single else actions
