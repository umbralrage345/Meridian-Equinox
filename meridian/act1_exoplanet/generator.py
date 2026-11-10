"""Synthetic exoplanet transit lightcurve generator."""

from __future__ import annotations

from typing import Any

import numpy as np

N_POINTS: int = 1296
TOTAL_DAYS: float = 27.0
TIME_GRID: np.ndarray = np.linspace(0.0, TOTAL_DAYS, N_POINTS, dtype=np.float64)


def make_star(
    rng: np.random.Generator | int | None,
    has_planet: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate a synthetic stellar light curve of 1296 points across 27 days.

    Args:
        rng: Seed or numpy Generator instance for reproducible stochastic generation.
        has_planet: Whether to inject an exoplanet transit signal.

    Returns:
        tuple of (flux array of 1296 floats, truth dict with planet physical properties).
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Base normalized stellar flux
    flux = np.ones(N_POINTS, dtype=np.float64)

    # 1 to 2 random sinusoidal stellar variability components
    n_sinusoids = int(rng.integers(1, 3))
    for _ in range(n_sinusoids):
        p_sin = float(rng.uniform(1.5, 15.0))
        amp = float(rng.uniform(0.0005, 0.006))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        flux += amp * np.sin(2.0 * np.pi * TIME_GRID / p_sin + phase)

    # Gaussian photometric noise
    sigma = float(rng.uniform(0.0006, 0.002))
    flux += rng.normal(0.0, sigma, size=N_POINTS)

    # Stellar flares / outlier spikes with probability 0.3
    if rng.uniform(0.0, 1.0) < 0.3:
        n_flares = int(rng.integers(2, 8))
        flare_indices = rng.choice(N_POINTS, size=n_flares, replace=False)
        flux[flare_indices] += rng.uniform(0.003, 0.012, size=n_flares)

    # Transit injection
    if has_planet:
        period = float(rng.uniform(0.8, 9.0))
        duration = float(rng.uniform(0.05, 0.22))
        depth = float(rng.uniform(0.002, 0.012))
        epoch = float(rng.uniform(0.0, period))

        # Box transit shape around periodic transit epochs
        dt = np.abs((TIME_GRID - epoch + 0.5 * period) % period - 0.5 * period)
        in_transit = dt <= (duration / 2.0)
        flux[in_transit] -= depth

        truth = {
            "has_planet": True,
            "period": period,
            "duration": duration,
            "depth": depth,
            "epoch": epoch,
        }
    else:
        truth = {
            "has_planet": False,
            "period": 0.0,
            "duration": 0.0,
            "depth": 0.0,
            "epoch": 0.0,
        }

    return flux, truth
