"""Procedural galaxy image generator and survey dataset construction for Act 3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from meridian.config import is_fast
from meridian.seeding import seed_for

CLASSES: Final[list[str]] = ["elliptical", "spiral", "edge_on", "merger"]
CLASS_PRIORS: Final[list[float]] = [0.45, 0.30, 0.15, 0.10]
VALUE_WEIGHTS: Final[dict[int, int]] = {0: 1, 1: 3, 2: 2, 3: 5}
IMAGE_SIZE: Final[int] = 48
DOWNLINK_BUDGET: Final[int] = 20


@dataclass
class GalaxySurvey:
    """Survey bundle containing training, test, and downlink candidate sky streams."""

    team_id: str
    is_fast: bool
    train_images: np.ndarray  # (N, 48, 48), float32
    train_labels: np.ndarray  # (N,), int64
    test_images: np.ndarray   # (N_test, 48, 48), float32
    test_labels: np.ndarray   # (N_test,), int64
    stream_images: np.ndarray # (200, 48, 48), float32
    stream_labels: np.ndarray # (200,), int64 (hidden truth)


def make_galaxy(rng: np.random.Generator, cls: int) -> np.ndarray:
    """Generate a single 48x48 float32 procedural galaxy image in [0, 1].

    Classes:
        0 - elliptical: Sersic-like profile exp(-2.2*(r/re)^0.5), ellipticity q~U(0.5,1), random rotation.
        1 - spiral: bulge + exponential disk modulated by 2-3 logarithmic spiral arms, inclination q~U(0.75,1.0).
        2 - edge_on: very flattened disk (axis ratio 0.08-0.22) with a small bulge and dust lane.
        3 - merger: two overlapping blobs with random offset and brightness ratio + curved tidal tail.
    """
    size = IMAGE_SIZE
    y, x = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = 23.5, 23.5
    theta = float(rng.uniform(0.0, 2.0 * np.pi))
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))

    # Rotated coordinates
    xr = cos_t * (x - cx) + sin_t * (y - cy)
    yr = -sin_t * (x - cx) + cos_t * (y - cy)
    r_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    aperture = np.clip((20.0 - r_center) / 3.0, 0.0, 1.0)

    if cls == 0:
        # Elliptical: Sersic-like profile exp(-2.2*(r/re)^0.5), ellipticity q~U(0.5,1), random rotation
        q = float(rng.uniform(0.50, 0.95))
        re = float(rng.uniform(4.5, 7.5))
        r = np.sqrt(xr**2 + (yr / q)**2)
        img = np.exp(-2.2 * (np.maximum(0.1, r / re))**0.5)

    elif cls == 1:
        # Spiral: bulge + exponential disk modulated by 2-3 logarithmic spiral arms, inclination q~U(0.80,1.0)
        q = float(rng.uniform(0.80, 1.0))
        r = np.sqrt(xr**2 + (yr / q)**2)
        phi = np.arctan2(yr / q, xr)

        rb = float(rng.uniform(0.7, 1.3))
        bulge = float(rng.uniform(4.5, 6.5)) * np.exp(-2.2 * (np.maximum(0.04, r / rb))**0.5)

        rd = float(rng.uniform(4.5, 6.5))
        disk = np.exp(-r / rd)
        n_arms = int(rng.choice([2, 3]))
        handedness = float(rng.choice([-1.0, 1.0]))
        pitch = float(rng.uniform(2.2, 3.5))
        arm_phase = n_arms * phi + handedness * pitch * np.log(np.maximum(1.0, r / 2.0))
        arm_raw = (0.5 + 0.5 * np.cos(arm_phase)) ** 2.2
        arm_pattern = 0.15 + 0.85 * arm_raw
        arm_env = np.clip(r / 2.0, 0.0, 1.0) * np.exp(-r / rd)
        img = bulge + 0.08 * disk + 1.5 * arm_pattern * arm_env

    elif cls == 2:
        # Edge-on: very flattened disk (axis ratio 0.08-0.22) with a small bulge
        q = float(rng.uniform(0.10, 0.22))
        r_maj = np.abs(xr)
        r_min = np.abs(yr)
        rd = float(rng.uniform(5.5, 8.5))
        z0 = float(rng.uniform(1.8, 2.8))
        disk = np.exp(-r_maj / rd) / (np.cosh(r_min / z0)**2)

        rb = float(rng.uniform(1.5, 2.5))
        bulge = float(rng.uniform(0.5, 0.9)) * np.exp(
            -2.2 * (np.maximum(0.1, np.sqrt(xr**2 + (yr / 0.4)**2) / rb))**0.5
        )
        dust = 1.0 - 0.25 * np.exp(-r_min / 0.8) * np.exp(-r_maj / (0.8 * rd))
        env = 0.15 * np.exp(-r_center / 8.0)
        img = (bulge + disk) * dust + env

    else:
        # Merger: two overlapping blobs with random offset and brightness ratio + curved tidal tail
        d = float(rng.uniform(5.0, 9.0))
        phi_d = float(rng.uniform(0.0, 2.0 * np.pi))
        c1x = float(cx - 0.5 * d * np.cos(phi_d))
        c1y = float(cy - 0.5 * d * np.sin(phi_d))
        c2x = float(cx + 0.5 * d * np.cos(phi_d))
        c2y = float(cy + 0.5 * d * np.sin(phi_d))

        r1 = np.sqrt((x - c1x)**2 + (y - c1y)**2)
        r2 = np.sqrt((x - c2x)**2 + (y - c2y)**2)
        b1 = float(rng.uniform(1.0, 1.5)) * np.exp(-2.0 * (np.maximum(0.1, r1 / float(rng.uniform(2.5, 4.0))))**0.5)
        ratio = float(rng.uniform(0.45, 0.90))
        b2 = ratio * np.exp(-2.0 * (np.maximum(0.1, r2 / float(rng.uniform(2.5, 3.5))))**0.5)

        # Faint curved tidal tail
        n_steps = 35
        tail_theta = np.linspace(0.0, float(rng.uniform(0.6 * np.pi, 1.1 * np.pi)), n_steps)
        tail_r = np.linspace(2.0, float(rng.uniform(10.0, 16.0)), n_steps)
        tail_sign = float(rng.choice([-1.0, 1.0]))
        tx = c1x + tail_r * np.cos(phi_d + tail_sign * tail_theta)
        ty = c1y + tail_r * np.sin(phi_d + tail_sign * tail_theta)
        tail_img = np.zeros((size, size), dtype=np.float32)
        tail_b = float(rng.uniform(0.12, 0.30))
        for px, py, intensity in zip(tx, ty, np.linspace(tail_b, 0.03, n_steps)):
            if 0.0 <= px < size and 0.0 <= py < size:
                dist_sq = (x - px)**2 + (y - py)**2
                tail_img += float(intensity) * np.exp(-0.5 * dist_sq / 1.8**2)

        img = b1 + b2 + tail_img

    img = img * aperture

    # Total-flux normalization plus a fixed global scale
    tot = float(np.sum(img))
    if tot > 1e-4:
        img = img / tot

    # Add Gaussian noise with the SAME absolute sigma for every class
    noisy = (img * 125.0) + rng.normal(0.0, 0.025, size=(size, size))
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def build_survey(team_id: str | int, fast: bool | None = None) -> GalaxySurvey:
    """Construct deterministic survey of training, test, and downlink candidate galaxy images."""
    if fast is None:
        fast = is_fast()

    seed = seed_for(team_id, "act3_survey")
    rng = np.random.default_rng(seed)

    n_train = 2000 if fast else 4000
    n_test = 500 if fast else 1000
    n_stream = 200

    y_train = rng.choice(4, size=n_train, p=CLASS_PRIORS).astype(np.int64)
    train_imgs = np.empty((n_train, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, c in enumerate(y_train):
        train_imgs[i] = make_galaxy(rng, int(c))

    y_test = rng.choice(4, size=n_test, p=CLASS_PRIORS).astype(np.int64)
    test_imgs = np.empty((n_test, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, c in enumerate(y_test):
        test_imgs[i] = make_galaxy(rng, int(c))

    y_stream = rng.choice(4, size=n_stream, p=CLASS_PRIORS).astype(np.int64)
    stream_imgs = np.empty((n_stream, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for i, c in enumerate(y_stream):
        stream_imgs[i] = make_galaxy(rng, int(c))

    return GalaxySurvey(
        team_id=str(team_id),
        is_fast=bool(fast),
        train_images=train_imgs,
        train_labels=y_train,
        test_images=test_imgs,
        test_labels=y_test,
        stream_images=stream_imgs,
        stream_labels=y_stream,
    )
