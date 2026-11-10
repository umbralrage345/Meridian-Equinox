"""Synthetic telemetry stream generation with fault injection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from meridian.coupling import n_faults, stream_steps
from meridian.seeding import seed_for

CHANNELS: list[str] = ["bus_voltage", "battery_temp", "wheel_rpm", "solar_current"]
CHANNEL_AMPLITUDES: dict[str, float] = {
    "bus_voltage": 1.0,
    "battery_temp": 1.2,
    "wheel_rpm": 1.1,
    "solar_current": 0.9,
}
FAULT_TYPES: list[str] = ["level_shift", "stuck_sensor", "noisy_burst", "slow_drift"]


@dataclass
class TelemetryStream:
    """Telemetry stream bundle containing nominal training data and mission stream with injected faults."""

    team_id: str
    trip_days: int
    is_fast: bool
    train_data: dict[str, np.ndarray]
    mission_data: dict[str, np.ndarray]
    faults: list[dict[str, Any]] = field(default_factory=list)


def make_stream(
    team_id: str | int,
    trip_days: int,
    fast: bool = False,
) -> TelemetryStream:
    """Generate 4-channel telemetry streams for baseline training and mission execution with seeded faults."""
    seed = seed_for(team_id, "act2")
    rng = np.random.default_rng(seed)

    n_train_steps = 3000 if fast else 6000
    n_mission_steps = min(3000 if fast else 6000, stream_steps(trip_days))

    t_tr = np.arange(n_train_steps, dtype=np.float64)
    # Continuous phase from training stream into mission cruise
    t_mi = np.arange(n_train_steps, n_train_steps + n_mission_steps, dtype=np.float64)

    train_data: dict[str, np.ndarray] = {}
    mission_data: dict[str, np.ndarray] = {}

    for ch in CHANNELS:
        amp = CHANNEL_AMPLITUDES[ch]
        p1 = float(rng.uniform(198.0, 202.0))
        p2 = float(rng.uniform(46.0, 48.0))
        phi1 = float(rng.uniform(0.0, 2.0 * np.pi))
        phi2 = float(rng.uniform(0.0, 2.0 * np.pi))
        sig = 0.08 * amp

        y_tr = (
            0.6 * amp * np.sin(2.0 * np.pi * t_tr / p1 + phi1)
            + 0.4 * amp * np.sin(2.0 * np.pi * t_tr / p2 + phi2)
            + rng.normal(0.0, sig, n_train_steps)
        )
        y_mi = (
            0.6 * amp * np.sin(2.0 * np.pi * t_mi / p1 + phi1)
            + 0.4 * amp * np.sin(2.0 * np.pi * t_mi / p2 + phi2)
            + rng.normal(0.0, sig, n_mission_steps)
        )

        train_data[ch] = y_tr
        mission_data[ch] = y_mi

    nf = n_faults(trip_days)
    fault_list: list[dict[str, Any]] = []

    # Inject faults only into mission stream: non-overlapping, none in first 500 steps
    min_start = 500
    available_steps = n_mission_steps - min_start - 60
    seg_len = available_steps // nf

    for i in range(nf):
        ftype = FAULT_TYPES[i % len(FAULT_TYPES)]
        ch = str(rng.choice(CHANNELS))
        amp = CHANNEL_AMPLITUDES[ch]

        if ftype == "level_shift":
            dur = int(rng.integers(60, 151))
        elif ftype == "stuck_sensor":
            dur = int(rng.integers(60, 121))
        elif ftype == "noisy_burst":
            dur = int(rng.integers(150, 301))
        else:  # slow_drift
            dur = int(rng.integers(200, 401))

        dur = min(dur, seg_len - 25)
        seg_start = min_start + i * seg_len
        max_offset = max(11, seg_len - dur - 15)
        offset = int(rng.integers(10, max_offset))
        start = seg_start + offset
        end = start + dur

        if ftype == "level_shift":
            mission_data[ch][start:end] += 1.2 * amp
        elif ftype == "stuck_sensor":
            frozen_val = float(mission_data[ch][start])
            mission_data[ch][start:end] = frozen_val + rng.normal(0.0, 0.005 * amp, dur)
        elif ftype == "noisy_burst":
            sig = 0.08 * amp
            mission_data[ch][start:end] += rng.normal(0.0, sig * 5.0, dur)
        elif ftype == "slow_drift":
            ramp = np.linspace(0.0, 1.0 * amp, dur)
            mission_data[ch][start:end] += ramp

        fault_list.append({
            "type": ftype,
            "channel": ch,
            "start": int(start),
            "end": int(end),
        })

    return TelemetryStream(
        team_id=str(team_id),
        trip_days=trip_days,
        is_fast=fast,
        train_data=train_data,
        mission_data=mission_data,
        faults=fault_list,
    )
