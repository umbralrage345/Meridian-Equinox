"""Cross-act parameter coupling and physical state transitions.

ALL cross-act formulas live here, nowhere else.
"""

from __future__ import annotations

from typing import Any


def _clip_float(val: float, low: float, high: float) -> float:
    return max(low, min(high, float(val)))


def gravity_from_depth(depth: float) -> float:
    """Calculate planetary surface gravity in m/s^2 from transit depth."""
    return _clip_float(2.0 + 800.0 * depth, 3.5, 12.0)


def wind_from_period(period_days: float) -> float:
    """Calculate target atmospheric wind speed in m/s from orbital period."""
    if period_days <= 0:
        return 6.0
    return _clip_float(6.0 / float(period_days), 0.5, 6.0)


def trip_days_from_period(period_days: float) -> int:
    """Calculate probe transit duration in days from orbital period."""
    return int(60 + 10 * period_days)


def n_faults(trip_days: int) -> int:
    """Calculate expected number of telemetry anomaly faults during transit."""
    return int(3 + trip_days // 50)


def stream_steps(trip_days: int) -> int:
    """Calculate total telemetry stream time steps from cruise duration."""
    return int(trip_days * 40)


def false_positive_target() -> dict[str, Any]:
    """Return environment parameters for a false-positive target selection."""
    return {
        "gravity": 14.0,
        "wind": 6.0,
        "trip_days": 150,
        "is_real_planet": False,
        "label": "dead world",
    }


def act2_resources(missed: int, false_alarms: int) -> dict[str, float]:
    """Calculate updated probe resource levels following telemetry anomaly detection."""
    fuel = _clip_float(100.0 - 7.0 * missed - 0.5 * false_alarms, 40.0, 100.0)
    noise = min(0.5, 0.08 * missed)
    hull = _clip_float(100.0 - 10.0 * missed, 20.0, 100.0)
    return {
        "fuel_pct": fuel,
        "sensor_noise": noise,
        "hull_health": hull,
    }
