"""Meridian: Workshop simulation package for autonomous probe missions."""

__version__ = "0.1.0"

from meridian.config import FAST_BUDGETS, MAX_SCORES, SECRET, is_fast
from meridian.coupling import (
    act2_resources,
    false_positive_target,
    gravity_from_depth,
    n_faults,
    stream_steps,
    trip_days_from_period,
    wind_from_period,
)
from meridian.scoring import export_code, verify_code
from meridian.seeding import seed_for
from meridian.state import (
    Act1State,
    Act1Target,
    Act2State,
    Act3State,
    Act4State,
    MissionState,
    ResourcesState,
    load_state,
    save_state,
)

__all__ = [
    "FAST_BUDGETS",
    "MAX_SCORES",
    "SECRET",
    "is_fast",
    "seed_for",
    "MissionState",
    "ResourcesState",
    "Act1State",
    "Act1Target",
    "Act2State",
    "Act3State",
    "Act4State",
    "load_state",
    "save_state",
    "gravity_from_depth",
    "wind_from_period",
    "trip_days_from_period",
    "n_faults",
    "stream_steps",
    "false_positive_target",
    "act2_resources",
    "export_code",
    "verify_code",
]
