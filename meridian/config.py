"""Runtime configuration and constants for Project Meridian."""

import os
from typing import Final

FAST_BUDGETS: Final[dict[str, int]] = {
    "act1": 240,
    "act2": 120,
    "act3": 240,
    "act4": 300,
}

MAX_SCORES: Final[dict[str, int]] = {
    "act1": 200,
    "act2": 200,
    "act3": 200,
    "act4": 400,
}

SECRET: Final[str] = "meridian-probe-telemetry-auth-secret-2026"


def is_fast() -> bool:
    """Check if fast execution mode is enabled via MERIDIAN_FAST env var."""
    val = os.environ.get("MERIDIAN_FAST", "0").strip().lower()
    return val in ("1", "true", "yes")
