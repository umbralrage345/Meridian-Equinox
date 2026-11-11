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


def get_max_jobs(cap: int = 4) -> int:
    """Return max worker threads/processes capped to min(cap, cpu_count, affinity)."""
    count = os.cpu_count() or 1
    try:
        if hasattr(os, "sched_getaffinity"):
            count = len(os.sched_getaffinity(0))
        else:
            import psutil
            count = len(psutil.Process().cpu_affinity())
    except Exception:
        pass

    for env_var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS"):
        if env_var in os.environ:
            try:
                count = min(count, int(os.environ[env_var]))
            except ValueError:
                pass

    return max(1, min(cap, count))
