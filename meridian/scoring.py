"""Score export and cryptographic verification utilities."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from typing import TYPE_CHECKING, Any

from meridian.config import SECRET

if TYPE_CHECKING:
    from meridian.state import MissionState


def export_code(state: MissionState) -> str:
    """Export mission completion code with HMAC-SHA256 integrity token."""
    payload: dict[str, Any] = {
        "team_id": str(state.team_id),
        "scores": {
            "act1": float(state.act1.score),
            "act2": float(state.act2.score),
            "act3": float(state.act3.score),
            "act4": float(state.act4.score),
        },
        "act1": float(state.act1.score),
        "act2": float(state.act2.score),
        "act3": float(state.act3.score),
        "act4": float(state.act4.score),
        "total": float(state.total_score),
        "resources": {
            "fuel_pct": float(state.resources.fuel_pct),
            "sensor_noise": float(state.resources.sensor_noise),
            "hull_health": float(state.resources.hull_health),
        },
    }

    json_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    b64_str = base64.urlsafe_b64encode(json_bytes).decode("ascii").rstrip("=")
    sig = hmac.new(SECRET.encode("utf-8"), b64_str.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    return f"MER1-{b64_str}-{sig}"


def verify_code(code: str) -> dict[str, Any] | None:
    """Verify an exported mission code, returning payload dict or None if invalid/tampered."""
    if not isinstance(code, str):
        return None

    code_clean = code.strip()
    parts = code_clean.split("-")
    if len(parts) != 3:
        return None

    prefix, b64_str, sig = parts
    if prefix != "MER1":
        return None

    expected_sig = hmac.new(SECRET.encode("utf-8"), b64_str.encode("utf-8"), hashlib.sha256).hexdigest()[:12]
    if not hmac.compare_digest(sig.lower(), expected_sig.lower()):
        return None

    try:
        padding = "=" * (-len(b64_str) % 4)
        raw_json = base64.urlsafe_b64decode((b64_str + padding).encode("ascii")).decode("utf-8")
        data = json.loads(raw_json)
        if not isinstance(data, dict):
            return None
        if "team_id" not in data or "total" not in data:
            return None
        return data
    except Exception:
        return None
