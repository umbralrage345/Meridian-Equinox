"""Deterministic seeding utility for reproducibility across teams and runs."""

import hashlib


def seed_for(team_id: str | int, salt: str | int = "") -> int:
    """Generate a deterministic 32-bit unsigned integer seed from team_id and salt."""
    key = f"{team_id}:{salt}".encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:4], byteorder="big", signed=False)
