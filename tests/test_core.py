"""Acceptance tests for meridian core scaffold, configuration, state, seeding, coupling, scoring, and leaderboard."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

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
from meridian.state import MissionState
from tools.build_leaderboard import build_leaderboard


def test_seed_determinism_and_uniqueness() -> None:
    """Test seed determinism and uniqueness across teams, salts, and types."""
    # Determinism
    seed1 = seed_for("team_orion", "act1")
    seed2 = seed_for("team_orion", "act1")
    assert seed1 == seed2, "Same team_id and salt must yield identical seed"

    # Difference across team_ids
    seed_other_team = seed_for("team_aquila", "act1")
    assert seed1 != seed_other_team, "Different team_ids must yield different seeds"

    # Difference across salts
    seed_other_salt = seed_for("team_orion", "act2")
    assert seed1 != seed_other_salt, "Different salts must yield different seeds"

    # 32-bit unsigned range check
    assert 0 <= seed1 <= 0xFFFFFFFF, "Seed must fit in an unsigned 32-bit integer"
    assert isinstance(seed1, int)

    # Integer team_id support
    seed_int = seed_for(42, "salt")
    seed_int2 = seed_for(42, "salt")
    assert seed_int == seed_int2


def test_state_defaults_save_load_roundtrip(tmp_path: Path) -> None:
    """Test state defaults, schema persistence, and load/save roundtrip."""
    team_id = "team_pulsar"
    state_file = tmp_path / f"mission_state_{team_id}.json"

    # Loading non-existent file should create default state and write it to disk
    state = MissionState.load(team_id=team_id, path=state_file)
    assert state_file.exists(), "MissionState.load should create default file when absent"
    assert state.team_id == team_id
    assert state.schema_version == 1
    assert state.total_score == 0.0

    # Verify default unscouted values
    assert state.resources.fuel_pct == 100.0
    assert state.resources.sensor_noise == 0.0
    assert state.resources.hull_health == 100.0
    assert state.act1.done is False
    assert state.act1.score == 0.0
    assert state.act1.target.gravity == 7.0
    assert state.act1.target.wind == 2.0
    assert state.act1.target.trip_days == 100
    assert state.act1.target.label == "unscouted world"
    assert state.act1.target.is_real_planet is False

    # Also test dict-subscript access
    assert state.resources["fuel_pct"] == 100.0
    assert state.act1.target["gravity"] == 7.0

    # Modify state fields across acts
    state.act1.done = True
    state.act1.score = 175.5
    state.act1.auc = 0.94
    state.act1.target.star_id = "STAR_77"
    state.act1.target.is_real_planet = True
    state.act1.target.gravity = 5.2
    state.act1.target.wind = 1.8
    state.act1.target.trip_days = 90
    state.act1.target.label = "confirmed planet"

    state.act2.done = True
    state.act2.score = 160.0
    state.act2.n_faults = 4
    state.act2.caught = 3
    state.act2.missed = 1
    state.act2.false_alarms = 2
    state.act2.mean_delay = 1.25

    state.act3.done = True
    state.act3.score = 190.0
    state.act3.accuracy = 0.88
    state.act3.value_ratio = 1.15
    state.act3.backend = "sklearn"

    state.act4.done = True
    state.act4.score = 340.0
    state.act4.landing_rate = 0.85
    state.act4.mean_reward = 210.0
    state.act4.conditions = {"gravity": 5.2, "wind": 1.8}

    state.resources.fuel_pct = 92.0
    state.resources.sensor_noise = 0.08
    state.resources.hull_health = 90.0

    expected_total = 175.5 + 160.0 + 190.0 + 340.0
    assert state.total_score == pytest.approx(expected_total)

    # Save and reload
    state.save(state_file)
    reloaded = MissionState.load(team_id=team_id, path=state_file)

    assert reloaded.team_id == team_id
    assert reloaded.schema_version == 1
    assert reloaded.total_score == pytest.approx(expected_total)
    assert reloaded.act1.score == pytest.approx(175.5)
    assert reloaded.act1.target.label == "confirmed planet"
    assert reloaded.act2.caught == 3
    assert reloaded.act3.backend == "sklearn"
    assert reloaded.act4.landing_rate == pytest.approx(0.85)
    assert reloaded.resources.fuel_pct == pytest.approx(92.0)


def test_export_and_verify_tamper_detection() -> None:
    """Test mission code export, verification, and tamper detection."""
    state = MissionState(team_id="team_centauri")
    state.act1.score = 180.0
    state.act2.score = 150.0
    state.act3.score = 170.0
    state.act4.score = 360.0
    state.resources.fuel_pct = 85.0
    state.resources.sensor_noise = 0.05
    state.resources.hull_health = 95.0

    code = export_code(state)
    assert isinstance(code, str)
    assert code.startswith("MER1-")

    # Round trip verify
    payload = verify_code(code)
    assert payload is not None
    assert payload["team_id"] == "team_centauri"
    assert payload["total"] == pytest.approx(860.0)
    assert payload["act1"] == pytest.approx(180.0)
    assert payload["resources"]["fuel_pct"] == pytest.approx(85.0)

    # Tamper with signature
    parts = code.split("-")
    tampered_sig = parts[0] + "-" + parts[1] + "-" + ("0" * len(parts[2]))
    assert verify_code(tampered_sig) is None, "Tampered signature must fail verification"

    # Tamper with base64 payload
    tampered_b64 = parts[0] + "-" + parts[1][:-1] + "A-" + parts[2]
    assert verify_code(tampered_b64) is None, "Tampered payload must fail verification"

    # Malformed prefix
    bad_prefix = "MER2-" + parts[1] + "-" + parts[2]
    assert verify_code(bad_prefix) is None, "Invalid prefix must fail verification"

    # Non-code strings
    assert verify_code("invalid_garbage") is None
    assert verify_code("") is None


def test_coupling_functions_bounds_and_monotonicity() -> None:
    """Test coupling formulas for monotonicity, clipping bounds, and exact behavior."""
    # gravity_from_depth(d) = clip(2.0 + 800*d, 3.5, 12.0)
    assert gravity_from_depth(0.0) == 3.5, "Lower bound clip must be 3.5"
    assert gravity_from_depth(0.001) == 3.5  # 2.0 + 0.8 = 2.8 -> clipped to 3.5
    assert gravity_from_depth(0.005) == pytest.approx(6.0)
    assert gravity_from_depth(0.010) == pytest.approx(10.0)
    assert gravity_from_depth(0.020) == 12.0, "Upper bound clip must be 12.0"
    # Monotonicity check
    depths = [0.002, 0.005, 0.008, 0.011, 0.015]
    gravities = [gravity_from_depth(d) for d in depths]
    assert all(g1 <= g2 for g1, g2 in zip(gravities, gravities[1:])), "Gravity must be non-decreasing"

    # wind_from_period(p) = clip(6.0/p, 0.5, 6.0)
    assert wind_from_period(0.5) == 6.0, "Upper bound clip must be 6.0"
    assert wind_from_period(1.0) == 6.0
    assert wind_from_period(2.0) == pytest.approx(3.0)
    assert wind_from_period(6.0) == pytest.approx(1.0)
    assert wind_from_period(20.0) == 0.5, "Lower bound clip must be 0.5"
    # Monotonicity check
    periods = [0.8, 1.5, 3.0, 6.0, 15.0]
    winds = [wind_from_period(p) for p in periods]
    assert all(w1 >= w2 for w1, w2 in zip(winds, winds[1:])), "Wind must be non-increasing with period"

    # trip_days_from_period(p) = int(60 + 10*p)
    assert trip_days_from_period(1.0) == 70
    assert trip_days_from_period(5.5) == 115
    assert trip_days_from_period(10.0) == 160

    # n_faults(trip_days) = 3 + trip_days//50
    assert n_faults(49) == 3
    assert n_faults(50) == 4
    assert n_faults(99) == 4
    assert n_faults(100) == 5

    # stream_steps(trip_days) = trip_days * 40
    assert stream_steps(100) == 4000
    assert stream_steps(150) == 6000

    # false-positive target
    fp = false_positive_target()
    assert fp["gravity"] == 14.0
    assert fp["wind"] == 6.0
    assert fp["trip_days"] == 150
    assert fp["is_real_planet"] is False
    assert fp["label"] == "dead world"

    # act2_resources(missed, false_alarms)
    res_clean = act2_resources(0, 0)
    assert res_clean["fuel_pct"] == 100.0
    assert res_clean["sensor_noise"] == 0.0
    assert res_clean["hull_health"] == 100.0

    res_mild = act2_resources(2, 4)
    # fuel: 100 - 7*2 - 0.5*4 = 100 - 14 - 2 = 84.0
    # noise: min(0.5, 0.08*2) = 0.16
    # hull: 100 - 10*2 = 80.0
    assert res_mild["fuel_pct"] == pytest.approx(84.0)
    assert res_mild["sensor_noise"] == pytest.approx(0.16)
    assert res_mild["hull_health"] == pytest.approx(80.0)

    # Clipping bounds
    res_heavy = act2_resources(20, 20)
    assert res_heavy["fuel_pct"] == 40.0, "Fuel lower bound clip must be 40"
    assert res_heavy["sensor_noise"] == 0.5, "Sensor noise upper bound clip must be 0.5"
    assert res_heavy["hull_health"] == 20.0, "Hull health lower bound clip must be 20"


def test_leaderboard_tool_five_codes(tmp_path: Path) -> None:
    """Test leaderboard tool on 5 fake codes (including one tampered and one duplicate team)."""
    # Create 4 valid states:
    # Team Alpha: 620.0
    s_alpha = MissionState(team_id="team_alpha")
    s_alpha.act1.score = 150.0
    s_alpha.act2.score = 150.0
    s_alpha.act3.score = 160.0
    s_alpha.act4.score = 160.0

    # Team Beta: 710.0
    s_beta = MissionState(team_id="team_beta")
    s_beta.act1.score = 180.0
    s_beta.act2.score = 170.0
    s_beta.act3.score = 180.0
    s_beta.act4.score = 180.0

    # Team Gamma submission 1: 500.0
    s_gamma_v1 = MissionState(team_id="team_gamma")
    s_gamma_v1.act1.score = 100.0
    s_gamma_v1.act2.score = 100.0
    s_gamma_v1.act3.score = 150.0
    s_gamma_v1.act4.score = 150.0

    # Team Gamma submission 2 (improved duplicate): 830.0
    s_gamma_v2 = MissionState(team_id="team_gamma")
    s_gamma_v2.act1.score = 200.0
    s_gamma_v2.act2.score = 190.0
    s_gamma_v2.act3.score = 190.0
    s_gamma_v2.act4.score = 250.0

    code_alpha = export_code(s_alpha)
    code_beta = export_code(s_beta)
    code_gamma_v1 = export_code(s_gamma_v1)
    code_gamma_v2 = export_code(s_gamma_v2)

    # 5th code: tampered
    parts = code_alpha.split("-")
    code_tampered = f"MER1-{parts[1]}-deadbeef1234"

    codes = [
        code_alpha,
        code_gamma_v1,
        code_tampered,
        code_beta,
        code_gamma_v2,
    ]

    codes_file = tmp_path / "submission_codes.txt"
    codes_file.write_text("\n".join(codes) + "\n", encoding="utf-8")

    out_dir = tmp_path / "leaderboard_out"
    csv_path, html_path = build_leaderboard(codes_file=codes_file, out_dir=out_dir)

    assert csv_path.exists()
    assert html_path.exists()

    # Read CSV
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    # Exactly 3 distinct teams (tampered excluded, duplicate deduplicated)
    assert len(reader) == 3, f"Expected 3 entries in leaderboard, found {len(reader)}"

    # Check rank order: Gamma (830.0) > Beta (710.0) > Alpha (620.0)
    assert reader[0]["team_id"] == "team_gamma"
    assert reader[0]["rank"] == "1"
    assert float(reader[0]["total"]) == pytest.approx(830.0)

    assert reader[1]["team_id"] == "team_beta"
    assert reader[1]["rank"] == "2"
    assert float(reader[1]["total"]) == pytest.approx(710.0)

    assert reader[2]["team_id"] == "team_alpha"
    assert reader[2]["rank"] == "3"
    assert float(reader[2]["total"]) == pytest.approx(620.0)

    # Verify HTML content
    html_text = html_path.read_text(encoding="utf-8")
    assert "team_gamma" in html_text
    assert "830.00" in html_text
    assert "team_beta" in html_text
    assert "team_alpha" in html_text
    assert "deadbeef" not in html_text
