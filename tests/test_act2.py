"""Acceptance tests for Act 2 Telemetry stream generator, anomaly detectors, and cross-act state updates."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from meridian.act2_telemetry import (
    CHANNELS,
    TelemetryDetector,
    TelemetryResult,
    animate_stream,
    apply_to_state,
    build_stream,
    make_stream,
    plot_result,
    run,
    train,
    validate_knobs,
)
from meridian.act2_telemetry.detectors import STRIDE
from meridian.coupling import act2_resources
from meridian.state import MissionState


def test_act2_determinism_and_uniqueness() -> None:
    """Acceptance 1: Telemetry generation determinism by team_id."""
    team_a = "team_alpha_telem"
    team_b = "team_beta_telem"

    stream_a1 = make_stream(team_a, trip_days=100, fast=True)
    stream_a2 = make_stream(team_a, trip_days=100, fast=True)
    stream_b = make_stream(team_b, trip_days=100, fast=True)

    # Check identical channels
    for ch in CHANNELS:
        assert np.array_equal(stream_a1.train_data[ch], stream_a2.train_data[ch])
        assert np.array_equal(stream_a1.mission_data[ch], stream_a2.mission_data[ch])
        assert not np.array_equal(stream_a1.mission_data[ch], stream_b.mission_data[ch])

    # Check identical fault structures
    assert len(stream_a1.faults) == len(stream_a2.faults)
    for f1, f2 in zip(stream_a1.faults, stream_a2.faults):
        assert f1["type"] == f2["type"]
        assert f1["channel"] == f2["channel"]
        assert f1["start"] == f2["start"]
        assert f1["end"] == f2["end"]


def test_act2_default_knobs_catch_rate_and_false_alarms() -> None:
    """Acceptance 2: Default knobs over 5 team_ids achieve mean catch rate >= 0.80 and false alarms <= 4."""
    team_ids = [
        "team_sirius_01",
        "team_vega_02",
        "team_rigel_03",
        "team_altair_04",
        "team_deneb_05",
    ]

    catch_rates: list[float] = []
    false_alarm_counts: list[int] = []

    for tid in team_ids:
        stream = make_stream(tid, trip_days=100, fast=True)
        detector = train(stream)  # default knobs: detector='both', window=64, percentile=99, persistence=1
        result = run(detector, stream)

        cr = result.caught / result.n_faults if result.n_faults > 0 else 1.0
        catch_rates.append(cr)
        false_alarm_counts.append(result.false_alarms)

    mean_catch_rate = float(np.mean(catch_rates))
    mean_false_alarms = float(np.mean(false_alarm_counts))

    print(f"\nMeasured Mean Catch Rate across 5 teams: {mean_catch_rate * 100:.1f}%")
    print(f"Measured Mean False Alarms across 5 teams: {mean_false_alarms:.2f}")

    assert mean_catch_rate >= 0.80, f"Mean catch rate {mean_catch_rate:.2f} < 0.80"
    assert mean_false_alarms <= 4.0, f"Mean false alarms {mean_false_alarms:.2f} > 4.0"


def test_act2_window_level_roc_auc_separately() -> None:
    """Acceptance 3: Window-level ROC-AUC >= 0.90 for iforest and autoencoder separately, pooled over channels."""
    team_id = "team_auc_benchmark"
    stream = make_stream(team_id, trip_days=100, fast=True)
    window = 64

    # Train iforest separately
    det_if = train(stream, detector="iforest", window=window)
    res_if = run(det_if, stream)

    # Train autoencoder separately
    det_ae = train(stream, detector="autoencoder", window=window)
    res_ae = run(det_ae, stream)

    # Build ground-truth binary labels per channel-window
    y_true_pooled: list[int] = []
    scores_if_pooled: list[float] = []
    scores_ae_pooled: list[float] = []

    window_starts = res_if.window_starts

    for ch in CHANNELS:
        ch_faults = [f for f in stream.faults if f["channel"] == ch]
        sc_if = res_if.channel_scores[ch]["iforest"]
        sc_ae = res_ae.channel_scores[ch]["autoencoder"]

        for i, ws in enumerate(window_starts):
            we = ws + window
            is_anomaly = int(any(max(ws, f["start"]) < min(we, f["end"]) for f in ch_faults))

            y_true_pooled.append(is_anomaly)
            scores_if_pooled.append(float(sc_if[i]))
            scores_ae_pooled.append(float(sc_ae[i]))

    auc_if = float(roc_auc_score(y_true_pooled, scores_if_pooled))
    auc_ae = float(roc_auc_score(y_true_pooled, scores_ae_pooled))

    print(f"\nMeasured Window-Level ROC-AUC (iForest pooled): {auc_if:.4f}")
    print(f"Measured Window-Level ROC-AUC (Autoencoder pooled): {auc_ae:.4f}")

    assert auc_if >= 0.90, f"iForest ROC-AUC {auc_if:.4f} < 0.90"
    assert auc_ae >= 0.90, f"Autoencoder ROC-AUC {auc_ae:.4f} < 0.90"


def test_act2_runtime_budget_fast() -> None:
    """Acceptance 4: End-to-end execution runtime <= 120s in FAST mode."""
    state = MissionState(team_id="team_runtime_perf")
    state.act1.done = True
    state.act1.target.trip_days = 100

    t0 = time.time()
    stream = build_stream(state, fast=True)
    t_stream = time.time()

    detector = train(stream)
    t_train = time.time()

    result = run(detector, stream)
    t_run = time.time()

    apply_to_state(state, result)
    t_apply = time.time()

    total_time = t_apply - t0
    print(
        f"\nMeasured Act 2 Runtime: {total_time:.2f}s "
        f"(Stream: {t_stream - t0:.2f}s, Train: {t_train - t_stream:.2f}s, "
        f"Run: {t_run - t_train:.2f}s, Apply: {t_apply - t_run:.2f}s)"
    )

    assert total_time <= 120.0, f"Act 2 execution time {total_time:.2f}s exceeded 120s budget"


def test_act2_resource_formulas_handmade_result() -> None:
    """Acceptance 5: Verify resource coupling formulas with a hand-made Result."""
    state = MissionState(team_id="team_handmade_test")

    # Hand-made test result: 5 faults total, 3 caught, 2 missed, 4 false alarms
    fake_result = TelemetryResult(
        score=140.0,
        n_faults=5,
        caught=3,
        missed=2,
        false_alarms=4,
        mean_delay=45.0,
        alarm_runs=[(600, 750), (1400, 1600)],
        fault_details=[],
        window_starts=np.array([0, 8]),
        channel_scores={},
    )

    apply_to_state(state, fake_result)

    # coupling formulas:
    # fuel_pct = clip(100 - 7*2 - 0.5*4, 40, 100) = 100 - 14 - 2 = 84.0
    # sensor_noise = min(0.5, 0.08*2) = 0.16
    # hull_health = clip(100 - 10*2, 20, 100) = 80.0
    expected_res = act2_resources(missed=2, false_alarms=4)

    assert state.resources.fuel_pct == pytest.approx(84.0)
    assert state.resources.fuel_pct == pytest.approx(expected_res["fuel_pct"])
    assert state.resources.sensor_noise == pytest.approx(0.16)
    assert state.resources.sensor_noise == pytest.approx(expected_res["sensor_noise"])
    assert state.resources.hull_health == pytest.approx(80.0)
    assert state.resources.hull_health == pytest.approx(expected_res["hull_health"])

    assert state.act2.done is True
    assert state.act2.score == pytest.approx(140.0)
    assert state.act2.caught == 3
    assert state.act2.missed == 2
    assert state.act2.false_alarms == 4
    assert state.act2.mean_delay == pytest.approx(45.0)


def test_act2_artifact_exists_and_animation() -> None:
    """Acceptance 6: Verify artifacts/act2_result.png exists and animation generation functions cleanly."""
    artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "act2_result.png"
    assert artifact_path.exists(), f"act2_result.png must exist at {artifact_path}"
    assert artifact_path.stat().st_size > 50_000, "act2_result.png must be non-empty"

    # Test animation rendering to JSHTML
    stream = make_stream("anim_test", trip_days=70, fast=True)
    html_str = animate_stream(stream)
    assert isinstance(html_str, str)
    assert "anim" in html_str.lower() or "javascript" in html_str.lower()
