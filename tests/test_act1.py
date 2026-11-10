"""Acceptance tests for Act 1 Exoplanet transit detection pipeline, models, and coupling."""

from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from meridian.act1_exoplanet import (
    build_survey,
    choose_target,
    rank_candidates,
    train,
    validate_knobs,
)
from meridian.act1_exoplanet.generator import TIME_GRID, make_star
from meridian.coupling import (
    false_positive_target,
    gravity_from_depth,
    trip_days_from_period,
    wind_from_period,
)
from meridian.state import MissionState


def test_generator_shape_and_physics() -> None:
    """Test lightcurve generation shapes, noise characteristics, and transit injection."""
    rng = np.random.default_rng(42)
    flux_clean, truth_clean = make_star(rng, has_planet=False)
    assert len(flux_clean) == 1296
    assert truth_clean["has_planet"] is False
    assert truth_clean["depth"] == 0.0

    flux_pl, truth_pl = make_star(rng, has_planet=True)
    assert len(flux_pl) == 1296
    assert truth_pl["has_planet"] is True
    assert 0.8 <= truth_pl["period"] <= 9.0
    assert 0.05 <= truth_pl["duration"] <= 0.22
    assert 0.002 <= truth_pl["depth"] <= 0.012


def test_survey_determinism_and_uniqueness() -> None:
    """Acceptance 1: Same team_id => identical survey (hash arrays); different team_id => different."""
    survey_a1 = build_survey("team_pegasus", fast=True)
    survey_a2 = build_survey("team_pegasus", fast=True)
    survey_b = build_survey("team_cygnus", fast=True)

    hash_a1 = hashlib.sha256(survey_a1.X_train.tobytes()).hexdigest()
    hash_a2 = hashlib.sha256(survey_a2.X_train.tobytes()).hexdigest()
    hash_b = hashlib.sha256(survey_b.X_train.tobytes()).hexdigest()

    assert hash_a1 == hash_a2, "Identical team_id must yield byte-for-byte identical survey features"
    assert np.array_equal(survey_a1.y_train, survey_a2.y_train)
    assert hash_a1 != hash_b, "Different team_ids must yield distinct survey datasets"


def test_knobs_validation_clips_and_warns() -> None:
    """Test that validate_knobs clips extreme values and emits warnings without raising."""
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        knobs = validate_knobs(
            n_estimators=1000,
            max_depth=1,
            feature_set="unsupported_feature_type",
            threshold=0.95,
        )
        assert len(recorded_warnings) >= 4
        assert knobs["n_estimators"] == 500
        assert knobs["max_depth"] == 3
        assert knobs["feature_set"] == "full"
        assert knobs["threshold"] == pytest.approx(0.8)


def test_act1_end_to_end_runtime_auc_and_coupling(tmp_path: Path) -> None:
    """Acceptance 2, 3, 4: Hidden eval ROC-AUC >= 0.90, FAST runtime <= 240s, coupling on FP & TP."""
    t0 = time.time()
    team_id = "team_andromeda"

    # Step 1: Build survey in FAST mode
    survey = build_survey(team_id=team_id, fast=True)
    t_survey = time.time()

    # Step 2: Train baseline model
    model = train(survey)
    t_train = time.time()

    # Step 3: Evaluate hidden AUC
    eval_probs = model.predict_proba(survey._X_eval)[:, 1]
    auc = float(roc_auc_score(survey._y_eval, eval_probs))

    # Acceptance 2: AUC >= 0.90
    print(f"\nMeasured Hidden Evaluation ROC-AUC: {auc:.4f}")
    assert auc >= 0.90, f"Expected hidden-eval ROC-AUC >= 0.90, got {auc:.4f}"

    # Step 4: Rank candidates
    ranked = rank_candidates(model, survey)
    assert len(ranked) == 60
    assert list(ranked.columns) == ["star_id", "p_planet", "bls_period", "bls_depth"]
    assert ranked["p_planet"].is_monotonic_decreasing

    # Step 5: Test target choice on known False Positive
    fp_star_id = next(sid for sid, t in survey._candidate_truth.items() if not t["has_planet"])
    state_fp_file = tmp_path / "mission_state_fp.json"
    state_fp = MissionState(team_id=team_id)
    choose_target(state_fp, survey, model, fp_star_id)
    state_fp.save(state_fp_file)

    # Acceptance 4 (FP): penalty conditions applied
    fp_penalty = false_positive_target()
    assert state_fp.act1.done is True
    assert state_fp.act1.target.is_real_planet is False
    assert state_fp.act1.target.gravity == fp_penalty["gravity"]
    assert state_fp.act1.target.wind == fp_penalty["wind"]
    assert state_fp.act1.target.trip_days == fp_penalty["trip_days"]
    assert state_fp.act1.target.label == "dead world"
    expected_fp_score = float(round(120.0 * np.clip((auc - 0.5) / 0.5, 0.0, 1.0) + 0.0))
    assert state_fp.act1.score == pytest.approx(expected_fp_score)

    # Step 6: Test target choice on known True Planet
    tp_star_id = next(sid for sid, t in survey._candidate_truth.items() if t["has_planet"])
    state_tp = MissionState(team_id=team_id)
    choose_target(state_tp, survey, model, tp_star_id)
    t_choose = time.time()

    # Acceptance 4 (TP): coupling formulas applied
    tp_truth = survey._candidate_truth[tp_star_id]
    assert state_tp.act1.done is True
    assert state_tp.act1.target.is_real_planet is True
    assert state_tp.act1.target.gravity == pytest.approx(gravity_from_depth(tp_truth["depth"]))
    assert state_tp.act1.target.wind == pytest.approx(wind_from_period(tp_truth["period"]))
    assert state_tp.act1.target.trip_days == trip_days_from_period(tp_truth["period"])
    assert state_tp.act1.target.label == f"confirmed planet ({tp_star_id})"
    expected_tp_score = float(round(120.0 * np.clip((auc - 0.5) / 0.5, 0.0, 1.0) + 80.0))
    assert state_tp.act1.score == pytest.approx(expected_tp_score)

    # Acceptance 3: Total runtime <= 240 s
    total_elapsed = t_choose - t0
    print(f"Measured Total Runtime: {total_elapsed:.2f} s (Survey: {t_survey - t0:.2f}s, Train: {t_train - t_survey:.2f}s, Choose: {t_choose - t_train:.2f}s)")
    assert total_elapsed <= 240.0, f"Total FAST runtime {total_elapsed:.2f}s exceeded 240s budget"


def test_act1_samples_artifact_exists() -> None:
    """Acceptance 5: Verify artifacts/act1_samples.png exists and has valid size."""
    artifact_path = Path(__file__).resolve().parent.parent / "artifacts" / "act1_samples.png"
    assert artifact_path.exists(), f"act1_samples.png must exist at {artifact_path}"
    assert artifact_path.stat().st_size > 50_000, "Artifact act1_samples.png must be non-empty"
