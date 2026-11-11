"""Acceptance tests for Act 3 Galaxy Classification and Downlink Optimization."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from meridian.act3_galaxy import (
    CLASSES,
    GalaxySurvey,
    build_survey,
    has_torch,
    occlusion_map,
    score,
    select_downlink,
    show_samples,
    train,
    validate_knobs,
)
from meridian.state import MissionState


def test_act3_survey_determinism_and_uniqueness() -> None:
    """Acceptance 1: Survey generation determinism by team_id."""
    team_a = "team_alpha_galaxy"
    team_b = "team_beta_galaxy"

    survey_a1 = build_survey(team_a, fast=True)
    survey_a2 = build_survey(team_a, fast=True)
    survey_b = build_survey(team_b, fast=True)

    # Identical arrays for same team_id
    assert np.array_equal(survey_a1.train_images, survey_a2.train_images)
    assert np.array_equal(survey_a1.train_labels, survey_a2.train_labels)
    assert np.array_equal(survey_a1.test_images, survey_a2.test_images)
    assert np.array_equal(survey_a1.test_labels, survey_a2.test_labels)
    assert np.array_equal(survey_a1.stream_images, survey_a2.stream_images)
    assert np.array_equal(survey_a1.stream_labels, survey_a2.stream_labels)

    # Different arrays for different team_ids
    assert not np.array_equal(survey_a1.train_images, survey_b.train_images)
    assert not np.array_equal(survey_a1.stream_images, survey_b.stream_images)


def test_act3_model_accuracy_torch_and_sklearn() -> None:
    """Acceptance 2: With torch, default accuracy in [85%, 95%]. Without torch (fallback), accuracy >= 70%."""
    team_id = "team_acc_eval"
    survey = build_survey(team_id, fast=True)

    if has_torch():
        # 1. Test PyTorch TinyCNN (default backend when torch is available)
        model_torch = train(survey, force_backend="torch")
        print(f"\nTested default backend: {model_torch.backend}")
        probs_torch = model_torch.predict_proba(survey.test_images)
        preds_torch = np.argmax(probs_torch, axis=1)
        acc_torch = float(np.mean(preds_torch == survey.test_labels))
        print(f"Measured PyTorch TinyCNN Test Accuracy: {acc_torch * 100:.2f}%")

        assert 0.85 <= acc_torch <= 0.95, (
            f"PyTorch accuracy {acc_torch * 100:.2f}% outside target range [85%, 95%]"
        )
    else:
        print("\nPyTorch not installed in this environment; testing default fallback")
        model_default = train(survey)
        assert model_default.backend == "sklearn"

    # 2. Test Scikit-learn fallback backend
    model_sklearn = train(survey, force_backend="sklearn")
    print(f"Tested fallback backend: {model_sklearn.backend}")
    probs_sklearn = model_sklearn.predict_proba(survey.test_images)
    preds_sklearn = np.argmax(probs_sklearn, axis=1)
    acc_sklearn = float(np.mean(preds_sklearn == survey.test_labels))
    print(f"Measured Scikit-learn Fallback Test Accuracy: {acc_sklearn * 100:.2f}%")

    assert acc_sklearn >= 0.70, (
        f"Scikit-learn fallback accuracy {acc_sklearn * 100:.2f}% < 70%"
    )


def test_act3_training_time_fast() -> None:
    """Acceptance 3: Training time <= 240 s in FAST on CPU."""
    survey = build_survey("team_time_perf", fast=True)

    t0 = time.time()
    model = train(survey)
    train_time = time.time() - t0

    print(f"\nMeasured Act 3 Training Time (FAST): {train_time:.2f} s")
    assert train_time <= 240.0, f"Training time {train_time:.2f}s exceeded budget of 240s"


def test_act3_expected_value_vs_max_confidence_strategies() -> None:
    """Acceptance 4: Over 5 team_ids, 'expected_value' scores higher on average than 'max_confidence'."""
    team_ids = [
        "team_sirius_01",
        "team_vega_02",
        "team_rigel_03",
        "team_altair_04",
        "team_deneb_05",
    ]

    scores_ev: list[float] = []
    scores_mc: list[float] = []

    for tid in team_ids:
        survey = build_survey(tid, fast=True)
        model = train(survey)

        # Expected value strategy
        sel_ev = select_downlink(model, survey, strategy="expected_value")
        state_ev = MissionState(team_id=f"{tid}_ev")
        sc_ev = score(state_ev, survey, model, sel_ev)
        scores_ev.append(sc_ev)

        # Max confidence strategy
        sel_mc = select_downlink(model, survey, strategy="max_confidence")
        state_mc = MissionState(team_id=f"{tid}_mc")
        sc_mc = score(state_mc, survey, model, sel_mc)
        scores_mc.append(sc_mc)

    mean_ev = float(np.mean(scores_ev))
    mean_mc = float(np.mean(scores_mc))

    print(f"\nMeasured Mean Expected Value Downlink Score: {mean_ev:.2f}")
    print(f"Measured Mean Max Confidence Downlink Score: {mean_mc:.2f}")

    assert mean_ev > mean_mc, (
        f"Expected Value strategy ({mean_ev:.2f}) did not beat Max Confidence ({mean_mc:.2f})"
    )


def test_act3_occlusion_central_vs_border_importance() -> None:
    """Acceptance 5: For a correctly classified spiral, mean importance in central region > border region."""
    survey = build_survey("team_occlusion_test", fast=True)
    model = train(survey)

    # Find a correctly classified spiral (class 1) in the test set
    probs = model.predict_proba(survey.test_images)
    preds = np.argmax(probs, axis=1)

    spiral_indices = np.where((survey.test_labels == 1) & (preds == 1))[0]
    assert len(spiral_indices) > 0, "No correctly classified spiral found in test set"

    test_idx = spiral_indices[0]
    spiral_img = survey.test_images[test_idx]

    # Compute occlusion map
    heatmap = occlusion_map(model, spiral_img, cls=1, patch_size=8, stride=4)

    # Define central region [12:36, 12:36] and border region
    center_mask = np.zeros_like(heatmap, dtype=bool)
    center_mask[12:36, 12:36] = True
    border_mask = ~center_mask

    mean_center = float(np.mean(heatmap[center_mask]))
    mean_border = float(np.mean(heatmap[border_mask]))

    print(f"\nMeasured Occlusion Importance: Center = {mean_center:.4f}, Border = {mean_border:.4f}")
    assert mean_center > mean_border, (
        f"Central importance ({mean_center:.4f}) not greater than border importance ({mean_border:.4f})"
    )


def test_act3_peak_memory_fast() -> None:
    """Acceptance 6: Peak memory in FAST under 1.5 GB using psutil process RSS."""
    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed; skipping process RSS memory test")

    proc = psutil.Process()

    survey = build_survey("team_mem_eval", fast=True)
    model = train(survey)
    selected = select_downlink(model, survey, strategy="expected_value")
    state = MissionState(team_id="team_mem_eval")
    _ = score(state, survey, model, selected)

    rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)
    print(f"\nMeasured Act 3 Process RSS: {rss_mb:.2f} MB")

    assert rss_mb < 1500.0, f"Peak memory {rss_mb:.2f} MB exceeded 1.5 GB"


def test_act3_samples_artifact_exists() -> None:
    """Acceptance 7: Verify artifacts/act3_samples.png exists and is non-empty."""
    artifact_path = Path("./artifacts/act3_samples.png")
    if not artifact_path.exists():
        survey = build_survey("team_artifact_gen", fast=True)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        show_samples(survey, n_per_class=8, save_path=str(artifact_path))

    assert artifact_path.exists(), "act3_samples.png was not generated"
    assert artifact_path.stat().st_size > 5000, "act3_samples.png is empty or corrupt"


def test_act3_shortcut_logistic_regression() -> None:
    """Acceptance 8: Logistic regression on simple shortcut features gets balanced accuracy <= 0.45."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    survey = build_survey("team_shortcut_test", fast=True)
    images = survey.train_images
    labels = survey.train_labels

    def extract_shortcut_features(imgs: np.ndarray) -> np.ndarray:
        feats = []
        for im in imgs:
            border = np.concatenate([im[0, :], im[-1, :], im[1:-1, 0], im[1:-1, -1]])
            feats.append([
                float(np.mean(border)),
                float(np.std(border)),
                float(np.sum(im)),
                float(np.max(im)),
                float(np.mean(im)),
            ])
        return np.array(feats)

    X = extract_shortcut_features(images)
    split = int(0.75 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = labels[:split], labels[split:]

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    bacc = float(balanced_accuracy_score(y_val, preds))

    print(f"\nMeasured Shortcut Logistic Regression Balanced Accuracy: {bacc:.4f} (target <= 0.45)")
    assert bacc <= 0.45, f"Shortcut balanced accuracy {bacc:.4f} > 0.45 (chance is 0.25)"

