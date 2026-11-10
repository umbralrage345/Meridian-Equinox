"""Interactive plotting and Human-vs-AI benchmarking utilities for Act 1."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from meridian.act1_exoplanet.generator import TIME_GRID
from meridian.act1_exoplanet.pipeline import ExoplanetModel, Survey, train


def plot_lightcurve(
    t: np.ndarray,
    flux: np.ndarray,
    title: str = "Stellar Light Curve",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot normalized stellar flux against time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    ax.scatter(t, flux, s=3, color="#2563eb", alpha=0.7, edgecolors="none")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def plot_bls(
    periods: np.ndarray,
    powers: np.ndarray,
    best_period: float | None = None,
    title: str = "Box Least Squares Periodogram",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot BLS power spectrum over trial periods."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(periods, powers, color="#059669", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("BLS SNR Power")
    ax.set_title(title)
    if best_period is not None:
        ax.axvline(best_period, color="#dc2626", linestyle="--", label=f"Best P = {best_period:.3f} d")
        ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def plot_fold(
    t: np.ndarray,
    flux: np.ndarray,
    period: float,
    epoch: float = 0.0,
    title: str = "Phase-Folded Light Curve",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot light curve folded around a candidate orbital period and epoch."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    phase = ((t - epoch + 0.5 * period) % period) / period - 0.5
    sort_idx = np.argsort(phase)
    ax.scatter(phase[sort_idx], flux[sort_idx], s=3, color="#7c3aed", alpha=0.6, edgecolors="none")
    ax.set_xlabel("Orbital Phase")
    ax.set_ylabel("Normalized Flux")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def plot_candidates(
    ranked_df: pd.DataFrame,
    top_n: int = 10,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot candidate exoplanet stars ranked by model probability."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))
    top = ranked_df.head(top_n).iloc[::-1]
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["p_planet"], color="#0284c7", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["star_id"])
    ax.set_xlabel("Predicted Planet Probability")
    ax.set_xlim(0.0, 1.0)
    ax.axvline(0.5, color="#ef4444", linestyle="--", alpha=0.7, label="Threshold (0.5)")
    ax.set_title(f"Top {top_n} Exoplanet Candidates")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    return ax


def human_vs_ai(survey: Survey, n: int = 6) -> list[str]:
    """Display n candidate light curves for visual human inspection."""
    shown_ids = survey.candidate_star_ids[:n]
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, star_id in enumerate(shown_ids):
        flux = survey.candidate_fluxes[i]
        axes[i].scatter(TIME_GRID, flux, s=2, color="#334155", alpha=0.7)
        axes[i].set_ylabel("Flux")
        axes[i].set_title(f"Candidate #{i+1}: {star_id}", fontsize=10, loc="left")
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Time (days)")
    plt.tight_layout()
    plt.show()

    print("\n--- HUMAN VS AI: EXOPLANET CHALLENGE ---")
    print(f"Inspect the {n} light curves displayed above.")
    print("Record your binary guesses (1 = has planet transit, 0 = stellar noise/variability).")
    print("Then call: reveal(guesses=[...], survey=survey)")
    print(f"Target IDs: {shown_ids}\n")
    return shown_ids


def reveal(
    guesses: list[int] | np.ndarray,
    survey: Survey,
    model: ExoplanetModel | None = None,
    star_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compare human guesses and AI predictions against ground truth."""
    if star_ids is None:
        star_ids = survey.candidate_star_ids[:len(guesses)]

    if len(guesses) != len(star_ids):
        raise ValueError(f"Length of guesses ({len(guesses)}) does not match star_ids ({len(star_ids)}).")

    if model is None:
        model = train(survey)

    # Candidate indices in survey
    indices = [survey.candidate_star_ids.index(sid) for sid in star_ids]
    X_sub = survey.X_candidates[indices]
    ai_probs = model.predict_proba(X_sub)[:, 1]
    ai_preds = (ai_probs >= model.threshold).astype(int)

    human_correct = 0
    ai_correct = 0
    records = []

    for i, sid in enumerate(star_ids):
        truth = survey._candidate_truth[sid]
        actual = 1 if truth["has_planet"] else 0
        h_guess = int(guesses[i])
        ai_pred = int(ai_preds[i])

        if h_guess == actual:
            human_correct += 1
        if ai_pred == actual:
            ai_correct += 1

        records.append({
            "star_id": sid,
            "human_guess": h_guess,
            "ai_prob": float(ai_probs[i]),
            "ai_pred": ai_pred,
            "actual": actual,
            "match_human": h_guess == actual,
            "match_ai": ai_pred == actual,
        })

    total = len(star_ids)
    h_acc = human_correct / total if total > 0 else 0.0
    ai_acc = ai_correct / total if total > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"{'STAR ID':<10} | {'HUMAN':<7} | {'AI PROB':<8} | {'AI PRED':<7} | {'ACTUAL':<7}")
    print("-" * 65)
    for r in records:
        print(f"{r['star_id']:<10} | {r['human_guess']:<7} | {r['ai_prob']:<8.3f} | {r['ai_pred']:<7} | {r['actual']:<7}")
    print("=" * 65)
    print(f"Human Accuracy : {human_correct}/{total} ({h_acc * 100:.1f}%)")
    print(f"AI Model Accuracy: {ai_correct}/{total} ({ai_acc * 100:.1f}%)")
    print("=" * 65 + "\n")

    return {
        "human_accuracy": h_acc,
        "ai_accuracy": ai_acc,
        "records": records,
    }
