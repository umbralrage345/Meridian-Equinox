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


_LAST_HUMAN_VS_AI_IDS: list[str] | None = None


def human_vs_ai(survey: Survey, n: int = 6) -> list[str]:
    """Display n candidate light curves for visual human inspection."""
    global _LAST_HUMAN_VS_AI_IDS
    # Prioritize shallow transits (depth <= 4 ppt) to challenge human eyes
    shallow_planets = [
        sid for sid in survey.candidate_star_ids
        if survey._candidate_truth.get(sid, {}).get("has_planet")
        and survey._candidate_truth.get(sid, {}).get("depth", 1.0) <= 0.004
    ]
    other_candidates = [
        sid for sid in survey.candidate_star_ids
        if sid not in shallow_planets
    ]

    n_shallow = min(2, len(shallow_planets))
    chosen_shallow = shallow_planets[:n_shallow]
    needed_other = n - n_shallow
    chosen_other = other_candidates[:needed_other]
    shown_ids = chosen_shallow + chosen_other
    _LAST_HUMAN_VS_AI_IDS = list(shown_ids)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, star_id in enumerate(shown_ids):
        star_idx = survey.candidate_star_ids.index(star_id)
        flux = survey.candidate_fluxes[star_idx]
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
    guesses: list[int | None] | np.ndarray,
    survey: Survey,
    model: ExoplanetModel | None = None,
    star_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compare human guesses and AI predictions against ground truth."""
    if star_ids is None:
        if _LAST_HUMAN_VS_AI_IDS is not None and len(_LAST_HUMAN_VS_AI_IDS) >= len(guesses):
            star_ids = _LAST_HUMAN_VS_AI_IDS[:len(guesses)]
        else:
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
    human_attempted = 0
    ai_correct = 0
    records = []

    for i, sid in enumerate(star_ids):
        truth = survey._candidate_truth[sid]
        actual = 1 if truth["has_planet"] else 0
        raw_g = guesses[i]
        if raw_g is not None:
            h_guess = int(raw_g)
            human_attempted += 1
            if h_guess == actual:
                human_correct += 1
            match_human = (h_guess == actual)
        else:
            h_guess = None
            match_human = None

        ai_pred = int(ai_preds[i])
        if ai_pred == actual:
            ai_correct += 1

        records.append({
            "star_id": sid,
            "human_guess": h_guess,
            "ai_prob": float(ai_probs[i]),
            "ai_pred": ai_pred,
            "actual": actual,
            "match_human": match_human,
            "match_ai": ai_pred == actual,
        })

    total = len(star_ids)
    h_acc = human_correct / human_attempted if human_attempted > 0 else 0.0
    ai_acc = ai_correct / total if total > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"{'STAR ID':<10} | {'HUMAN':<7} | {'AI PROB':<8} | {'AI PRED':<7} | {'ACTUAL':<7}")
    print("-" * 65)
    for r in records:
        h_disp = str(r["human_guess"]) if r["human_guess"] is not None else "SKIP"
        print(f"{r['star_id']:<10} | {h_disp:<7} | {r['ai_prob']:<8.3f} | {r['ai_pred']:<7} | {r['actual']:<7}")
    print("=" * 65)
    if human_attempted > 0:
        print(f"Human Accuracy   : {human_correct}/{human_attempted} ({h_acc * 100:.1f}%) [Attempted {human_attempted}/{total}]")
    else:
        print(f"Human Accuracy   : SKIPPED (0/{total} attempted).")
        print(">> Instruction   : Edit the 'guesses' list above (e.g. [1, 0, 1, 0]) to test your human visual detection!")
    print(f"AI Model Accuracy: {ai_correct}/{total} ({ai_acc * 100:.1f}%)")
    print("=" * 65 + "\n")

    return {
        "human_accuracy": h_acc,
        "human_attempted": human_attempted,
        "ai_accuracy": ai_acc,
        "records": records,
    }


def generate_samples_plot(save_path: str | Any | None = None) -> plt.Figure:
    """Generate sample plot with 3 planet curves spanning depth bands, 3 no-planet curves, and 1 folded transit curve."""
    from meridian.act1_exoplanet.generator import make_star
    from meridian.act1_exoplanet.pipeline import flatten
    rng = np.random.default_rng(12345)
    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 1.2])

    # Sample 3 planets spanning depth bands: shallow (~2 ppt), medium (~6 ppt), deep (~11 ppt)
    shallow_planet = None
    med_planet = None
    deep_planet = None
    while shallow_planet is None or med_planet is None or deep_planet is None:
        f, tr = make_star(rng, has_planet=True)
        d = tr["depth"]
        if shallow_planet is None and 0.0020 <= d <= 0.0035:
            shallow_planet = (f, tr)
        elif med_planet is None and 0.0055 <= d <= 0.0075:
            med_planet = (f, tr)
        elif deep_planet is None and 0.0100 <= d <= 0.0120:
            deep_planet = (f, tr)

    planet_stars = [shallow_planet, med_planet, deep_planet]
    band_labels = ["Shallow Transit (~2.5 ppt)", "Medium Transit (~6.5 ppt)", "Deep Transit (~11 ppt)"]
    noplanet_stars = [make_star(rng, has_planet=False) for _ in range(3)]

    for i in range(3):
        ax_p = fig.add_subplot(gs[i, 0])
        flux_p, truth_p = planet_stars[i]
        ax_p.scatter(TIME_GRID, flux_p, s=2, color="#2563eb", alpha=0.6)
        p_val = truth_p["period"]
        d_val = truth_p["depth"]
        ax_p.set_title(
            f"Transiting Exoplanet #{i+1} [{band_labels[i]}] (P = {p_val:.2f} d, Depth = {d_val * 1000:.1f} ppt)",
            fontsize=10,
            fontweight="semibold",
        )
        ax_p.set_ylabel("Normalized Flux", fontsize=9)
        ax_p.set_xlabel("Time (days)", fontsize=8)
        ax_p.grid(True, linestyle="--", alpha=0.4)

        ax_np = fig.add_subplot(gs[i, 1])
        flux_np, _ = noplanet_stars[i]
        ax_np.scatter(TIME_GRID, flux_np, s=2, color="#64748b", alpha=0.6)
        ax_np.set_title(f"No-Planet Stellar Light Curve #{i+1} (Variability + Noise)", fontsize=10)
        ax_np.set_ylabel("Normalized Flux", fontsize=9)
        ax_np.set_xlabel("Time (days)", fontsize=8)
        ax_np.grid(True, linestyle="--", alpha=0.4)

    # Folded light curve of the deep transit candidate, FLATTENED to remove stellar variability
    ax_fold = fig.add_subplot(gs[3, :])
    flux_deep, truth_deep = planet_stars[2]
    flat_deep = flatten(flux_deep)
    p_deep = truth_deep["period"]
    epoch_deep = truth_deep.get("epoch", 0.0)
    phase = ((TIME_GRID - epoch_deep + 0.5 * p_deep) % p_deep) / p_deep - 0.5
    sort_idx = np.argsort(phase)

    ax_fold.scatter(phase[sort_idx], flat_deep[sort_idx], s=3, color="#7c3aed", alpha=0.6)
    d_deep_ppt = truth_deep["depth"] * 1000.0
    ax_fold.set_title(
        f"Phase-Folded Flattened Light Curve for Deep Transit Candidate (P = {p_deep:.3f} d, Depth ~ {d_deep_ppt:.1f} ppt)",
        fontsize=11,
        fontweight="bold",
    )
    ax_fold.set_xlabel("Orbital Phase [-0.5, 0.5]", fontsize=10)
    ax_fold.set_ylabel("Detrended Flux (f / median - 1)", fontsize=10)
    ax_fold.axhline(0.0, color="#94a3b8", linestyle="--", alpha=0.5)
    ax_fold.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("Act 1 Exoplanet Transit Survey: Planet vs Non-Planet Samples", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout()
    if save_path:
        import pathlib
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=120, bbox_inches="tight")
    return fig


show_samples = generate_samples_plot
