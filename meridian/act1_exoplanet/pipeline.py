"""Exoplanet detection feature pipeline, model training, candidate ranking, and target selection."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy.signal
from astropy.timeseries import BoxLeastSquares
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from meridian.act1_exoplanet.generator import N_POINTS, TIME_GRID, make_star
from meridian.config import get_max_jobs, is_fast
from meridian.coupling import (
    false_positive_target,
    gravity_from_depth,
    trip_days_from_period,
    wind_from_period,
)
from meridian.seeding import seed_for

if TYPE_CHECKING:
    from meridian.state import MissionState

BLS_PERIODS: np.ndarray = np.exp(np.linspace(np.log(0.6), np.log(13.0), 1500))
BLS_DURATIONS: list[float] = [0.04, 0.08, 0.15, 0.25]
N_PHASE_BINS: int = 40
PHASE_BIN_EDGES: np.ndarray = np.linspace(-0.5, 0.5, N_PHASE_BINS + 1)

TRAIN_KNOBS: dict[str, tuple[Any, ...]] = {
    "n_estimators": (50, 500, 300),
    "max_depth": (3, 30, None),
    "feature_set": ("full", "folded_only", "bls_only"),
    "threshold": (0.2, 0.8, 0.5),
}


def flatten(flux: np.ndarray) -> np.ndarray:
    """Flatten light curve using median filtering and clip bounds to [-0.05, 0.02]."""
    med = scipy.signal.medfilt(flux, 97)
    med_safe = np.where(med <= 0.0, 1.0, med)
    flat = flux / med_safe - 1.0
    return np.clip(flat, -0.05, 0.02)


def extract_features(flux: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Extract 46 features from a stellar light curve using BLS and phase folding."""
    f_flat = flatten(flux)
    std_y = float(np.std(f_flat))
    std_div = std_y if std_y > 1e-12 else 1e-12

    bls = BoxLeastSquares(TIME_GRID, f_flat)
    periodogram = bls.power(BLS_PERIODS, BLS_DURATIONS, objective="snr")

    best_idx = int(np.nanargmax(periodogram.power))
    best_period = float(periodogram.period[best_idx])
    best_duration = float(periodogram.duration[best_idx])
    best_transit_time = float(periodogram.transit_time[best_idx])
    best_power = float(periodogram.power[best_idx])
    best_depth = float(periodogram.depth[best_idx])
    best_depth_snr = float(periodogram.depth_snr[best_idx])

    # Phase fold to [-0.5, 0.5]
    phase = ((TIME_GRID - best_transit_time + 0.5 * best_period) % best_period) / best_period - 0.5
    bin_idx = np.clip(np.floor((phase + 0.5) * N_PHASE_BINS).astype(int), 0, N_PHASE_BINS - 1)

    folded_features = np.zeros(N_PHASE_BINS, dtype=np.float64)
    for b in range(N_PHASE_BINS):
        in_bin = f_flat[bin_idx == b]
        if in_bin.size > 0:
            folded_features[b] = np.median(in_bin) / std_div
        else:
            folded_features[b] = 0.0

    scalar_features = np.array([
        best_depth_snr,
        best_power,
        best_period,
        best_duration,
        best_depth / std_div,
        std_y,
    ], dtype=np.float64)

    all_features = np.concatenate([folded_features, scalar_features])
    bls_info = {
        "period": best_period,
        "duration": best_duration,
        "depth": best_depth,
        "power": best_power,
        "depth_snr": best_depth_snr,
        "transit_time": best_transit_time,
    }
    return all_features, bls_info


def _process_star_worker(
    star_seed: int,
    has_planet: bool,
    star_id: str,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, dict[str, float]]:
    """Worker function for parallel star generation and feature caching."""
    rng = np.random.default_rng(star_seed)
    flux, truth = make_star(rng, has_planet)
    features, bls_info = extract_features(flux)
    return flux, truth, features, bls_info


@dataclass
class Survey:
    """Survey containing training stars, candidate stars, and hidden evaluation set."""

    team_id: str
    is_fast: bool
    seed: int
    train_star_ids: list[str]
    train_fluxes: list[np.ndarray]
    X_train: np.ndarray
    y_train: np.ndarray
    candidate_star_ids: list[str]
    candidate_fluxes: list[np.ndarray]
    X_candidates: np.ndarray
    candidate_bls_info: list[dict[str, float]]
    _candidate_truth: dict[str, dict[str, Any]]
    _X_eval: np.ndarray
    _y_eval: np.ndarray

    def get_features(self, X: np.ndarray, feature_set: str) -> np.ndarray:
        """Slice feature matrix according to selected feature_set."""
        if feature_set == "folded_only":
            return X[:, :40]
        elif feature_set == "bls_only":
            return X[:, 40:]
        return X


def build_survey(team_id: str | int, fast: bool | None = None) -> Survey:
    """Construct deterministic survey datasets with parallel feature extraction and caching."""
    if fast is None:
        fast = is_fast()

    team_str = str(team_id)
    cache_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"survey_{team_str}_fast{int(fast)}.npz"

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as data:
                meta = json.loads(str(data["meta"]))
                train_fluxes_arr = data["train_fluxes"]
                candidate_fluxes_arr = data["candidate_fluxes"]
                return Survey(
                    team_id=team_str,
                    is_fast=bool(meta["is_fast"]),
                    seed=int(meta["seed"]),
                    train_star_ids=meta["train_star_ids"],
                    train_fluxes=[train_fluxes_arr[i] for i in range(len(train_fluxes_arr))],
                    X_train=data["X_train"],
                    y_train=data["y_train"],
                    candidate_star_ids=meta["candidate_star_ids"],
                    candidate_fluxes=[candidate_fluxes_arr[i] for i in range(len(candidate_fluxes_arr))],
                    X_candidates=data["X_candidates"],
                    candidate_bls_info=meta["candidate_bls_info"],
                    _candidate_truth=meta["candidate_truth"],
                    _X_eval=data["X_eval"],
                    _y_eval=data["y_eval"],
                )
        except Exception:
            pass  # Fall back to re-generating if cache corrupted

    seed = seed_for(team_id, "act1")
    rng = np.random.default_rng(seed)

    n_train = 600 if fast else 1500
    n_candidates = 60
    n_eval = 300

    # Training set labels (~50% balanced)
    train_planets = [bool(i % 2 == 0) for i in range(n_train)]
    rng.shuffle(train_planets)

    # Candidate set labels: exactly 40% (24 stars) have planets
    n_cand_planets = int(round(0.4 * n_candidates))
    cand_planets = [True] * n_cand_planets + [False] * (n_candidates - n_cand_planets)
    rng.shuffle(cand_planets)

    # Hidden evaluation set (50% balanced)
    eval_planets = [bool(i % 2 == 0) for i in range(n_eval)]
    rng.shuffle(eval_planets)

    # Generate seeds for each individual star to allow parallel deterministic execution
    train_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_train)]
    cand_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_candidates)]
    eval_seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(n_eval)]

    # Run parallel generation and feature caching
    all_tasks = (
        [(train_seeds[i], train_planets[i], f"TRAIN_{i:04d}") for i in range(n_train)]
        + [(cand_seeds[i], cand_planets[i], f"CAND_{i:02d}") for i in range(n_candidates)]
        + [(eval_seeds[i], eval_planets[i], f"EVAL_{i:04d}") for i in range(n_eval)]
    )

    results = Parallel(n_jobs=min(2, get_max_jobs()), batch_size=16, prefer="threads")(
        delayed(_process_star_worker)(s, hp, sid) for s, hp, sid in all_tasks
    )

    train_res = results[:n_train]
    cand_res = results[n_train : n_train + n_candidates]
    eval_res = results[n_train + n_candidates :]

    train_fluxes = [r[0] for r in train_res]
    X_train = np.array([r[2] for r in train_res], dtype=np.float64)
    y_train = np.array([1 if p else 0 for p in train_planets], dtype=np.int32)

    candidate_star_ids = [f"CAND_{i:02d}" for i in range(n_candidates)]
    candidate_fluxes = [r[0] for r in cand_res]
    X_candidates = np.array([r[2] for r in cand_res], dtype=np.float64)
    candidate_bls_info = [r[3] for r in cand_res]
    candidate_truth = {candidate_star_ids[i]: cand_res[i][1] for i in range(n_candidates)}

    X_eval = np.array([r[2] for r in eval_res], dtype=np.float64)
    y_eval = np.array([1 if p else 0 for p in eval_planets], dtype=np.int32)

    # Save to disk cache
    meta_json = json.dumps({
        "team_id": team_str,
        "is_fast": fast,
        "seed": seed,
        "train_star_ids": [f"TRAIN_{i:04d}" for i in range(n_train)],
        "candidate_star_ids": candidate_star_ids,
        "candidate_bls_info": candidate_bls_info,
        "candidate_truth": candidate_truth,
    })

    np.savez_compressed(
        cache_path,
        meta=np.array(meta_json),
        train_fluxes=np.array(train_fluxes, dtype=np.float64),
        X_train=X_train,
        y_train=y_train,
        candidate_fluxes=np.array(candidate_fluxes, dtype=np.float64),
        X_candidates=X_candidates,
        X_eval=X_eval,
        y_eval=y_eval,
    )

    return Survey(
        team_id=team_str,
        is_fast=fast,
        seed=seed,
        train_star_ids=[f"TRAIN_{i:04d}" for i in range(n_train)],
        train_fluxes=train_fluxes,
        X_train=X_train,
        y_train=y_train,
        candidate_star_ids=candidate_star_ids,
        candidate_fluxes=candidate_fluxes,
        X_candidates=X_candidates,
        candidate_bls_info=candidate_bls_info,
        _candidate_truth=candidate_truth,
        _X_eval=X_eval,
        _y_eval=y_eval,
    )


def validate_knobs(**knobs: Any) -> dict[str, Any]:
    """Validate and clip training hyperparameters, warning on adjustments without raising."""
    cleaned: dict[str, Any] = {}

    # n_estimators (50, 500, 300)
    raw_n_est = knobs.get("n_estimators", 300)
    try:
        n_est = int(raw_n_est)
    except (ValueError, TypeError):
        n_est = 300
    if n_est < 50:
        warnings.warn(f"n_estimators={raw_n_est} is below 50, clipping to 50", UserWarning)
        n_est = 50
    elif n_est > 500:
        warnings.warn(f"n_estimators={raw_n_est} is above 500, clipping to 500", UserWarning)
        n_est = 500
    cleaned["n_estimators"] = n_est

    # max_depth (3, 30, None)
    raw_depth = knobs.get("max_depth", None)
    if raw_depth is not None:
        try:
            depth = int(raw_depth)
            if depth < 3:
                warnings.warn(f"max_depth={raw_depth} is below 3, clipping to 3", UserWarning)
                depth = 3
            elif depth > 30:
                warnings.warn(f"max_depth={raw_depth} is above 30, clipping to 30", UserWarning)
                depth = 30
            cleaned["max_depth"] = depth
        except (ValueError, TypeError):
            cleaned["max_depth"] = None
    else:
        cleaned["max_depth"] = None

    # feature_set in {"full", "folded_only", "bls_only"}
    raw_fset = knobs.get("feature_set", "full")
    if raw_fset not in {"full", "folded_only", "bls_only"}:
        warnings.warn(f"feature_set='{raw_fset}' invalid, defaulting to 'full'", UserWarning)
        raw_fset = "full"
    cleaned["feature_set"] = raw_fset

    # threshold (0.2, 0.8, 0.5)
    raw_thresh = knobs.get("threshold", 0.5)
    try:
        thresh = float(raw_thresh)
    except (ValueError, TypeError):
        thresh = 0.5
    if thresh < 0.2:
        warnings.warn(f"threshold={raw_thresh} is below 0.2, clipping to 0.2", UserWarning)
        thresh = 0.2
    elif thresh > 0.8:
        warnings.warn(f"threshold={raw_thresh} is above 0.8, clipping to 0.8", UserWarning)
        thresh = 0.8
    cleaned["threshold"] = thresh

    return cleaned


class ExoplanetModel:
    """Trained classifier wrapper with feature slicing and probability prediction."""

    def __init__(self, clf: RandomForestClassifier, feature_set: str, threshold: float, knobs: dict[str, Any]):
        self.clf = clf
        self.feature_set = feature_set
        self.threshold = threshold
        self.knobs = knobs

    def _slice_features(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] == 46:
            if self.feature_set == "folded_only":
                return X[:, :40]
            elif self.feature_set == "bls_only":
                return X[:, 40:]
        return X

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_sub = self._slice_features(X)
        return self.clf.predict_proba(X_sub)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary classification using configured threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(np.int32)


def train(survey: Survey, **knobs: Any) -> ExoplanetModel:
    """Train baseline RandomForestClassifier with validated knobs."""
    valid_knobs = validate_knobs(**knobs)
    fset = valid_knobs["feature_set"]
    X_tr = survey.get_features(survey.X_train, fset)

    clf = RandomForestClassifier(
        n_estimators=valid_knobs["n_estimators"],
        max_depth=valid_knobs["max_depth"],
        random_state=survey.seed,
        n_jobs=get_max_jobs(),
    )
    clf.fit(X_tr, survey.y_train)

    return ExoplanetModel(
        clf=clf,
        feature_set=fset,
        threshold=valid_knobs["threshold"],
        knobs=valid_knobs,
    )


def rank_candidates(model: ExoplanetModel, survey: Survey) -> pd.DataFrame:
    """Rank candidate stars descending by predicted probability of hosting a planet."""
    probs = model.predict_proba(survey.X_candidates)[:, 1]
    df = pd.DataFrame({
        "star_id": survey.candidate_star_ids,
        "p_planet": probs,
        "bls_period": [info["period"] for info in survey.candidate_bls_info],
        "bls_depth": [info["depth"] for info in survey.candidate_bls_info],
    })
    return df.sort_values(by=["p_planet", "star_id"], ascending=[False, True]).reset_index(drop=True)


def choose_target(
    state: MissionState,
    survey: Survey,
    model: ExoplanetModel,
    star_id: str,
) -> MissionState:
    """Select destination star, evaluate hidden AUC, apply cross-act coupling, and save state."""
    if star_id not in survey._candidate_truth:
        raise ValueError(f"Target star_id '{star_id}' not found in candidate pool.")

    eval_probs = model.predict_proba(survey._X_eval)[:, 1]
    auc = float(roc_auc_score(survey._y_eval, eval_probs))

    truth = survey._candidate_truth[star_id]
    is_real = bool(truth["has_planet"])

    if not is_real:
        penalty = false_positive_target()
        gravity = float(penalty["gravity"])
        wind = float(penalty["wind"])
        trip_days = int(penalty["trip_days"])
        label = str(penalty["label"])
        target_real = False
        depth = float(truth.get("depth", 0.0))
        period = float(truth.get("period", 0.0))
    else:
        depth = float(truth["depth"])
        period = float(truth["period"])
        gravity = float(gravity_from_depth(depth))
        wind = float(wind_from_period(period))
        trip_days = int(trip_days_from_period(period))
        label = f"confirmed planet ({star_id})"
        target_real = True

    auc_norm = float(np.clip((auc - 0.5) / 0.5, 0.0, 1.0))
    planet_bonus = 80.0 if is_real else 0.0
    score = float(round(120.0 * auc_norm + planet_bonus))

    state.act1.done = True
    state.act1.score = score
    state.act1.auc = auc
    state.act1.target.star_id = str(star_id)
    state.act1.target.is_real_planet = target_real
    state.act1.target.depth = depth
    state.act1.target.period_days = period
    state.act1.target.gravity = gravity
    state.act1.target.wind = wind
    state.act1.target.trip_days = trip_days
    state.act1.target.label = label

    state.save()
    return state
