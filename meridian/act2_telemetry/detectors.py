"""Anomaly detection models, scoring routines, and mission state transitions for Act 2."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from meridian.act2_telemetry.generator import CHANNELS, TelemetryStream, make_stream
from meridian.config import is_fast
from meridian.coupling import act2_resources
from meridian.seeding import seed_for

if TYPE_CHECKING:
    from meridian.state import MissionState

STRIDE: int = 8

TRAIN_KNOBS: dict[str, Any] = {
    "detector": ("iforest", "autoencoder", "both"),
    "window": (32, 64, 128),
    "percentile": (95.0, 99.9, 99.0),
    "persistence": (1, 5, 1),
}


def validate_knobs(**knobs: Any) -> dict[str, Any]:
    """Validate and clip hyperparameter knobs with warning emissions, never raising."""
    cleaned: dict[str, Any] = {}

    det = str(knobs.get("detector", "both")).lower()
    if det not in ("iforest", "autoencoder", "both"):
        warnings.warn(f"detector='{det}' invalid, defaulting to 'both'", UserWarning)
        det = "both"
    cleaned["detector"] = det

    win = knobs.get("window", 64)
    try:
        win_int = int(win)
        if win_int not in (32, 64, 128):
            closest = min((32, 64, 128), key=lambda x: abs(x - win_int))
            warnings.warn(f"window={win} not in {{32,64,128}}, snapping to {closest}", UserWarning)
            win_int = closest
        cleaned["window"] = win_int
    except (ValueError, TypeError):
        cleaned["window"] = 64

    pct = knobs.get("percentile", 99.0)
    try:
        pct_float = float(pct)
        if pct_float < 95.0:
            warnings.warn(f"percentile={pct} below 95.0, clipping to 95.0", UserWarning)
            pct_float = 95.0
        elif pct_float > 99.9:
            warnings.warn(f"percentile={pct} above 99.9, clipping to 99.9", UserWarning)
            pct_float = 99.9
        cleaned["percentile"] = pct_float
    except (ValueError, TypeError):
        cleaned["percentile"] = 99.0

    pers = knobs.get("persistence", 1)
    try:
        pers_int = int(pers)
        if pers_int < 1:
            warnings.warn(f"persistence={pers} below 1, clipping to 1", UserWarning)
            pers_int = 1
        elif pers_int > 5:
            warnings.warn(f"persistence={pers} above 5, clipping to 5", UserWarning)
            pers_int = 5
        cleaned["persistence"] = pers_int
    except (ValueError, TypeError):
        cleaned["persistence"] = 1

    return cleaned


def extract_iforest_features(series: np.ndarray, window: int, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Extract 7 features per window for IsolationForest."""
    starts = np.arange(0, len(series) - window + 1, stride)
    x_axis = np.arange(window, dtype=np.float64)
    feats = np.empty((len(starts), 7), dtype=np.float64)

    for i, ws in enumerate(starts):
        win = series[ws : ws + window]
        m = float(np.mean(win))
        s = float(np.std(win))
        mn = float(np.min(win))
        mx = float(np.max(win))
        diffs = np.diff(win)
        slope = float(np.polyfit(x_axis, win, 1)[0])
        std_diff = float(np.std(diffs))
        max_diff = float(np.max(np.abs(diffs)))
        feats[i] = [m, s, mn, mx, slope, std_diff, max_diff]

    return feats, starts


def extract_autoencoder_features(series: np.ndarray, window: int, stride: int = STRIDE) -> tuple[np.ndarray, np.ndarray]:
    """Extract subsampled window sequence (every 2nd point) for Autoencoder."""
    starts = np.arange(0, len(series) - window + 1, stride)
    half_w = window // 2
    feats = np.empty((len(starts), half_w), dtype=np.float64)

    for i, ws in enumerate(starts):
        feats[i] = series[ws : ws + window : 2]

    return feats, starts


@dataclass
class ChannelModels:
    """Trained models, scalers, and thresholds for a single telemetry channel."""

    channel: str
    iforest_scaler: StandardScaler | None = None
    iforest_model: IsolationForest | None = None
    iforest_threshold: float = 0.0
    ae_scaler: StandardScaler | None = None
    ae_model: MLPRegressor | None = None
    ae_threshold: float = 0.0


@dataclass
class TelemetryDetector:
    """Trained telemetry detector bundle across all channels."""

    knobs: dict[str, Any]
    models: dict[str, ChannelModels]
    train_scores: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


@dataclass
class TelemetryResult:
    """Telemetry anomaly evaluation metrics and alarm detection intervals."""

    score: float
    n_faults: int
    caught: int
    missed: int
    false_alarms: int
    mean_delay: float
    alarm_runs: list[tuple[int, int]]
    fault_details: list[dict[str, Any]]
    window_starts: np.ndarray
    channel_scores: dict[str, dict[str, np.ndarray]]


def build_stream(state: MissionState, fast: bool | None = None) -> TelemetryStream:
    """Construct deterministic telemetry stream from current mission state."""
    if fast is None:
        fast = is_fast()

    trip_days = int(getattr(state.act1.target, "trip_days", 100))
    return make_stream(state.team_id, trip_days=trip_days, fast=fast)


def train(stream: TelemetryStream, **knobs: Any) -> TelemetryDetector:
    """Train per-channel IsolationForest and/or Autoencoder models on training stream."""
    valid_knobs = validate_knobs(**knobs)
    det_mode = valid_knobs["detector"]
    window = valid_knobs["window"]
    percentile = valid_knobs["percentile"]

    seed = seed_for(stream.team_id, "act2_train")
    channel_models: dict[str, ChannelModels] = {}
    train_scores: dict[str, dict[str, np.ndarray]] = {}

    for ch_idx, ch in enumerate(CHANNELS):
        ch_seed = seed + ch_idx * 17
        data = stream.train_data[ch]
        c_mod = ChannelModels(channel=ch)
        train_scores[ch] = {}

        if det_mode in ("iforest", "both"):
            X_tr, _ = extract_iforest_features(data, window)
            s_if = StandardScaler()
            X_tr_s = s_if.fit_transform(X_tr)
            clf_if = IsolationForest(n_estimators=300, random_state=ch_seed, n_jobs=-1)
            clf_if.fit(X_tr_s)
            tr_sc_if = -clf_if.score_samples(X_tr_s)

            c_mod.iforest_scaler = s_if
            c_mod.iforest_model = clf_if
            c_mod.iforest_threshold = float(np.percentile(tr_sc_if, percentile))
            train_scores[ch]["iforest"] = tr_sc_if

        if det_mode in ("autoencoder", "both"):
            X_tr, _ = extract_autoencoder_features(data, window)
            s_ae = StandardScaler()
            X_tr_s = s_ae.fit_transform(X_tr)
            mlp = MLPRegressor(
                hidden_layer_sizes=(16, 6, 16),
                activation="tanh",
                max_iter=300,
                random_state=ch_seed,
            )
            mlp.fit(X_tr_s, X_tr_s)
            recons_tr = mlp.predict(X_tr_s)
            tr_sc_ae = np.mean((X_tr_s - recons_tr) ** 2, axis=1)

            c_mod.ae_scaler = s_ae
            c_mod.ae_model = mlp
            c_mod.ae_threshold = float(np.percentile(tr_sc_ae, percentile))
            train_scores[ch]["autoencoder"] = tr_sc_ae

        channel_models[ch] = c_mod

    return TelemetryDetector(
        knobs=valid_knobs,
        models=channel_models,
        train_scores=train_scores,
    )


def run(detector: TelemetryDetector, stream: TelemetryStream) -> TelemetryResult:
    """Run anomaly detector on mission stream, merge alarm runs, and evaluate fault metrics."""
    window = detector.knobs["window"]
    det_mode = detector.knobs["detector"]
    persistence = detector.knobs["persistence"]

    channel_alarms: list[np.ndarray] = []
    channel_scores: dict[str, dict[str, np.ndarray]] = {}
    window_starts = np.array([])

    for ch in CHANNELS:
        c_mod = detector.models[ch]
        data = stream.mission_data[ch]
        channel_scores[ch] = {}
        ch_triggers: list[np.ndarray] = []

        if det_mode in ("iforest", "both") and c_mod.iforest_model is not None and c_mod.iforest_scaler is not None:
            X_mi, window_starts = extract_iforest_features(data, window)
            X_mi_s = c_mod.iforest_scaler.transform(X_mi)
            sc_if = -c_mod.iforest_model.score_samples(X_mi_s)
            channel_scores[ch]["iforest"] = sc_if
            ch_triggers.append(sc_if > c_mod.iforest_threshold)

        if det_mode in ("autoencoder", "both") and c_mod.ae_model is not None and c_mod.ae_scaler is not None:
            X_mi, window_starts = extract_autoencoder_features(data, window)
            X_mi_s = c_mod.ae_scaler.transform(X_mi)
            recons_mi = c_mod.ae_model.predict(X_mi_s)
            sc_ae = np.mean((X_mi_s - recons_mi) ** 2, axis=1)
            channel_scores[ch]["autoencoder"] = sc_ae
            ch_triggers.append(sc_ae > c_mod.ae_threshold)

        # "both" = either triggers
        ch_over = np.any(ch_triggers, axis=0) if ch_triggers else np.zeros(len(window_starts), dtype=bool)

        # Persistence check: score must exceed threshold for `persistence` consecutive windows
        if persistence > 1:
            pers_alarm = np.zeros_like(ch_over, dtype=bool)
            for w_idx in range(persistence - 1, len(ch_over)):
                if np.all(ch_over[w_idx - persistence + 1 : w_idx + 1]):
                    pers_alarm[w_idx] = True
            ch_over = pers_alarm

        channel_alarms.append(ch_over)

    # An alarm fires when any channel's window score triggers
    combined_alarms = np.any(channel_alarms, axis=0) if channel_alarms else np.zeros(len(window_starts), dtype=bool)
    alarm_indices = np.where(combined_alarms)[0]

    # Merge overlapping/consecutive alarm windows into alarm runs (bridging gaps <= 2 * STRIDE)
    merged_runs: list[tuple[int, int]] = []
    for w in alarm_indices:
        w_start = int(w * STRIDE)
        w_end = int(w_start + window)
        if not merged_runs or w_start > merged_runs[-1][1] + 2 * STRIDE:
            merged_runs.append((w_start, w_end))
        else:
            prev_start, prev_end = merged_runs[-1]
            merged_runs[-1] = (prev_start, max(prev_end, w_end))

    # Evaluate faults caught vs missed and delays
    n_faults_total = len(stream.faults)
    caught_count = 0
    delays: list[float] = []
    fault_details: list[dict[str, Any]] = []

    for f in stream.faults:
        f_start = f["start"]
        f_end_extended = f["end"] + 100

        # Fault is caught if an alarm run overlaps [start, end+100]
        overlapping_runs = [
            r for r in merged_runs if max(r[0], f_start) <= min(r[1], f_end_extended)
        ]

        if overlapping_runs:
            caught_count += 1
            first_alarm_step = min(r[0] for r in overlapping_runs)
            delay = float(max(0, first_alarm_step - f_start))
            delays.append(delay)

            if delay <= 100.0:
                credit = 1.0
            elif delay <= 400.0:
                credit = 1.0 - 0.7 * (delay - 100.0) / 300.0
            else:
                credit = 0.3

            fault_details.append({
                "fault": f,
                "caught": True,
                "delay": delay,
                "credit": credit,
                "first_alarm_step": first_alarm_step,
            })
        else:
            fault_details.append({
                "fault": f,
                "caught": False,
                "delay": None,
                "credit": 0.0,
                "first_alarm_step": None,
            })

    # A false alarm is an alarm run overlapping no fault window
    false_alarm_count = 0
    for r in merged_runs:
        overlaps_any = any(
            max(r[0], f["start"]) <= min(r[1], f["end"] + 100) for f in stream.faults
        )
        if not overlaps_any:
            false_alarm_count += 1

    total_credits = sum(fd["credit"] for fd in fault_details)
    raw_score = (
        200.0 * (total_credits / n_faults_total) - min(60.0, 6.0 * false_alarm_count)
        if n_faults_total > 0
        else 0.0
    )
    score = float(np.clip(raw_score, 0.0, 200.0))
    missed_count = n_faults_total - caught_count
    mean_delay = float(np.mean(delays)) if delays else 0.0

    return TelemetryResult(
        score=score,
        n_faults=n_faults_total,
        caught=caught_count,
        missed=missed_count,
        false_alarms=false_alarm_count,
        mean_delay=mean_delay,
        alarm_runs=merged_runs,
        fault_details=fault_details,
        window_starts=window_starts,
        channel_scores=channel_scores,
    )


def apply_to_state(state_or_result: Any, result_or_state: Any) -> MissionState:
    """Update mission state resources and Act 2 statistics using coupling formulas and save."""
    from meridian.state import MissionState
    if isinstance(state_or_result, MissionState):
        state = state_or_result
        result = result_or_state
    else:
        result = state_or_result
        state = result_or_state

    res_update = act2_resources(result.missed, result.false_alarms)

    state.resources.fuel_pct = float(res_update["fuel_pct"])
    state.resources.sensor_noise = float(res_update["sensor_noise"])
    state.resources.hull_health = float(res_update["hull_health"])

    state.act2.done = True
    state.act2.score = float(result.score)
    state.act2.n_faults = int(result.n_faults)
    state.act2.caught = int(result.caught)
    state.act2.missed = int(result.missed)
    state.act2.false_alarms = int(result.false_alarms)
    state.act2.mean_delay = float(result.mean_delay)

    state.save()
    return state
