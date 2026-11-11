"""Act 2: Telemetry Anomaly Detection across Multichannel Subsystems."""

from meridian.act2_telemetry.detectors import (
    STRIDE,
    TRAIN_KNOBS,
    TelemetryDetector,
    TelemetryResult,
    apply_to_state,
    build_stream,
    extract_autoencoder_features,
    extract_iforest_features,
    run,
    train,
    validate_knobs,
)
from meridian.act2_telemetry.generator import (
    CHANNEL_AMPLITUDES,
    CHANNELS,
    FAULT_TYPES,
    TelemetryStream,
    make_stream,
)
from meridian.act2_telemetry.play import animate_stream, human_vs_ai, plot_result, reveal

__all__ = [
    "CHANNELS",
    "CHANNEL_AMPLITUDES",
    "FAULT_TYPES",
    "STRIDE",
    "TelemetryStream",
    "make_stream",
    "build_stream",
    "TRAIN_KNOBS",
    "validate_knobs",
    "TelemetryDetector",
    "TelemetryResult",
    "train",
    "run",
    "apply_to_state",
    "extract_iforest_features",
    "extract_autoencoder_features",
    "animate_stream",
    "plot_result",
    "human_vs_ai",
    "reveal",
]
