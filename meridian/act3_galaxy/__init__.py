"""Act 3: Galaxy Classification and Downlink Prioritization."""

from __future__ import annotations

from meridian.act3_galaxy.generator import (
    CLASS_PRIORS,
    CLASSES,
    DOWNLINK_BUDGET,
    GalaxySurvey,
    IMAGE_SIZE,
    VALUE_WEIGHTS,
    build_survey,
    make_galaxy,
)
from meridian.act3_galaxy.models import (
    KNOB_DEFAULTS_FAST,
    KNOB_DEFAULTS_NORMAL,
    SklearnModel,
    TorchModel,
    has_torch,
    score,
    select_downlink,
    train,
    validate_knobs,
)
from meridian.act3_galaxy.play import (
    human_vs_ai,
    occlusion_map,
    plot_confusion,
    show_downlink,
    show_samples,
)

__all__ = [
    "CLASSES",
    "CLASS_PRIORS",
    "DOWNLINK_BUDGET",
    "GalaxySurvey",
    "IMAGE_SIZE",
    "KNOB_DEFAULTS_FAST",
    "KNOB_DEFAULTS_NORMAL",
    "SklearnModel",
    "TorchModel",
    "VALUE_WEIGHTS",
    "build_survey",
    "has_torch",
    "human_vs_ai",
    "make_galaxy",
    "occlusion_map",
    "plot_confusion",
    "score",
    "select_downlink",
    "show_downlink",
    "show_samples",
    "train",
    "validate_knobs",
]
