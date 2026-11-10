"""Act 1: Exoplanet Detection using Box Least Squares and Random Forests."""

from meridian.act1_exoplanet.generator import TIME_GRID, make_star
from meridian.act1_exoplanet.pipeline import (
    TRAIN_KNOBS,
    ExoplanetModel,
    Survey,
    build_survey,
    choose_target,
    extract_features,
    flatten,
    rank_candidates,
    train,
    validate_knobs,
)
from meridian.act1_exoplanet.play import (
    human_vs_ai,
    plot_bls,
    plot_candidates,
    plot_fold,
    plot_lightcurve,
    reveal,
)

__all__ = [
    "TIME_GRID",
    "make_star",
    "flatten",
    "extract_features",
    "TRAIN_KNOBS",
    "validate_knobs",
    "Survey",
    "build_survey",
    "ExoplanetModel",
    "train",
    "rank_candidates",
    "choose_target",
    "plot_lightcurve",
    "plot_bls",
    "plot_fold",
    "plot_candidates",
    "human_vs_ai",
    "reveal",
]
