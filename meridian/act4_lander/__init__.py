"""Act 4: Lunar Lander Simulation and Evolutionary Training."""

from meridian.act4_lander.env import VecLander
from meridian.act4_lander.policy import MLPPolicy, pd_controller
from meridian.act4_lander.train import (
    TRAIN_KNOBS,
    validate_knobs,
    build_conditions,
    evaluate,
    train,
    apply_to_state,
)
from meridian.act4_lander.play import (
    human_vs_ai,
    plot_learning_curve,
    record_episode,
    reveal,
    stress_test,
)

__all__ = [
    "VecLander",
    "MLPPolicy",
    "pd_controller",
    "TRAIN_KNOBS",
    "validate_knobs",
    "build_conditions",
    "evaluate",
    "train",
    "apply_to_state",
    "record_episode",
    "plot_learning_curve",
    "stress_test",
    "human_vs_ai",
    "reveal",
]
