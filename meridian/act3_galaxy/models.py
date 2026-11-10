"""Galaxy classification models (TinyCNN in PyTorch with scikit-learn MLP fallback) and downlink scoring."""

from __future__ import annotations

import os
import warnings
from typing import Any, Final

import numpy as np
from sklearn.neural_network import MLPClassifier

from meridian.act3_galaxy.generator import DOWNLINK_BUDGET, GalaxySurvey, VALUE_WEIGHTS
from meridian.config import is_fast
from meridian.seeding import seed_for
from meridian.state import MissionState

# Optional PyTorch import guard
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def has_torch() -> bool:
    """Return True if PyTorch is installed and available."""
    return HAS_TORCH

KNOB_DEFAULTS_NORMAL: Final[dict[str, Any]] = {
    "epochs": 8,
    "augment": True,
    "class_weighting": True,
    "strategy": "expected_value",
}

KNOB_DEFAULTS_FAST: Final[dict[str, Any]] = {
    "epochs": 5,
    "augment": True,
    "class_weighting": True,
    "strategy": "expected_value",
}

VALID_STRATEGIES: Final[set[str]] = {"expected_value", "max_confidence", "rarity_first"}


def validate_knobs(fast: bool = False, **knobs: Any) -> dict[str, Any]:
    """Validate, clip, and set defaults for training knobs with warnings on deviation."""
    defaults = KNOB_DEFAULTS_FAST if fast else KNOB_DEFAULTS_NORMAL
    result = dict(defaults)

    if "epochs" in knobs:
        raw_epochs = knobs["epochs"]
        try:
            ep = int(raw_epochs)
            if ep < 2 or ep > 20:
                warnings.warn(
                    f"epochs {raw_epochs} outside valid range [2, 20]; clipping.",
                    UserWarning,
                    stacklevel=2,
                )
            result["epochs"] = int(np.clip(ep, 2, 20))
        except (ValueError, TypeError):
            warnings.warn(f"Invalid epochs {raw_epochs}; using default {defaults['epochs']}.", UserWarning, stacklevel=2)

    if "augment" in knobs:
        result["augment"] = bool(knobs["augment"])

    if "class_weighting" in knobs:
        result["class_weighting"] = bool(knobs["class_weighting"])

    if "strategy" in knobs:
        strat = str(knobs["strategy"])
        if strat not in VALID_STRATEGIES:
            warnings.warn(
                f"Unknown strategy '{strat}'; falling back to '{defaults['strategy']}'.",
                UserWarning,
                stacklevel=2,
            )
            result["strategy"] = defaults["strategy"]
        else:
            result["strategy"] = strat

    return result


if HAS_TORCH:
    class _TinyCNNModule(nn.Module):
        """Tiny 3-layer convolutional network for 48x48 galaxy image classification."""

        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 24x24
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 12x12
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),        # 12x12
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 4),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


class TorchModel:
    """Wrapper exposing standard predict_proba API for trained PyTorch TinyCNN."""

    def __init__(self, net: Any) -> None:
        self.net = net
        self.backend: str = "torch"

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Compute softmax class probabilities for a batch of 48x48 images."""
        if images.ndim == 2:
            images = images[None, :, :]
        # Ensure (N, 1, 48, 48) shape
        if images.ndim == 3:
            images_t = torch.from_numpy(images[:, None, :, :].astype(np.float32))
        else:
            images_t = torch.from_numpy(images.astype(np.float32))

        self.net.eval()
        with torch.no_grad():
            logits = self.net(images_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs.astype(np.float32)


class SklearnModel:
    """Fallback MLP classifier on 24x24 downsampled pixels."""

    def __init__(self, clf: MLPClassifier) -> None:
        self.clf = clf
        self.backend: str = "sklearn"

    @staticmethod
    def _downsample(images: np.ndarray) -> np.ndarray:
        """Downsample 48x48 images to 24x24 by 2x2 average pooling and flatten."""
        if images.ndim == 2:
            images = images[None, :, :]
        ds = (
            images[:, 0::2, 0::2]
            + images[:, 1::2, 0::2]
            + images[:, 0::2, 1::2]
            + images[:, 1::2, 1::2]
        ) * 0.25
        return ds.reshape(len(images), -1).astype(np.float32)

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        """Compute class probabilities via MLP on downsampled pixels."""
        flat = self._downsample(images)
        return self.clf.predict_proba(flat).astype(np.float32)


def _augment_batch(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Apply random horizontal/vertical flips and 90-degree rotations."""
    b = x.clone()
    for i in range(len(b)):
        if rng.random() > 0.5:
            b[i] = torch.flip(b[i], [2])
        if rng.random() > 0.5:
            b[i] = torch.flip(b[i], [1])
        k = int(rng.integers(0, 4))
        if k > 0:
            b[i] = torch.rot90(b[i], k, [1, 2])
    return b


def train(
    survey: GalaxySurvey,
    force_backend: str | None = None,
    **knobs: Any,
) -> TorchModel | SklearnModel:
    """Train galaxy classifier on survey training set using PyTorch or scikit-learn fallback."""
    fast = survey.is_fast
    valid_knobs = validate_knobs(fast=fast, **knobs)
    epochs = int(valid_knobs["epochs"])
    use_augment = bool(valid_knobs["augment"])
    use_class_weighting = bool(valid_knobs["class_weighting"])

    # Backend selection: use PyTorch if importable unless force_backend == 'sklearn'
    use_torch = HAS_TORCH and (force_backend != "sklearn") and (os.environ.get("FORCE_SKLEARN", "0") != "1")

    if use_torch:
        print("[Meridian Act 3] Active backend: torch (TinyCNN on CPU)")
        torch.set_num_threads(min(4, os.cpu_count() or 1))

        train_seed = seed_for(survey.team_id, "act3_train")
        torch.manual_seed(train_seed)
        aug_rng = np.random.default_rng(train_seed + 11)

        X_tr = survey.train_images[:, None, :, :].astype(np.float32)
        y_tr = survey.train_labels.astype(np.int64)

        if use_class_weighting:
            counts = np.bincount(y_tr, minlength=4)
            weights = torch.tensor(len(y_tr) / (4.0 * np.maximum(1, counts)), dtype=torch.float32)
            criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            criterion = nn.CrossEntropyLoss()

        model = _TinyCNNModule()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)

        dataset = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for _ in range(epochs):
            model.train()
            for bx, by in loader:
                if use_augment:
                    bx = _augment_batch(bx, aug_rng)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        return TorchModel(model)

    else:
        print("[Meridian Act 3] Active backend: sklearn (MLPClassifier on 24x24 downsampled pixels)")
        train_seed = seed_for(survey.team_id, "act3_train")
        X_tr_flat = SklearnModel._downsample(survey.train_images)
        y_tr = survey.train_labels.astype(np.int64)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=max(30, epochs * 6),
            random_state=train_seed,
            early_stopping=False,
        )
        mlp.fit(X_tr_flat, y_tr)
        return SklearnModel(mlp)


def select_downlink(
    model: TorchModel | SklearnModel,
    survey: GalaxySurvey,
    strategy: str = "expected_value",
) -> list[int]:
    """Select top K candidate images from sky stream according to chosen prioritization strategy."""
    probs = model.predict_proba(survey.stream_images)
    k = min(DOWNLINK_BUDGET, len(survey.stream_images))

    if strategy == "expected_value":
        # Values: elliptical 1, spiral 3, edge_on 2, merger 5
        values = np.array([
            VALUE_WEIGHTS[0],
            VALUE_WEIGHTS[1],
            VALUE_WEIGHTS[2],
            VALUE_WEIGHTS[3],
        ], dtype=np.float32)
        ranks = probs @ values
    elif strategy == "max_confidence":
        ranks = np.max(probs, axis=1)
    elif strategy == "rarity_first":
        # Rank by p_merger + 0.5 * p_spiral
        ranks = probs[:, 3] + 0.5 * probs[:, 1]
    else:
        values = np.array([1.0, 3.0, 2.0, 5.0], dtype=np.float32)
        ranks = probs @ values

    selected_indices = np.argsort(-ranks)[:k].tolist()
    return selected_indices


def score(
    state: MissionState,
    survey: GalaxySurvey,
    model: TorchModel | SklearnModel,
    selected: list[int],
) -> float:
    """Evaluate downlink selection value ratio, test accuracy, update mission state, and save."""
    stream_values = np.array([VALUE_WEIGHTS[int(c)] for c in survey.stream_labels], dtype=np.float32)
    selected_value = float(np.sum(stream_values[selected]))

    k = min(DOWNLINK_BUDGET, len(stream_values))
    best_possible_value = float(np.sum(np.sort(stream_values)[-k:]))
    value_ratio = (selected_value / best_possible_value) if best_possible_value > 0.0 else 1.0

    act3_score = float(np.clip(round(200.0 * value_ratio, 2), 0.0, 200.0))

    # Test set classification accuracy
    test_probs = model.predict_proba(survey.test_images)
    test_preds = np.argmax(test_probs, axis=1)
    test_accuracy = float(np.mean(test_preds == survey.test_labels))

    # Update state
    state.act3.done = True
    state.act3.score = act3_score
    state.act3.accuracy = round(test_accuracy, 4)
    state.act3.value_ratio = round(value_ratio, 4)
    state.act3.backend = model.backend

    state.save()
    return act3_score
