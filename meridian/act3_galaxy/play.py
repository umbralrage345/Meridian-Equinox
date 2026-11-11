"""Interactive visualization and analysis tools for Act 3 Galaxy Classification."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from meridian.act3_galaxy.generator import CLASSES, GalaxySurvey, VALUE_WEIGHTS


def show_samples(
    survey: GalaxySurvey,
    n_per_class: int = 4,
    save_path: str | None = None,
) -> plt.Figure:
    """Display a grid of sample galaxy images grouped by class."""
    fig, axes = plt.subplots(4, n_per_class, figsize=(2.2 * n_per_class, 9.0))
    fig.suptitle(f"Act 3 Galaxy Survey Samples (Team: {survey.team_id})", fontsize=14, fontweight="bold")

    for cls_idx, cls_name in enumerate(CLASSES):
        indices = np.where(survey.train_labels == cls_idx)[0][:n_per_class]
        for col_idx, img_idx in enumerate(indices):
            ax = axes[cls_idx, col_idx]
            ax.imshow(survey.train_images[img_idx], cmap="inferno", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                val = VALUE_WEIGHTS[cls_idx]
                ax.set_ylabel(f"{cls_name.capitalize()}\n(Val: {val})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_confusion(
    model: Any,
    survey: GalaxySurvey,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot normalized test-set confusion matrix for the trained classifier."""
    probs = model.predict_proba(survey.test_images)
    preds = np.argmax(probs, axis=1)
    cm = confusion_matrix(survey.test_labels, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cax = ax.imshow(cm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CLASSES, fontsize=10)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_title(f"Test Confusion Matrix ({model.backend.upper()} backend)", fontsize=13, fontweight="bold")

    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", color=color, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def occlusion_map(
    model: Any,
    image: np.ndarray,
    cls: int,
    patch_size: int = 8,
    stride: int = 4,
) -> np.ndarray:
    """Compute 2D spatial feature importance map by sliding zero occlusion patch across image."""
    size = image.shape[0]
    p_base = float(model.predict_proba(image[None, :, :])[0, cls])

    starts = list(range(0, size - patch_size + 1, stride))
    occluded_batch = []
    coords = []

    for r in starts:
        for c in starts:
            occ = image.copy()
            occ[r : r + patch_size, c : c + patch_size] = 0.0
            occluded_batch.append(occ)
            coords.append((r, c))

    batch_arr = np.stack(occluded_batch)
    p_occs = model.predict_proba(batch_arr)[:, cls]
    diffs = p_base - p_occs

    # Accumulate importance onto full 48x48 pixel array
    importance = np.zeros((size, size), dtype=np.float32)
    weights = np.zeros((size, size), dtype=np.float32)

    for (r, c), diff in zip(coords, diffs):
        importance[r : r + patch_size, c : c + patch_size] += float(diff)
        weights[r : r + patch_size, c : c + patch_size] += 1.0

    weights[weights == 0.0] = 1.0
    return importance / weights


def show_downlink(
    survey: GalaxySurvey,
    selected: list[int],
    model: Any | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Visualize selected downlink galaxies with predicted and true values."""
    k = len(selected)
    cols = 5
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.2 * rows))
    fig.suptitle(f"Selected Downlink Stream ({k} Targets)", fontsize=14, fontweight="bold")
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    probs = model.predict_proba(survey.stream_images[selected]) if model is not None else None

    for i, idx in enumerate(selected):
        ax = axes_flat[i]
        img = survey.stream_images[idx]
        true_cls = int(survey.stream_labels[idx])
        true_val = VALUE_WEIGHTS[true_cls]

        ax.imshow(img, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])

        if probs is not None:
            pred_cls = int(np.argmax(probs[i]))
            pred_conf = float(probs[i, pred_cls])
            ax.set_title(
                f"True: {CLASSES[true_cls]} (V={true_val})\nPred: {CLASSES[pred_cls]} ({pred_conf:.0%})",
                fontsize=9,
            )
        else:
            ax.set_title(f"Target #{idx}: {CLASSES[true_cls]} (V={true_val})", fontsize=10)

    for j in range(k, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def human_vs_ai(survey: GalaxySurvey, n: int = 6) -> dict[str, Any]:
    """Display n unlabeled galaxies from the sky stream for human classification guessing."""
    indices = np.arange(min(n, len(survey.stream_images)))
    imgs = survey.stream_images[indices]
    truths = survey.stream_labels[indices]

    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
    fig.suptitle("Human Challenge: Label each galaxy (0: Elliptical, 1: Spiral, 2: Edge-on, 3: Merger)", fontsize=12)

    for i in range(n):
        ax = axes[i]
        ax.imshow(imgs[i], cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title(f"Image #{i+1}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

    def _reveal_fn(guesses: list[int | None] | np.ndarray) -> dict[str, Any]:
        """Check participant guesses against true galaxy labels."""
        guesses_list = list(guesses[:n]) if hasattr(guesses, "__iter__") else [guesses]
        attempted = 0
        correct = 0
        records = []
        for i, (g, t) in enumerate(zip(guesses_list, truths[:len(guesses_list)])):
            if g is not None:
                attempted += 1
                g_int = int(g)
                is_corr = (g_int == int(t))
                if is_corr:
                    correct += 1
                status = "CORRECT" if is_corr else "MISSED"
                g_name = CLASSES[g_int] if 0 <= g_int < len(CLASSES) else str(g_int)
                print(f"Image #{i+1}: Guess={g_name:<10} | True={CLASSES[int(t)]:<10} [{status}]")
                records.append({"image": i + 1, "guess": g_name, "true": CLASSES[int(t)], "status": status})
            else:
                print(f"Image #{i+1}: Guess=SKIP       | True={CLASSES[int(t)]:<10} [SKIPPED]")
                records.append({"image": i + 1, "guess": "SKIP", "true": CLASSES[int(t)], "status": "SKIPPED"})

        acc = float(correct / attempted) if attempted > 0 else 0.0
        print("-" * 55)
        if attempted > 0:
            print(f"Human Classification Result: {correct}/{attempted} correct ({acc:.1%}) [Attempted {attempted}/{len(guesses_list)}]")
        else:
            print(f"Human Classification Result: SKIPPED (0/{len(guesses_list)} attempted).")
            print(">> Instruction: Edit guesses = [0, 1, 2, 3] (0:Elliptical, 1:Spiral, 2:Edge-on, 3:Merger) to test yourself!")
        return {"accuracy": acc, "correct": int(correct), "total": len(guesses_list), "attempted": attempted, "records": records}

    return {"figure": fig, "indices": indices, "truths": truths, "reveal": _reveal_fn}


def reveal(
    guesses: list[int | None] | np.ndarray,
    challenge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Top-level helper to evaluate human guesses from galaxy challenge."""
    if challenge is None or "reveal" not in challenge:
        raise ValueError("Must provide challenge dict returned by human_vs_ai(survey).")
    return challenge["reveal"](guesses)
