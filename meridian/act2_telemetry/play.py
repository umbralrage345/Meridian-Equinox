"""Visualization and interactive benchmarking tools for Act 2 Telemetry Anomaly Detection."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

from meridian.act2_telemetry.detectors import TelemetryResult
from meridian.act2_telemetry.generator import CHANNELS, TelemetryStream


def animate_stream(stream: TelemetryStream, result: TelemetryResult | None = None) -> str:
    """Generate HTML5/JS animation of telemetry mission stream (<= 60 subsampled frames)."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    n_steps = len(stream.mission_data[CHANNELS[0]])
    n_frames = min(60, n_steps)
    step_indices = np.linspace(0, n_steps - 1, n_frames, dtype=int)

    lines = []
    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        data = stream.mission_data[ch]
        (line,) = ax.plot([], [], lw=1.2, color="#0284c7")
        ax.set_xlim(0, n_steps)
        ax.set_ylim(float(np.min(data) - 0.5), float(np.max(data) + 0.5))
        ax.set_ylabel(ch, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        lines.append(line)

        # Draw static fault spans
        for f in stream.faults:
            if f["channel"] == ch:
                ax.axvspan(f["start"], f["end"], color="#ef4444", alpha=0.25)

        # Draw detected alarm runs
        if result is not None:
            for r_start, r_end in result.alarm_runs:
                ax.axvspan(r_start, r_end, color="#f97316", alpha=0.2)

    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()

    def update(frame_idx: int):
        cur_step = step_indices[frame_idx]
        for i, ch in enumerate(CHANNELS):
            y_data = stream.mission_data[ch][:cur_step]
            x_data = np.arange(len(y_data))
            lines[i].set_data(x_data, y_data)
        return lines

    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    html_output = anim.to_jshtml()
    plt.close(fig)
    return html_output


def plot_result(
    stream: TelemetryStream,
    result: TelemetryResult,
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot mission stream across 4 channels with true faults (red), channel alarms, and pooled alarm strip."""
    fig, axes = plt.subplots(
        5, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 2, 2, 2, 0.9]}
    )
    n_steps = len(stream.mission_data[CHANNELS[0]])
    time_steps = np.arange(n_steps)

    for i, ch in enumerate(CHANNELS):
        ax = axes[i]
        data = stream.mission_data[ch]
        ax.plot(time_steps, data, color="#334155", lw=0.9, alpha=0.85, label="Telemetry")

        # Shade true faults in red
        for f_idx, f in enumerate(stream.faults):
            if f["channel"] == ch:
                lbl = "True Fault" if f_idx == 0 else ""
                ax.axvspan(f["start"], f["end"], color="#ef4444", alpha=0.35, label=lbl)
                mid_x = (f["start"] + f["end"]) / 2
                ax.text(mid_x, float(np.max(data)), f["type"], fontsize=8, color="#991b1b", ha="center")

        # Draw each channel's alarm runs ONLY on the channel that triggered
        ch_runs = result.channel_alarm_runs.get(ch, [])
        for r_idx, (r_start, r_end) in enumerate(ch_runs):
            lbl = "Channel Alarm" if r_idx == 0 else ""
            ax.axvspan(r_start, r_end, color="#f97316", alpha=0.35, label=lbl)

        ax.set_ylabel(ch, fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Bottom strip: Pooled alarm and overall ground truth faults
    ax_pooled = axes[4]
    ax_pooled.set_xlim(0, n_steps)
    ax_pooled.set_ylim(0, 1)
    ax_pooled.set_yticks([])
    ax_pooled.set_ylabel("Pooled Alarm", fontsize=8, fontweight="bold")
    ax_pooled.grid(True, linestyle="--", alpha=0.3)

    for f_idx, f in enumerate(stream.faults):
        lbl = "True Fault" if f_idx == 0 else ""
        ax_pooled.axvspan(f["start"], f["end"], color="#ef4444", alpha=0.45, label=lbl)

    for r_idx, (r_start, r_end) in enumerate(result.alarm_runs):
        lbl = "Pooled Alarm" if r_idx == 0 else ""
        ax_pooled.axvspan(r_start, r_end, color="#f97316", alpha=0.55, label=lbl)

    ax_pooled.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Mission Time Steps")

    title_str = (
        f"Act 2 Telemetry Mission: Score = {result.score:.1f}/200 | "
        f"Caught = {result.caught}/{result.n_faults} | "
        f"Alarm Coverage = {result.alarm_coverage * 100:.1f}% | "
        f"False Alarms = {result.false_alarms} | Mean Delay = {result.mean_delay:.1f} steps"
    )
    fig.suptitle(title_str, fontsize=12, y=0.99)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120)

    return fig, axes


def human_vs_ai(stream: TelemetryStream) -> dict[str, Any]:
    """Display 3 short telemetry segments (one containing a real fault) for human identification."""
    if not stream.faults:
        raise ValueError("Mission stream has no faults to benchmark against.")

    target_fault = stream.faults[0]
    target_ch = target_fault["channel"]
    f_start = target_fault["start"]
    f_end = target_fault["end"]

    # Build 3 segments of length 300
    seg_len = 300
    f_mid = (f_start + f_end) // 2
    seg_fault_start = max(0, f_mid - seg_len // 2)
    seg_fault = stream.mission_data[target_ch][seg_fault_start : seg_fault_start + seg_len]

    # Two nominal segments from training data
    tr_data = stream.train_data[target_ch]
    seg_nom1 = tr_data[100 : 100 + seg_len]
    seg_nom2 = tr_data[800 : 800 + seg_len]

    segments = [("A", seg_nom1, False), ("B", seg_fault, True), ("C", seg_nom2, False)]
    # Deterministic rotation based on fault length
    rot = f_start % 3
    segments = segments[rot:] + segments[:rot]
    correct_choice = next(name for name, _, has_f in segments if has_f)

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for idx, (name, seg, _) in enumerate(segments):
        axes[idx].plot(seg, color="#2563eb", lw=1.2)
        axes[idx].set_title(f"Segment {name} (Channel: {target_ch})", fontsize=10, loc="left")
        axes[idx].grid(True, linestyle="--", alpha=0.4)
    axes[-1].set_xlabel("Relative Time Step")
    plt.tight_layout()
    plt.show()

    print("\n--- HUMAN VS AI TELEMETRY CHALLENGE ---")
    print(f"One of the segments above contains a genuine probe fault ({target_fault['type']}).")
    print("Which segment contains the anomaly? Choose 'A', 'B', or 'C'.")
    print("Call reveal(guesses=['A']) or leave guesses=[None] to skip.\n")

    def _reveal_fn(guesses: list[Any] | Any = None) -> dict[str, Any]:
        return reveal(guesses=guesses, challenge={"correct_segment": correct_choice, "fault_type": target_fault["type"], "channel": target_ch})

    return {
        "correct_segment": correct_choice,
        "fault_type": target_fault["type"],
        "channel": target_ch,
        "reveal": _reveal_fn,
    }


def reveal(
    guesses: list[Any] | Any = None,
    challenge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare human anomaly guess against ground truth fault segment."""
    if challenge is None:
        raise ValueError("Must provide challenge dict from human_vs_ai(stream).")

    correct = challenge["correct_segment"]
    f_type = challenge["fault_type"]
    ch = challenge["channel"]

    # Normalize guess
    if isinstance(guesses, list):
        guess = guesses[0] if len(guesses) > 0 else None
    else:
        guess = guesses

    if guess is not None:
        guess_str = str(guess).strip().upper()
        is_correct = (guess_str == correct)
        status = "CORRECT" if is_correct else "INCORRECT"
        print("=" * 60)
        print(f"TELEMETRY FAULT DIAGNOSIS RESULT: [{status}]")
        print(f"Your Guess    : Segment {guess_str}")
        print(f"Ground Truth  : Segment {correct} ({f_type} on {ch})")
        print("=" * 60 + "\n")
        return {"correct": is_correct, "guess": guess_str, "actual": correct, "skipped": False}
    else:
        print("=" * 60)
        print("TELEMETRY FAULT DIAGNOSIS: [SKIPPED]")
        print(f"Ground Truth  : Segment {correct} ({f_type} on {ch})")
        print(">> Instruction: Edit guesses = ['A'] (or 'B', 'C') to test your visual diagnosis!")
        print("=" * 60 + "\n")
        return {"correct": False, "guess": None, "actual": correct, "skipped": True}
