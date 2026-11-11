"""Script to generate all workshop visual artifacts into artifacts/.

Generates:
- artifacts/act1_samples.png: 3 planet, 3 no-planet curves, and 1 folded transit curve.
- artifacts/act2_result.png: Mission telemetry with true faults (red) and alarms (orange).
- artifacts/act3_samples.png: Grid of 4 galaxy classes.
- artifacts/act4_demo.html: HTML animation of landing episode (<= 600 KB).
"""

from __future__ import annotations

import os
import pathlib
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meridian.act1_exoplanet import generate_samples_plot
from meridian.act2_telemetry import make_stream, train as train_telem, run as run_telem, plot_result
from meridian.act3_galaxy import build_survey, show_samples
from meridian.act4_lander import MLPPolicy, record_episode


def main() -> None:
    artifacts_dir = pathlib.Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Act 1 artifact: artifacts/act1_samples.png ...")
    generate_samples_plot(save_path=artifacts_dir / "act1_samples.png")
    plt.close("all")

    print("Generating Act 2 artifact: artifacts/act2_result.png ...")
    stream = make_stream("artifacts_demo", trip_days=270, fast=True)
    det = train_telem(stream, detector="iforest")
    res = run_telem(det, stream)
    plot_result(stream, res, save_path=str(artifacts_dir / "act2_result.png"))
    plt.close("all")

    print("Generating Act 3 artifact: artifacts/act3_samples.png ...")
    survey = build_survey("artifacts_demo", fast=True)
    show_samples(survey, n_per_class=4, save_path=str(artifacts_dir / "act3_samples.png"))
    plt.close("all")

    print("Generating Act 4 artifact: artifacts/act4_demo.html ...")
    ckpt_path = pathlib.Path("checkpoints/act4_default.npz")
    if not ckpt_path.exists():
        from meridian.act4_lander.train import train
        policy, _ = train(n_iterations=5, n_rollouts=40, fast=True)
        policy.save(ckpt_path)
    else:
        policy = MLPPolicy.load(str(ckpt_path))

    cond = {"gravity": 7.0, "wind": 0.0, "fuel_pct": 100.0, "sensor_noise": 0.0}
    html_out = record_episode(
        policy,
        conditions=cond,
        max_frames=45,
        figsize=(4.5, 3.8),
        dpi=60,
    )
    (artifacts_dir / "act4_demo.html").write_text(html_out, encoding="utf-8")
    plt.close("all")

    html_size_kb = (artifacts_dir / "act4_demo.html").stat().st_size / 1024
    print(f"Act 4 demo size: {html_size_kb:.1f} KB (must be <= 600 KB)")

    print("All artifacts generated successfully.")


if __name__ == "__main__":
    main()
