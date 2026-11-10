"""Pre-workshop smoke test and environment validation script."""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Check core dependencies and report versions
VERSIONS: dict[str, str] = {}
for pkg in ["numpy", "scipy", "sklearn", "matplotlib", "pandas", "astropy", "IPython", "nbformat"]:
    try:
        mod = __import__(pkg)
        VERSIONS[pkg] = getattr(mod, "__version__", "unknown")
    except ImportError:
        VERSIONS[pkg] = "MISSING"

# Check optional PyTorch
TORCH_PRESENT = False
TORCH_VERSION = "Not installed (CPU sklearn fallback active)"
try:
    import torch
    TORCH_PRESENT = True
    TORCH_VERSION = f"v{torch.__version__} (CPU)"
except ImportError:
    pass


def run_smoke_test() -> dict[str, Any]:
    """Execute quick smoke test across all four acts at tiny scale."""
    results: dict[str, bool] = {}
    timings: dict[str, float] = {}

    # Act 1 Smoke Test
    t0 = time.perf_counter()
    try:
        from meridian.act1_exoplanet import build_survey, train, rank_candidates, choose_target
        from meridian.state import MissionState
        s1 = MissionState("smoke_team")
        survey = build_survey("smoke_team", fast=True)
        model = train(survey, n_estimators=50, max_depth=5, feature_set="bls_only")
        ranked = rank_candidates(model, survey)
        s1 = choose_target(s1, survey, model, ranked.iloc[0]["star_id"])
        results["Act 1 (Exoplanet)"] = bool(s1.act1.done and s1.act1.score >= 0)
    except Exception as e:
        print(f"  [!] Act 1 error: {e}")
        results["Act 1 (Exoplanet)"] = False
    timings["Act 1 (Exoplanet)"] = time.perf_counter() - t0

    # Act 2 Smoke Test
    t0 = time.perf_counter()
    try:
        from meridian.act2_telemetry import make_stream, train, run, apply_to_state
        s2 = MissionState("smoke_team")
        stream = make_stream("smoke_team", trip_days=60, fast=True)
        det = train(stream, detector="iforest", window=32, threshold_q=99.0, persistence=2)
        res = run(det, stream)
        s2 = apply_to_state(s2, res)
        results["Act 2 (Telemetry)"] = bool(s2.act2.done and s2.act2.score >= 0)
    except Exception as e:
        print(f"  [!] Act 2 error: {e}")
        results["Act 2 (Telemetry)"] = False
    timings["Act 2 (Telemetry)"] = time.perf_counter() - t0

    # Act 3 Smoke Test
    t0 = time.perf_counter()
    try:
        from meridian.act3_galaxy import build_survey as build_galaxy_survey, train as train_galaxy, select_downlink, score as score_galaxy
        s3 = MissionState("smoke_team")
        gsurvey = build_galaxy_survey("smoke_team", fast=True)
        gmodel = train_galaxy(gsurvey, epochs=2, augment=False, class_weighting=False)
        selected = select_downlink(gmodel, gsurvey, strategy="expected_value")
        score_val = score_galaxy(s3, gsurvey, gmodel, selected)
        results["Act 3 (Galaxy)"] = bool(s3.act3.done and score_val >= 0)
    except Exception as e:
        print(f"  [!] Act 3 error: {e}")
        results["Act 3 (Galaxy)"] = False
    timings["Act 3 (Galaxy)"] = time.perf_counter() - t0

    # Act 4 Smoke Test
    t0 = time.perf_counter()
    try:
        from meridian.act4_lander import build_conditions, train as train_lander, evaluate as eval_lander, apply_to_state as apply_lander
        s4 = MissionState("smoke_team")
        cond = build_conditions(s4)
        policy, _ = train_lander(cond, iterations=5, pop_size=10, sigma=0.08, lr=0.04, warm_start=True)
        eval_res = eval_lander(policy, cond, n_episodes=10, seed=42)
        apply_lander(s4, eval_res)
        results["Act 4 (Lander)"] = bool(s4.act4.done and s4.act4.score >= 0)
    except Exception as e:
        print(f"  [!] Act 4 error: {e}")
        results["Act 4 (Lander)"] = False
    timings["Act 4 (Lander)"] = time.perf_counter() - t0

    # Clean up generated smoke test state files
    for p in [Path("mission_state_smoke_team.json"), Path("./mission_state_smoke_team.json")]:
        if p.exists():
            p.unlink()

    return {"results": results, "timings": timings}


def main():
    print("=" * 65)
    print("        PROJECT MERIDIAN — SYSTEM PRE-FLIGHT SELFCHECK         ")
    print("=" * 65)

    print(f"Platform       : {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version : {sys.version.split()[0]}")
    print(f"Logical CPUs   : {os.cpu_count() or 1}")
    print(f"PyTorch Status : {TORCH_VERSION}")
    print("-" * 65)
    print("Core Dependencies:")
    for lib, ver in VERSIONS.items():
        status = ver if ver != "MISSING" else "MISSING [FAIL]"
        print(f"  * {lib:<15}: {status}")

    print("-" * 65)
    print("Executing Act 1-4 Smoke Test (Runtime budget: <= 60s)...")
    t_start = time.perf_counter()
    data = run_smoke_test()
    t_total = time.perf_counter() - t_start

    print("-" * 65)
    all_passed = True
    for act_name, passed in data["results"].items():
        status_str = "PASS" if passed else "FAIL"
        t_s = data["timings"].get(act_name, 0.0)
        print(f"  * {act_name:<20}: [{status_str}] in {t_s:.2f}s")
        if not passed:
            all_passed = False

    print("=" * 65)
    print(f"Selfcheck Completed in {t_total:.2f}s (Budget: <= 60.0s)")
    if all_passed and t_total <= 60.0:
        print("OVERALL STATUS: ALL ACTS PASSED — MERIDIAN WORKSHOP READY!")
    else:
        print("OVERALL STATUS: CHECK FAILED — REVIEW OUTPUT ABOVE.")
        sys.exit(1)


if __name__ == "__main__":
    main()
