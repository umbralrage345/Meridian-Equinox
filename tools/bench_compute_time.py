"""Measure total mission compute across Acts 1-4 in FAST mode (restricted to 2 CPUs) and NORMAL mode."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Enforce 2-CPU low-core configuration per workshop specifications
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["LOKY_MAX_CPU_COUNT"] = "2"

try:
    import psutil
    p = psutil.Process()
    # Restrict process to 2 logical CPUs
    p.cpu_affinity([0, 1])
except Exception:
    pass

from meridian.config import FAST_BUDGETS
from meridian.state import MissionState


def run_full_mission(fast: bool) -> tuple[dict[str, float], float]:
    """Execute complete Acts 1-4 mission pipeline and return per-act and total execution times."""
    os.environ["MERIDIAN_FAST"] = "1" if fast else "0"
    timings: dict[str, float] = {}
    team_name = f"bench_{'fast' if fast else 'normal'}"
    state = MissionState(team_name)

    # Act 1: Exoplanet
    t0 = time.perf_counter()
    from meridian.act1_exoplanet import build_survey as build_s1, train as train_s1, rank_candidates, choose_target
    survey1 = build_s1(team_name, fast=fast)
    m1 = train_s1(survey1)
    ranked = rank_candidates(m1, survey1)
    state = choose_target(state, survey1, m1, ranked.iloc[0]["star_id"])
    timings["Act 1 (Exoplanet)"] = time.perf_counter() - t0

    # Act 2: Telemetry
    t0 = time.perf_counter()
    from meridian.act2_telemetry import make_stream, train as train_s2, run as run_s2, apply_to_state as apply_s2
    trip_days = state.act1.target.trip_days
    stream2 = make_stream(team_name, trip_days=trip_days, fast=fast)
    det2 = train_s2(stream2)
    res2 = run_s2(det2, stream2)
    state = apply_s2(state, res2)
    timings["Act 2 (Telemetry)"] = time.perf_counter() - t0

    # Act 3: Galaxy
    t0 = time.perf_counter()
    from meridian.act3_galaxy import build_survey as build_s3, train as train_s3, select_downlink, score as score_s3
    survey3 = build_s3(team_name, fast=fast)
    epochs = 5 if fast else 8
    m3 = train_s3(survey3, epochs=epochs)
    selected = select_downlink(m3, survey3)
    score_s3(state, survey3, m3, selected)
    timings["Act 3 (Galaxy)"] = time.perf_counter() - t0

    # Act 4: Lander
    t0 = time.perf_counter()
    from meridian.act4_lander import build_conditions, train as train_s4, evaluate as eval_s4, apply_to_state as apply_s4
    cond4 = build_conditions(state)
    iters = 15 if fast else 35
    pop = 20 if fast else 30
    p4, _ = train_s4(cond4, iterations=iters, pop_size=pop, warm_start=True)
    eval_res4 = eval_s4(p4, cond4, n_episodes=100, seed=42)
    apply_s4(state, eval_res4)
    timings["Act 4 (Lander)"] = time.perf_counter() - t0

    # Clean up state file
    for p_path in Path(".").glob(f"mission_state_{team_name}.json"):
        p_path.unlink()

    total_time = sum(timings.values())
    return timings, total_time


def main():
    print("=" * 75)
    print("MISSION COMPUTE BENCHMARK: FAST MODE (2-CPU LOW-CORE RESTRICTION)")
    print("Budgets: Act 1 <= 240s, Act 2 <= 120s, Act 3 <= 240s, Act 4 <= 300s, Total <= 720s")
    print("=" * 75)

    # In FAST mode, delete pre-computed bench_fast survey cache to measure cold execution time
    cache_f = Path("artifacts/cache/survey_bench_fast_fast1.npz")
    if cache_f.exists():
        cache_f.unlink()

    fast_timings, fast_total = run_full_mission(fast=True)
    budgets = {"Act 1 (Exoplanet)": 240, "Act 2 (Telemetry)": 120, "Act 3 (Galaxy)": 240, "Act 4 (Lander)": 300}
    for act, t in fast_timings.items():
        b = budgets.get(act, 300)
        status = "PASS" if t <= b else "FAIL"
        print(f"  {act:<22}: {t:6.2f} s / {b} s max ({t/60:4.2f} min) [{status}]")
    total_status = "PASS" if fast_total <= 720 else "FAIL"
    print(f"  {'Total FAST Compute':<22}: {fast_total:6.2f} s / 720 s max ({fast_total/60:4.2f} min) [{total_status}]")

    print("\n" + "=" * 75)
    print("MISSION COMPUTE BENCHMARK: NORMAL MODE (Budget <= 20 min / 1200 s)")
    print("=" * 75)
    norm_timings, norm_total = run_full_mission(fast=False)
    for act, t in norm_timings.items():
        print(f"  {act:<22}: {t:6.2f} s ({t/60:4.2f} min)")
    print(f"  {'Total NORMAL Compute':<22}: {norm_total:6.2f} s / 1200 s max ({norm_total/60:4.2f} min) [{'PASS' if norm_total <= 1200 else 'FAIL'}]")
    print("=" * 75)


if __name__ == "__main__":
    main()
