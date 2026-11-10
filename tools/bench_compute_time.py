"""Measure total mission compute across Acts 1-4 in FAST and NORMAL modes."""

from __future__ import annotations

import os
import time
from pathlib import Path

from meridian.state import MissionState


def run_full_mission(fast: bool) -> tuple[dict[str, float], float]:
    """Execute complete Acts 1-4 mission pipeline and return per-act and total execution times."""
    os.environ["MERIDIAN_FAST"] = "1" if fast else "0"
    timings = {}
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
    for p in Path(".").glob(f"mission_state_{team_name}.json"):
        p.unlink()

    total_time = sum(timings.values())
    return timings, total_time


def main():
    print("=" * 70)
    print("MISSION COMPUTE BENCHMARK: FAST MODE (Budget <= 12 min / 720 s)")
    print("=" * 70)
    fast_timings, fast_total = run_full_mission(fast=True)
    for act, t in fast_timings.items():
        print(f"  {act:<22}: {t:6.2f} s ({t/60:4.2f} min)")
    print(f"  {'Total FAST Compute':<22}: {fast_total:6.2f} s ({fast_total/60:4.2f} min) [{'PASS' if fast_total <= 720 else 'FAIL'}]")

    print("\n" + "=" * 70)
    print("MISSION COMPUTE BENCHMARK: NORMAL MODE (Budget <= 20 min / 1200 s)")
    print("=" * 70)
    norm_timings, norm_total = run_full_mission(fast=False)
    for act, t in norm_timings.items():
        print(f"  {act:<22}: {t:6.2f} s ({t/60:4.2f} min)")
    print(f"  {'Total NORMAL Compute':<22}: {norm_total:6.2f} s ({norm_total/60:4.2f} min) [{'PASS' if norm_total <= 1200 else 'FAIL'}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
