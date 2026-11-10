"""Benchmark peak memory usage for each act in FAST mode."""

from __future__ import annotations

import os
import tracemalloc
from pathlib import Path

# Ensure FAST mode
os.environ["MERIDIAN_FAST"] = "1"

from meridian.state import MissionState


def bench_act1() -> float:
    from meridian.act1_exoplanet import build_survey, train, rank_candidates, choose_target
    tracemalloc.start()
    survey = build_survey("bench_mem", fast=True)
    model = train(survey)
    ranked = rank_candidates(model, survey)
    state = MissionState("bench_mem")
    choose_target(state, survey, model, ranked.iloc[0]["star_id"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def bench_act2() -> float:
    from meridian.act2_telemetry import make_stream, train, run, apply_to_state
    tracemalloc.start()
    stream = make_stream("bench_mem", trip_days=100, fast=True)
    det = train(stream)
    res = run(det, stream)
    state = MissionState("bench_mem")
    apply_to_state(state, res)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def bench_act3() -> float:
    from meridian.act3_galaxy import build_survey, train, select_downlink, score
    tracemalloc.start()
    survey = build_survey("bench_mem", fast=True)
    model = train(survey)
    selected = select_downlink(model, survey)
    state = MissionState("bench_mem")
    score(state, survey, model, selected)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def bench_act4() -> float:
    from meridian.act4_lander import build_conditions, train, evaluate, apply_to_state
    tracemalloc.start()
    state = MissionState("bench_mem")
    cond = build_conditions(state)
    policy, _ = train(cond, iterations=15, pop_size=20, warm_start=True)
    res = evaluate(policy, cond, n_episodes=50, seed=42)
    apply_to_state(state, res)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / (1024 * 1024)


def main():
    print("=" * 60)
    print("PEAK MEMORY BENCHMARK (FAST Mode, Budget < 1536 MB / 1.5 GB)")
    print("=" * 60)

    acts = [
        ("Act 1: Exoplanet", bench_act1),
        ("Act 2: Telemetry", bench_act2),
        ("Act 3: Galaxy", bench_act3),
        ("Act 4: Lander", bench_act4),
    ]

    for name, func in acts:
        mem_mb = func()
        status = "PASS" if mem_mb < 1536.0 else "FAIL"
        print(f"{name:<20}: {mem_mb:8.2f} MB  [{status}]")

    for p in Path(".").glob("mission_state_bench_mem.json"):
        p.unlink()

    print("=" * 60)


if __name__ == "__main__":
    main()
