"""Benchmark peak memory usage (process + child RSS) sampled every 100 ms via psutil."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

# Set FAST mode
os.environ["MERIDIAN_FAST"] = "1"

import psutil

from meridian.state import MissionState


class MemoryMonitor:
    """Threaded sampler tracking peak RSS of process and all its children every 100 ms."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_bytes: int = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.proc = psutil.Process()

    def _sample_rss(self) -> int:
        try:
            total = self.proc.memory_info().rss
            for child in self.proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._sample_rss()
            if rss > self.peak_bytes:
                self.peak_bytes = rss
            time.sleep(self.interval)

    def __enter__(self) -> MemoryMonitor:
        self.peak_bytes = self._sample_rss()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.5)
        rss = self._sample_rss()
        if rss > self.peak_bytes:
            self.peak_bytes = rss

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024.0 * 1024.0)


def bench_act1() -> float:
    from meridian.act1_exoplanet import build_survey, train, rank_candidates, choose_target
    with MemoryMonitor() as mon:
        survey = build_survey("bench_mem", fast=True)
        model = train(survey)
        ranked = rank_candidates(model, survey)
        state = MissionState("bench_mem")
        choose_target(state, survey, model, ranked.iloc[0]["star_id"])
    return mon.peak_mb


def bench_act2() -> float:
    from meridian.act2_telemetry import make_stream, train, run, apply_to_state
    with MemoryMonitor() as mon:
        stream = make_stream("bench_mem", trip_days=100, fast=True)
        det = train(stream)
        res = run(det, stream)
        state = MissionState("bench_mem")
        apply_to_state(state, res)
    return mon.peak_mb


def bench_act3() -> float:
    from meridian.act3_galaxy import build_survey, train, select_downlink, score
    with MemoryMonitor() as mon:
        survey = build_survey("bench_mem", fast=True)
        model = train(survey)
        selected = select_downlink(model, survey)
        state = MissionState("bench_mem")
        score(state, survey, model, selected)
    return mon.peak_mb


def bench_act4() -> float:
    from meridian.act4_lander import build_conditions, train, evaluate, apply_to_state
    with MemoryMonitor() as mon:
        state = MissionState("bench_mem")
        cond = build_conditions(state)
        policy, _ = train(cond, iterations=15, pop_size=20, warm_start=True)
        res = evaluate(policy, cond, n_episodes=50, seed=42)
        apply_to_state(state, res)
    return mon.peak_mb


def bench_notebook03() -> float:
    nb_path = Path("notebooks/03_deep_field.ipynb")
    if not nb_path.exists():
        return 0.0

    env = dict(os.environ)
    env["MERIDIAN_FAST"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        str(nb_path),
        "--output",
        "tmp_bench_nb03.ipynb",
    ]

    mon_peak = 0
    stop_event = threading.Event()

    def _monitor_proc(proc_obj: psutil.Process) -> None:
        nonlocal mon_peak
        while not stop_event.is_set():
            try:
                tot = proc_obj.memory_info().rss
                for ch in proc_obj.children(recursive=True):
                    try:
                        tot += ch.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if tot > mon_peak:
                    mon_peak = tot
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(0.1)

    p = subprocess.Popen(cmd, env=env)
    t = threading.Thread(target=_monitor_proc, args=(psutil.Process(p.pid),), daemon=True)
    t.start()
    p.wait()
    stop_event.set()
    t.join(timeout=1.0)

    tmp_out = Path("tmp_bench_nb03.ipynb")
    if tmp_out.exists():
        tmp_out.unlink()

    return mon_peak / (1024.0 * 1024.0)


def main() -> None:
    print("=" * 65)
    print("PEAK PROCESS RSS BENCHMARK (PSUTIL, 100ms SAMPLING, LIMIT <= 1.5 GB)")
    print("=" * 65)

    act1_mb = bench_act1()
    print(f"Act 1 (Exoplanet)   Peak RSS: {act1_mb:7.2f} MB [{'PASS' if act1_mb <= 1500 else 'FAIL'}]")

    act2_mb = bench_act2()
    print(f"Act 2 (Telemetry)   Peak RSS: {act2_mb:7.2f} MB [{'PASS' if act2_mb <= 1500 else 'FAIL'}]")

    act3_mb = bench_act3()
    print(f"Act 3 (Galaxy)      Peak RSS: {act3_mb:7.2f} MB [{'PASS' if act3_mb <= 1500 else 'FAIL'}]")

    act4_mb = bench_act4()
    print(f"Act 4 (Lander)      Peak RSS: {act4_mb:7.2f} MB [{'PASS' if act4_mb <= 1500 else 'FAIL'}]")

    nb03_mb = bench_notebook03()
    print(f"Notebook 03 (torch) Peak RSS: {nb03_mb:7.2f} MB [{'PASS' if nb03_mb <= 1500 else 'FAIL'}]")

    # Clean up state files
    for p in Path(".").glob("mission_state_bench_mem.json"):
        p.unlink()

    print("=" * 65)


if __name__ == "__main__":
    main()
