"""Acceptance test runner for Project Meridian workshop notebooks."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_notebook(nb_path: Path, env: dict) -> tuple[float, int, str]:
    """Execute a single notebook with nbconvert and measure runtime and capture output."""
    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--execute",
        "--to",
        "notebook",
        str(nb_path),
        "--output",
        f"executed_{nb_path.name}",
    ]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, cwd=nb_path.parent, capture_output=True, text=True)
    t1 = time.perf_counter()
    runtime = t1 - t0
    output = res.stdout + "\n" + res.stderr
    return runtime, res.returncode, output


def main():
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"

    env = os.environ.copy()
    env["MERIDIAN_FAST"] = "1"
    env["MERIDIAN_TEAM_ID"] = "test-team"
    if str(root) not in env.get("PYTHONPATH", ""):
        env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")

    # Clean any existing test-team state in root or notebooks
    for p in root.glob("mission_state_test-team.json"):
        p.unlink()
    for p in nb_dir.glob("mission_state_test-team.json"):
        p.unlink()

    notebooks_order = [
        "00_start.ipynb",
        "01_destination.ipynb",
        "02_cruise_watch.ipynb",
        "03_deep_field.ipynb",
        "04_descent.ipynb",
        "05_finale.ipynb",
    ]

    print("=" * 70)
    print("STAGE 1: Sequential Execution of all 6 Notebooks (MERIDIAN_FAST=1)")
    print("=" * 70)

    seq_results = {}
    for nb_name in notebooks_order:
        nb_file = nb_dir / nb_name
        print(f"Running {nb_name}...", flush=True)
        runtime, retcode, out = run_notebook(nb_file, env)
        warnings = [line for line in out.splitlines() if "warning" in line.lower() or "warn" in line.lower()]
        seq_results[nb_name] = {
            "runtime": runtime,
            "returncode": retcode,
            "warnings": warnings,
            "output": out,
        }
        status = "PASSED" if retcode == 0 else "FAILED"
        print(f"  -> {status} in {runtime:.2f}s ({len(warnings)} warnings)", flush=True)
        if retcode != 0:
            print(f"ERROR OUTPUT:\n{out[-1000:]}")
            sys.exit(1)

    print("\n" + "=" * 70)
    print("STAGE 2: Standalone Execution in Fresh Empty Directories")
    print("=" * 70)

    standalone_results = {}
    for nb_name in ["03_deep_field.ipynb", "04_descent.ipynb"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            target_nb = tmp_path / nb_name
            shutil.copy(nb_dir / nb_name, target_nb)

            print(f"Running standalone {nb_name} in fresh empty directory {tmp_path.name}...", flush=True)
            runtime, retcode, out = run_notebook(target_nb, env)
            warnings = [line for line in out.splitlines() if "warning" in line.lower() or "warn" in line.lower()]
            standalone_results[nb_name] = {
                "runtime": runtime,
                "returncode": retcode,
                "warnings": warnings,
            }
            status = "PASSED" if retcode == 0 else "FAILED"
            print(f"  -> {status} in {runtime:.2f}s ({len(warnings)} warnings)", flush=True)
            if retcode != 0:
                print(f"ERROR OUTPUT:\n{out[-1000:]}")
                sys.exit(1)

    # Clean up generated executed_*.ipynb files in notebooks dir
    for f in nb_dir.glob("executed_*.ipynb"):
        f.unlink()

    print("\n" + "=" * 70)
    print("FINAL ACCEPTANCE SUMMARY:")
    print("=" * 70)
    print(f"{'Notebook':<25} | {'Mode':<12} | {'Time (s)':<10} | {'Status':<8} | Warnings")
    print("-" * 70)
    for nb_name in notebooks_order:
        r = seq_results[nb_name]
        w_count = len(r["warnings"])
        print(f"{nb_name:<25} | {'Sequential':<12} | {r['runtime']:<10.2f} | {'OK' if r['returncode'] == 0 else 'FAIL':<8} | {w_count} warnings")

    for nb_name, r in standalone_results.items():
        w_count = len(r["warnings"])
        print(f"{nb_name:<25} | {'Standalone':<12} | {r['runtime']:<10.2f} | {'OK' if r['returncode'] == 0 else 'FAIL':<8} | {w_count} warnings")

    print("=" * 70)


if __name__ == "__main__":
    main()
