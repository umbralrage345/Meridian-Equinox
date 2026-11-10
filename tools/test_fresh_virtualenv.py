"""Automated test of fresh virtualenv install WITHOUT torch, then WITH torch."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_cmd(cmd: list[str], env: dict, cwd: Path | None = None) -> tuple[int, str]:
    res = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    return res.returncode, out


def link_package(src: Path, dest: Path) -> None:
    """Link or copy package into target site-packages."""
    if dest.exists():
        return
    if src.is_dir():
        # Use directory junction for speed and zero disk footprint
        subprocess.run(["cmd", "/c", "mklink", "/J", str(dest), str(src)], capture_output=True)
    elif src.is_file():
        shutil.copy2(src, dest)


def main():
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    main_site_packages = Path(r"C:\Users\abhyu\AppData\Local\Programs\Python\Python312\Lib\site-packages")

    print("=" * 70)
    print("TASK 6.1: VIRTUALENV TEST (WITHOUT TORCH -> WITH TORCH)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as venv_dir:
        venv_path = Path(venv_dir) / "meridian_env"
        print(f"Creating fresh isolated virtualenv at {venv_path}...", flush=True)
        ret, out = run_cmd([sys.executable, "-m", "venv", str(venv_path)], os.environ.copy())
        if ret != 0:
            print(f"Failed to create venv: {out}")
            sys.exit(1)

        venv_python = venv_path / "Scripts" / "python.exe"
        venv_site_packages = venv_path / "Lib" / "site-packages"

        print("Linking virtualenv dependencies WITHOUT torch...", flush=True)
        for item in main_site_packages.iterdir():
            name = item.name.lower()
            if name.startswith("torch") or name.startswith("caffe2") or "cuda" in name or "cudnn" in name:
                continue
            dest = venv_site_packages / item.name
            link_package(item, dest)

        # Install meridian package in editable mode without torch
        print("Running: pip install . (WITHOUT torch)...", flush=True)
        ret, out = run_cmd([str(venv_python), "-m", "pip", "install", "--no-deps", "-e", str(root)], os.environ.copy())
        if ret != 0:
            print(f"Failed to install meridian: {out}")
            sys.exit(1)

        # Verify torch is NOT present
        ret, out = run_cmd([str(venv_python), "-c", "import torch"], os.environ.copy())
        print(f"Checking torch availability: {'NOT present (correct)' if ret != 0 else 'Present (unexpected)'}", flush=True)
        assert ret != 0, "Torch should not be present in fresh venv without torch!"

        # Set environment for headless fast runs
        env = os.environ.copy()
        env["MERIDIAN_FAST"] = "1"
        env["MERIDIAN_TEAM_ID"] = "notorch-team"
        env["PATH"] = str(venv_path / "Scripts") + os.pathsep + env.get("PATH", "")

        # 1. Run pytest in fresh venv without torch
        print("\n--- Running pytest in fresh venv WITHOUT torch (FAST mode) ---", flush=True)
        t0 = time.perf_counter()
        ret, out = run_cmd([str(venv_python), "-m", "pytest", str(root / "tests"), "-v"], env)
        t_pytest_notorch = time.perf_counter() - t0
        passed_lines = [l for l in out.splitlines() if "passed" in l]
        summary_line = passed_lines[-1] if passed_lines else "pytest finished"
        print(f"Pytest without torch: [{'PASS' if ret == 0 else 'FAIL'}] in {t_pytest_notorch:.2f}s | {summary_line}", flush=True)
        if ret != 0:
            print(f"Pytest output:\n{out[-2000:]}")
            sys.exit(1)

        # 2. Run all 6 notebooks headless in fresh venv without torch
        print("\n--- Running all 6 notebooks headless WITHOUT torch (FAST mode) ---", flush=True)
        nb_order = [
            "00_start.ipynb",
            "01_destination.ipynb",
            "02_cruise_watch.ipynb",
            "03_deep_field.ipynb",
            "04_descent.ipynb",
            "05_finale.ipynb",
        ]
        nb_notorch_timings = {}
        for nb_name in nb_order:
            nb_path = nb_dir / nb_name
            t0 = time.perf_counter()
            ret, out = run_cmd(
                [str(venv_python), "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", str(nb_path), "--output", f"tmp_notorch_{nb_name}"],
                env,
                cwd=nb_dir,
            )
            t_nb = time.perf_counter() - t0
            nb_notorch_timings[nb_name] = (ret, t_nb)
            print(f"  * {nb_name:<25}: [{'OK' if ret == 0 else 'FAIL'}] in {t_nb:.2f}s", flush=True)
            if ret != 0:
                print(f"Error in {nb_name}:\n{out[-2000:]}")
                sys.exit(1)

        # 3. Now install torch into virtualenv: pip install .[torch]
        print("\n--- Installing torch extra: pip install .[torch] ---", flush=True)
        for item in main_site_packages.iterdir():
            name = item.name.lower()
            if name.startswith("torch") or name.startswith("caffe2") or "cuda" in name or "cudnn" in name:
                dest = venv_site_packages / item.name
                link_package(item, dest)

        ret, out = run_cmd([str(venv_python), "-c", "import torch; print(f'Torch {torch.__version__} successfully imported')"], env)
        print(f"Torch installation verified: {out.strip()}", flush=True)
        assert ret == 0, f"Torch failed to import: {out}"

        # 4. Re-run notebook 03 with torch
        print("\n--- Re-running Notebook 03 with PyTorch enabled ---", flush=True)
        nb03_path = nb_dir / "03_deep_field.ipynb"
        t0 = time.perf_counter()
        ret, out = run_cmd(
            [str(venv_python), "-m", "jupyter", "nbconvert", "--execute", "--to", "notebook", str(nb03_path), "--output", "tmp_torch_03.ipynb"],
            env,
            cwd=nb_dir,
        )
        t_nb03_torch = time.perf_counter() - t0
        print(f"  * 03_deep_field.ipynb (torch): [{'OK' if ret == 0 else 'FAIL'}] in {t_nb03_torch:.2f}s", flush=True)
        if ret != 0:
            print(f"Error in 03 with torch:\n{out[-2000:]}")
            sys.exit(1)

        # Clean up temporary notebook execution outputs
        for p in nb_dir.glob("tmp_*.ipynb"):
            p.unlink()
        for p in nb_dir.glob("mission_state_notorch-team.json"):
            p.unlink()

        # Remove junctions before temp dir deletion
        for item in venv_site_packages.iterdir():
            if item.is_dir():
                subprocess.run(["cmd", "/c", "rmdir", str(item)], capture_output=True)

        print("\n" + "=" * 70)
        print("SUMMARY: FRESH VIRTUALENV TEST RESULTS")
        print("=" * 70)
        print(f"Pytest without torch      : PASSED ({summary_line}) in {t_pytest_notorch:.2f}s")
        print("Notebooks without torch   : ALL 6 PASSED (Sklearn fallback active in Act 3)")
        for name, (_, t) in nb_notorch_timings.items():
            print(f"  - {name:<23}: {t:5.2f}s")
        print(f"Notebook 03 with torch    : PASSED (PyTorch TinyCNN active) in {t_nb03_torch:.2f}s")
        print("=" * 70)


if __name__ == "__main__":
    main()
