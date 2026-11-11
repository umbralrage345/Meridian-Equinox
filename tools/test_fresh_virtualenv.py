"""Automated test of true fresh virtualenvs from package index without junctions."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_cmd(cmd: list[str], env: dict, cwd: Path | None = None) -> tuple[int, str]:
    res = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    return res.returncode, out


def get_venv_executables(venv_dir: Path) -> tuple[Path, Path]:
    if sys.platform == "win32":
        python_bin = venv_dir / "Scripts" / "python.exe"
        pip_bin = venv_dir / "Scripts" / "pip.exe"
    else:
        python_bin = venv_dir / "bin" / "python"
        pip_bin = venv_dir / "bin" / "pip"
    return python_bin, pip_bin


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"

    print("=" * 75)
    print("TASK 7.2: TRUE CLEAN INSTALL VERIFICATION IN FRESH VENV")
    print("=" * 75)
    print(f"Base Python executable: {sys.executable} (version {sys.version.split()[0]})")
    print(f"Repository root        : {root}\n")

    # -------------------------------------------------------------------------
    # PART 1: Fresh venv WITHOUT torch
    # -------------------------------------------------------------------------
    print(">>> 1. Creating fresh isolated virtualenv WITHOUT torch...")
    with tempfile.TemporaryDirectory() as venv_dir_str:
        venv_path = Path(venv_dir_str) / "venv_no_torch"
        ret, out = run_cmd([sys.executable, "-m", "venv", str(venv_path)], os.environ.copy())
        if ret != 0:
            print(f"ERROR creating venv:\n{out}")
            sys.exit(1)

        v_python, _ = get_venv_executables(venv_path)

        print(f"    Installing meridian with dev dependencies: pip install .[dev] from index/cache...")
        t0_install = time.perf_counter()
        ret, out = run_cmd([str(v_python), "-m", "pip", "install", f"{root}[dev]"], os.environ.copy())
        t_install = time.perf_counter() - t0_install
        if ret != 0:
            print(f"ERROR installing package:\n{out}")
            sys.exit(1)
        print(f"    Install complete in {t_install:.2f}s.")

        # Ensure ipykernel is available for Jupyter nbconvert
        run_cmd([str(v_python), "-m", "ipykernel", "install", "--sys-prefix"], os.environ.copy())

        # Verify torch is NOT present
        ret, out = run_cmd([str(v_python), "-c", "import torch"], os.environ.copy())
        assert ret != 0, "PyTorch was unexpectedly found in non-torch venv!"
        print("    Verified: PyTorch is NOT installed in venv 1.")

        env = os.environ.copy()
        env["MERIDIAN_FAST"] = "1"
        env["MERIDIAN_TEAM_ID"] = "clean-test-team"

        # Run full pytest suite in fresh venv
        print("\n>>> Running full pytest suite in fresh venv WITHOUT torch...")
        t0_pytest = time.perf_counter()
        ret, out = run_cmd([str(v_python), "-m", "pytest", str(root / "tests"), "-v"], env, cwd=root)
        t_pytest = time.perf_counter() - t0_pytest
        if ret != 0:
            print(f"Pytest failed in fresh venv:\n{out}")
            sys.exit(1)
        passed_lines = [l for l in out.splitlines() if "passed" in l]
        summary_line = passed_lines[-1] if passed_lines else "all tests passed"
        print(f"    pytest result: PASS ({summary_line}) in {t_pytest:.2f}s")

        # Run all 6 notebooks headless
        print("\n>>> Running all 6 notebooks headless in fresh venv WITHOUT torch (FAST mode)...")
        notebooks = [
            "00_start.ipynb",
            "01_destination.ipynb",
            "02_cruise_watch.ipynb",
            "03_deep_field.ipynb",
            "04_descent.ipynb",
            "05_finale.ipynb",
        ]
        nb_timings = {}
        for nb_name in notebooks:
            nb_path = nb_dir / nb_name
            out_path = nb_dir / f"tmp_clean_{nb_name}"
            t0_nb = time.perf_counter()
            ret, out = run_cmd(
                [
                    str(v_python),
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--execute",
                    "--to",
                    "notebook",
                    str(nb_path),
                    "--output",
                    out_path.name,
                    "--ExecutePreprocessor.timeout=600",
                ],
                env,
                cwd=nb_dir,
            )
            t_nb = time.perf_counter() - t0_nb
            nb_timings[nb_name] = (ret == 0, t_nb)
            status = "PASS" if ret == 0 else "FAIL"
            print(f"    * {nb_name:<24}: [{status}] in {t_nb:5.2f}s")
            if out_path.exists():
                out_path.unlink()
            if ret != 0:
                print(f"Notebook {nb_name} failed:\n{out[-2500:]}")
                sys.exit(1)

        # Cleanup state
        for p in nb_dir.glob("mission_state_clean-test-team.json"):
            p.unlink()

    # -------------------------------------------------------------------------
    # PART 2: Second fresh venv WITH torch
    # -------------------------------------------------------------------------
    print("\n>>> 2. Creating SECOND fresh isolated virtualenv WITH torch...")
    print("    Standard Windows student command: pip install torch")
    with tempfile.TemporaryDirectory() as venv_dir_str_2:
        venv_path_2 = Path(venv_dir_str_2) / "venv_with_torch"
        ret, out = run_cmd([sys.executable, "-m", "venv", str(venv_path_2)], os.environ.copy())
        if ret != 0:
            print(f"ERROR creating venv 2:\n{out}")
            sys.exit(1)

        v_python_2, _ = get_venv_executables(venv_path_2)

        print(f"    Installing meridian with dev: pip install .[dev]...")
        run_cmd([str(v_python_2), "-m", "pip", "install", f"{root}[dev]"], os.environ.copy())

        # Exact command a student on Windows would run:
        student_torch_cmd = [str(v_python_2), "-m", "pip", "install", "torch"]
        print(f"    Running student command: {' '.join(student_torch_cmd)}...")
        t0_torch_install = time.perf_counter()
        ret, out = run_cmd(student_torch_cmd, os.environ.copy())
        t_torch_install = time.perf_counter() - t0_torch_install
        if ret != 0:
            print(f"ERROR installing torch:\n{out}")
            sys.exit(1)
        print(f"    Torch installation complete in {t_torch_install:.2f}s.")

        # Verify torch is present and check backend
        ret, out = run_cmd(
            [
                str(v_python_2),
                "-c",
                "import torch, meridian; print(f'Torch: {torch.__version__} | Act 3 HAS_TORCH={meridian.HAS_TORCH}')",
            ],
            os.environ.copy(),
        )
        print(f"    Environment verification: {out.strip()}")
        assert ret == 0, "Failed to import torch in venv 2!"

        env_2 = os.environ.copy()
        env_2["MERIDIAN_FAST"] = "1"
        env_2["MERIDIAN_TEAM_ID"] = "torch-test-team"

        # Re-run notebook 03 with torch
        print("\n>>> Re-running Notebook 03 in fresh venv WITH torch (FAST mode)...")
        nb03_path = nb_dir / "03_deep_field.ipynb"
        out_path = nb_dir / "tmp_torch_03.ipynb"
        t0_nb03 = time.perf_counter()
        ret, out = run_cmd(
            [
                str(v_python_2),
                "-m",
                "jupyter",
                "nbconvert",
                "--execute",
                "--to",
                "notebook",
                str(nb03_path),
                "--output",
                out_path.name,
                "--ExecutePreprocessor.timeout=600",
            ],
            env_2,
            cwd=nb_dir,
        )
        t_nb03 = time.perf_counter() - t0_nb03
        if out_path.exists():
            out_path.unlink()
        for p in nb_dir.glob("mission_state_torch-test-team.json"):
            p.unlink()

        status = "PASS" if ret == 0 else "FAIL"
        print(f"    * 03_deep_field.ipynb (torch): [{status}] in {t_nb03:5.2f}s")
        if ret != 0:
            print(f"Notebook 03 with torch failed:\n{out[-2500:]}")
            sys.exit(1)

    print("\n" + "=" * 75)
    print("TRUE CLEAN INSTALL VERIFICATION: ALL CHECKS PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()
