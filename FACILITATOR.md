# Project Meridian — Workshop Facilitator Guide

A concise operational guide for running the 2-hour Project Meridian simulation workshop.

---

## 1. Pre-Event Checklist (30 Minutes Before Kickoff)
- [ ] **Environment Verification**: On the presentation machine, execute:
  ```bash
  python -m meridian.selfcheck
  ```
  Ensure all four acts display `[PASS]` and total execution is under 60 seconds.
- [ ] **Distribution Prepared**: Ensure students have the GitHub repository URL or distributed zip archive.
- [ ] **Leaderboard Directory**: Confirm `tools/build_leaderboard.py` is executable and test generation:
  ```bash
  python tools/build_leaderboard.py
  ```
- [ ] **Hardware Requirements**: Confirm student laptops have standard Python 3.10–3.12, $\ge 4\text{ GB}$ available RAM, and Jupyter installed (or Google Colab access).

---

## 2. Workshop Timing Cheat Sheet (120 Minutes Total)

| Phase | Notebook | Time | Objective & ML Concept |
|---|---|---|---|
| **0. Orientation** | `00_start.ipynb` | 10 min | Team registration, pre-flight telemetry, probe diagnostics check. |
| **1. Target World** | `01_destination.ipynb` | 25 min | Exoplanet transit photometry, BLS periodograms, Random Forest tuning. |
| **2. Cruise Watch** | `02_cruise_watch.ipynb` | 20 min | Multichannel telemetry stream, Isolation Forest & Autoencoder anomaly detection. |
| **3. Deep Field** | `03_deep_field.ipynb` | 25 min | Galaxy morphology classification, CNN/MLP training, value-weighted downlink. |
| **4. Descent** | `04_descent.ipynb` | 25 min | Lunar lander physics, Evolution Strategies neural attitude and throttle control. |
| **5. Finale** | `05_finale.ipynb` | 15 min | Auto mission report, cryptographic verification code export, live leaderboard. |

---

## 3. How to Collect Submission Codes and Build the Leaderboard
1. **Student Code Export**: At the conclusion of `05_finale.ipynb`, students copy their exported verification code block:
   ```text
   MER1-eyJ0ZWFtX2lkIjoidGVhbV9hbHBoYSIs...-7f8a9b...
   ```
2. **Collect Codes**: Have students submit codes via a shared form, chat channel, or paste them into a text file named `codes.txt` (one code per line).
3. **Build Leaderboard**: Run the builder tool:
   ```bash
   python tools/build_leaderboard.py codes.txt --out-dir ./leaderboard
   ```
4. **Display**: Open `./leaderboard/leaderboard.html` in any web browser or project it onto the workshop screen. Tampered codes are automatically rejected; teams with multiple submissions are deduplicated keeping their highest total score.

---

## 4. Top 5 Things That Can Go Wrong & Immediate Fixes

### 1. Google Colab Disconnect / Session Timeout
- **Problem**: Student loses active Colab instance mid-session.
- **Fix**: All acts save progress to `mission_state_<team_id>.json`. Have students download their JSON file after each act. If reconnected, simply re-upload the JSON file or continue immediately—each act notebook works completely standalone even when prior acts are skipped.

### 2. `pip install` Dependency or Compiler Failure
- **Problem**: Missing Microsoft C++ Build Tools or wheel compilation errors.
- **Fix**: Project Meridian requires **no C++ compiler or native toolchain**. All dependencies (NumPy, SciPy, scikit-learn, Matplotlib, Pandas, Astropy) provide pre-compiled wheels. Run:
  ```bash
  pip install --only-binary :all: -e .
  ```

### 3. Student Laptop Running Slowly / High CPU Utilization
- **Problem**: Older 2-core laptop struggles with training iterations.
- **Fix**: Enable FAST mode in the terminal or notebook:
  ```bash
  # In terminal:
  set MERIDIAN_FAST=1      # Windows CMD
  $env:MERIDIAN_FAST="1"    # Windows PowerShell
  export MERIDIAN_FAST=1   # macOS / Linux
  ```
  Or add `%env MERIDIAN_FAST=1` at the top of the notebook. This reduces dataset sizes and iterations, dropping compute time under 4 minutes total.

### 4. PyTorch Missing or Incompatible Architecture
- **Problem**: Student does not have PyTorch or encounters CUDA driver version conflicts.
- **Fix**: PyTorch is strictly optional. Act 3 includes an automatic fallback to a CPU-optimized scikit-learn MLP classifier that runs identically and matches all baseline score thresholds. No action is required.

### 5. Lost or Corrupted State File
- **Problem**: Student accidentally deletes `mission_state_<team_id>.json` before Act 4 descent.
- **Fix**: Re-running `load_state(TEAM_ID)` automatically instantiates a default flight state ($g = 7.0\text{ m/s}^2, w = 2.0\text{ m/s}^2, \text{fuel} = 100\%$). The student can proceed directly without restarting earlier acts.
