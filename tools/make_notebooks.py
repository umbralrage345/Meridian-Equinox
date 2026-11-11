"""Generator script for Project Meridian workshop Jupyter notebooks using nbformat."""

from __future__ import annotations

import argparse
from pathlib import Path
import nbformat as nbf


SETUP_CODE = """# Setup and package installation
REPO_URL = "https://github.com/placeholder/meridian"

import os
import subprocess
import sys
from pathlib import Path

# Enable FAST mode by default to respect compute budgets (set to "0" to disable)
os.environ.setdefault("MERIDIAN_FAST", "1")

try:
    import meridian
except ImportError:
    if "google.colab" in sys.modules:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"git+{REPO_URL}"], check=True)
        import meridian
    else:
        repo_dir = Path.cwd()
        while repo_dir.parent != repo_dir and not (repo_dir / "meridian").exists():
            repo_dir = repo_dir.parent
        if (repo_dir / "meridian").exists() and str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        import meridian

fast_str = "ON" if os.environ.get("MERIDIAN_FAST") == "1" else "OFF"
backend_str = "torch" if getattr(meridian, "HAS_TORCH", False) else "sklearn"
print(f"Meridian v{meridian.__version__} | FAST mode: {fast_str} | Act 3 backend: {backend_str}")
"""

STATE_LOAD_CODE = """import os
TEAM_ID = os.environ.get("MERIDIAN_TEAM_ID", "your-team-name")

from meridian.state import load_state
from meridian.ui import briefing, dashboard, debrief
state = load_state(TEAM_ID)
"""


def _make_nb(cells: list) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "language_info": {"name": "python", "version": "3.12"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    }
    return nb


def create_00_start() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Project Meridian: Mission Orientation\n\n"
            "Welcome to Project Meridian! You are in command of an autonomous interstellar probe navigating across four distinct operational phases. "
            "Each act uses machine learning to solve a real-world mission challenge, from astronomical discovery to autonomous landing. "
            "Decisions made in early acts carry physical consequences into subsequent flight envelopes."
        ),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(
            "# Set your team identifier here\n"
            "TEAM_ID = \"team-alpha\"\n\n"
            "from meridian.state import load_state\n"
            "from meridian.ui import briefing, dashboard, debrief\n"
            "state = load_state(TEAM_ID)"
        ),
        nbf.v4.new_code_cell("briefing(0)"),
        nbf.v4.new_markdown_cell(
            "### Pre-Flight Telemetry Check\n\n"
            "Before departing, let us check probe subsystems and register telemetry communication channels. "
            "Our onboard computer monitors thruster fuel, hull integrity, and sensor noise floor. "
            "These resources will be depleted or degraded based on your flight performance in upcoming acts."
        ),
        nbf.v4.new_code_cell(
            "# Initial probe diagnostics check\n"
            "print('--- PROBE MERIDIAN PRE-FLIGHT SYSTEM CHECK ---')\n"
            "print(f'Team Identifier   : {TEAM_ID}')\n"
            "print(f'Propulsion Fuel   : {state.resources.fuel_pct:.1f}%')\n"
            "print(f'Hull Integrity    : {state.resources.hull_health:.1f}%')\n"
            "print(f'Sensor Noise Floor: σ = {state.resources.sensor_noise:.3f}')\n"
            "print('Status: All telemetry transceivers nominal.')"
        ),
        nbf.v4.new_code_cell(
            "# Initial mission dashboard\n"
            "dashboard(state)"
        ),
        nbf.v4.new_code_cell("debrief(state, 0)"),
        nbf.v4.new_code_cell(
            "state.save()\n"
            "print(f'Mission state saved to mission_state_{TEAM_ID}.json')"
        ),
    ]
    return _make_nb(cells)


def create_01_destination() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Act 1: Destination Selection via Exoplanet Transit Photometry\n\n"
            "Exoplanets are detected when they orbit in front of their host star, causing periodic dips in observed brightness called transits. "
            "We use the Box Least Squares (BLS) algorithm to scan for candidate orbital periods and durations across the light curve. "
            "A Random Forest classifier then examines folded transit shape features to distinguish genuine exoplanets from stellar noise. "
            "Tuning tree depth and detection thresholds balances false alarms against missed planetary targets."
        ),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(STATE_LOAD_CODE),
        nbf.v4.new_code_cell("briefing(1)"),
        nbf.v4.new_markdown_cell(
            "### Building the Stellar Survey\n\n"
            "We load training light curves and a catalog of candidate stars for our probe to scout. "
            "Each curve records brightness over a 27-day observation window. "
            "The survey extracts 46 transit features including BLS power spectra and phase-folded flux profiles."
        ),
        nbf.v4.new_code_cell(
            "from meridian.act1_exoplanet import (\n"
            "    build_survey, train, rank_candidates, choose_target,\n"
            "    plot_candidates, plot_lightcurve, human_vs_ai, reveal\n"
            ")\n"
            "survey = build_survey(TEAM_ID)\n"
            "print(f'Survey ready: {len(survey.candidate_star_ids)} candidates to evaluate.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Human vs AI Warm-up\n\n"
            "Can human eyes detect faint transit dips faster than an automated algorithm? "
            "Inspect the candidate curves below, make your prediction, and compare against baseline machine learning predictions. "
            "Notice how stellar noise and flares obscure true planetary signals."
        ),
        nbf.v4.new_code_cell(
            "# Inspect 4 candidate light curves\n"
            "sample_ids = human_vs_ai(survey, n=4)"
        ),
        nbf.v4.new_code_cell(
            "# YOUR GUESSES: Replace None with 1 (planet transit) or 0 (stellar variability/noise)\n"
            "# e.g. guesses = [1, 0, 1, 0]. Unedited run will treat None as skipped.\n"
            "guesses = [None, None, None, None]\n"
            "reveal(guesses=guesses, survey=survey, star_ids=sample_ids)"
        ),
        nbf.v4.new_markdown_cell(
            "### Inspecting Feature Extraction\n\n"
            "The Box Least Squares (BLS) algorithm fits a box-shaped transit model across thousands of trial periods. "
            "The peak in the periodogram reveals the most probable orbital period. "
            "Folding the time-series at this period stacks multiple transits together, boosting signal-to-noise."
        ),
        nbf.v4.new_code_cell(
            "import matplotlib.pyplot as plt\n"
            "from meridian.act1_exoplanet.generator import TIME_GRID\n"
            "plot_lightcurve(TIME_GRID, survey.candidate_fluxes[0], title=f'Candidate Star {survey.candidate_star_ids[0]}')\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### YOUR TURN: Tune the Transit Classifier\n\n"
            "Adjust your Random Forest hyperparameters and detection probability threshold below. "
            "A lower threshold flags more candidate planets but risks selecting a false positive star. "
            "Your highest-ranked candidate will become Meridian's official destination!"
        ),
        nbf.v4.new_code_cell(
            "# YOUR TURN: Adjust Act 1 Random Forest hyperparameters\n"
            "knobs = {\n"
            "    \"n_estimators\": 300,\n"
            "    \"max_depth\": None,\n"
            "    \"feature_set\": \"full\",\n"
            "    \"threshold\": 0.5,\n"
            "}"
        ),
        nbf.v4.new_code_cell(
            "# Train classifier and select primary exoplanet target\n"
            "model = train(survey, **knobs)\n"
            "ranked = rank_candidates(model, survey)\n"
            "target_star_id = ranked.iloc[0]['star_id']\n"
            "state = choose_target(state, survey, model, target_star_id)\n"
            "print(f'Target Selected: {target_star_id} | Score: {state.act1.score:.1f}/200 | AUC: {state.act1.auc:.3f}')"
        ),
        nbf.v4.new_code_cell(
            "# Display top candidate rankings\n"
            "plot_candidates(ranked, top_n=10)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell("debrief(state, 1)"),
        nbf.v4.new_code_cell(
            "state.save()\n"
            "dashboard(state)"
        ),
    ]
    return _make_nb(cells)


def create_02_cruise_watch() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Act 2: Cruise Watch Telemetry Anomaly Detection\n\n"
            "Spacecraft telemetry streams experience sudden equipment faults like stuck sensors, voltage drifts, and noise bursts. "
            "We extract sliding window statistics across multiple sensors and train Isolation Forests alongside neural Autoencoders to learn normal operational behavior. "
            "When observed signals deviate strongly from the learned baseline, the detector triggers an anomaly alarm. "
            "Tuning window size and detection persistence prevents spurious false alarms while catching dangerous faults quickly."
        ),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(STATE_LOAD_CODE),
        nbf.v4.new_code_cell("briefing(2)"),
        nbf.v4.new_markdown_cell(
            "### Generating Cruise Telemetry Stream\n\n"
            "The length of the cruise stream depends directly on the orbital period selected in Act 1. "
            "Four vital probe subsystems are monitored: bus voltage, battery temperature, reaction wheel speed, and solar current. "
            "We simulate unexpected hardware faults injected at unknown intervals along the flight path."
        ),
        nbf.v4.new_code_cell(
            "from meridian.act2_telemetry import make_stream, train, run, apply_to_state, plot_result, human_vs_ai, reveal\n"
            "trip_days = state.act1.target.trip_days if state.act1.done else 100\n"
            "stream = make_stream(TEAM_ID, trip_days=trip_days)\n"
            "print(f'Cruise telemetry stream generated: {len(stream.mission_data[\"bus_voltage\"])} steps, {len(stream.faults)} injected faults.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Human vs AI Warm-up\n\n"
            "Can you visually spot subtle equipment malfunctions within high-frequency sensor readings? "
            "Examine three segments from the probe telemetry stream below. "
            "One segment contains a genuine sensor fault while the other two display normal operating noise."
        ),
        nbf.v4.new_code_cell(
            "# Spot the faulty sensor stream segment\n"
            "challenge = human_vs_ai(stream)"
        ),
        nbf.v4.new_code_cell(
            "# YOUR GUESS: Choose 'A', 'B', or 'C' (e.g. guesses = ['A']).\n"
            "# Unedited run with None will be treated as skipped.\n"
            "guesses = [None]\n"
            "reveal(guesses=guesses, challenge=challenge)"
        ),
        nbf.v4.new_markdown_cell(
            "### YOUR TURN: Configure Anomaly Detection\n\n"
            "Select your anomaly detection model ('iforest', 'autoencoder', or 'both') and adjust window size and threshold. "
            "Higher threshold percentiles reduce false alarms but may delay catching slow drifts. "
            "Persistence requires consecutive anomalous windows before firing an alarm to filter out isolated noise spikes."
        ),
        nbf.v4.new_code_cell(
            "# YOUR TURN: Adjust Act 2 Anomaly Detection hyperparameters\n"
            "knobs = {\n"
            "    \"detector\": \"both\",\n"
            "    \"window\": 64,\n"
            "    \"threshold_q\": 99.0,\n"
            "    \"persistence\": 3,\n"
            "}"
        ),
        nbf.v4.new_code_cell(
            "# Train detector and evaluate autonomous anomaly detection on mission telemetry\n"
            "detector = train(stream, **knobs)\n"
            "result = run(detector, stream)\n"
            "apply_to_state(state, result)\n"
            "print(f'Act 2 Score: {result.score:.1f}/200 | Caught: {result.caught}/{result.n_faults} | False Alarms: {result.false_alarms}')"
        ),
        nbf.v4.new_code_cell(
            "# Plot detected alarms and actual faults across all 4 channels\n"
            "import matplotlib.pyplot as plt\n"
            "plot_result(stream, result)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell("debrief(state, 2)"),
        nbf.v4.new_code_cell(
            "state.save()\n"
            "dashboard(state)"
        ),
    ]
    return _make_nb(cells)


def create_03_deep_field() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Act 3: Deep Field Galaxy Classification & Downlink Optimization\n\n"
            "Distant galaxy surveys reveal diverse morphologies, including smooth ellipticals, spiral disks, edge-on dust lanes, and colliding mergers. "
            "Deep convolutional neural networks learn spatial filters that identify structural features like spiral arms and tidal tails directly from pixel arrays. "
            "Because downlink bandwidth is limited to 20 images, we must prioritize downlinking the most scientifically valuable galaxies. "
            "We balance classification probability against scientific utility weights to maximize discovery value."
        ),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(STATE_LOAD_CODE),
        nbf.v4.new_code_cell("briefing(3)"),
        nbf.v4.new_markdown_cell(
            "### Loading the Sky Survey\n\n"
            "The survey contains 48x48 pixel images categorized into 4 astronomical classes. "
            "Merger systems and spiral galaxies carry the highest scientific value weights for astrophysics research. "
            "Your model must accurately classify morphology despite faint signal levels and read noise."
        ),
        nbf.v4.new_code_cell(
            "from meridian.act3_galaxy import (\n"
            "    build_survey, train, select_downlink, score,\n"
            "    show_samples, plot_confusion, show_downlink, human_vs_ai, reveal\n"
            ")\n"
            "survey = build_survey(TEAM_ID)\n"
            "print(f'Galaxy survey loaded: {len(survey.train_images)} training images, {len(survey.stream_images)} sky stream images.')"
        ),
        nbf.v4.new_markdown_cell(
            "### Human vs AI Warm-up\n\n"
            "Can you visually distinguish between edge-on disks, smooth ellipticals, and colliding merger galaxies? "
            "Test your morphological classification skills on the four unlabeled sample images below. "
            "Automated classifiers evaluate subtle pixel gradients and asymmetry metrics."
        ),
        nbf.v4.new_code_cell(
            "# Challenge: Classify 4 galaxy morphology samples\n"
            "challenge = human_vs_ai(survey, n=4)"
        ),
        nbf.v4.new_code_cell(
            "# YOUR GUESSES: Replace None with class indices (0: Elliptical, 1: Spiral, 2: Edge-on, 3: Merger).\n"
            "# e.g. guesses = [0, 1, 2, 3]. Unedited run will treat None as skipped.\n"
            "guesses = [None, None, None, None]\n"
            "reveal(guesses=guesses, challenge=challenge)"
        ),
        nbf.v4.new_markdown_cell(
            "### Inspecting Morphological Classes\n\n"
            "Let us inspect reference examples from each morphological class in the training set. "
            "Notice how spirals display curved arm structures while mergers exhibit irregular asymmetric tidal debris. "
            "These spatial patterns guide the convolutional network during training."
        ),
        nbf.v4.new_code_cell(
            "# Inspect sample galaxy morphologies\n"
            "import matplotlib.pyplot as plt\n"
            "show_samples(survey, n_per_class=3)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### YOUR TURN: Train Classifier and Select Downlink Strategy\n\n"
            "Configure training epochs, data augmentation, and downlink selection strategy. "
            "The 'expected_value' strategy weights class probabilities by scientific value (mergers=5, spirals=3). "
            "Data augmentation rotates and flips training images to improve generalization."
        ),
        nbf.v4.new_code_cell(
            "# YOUR TURN: Adjust Act 3 Galaxy Classifier and Downlink hyperparameters\n"
            "knobs = {\n"
            "    \"epochs\": 5,\n"
            "    \"augment\": True,\n"
            "    \"class_weighting\": True,\n"
            "    \"strategy\": \"expected_value\",\n"
            "}"
        ),
        nbf.v4.new_code_cell(
            "# Train image classifier and optimize scientific downlink\n"
            "model = train(survey, **knobs)\n"
            "selected = select_downlink(model, survey, strategy=knobs[\"strategy\"])\n"
            "act3_score = score(state, survey, model, selected)\n"
            "print(f'Act 3 Score: {state.act3.score:.1f}/200 | Accuracy: {state.act3.accuracy:.1%} | Value Ratio: {state.act3.value_ratio:.1%}')"
        ),
        nbf.v4.new_code_cell(
            "# Display confusion matrix and prioritized downlink stream\n"
            "plot_confusion(model, survey)\n"
            "plt.show()\n"
            "show_downlink(survey, selected, model)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell("debrief(state, 3)"),
        nbf.v4.new_code_cell(
            "state.save()\n"
            "dashboard(state)"
        ),
    ]
    return _make_nb(cells)


def create_04_descent() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Act 4: Planetary Descent & Soft Landing\n\n"
            "Landing an autonomous probe requires continuous attitude control and rocket throttle regulation under unknown gravity and turbulent crosswinds. "
            "Because simulated physics engines are non-differentiable, we use Evolution Strategies (ES) to optimize a neural flight policy. "
            "ES perturbs candidate policy weights, evaluates their landing performance in parallel environments, and updates the network towards higher rewards. "
            "Fine-tuning population size, mutation noise, and learning rate steers the lander towards stable, fuel-efficient touchdowns."
        ),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(STATE_LOAD_CODE),
        nbf.v4.new_code_cell("briefing(4)"),
        nbf.v4.new_markdown_cell(
            "### Physical Flight Envelope\n\n"
            "The physical conditions of this landing reflect your mission history across earlier acts. "
            "Surface gravity and lateral wind were fixed by the exoplanet discovered in Act 1. "
            "Available thruster fuel and sensor noise reflect your anomaly handling during the Act 2 cruise."
        ),
        nbf.v4.new_code_cell(
            "from meridian.act4_lander import (\n"
            "    build_conditions, train, evaluate, apply_to_state,\n"
            "    plot_learning_curve, human_vs_ai, reveal, record_episode\n"
            ")\n"
            "from meridian import seeding\n"
            "conditions = build_conditions(state)\n"
            "print(f'Descent Conditions: Gravity={conditions[\"gravity\"]:.2f} m/s², Wind={conditions[\"wind\"]:.2f} m/s², Fuel={conditions[\"fuel_pct\"]:.1f}%, Noise=σ {conditions[\"sensor_noise\"]:.3f}')"
        ),
        nbf.v4.new_markdown_cell(
            "### Human vs AI Warm-up\n\n"
            "How does an unguided free-fall descent compare against an active flight control algorithm? "
            "We benchmark an unguided baseline against our hand-coded heuristic proportional-derivative (PD) controller. "
            "Notice how active feedback regulation stabilizes attitude and slows vertical touchdown velocity."
        ),
        nbf.v4.new_code_cell(
            "# Benchmark PD controller and free-fall across sample winds\n"
            "challenge = human_vs_ai(conditions)"
        ),
        nbf.v4.new_code_cell(
            "# YOUR GUESSES: Predict success (1) or crash (0) for the 3 test flight gusts.\n"
            "# e.g. guesses = [1, 0, 1]. Unedited run will treat None as skipped.\n"
            "guesses = [None, None, None]\n"
            "reveal(guesses=guesses, challenge=challenge)"
        ),
        nbf.v4.new_markdown_cell(
            "### YOUR TURN: Train Neural Flight Policy with Evolution Strategies\n\n"
            "Adjust your Evolution Strategies optimization hyperparameters below. "
            "A larger population explores more candidate weight perturbations per generation but requires more CPU steps. "
            "Setting warm_start=True initializes policy weights from our pre-trained baseline checkpoint."
        ),
        nbf.v4.new_code_cell(
            "# YOUR TURN: Adjust Act 4 Evolution Strategies hyperparameters\n"
            "knobs = {\n"
            "    \"iterations\": 15,\n"
            "    \"pop_size\": 20,\n"
            "    \"sigma\": 0.08,\n"
            "    \"lr\": 0.04,\n"
            "    \"warm_start\": True,\n"
            "}"
        ),
        nbf.v4.new_code_cell(
            "# Train neural flight controller via Evolution Strategies\n"
            "policy, history = train(conditions, **knobs)\n"
            "eval_seed = seeding.seed_for(TEAM_ID, 'act4_eval')\n"
            "eval_res = evaluate(policy, conditions, n_episodes=100, seed=eval_seed)\n"
            "apply_to_state(state, eval_res)\n"
            "print(f'Act 4 Score: {state.act4.score:.1f}/400 | Landing Rate: {eval_res[\"landing_rate\"] * 100.0:.1f}% | Mean Reward: {eval_res[\"mean_reward\"]:.1f}')"
        ),
        nbf.v4.new_code_cell(
            "# Plot ES training learning curve\n"
            "import matplotlib.pyplot as plt\n"
            "plot_learning_curve(history)\n"
            "plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "### Descent Flight Animations\n\n"
            "Inspect your autonomous lander's flight trajectory under nominal and stress-tested Martian conditions. "
            "The lander pulses its main engine and side thrusters to arrest descent and touch down on the pad. "
            "Pass `compare_with_random=True` to render a side-by-side comparison against unguided descent."
        ),
        nbf.v4.new_code_cell(
            "# Display descent animations: (1) Nominal episode, (2) Martian high-wind stress test\n"
            "from IPython.display import HTML\n\n"
            "# 1. Best nominal flight episode\n"
            "html_best = record_episode(policy, conditions, seed=eval_seed)\n"
            "display(HTML(html_best))\n\n"
            "# 2. Mars high-wind stress test (low gravity, turbulent crosswind)\n"
            "mars_cond = {\"gravity\": 3.7, \"wind\": 5.0, \"fuel_pct\": conditions.get(\"fuel_pct\", 100.0), \"sensor_noise\": conditions.get(\"sensor_noise\", 0.0)}\n"
            "html_mars = record_episode(policy, mars_cond, seed=eval_seed + 1)\n"
            "display(HTML(html_mars))"
        ),
        nbf.v4.new_code_cell("debrief(state, 4)"),
        nbf.v4.new_code_cell(
            "state.save()\n"
            "dashboard(state)"
        ),
    ]
    return _make_nb(cells)


def create_05_finale() -> nbf.NotebookNode:
    cells = [
        nbf.v4.new_markdown_cell(
            "# Mission Finale: Expedition Review & Official Certification\n\n"
            "Congratulations on completing the Meridian mission! Your probe has scouted an exoplanet, survived interplanetary cruise, surveyed distant galaxies, and touched down on alien soil. "
            "The final mission report visualizes your complete trajectory and cross-act coupling effects. "
            "Export your cryptographic score certificate to record your standing on the official workshop leaderboard."
        ),
        nbf.v4.new_code_cell(SETUP_CODE),
        nbf.v4.new_code_cell(STATE_LOAD_CODE),
        nbf.v4.new_code_cell("briefing(5)"),
        nbf.v4.new_markdown_cell(
            "### Mission Status Console\n\n"
            "Here is the comprehensive status of Probe Meridian at mission conclusion. "
            "All four act scores have been tallied toward your team's total cumulative score. "
            "Final propulsion fuel and hull integrity demonstrate overall spacecraft flight discipline."
        ),
        nbf.v4.new_code_cell(
            "# Final mission dashboard\n"
            "dashboard(state)"
        ),
        nbf.v4.new_code_cell(
            "# Auto Mission Report: 4-act timeline, scores, and coupling analysis\n"
            "import matplotlib.pyplot as plt\n"
            "from meridian.ui import mission_report\n"
            "fig = mission_report(state)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "# Export official cryptographic submission code\n"
            "from meridian.scoring import export_code\n"
            "code = export_code(state)\n"
            "print('=' * 70)\n"
            "print('MISSION COMPLETE — OFFICIAL VERIFICATION CODE:')\n"
            "print(code)\n"
            "print('=' * 70)\n"
            "print('Copy and submit this code to your workshop instructor or leaderboard server.')"
        ),
        nbf.v4.new_code_cell("debrief(state, 5)"),
        nbf.v4.new_code_cell(
            "state.save()\n"
            "print('Final state archived successfully.')"
        ),
    ]
    return _make_nb(cells)


NOTEBOOK_FACTORIES = {
    "00_start.ipynb": create_00_start,
    "01_destination.ipynb": create_01_destination,
    "02_cruise_watch.ipynb": create_02_cruise_watch,
    "03_deep_field.ipynb": create_03_deep_field,
    "04_descent.ipynb": create_04_descent,
    "05_finale.ipynb": create_05_finale,
}


def make_notebooks(out_dir: str | Path = "notebooks") -> list[Path]:
    """Generate all Project Meridian workshop Jupyter notebooks using nbformat."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []

    for filename, factory in NOTEBOOK_FACTORIES.items():
        nb = factory()
        out_path = output_dir / filename
        with open(out_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        generated_paths.append(out_path)

    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Project Meridian workshop Jupyter notebooks.")
    parser.add_argument("--out-dir", default="notebooks", help="Target output directory for notebooks.")
    args = parser.parse_args()

    paths = make_notebooks(args.out_dir)
    print(f"Generated {len(paths)} notebooks in {args.out_dir}:")
    for p in paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
