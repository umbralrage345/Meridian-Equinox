"""Mission narrative briefings and debrief templates for Project Meridian."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meridian.state import MissionState

# Mission briefings (strictly <= 120 words each, mission-control tone)
BRIEFINGS: dict[int, str] = {
    0: (
        "MISSION CONTROL // FLIGHT DIRECTIVE: ACT 0 — SYSTEM INITIALIZATION.\n"
        "Probe Meridian has established high-gain telemetry with Earth ground stations. "
        "All primary instruments, reaction control thrusters, and onboard compute arrays are online. "
        "Your flight team must now confirm core configuration, verify state serialization, "
        "and establish your mission identifier. As Meridian embarks on this multi-stage journey, "
        "each operational choice will shape subsequent flight envelopes: astronomical targeting determines "
        "surface gravity and wind, cruise anomaly management preserves fuel and sensor fidelity, "
        "and galaxy analysis prioritizes deep space science. Calibrate your terminal, inspect probe diagnostics, "
        "and prepare for destination acquisition."
    ),
    1: (
        "MISSION CONTROL // FLIGHT DIRECTIVE: ACT 1 — TARGET ACQUISITION.\n"
        "Probe Meridian has cleared the solar termination shock and is surveying candidate exoplanetary systems. "
        "Flight dynamics requires selecting our primary landing destination. You have access to high-precision stellar "
        "photometric time-series. Transit dips are faint, buried within stellar pulsation harmonics and flare noise. "
        "Train your transit classification pipeline, evaluate candidate stars, and select our target exoplanet. "
        "Be advised: the transit depth dictates surface gravity, and the orbital period dictates transit wind shear "
        "and interplanetary cruise duration. Choose wisely. Humanity's first interstellar touchdown begins with your astronomical analysis."
    ),
    2: (
        "MISSION CONTROL // FLIGHT DIRECTIVE: ACT 2 — CRUISE WATCH.\n"
        "Meridian has executed orbital insertion into the cruise corridor towards your selected destination. "
        "Four critical subsystem streams—bus voltage, battery temperature, reaction wheel speed, and solar array "
        "current—are broadcasting continuously. Interplanetary radiation, thermal stress, and debris pose severe hazard profiles. "
        "Your task: configure autonomous anomaly detection to flag equipment faults before structural failure occurs. "
        "Catching faults early preserves probe integrity. However, false alarms trigger evasive RCS burns that waste precious thruster fuel, "
        "while undetected faults permanently degrade sensor precision. Keep watch over our vessel."
    ),
    3: (
        "MISSION CONTROL // FLIGHT DIRECTIVE: ACT 3 — DEEP FIELD SURVEY.\n"
        "Approaching the target star system, Meridian's optical survey array captures thousands of distant galaxies. "
        "Deep space bandwidth is severely constrained: our transmitter budget allows downlinking only 20 high-resolution "
        "images back to Earth Science Operations. You must train an image classifier to identify galaxy morphology across four "
        "structural classes: elliptical, spiral, edge-on disk, and colliding merger systems. Mergers and spirals hold the highest "
        "astrophysical value. Balance classification confidence against scientific reward to maximize total downlink yield. "
        "The scientific legacy of the Meridian expedition rests on your downlink curation."
    ),
    4: (
        "MISSION CONTROL // FLIGHT DIRECTIVE: ACT 4 — PLANETARY DESCENT.\n"
        "Terminal phase. Meridian has entered the upper atmosphere of your chosen world. "
        "Gravity and lateral crosswinds—governed by your Act 1 astronomical discovery—are actively pulling the craft off vector. "
        "Onboard thruster fuel and sensor precision reflect your cruise performance in Act 2. Light speed delay precludes manual Earth piloting; "
        "you must train an autonomous neural flight controller using Evolution Strategies. Regulate main engine deceleration, "
        "compensate for turbulent wind gusts, and bring the lander down softly within pad boundaries. "
        "Touchdown must be gentle and upright. Godspeed, flight team."
    ),
    5: (
        "MISSION CONTROL // FLIGHT DIRECTIVE: ACT 5 — MISSION EVALUATION.\n"
        "The Meridian expedition has concluded its primary flight objectives. From stellar transit discovery to automated descent "
        "across an alien sky, every subsystem was tested under autonomous flight control. Flight computers have collated "
        "end-to-end performance metrics, telemetry logs, and scientific return. Review your mission timeline and cross-act "
        "coupling consequences in the final report. Generate your cryptographically signed mission certificate and transmit it to flight directors "
        "for verified leaderboard entry. Outstanding service, engineers. Meridian stands as a testament to autonomous space exploration."
    ),
}


def briefing_text(act_no: int) -> str:
    """Return the plain-text mission briefing for the specified act."""
    return BRIEFINGS.get(act_no, f"MISSION CONTROL // DIRECTIVE FOR ACT {act_no}: Stand by for telemetry.")


def debrief_text(state: MissionState, act_no: int) -> str:
    """Return a plain-language summary of what occurred in the act and what changed for subsequent acts."""
    if act_no == 0:
        return (
            f"ACT 0 COMPLETE — Systems initialized for Team '{state.team_id}'. "
            f"Propulsion reserves: {state.resources.fuel_pct:.1f}%, "
            f"Hull integrity: {state.resources.hull_health:.1f}%, "
            f"Sensor noise: {state.resources.sensor_noise:.3f}."
        )

    if act_no == 1:
        target = state.act1.target
        real_status = "CONFIRMED EXOPLANET" if target.is_real_planet else "FALSE POSITIVE"
        return (
            f"ACT 1 DEBRIEF — Target Selected: {target.star_id} ({target.label}, {real_status}).\n"
            f"Act 1 Score: {state.act1.score:.1f} / 200 (ROC-AUC: {state.act1.auc:.3f}).\n"
            f"Cross-Act Consequences:\n"
            f"  • Transit depth {target.depth:.4f} => Surface gravity set to {target.gravity:.2f} m/s² for Act 4 descent.\n"
            f"  • Orbital period {target.period_days:.2f} d => Surface wind set to {target.wind:.2f} m/s² for Act 4 descent.\n"
            f"  • Interplanetary cruise duration set to {target.trip_days} days, defining the Act 2 telemetry stream length."
        )

    if act_no == 2:
        res = state.resources
        return (
            f"ACT 2 DEBRIEF — Cruise Telemetry Monitoring Completed.\n"
            f"Act 2 Score: {state.act2.score:.1f} / 200 (Caught: {state.act2.caught}/{state.act2.n_faults}, "
            f"False Alarms: {state.act2.false_alarms}, Mean Delay: {state.act2.mean_delay:.1f} steps).\n"
            f"Cross-Act Consequences:\n"
            f"  • Thruster Fuel Remaining: {res.fuel_pct:.1f}% (reduced by false alarms triggering RCS burns).\n"
            f"  • Sensor Noise Level: {res.sensor_noise:.3f} (elevated by missed equipment faults).\n"
            f"  • Hull Integrity: {res.hull_health:.1f}%.\n"
            f"  Notice: Degraded sensors and reduced fuel will directly challenge your flight control during Act 4 descent!"
        )

    if act_no == 3:
        return (
            f"ACT 3 DEBRIEF — Deep Sky Survey & Downlink Completed.\n"
            f"Act 3 Score: {state.act3.score:.1f} / 200 (Test Accuracy: {state.act3.accuracy:.1%}, "
            f"Science Value Ratio: {state.act3.value_ratio:.1%}, Classifier: {state.act3.backend.upper()}).\n"
            f"Scientific return transmitted to Earth. Downlinked rare mergers and spirals yielded maximum research credits."
        )

    if act_no == 4:
        landing_status = "SUCCESSFUL TOUCHDOWN" if state.act4.landing_rate >= 0.8 else "CRASH / HAZARDOUS TOUCHDOWN"
        return (
            f"ACT 4 DEBRIEF — Terminal Descent & Landing Completed ({landing_status}).\n"
            f"Act 4 Score: {state.act4.score:.1f} / 400 (Landing Rate: {state.act4.landing_rate:.1%}, "
            f"Mean Reward: {state.act4.mean_reward:.1f}).\n"
            f"Conditions Encountered: Gravity = {state.act4.conditions.get('gravity', 7.0):.1f} m/s², "
            f"Wind = {state.act4.conditions.get('wind', 2.0):.1f} m/s², "
            f"Fuel = {state.act4.conditions.get('fuel_pct', 100.0):.1f}%, "
            f"Sensor Noise = {state.act4.conditions.get('sensor_noise', 0.0):.3f}."
        )

    if act_no == 5:
        return (
            f"MISSION FINALE — Expedition Summary for Team '{state.team_id}'.\n"
            f"Cumulative Score: {state.total_score:.1f} / 1000.\n"
            f"Act 1: {state.act1.score:.1f} | Act 2: {state.act2.score:.1f} | "
            f"Act 3: {state.act3.score:.1f} | Act 4: {state.act4.score:.1f}.\n"
            f"Verification certificate generated for official leaderboard recording."
        )

    return f"Act {act_no} concluded. Probe systems nominal."
