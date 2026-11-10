"""Interactive UI widgets and mission dashboard for Project Meridian."""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML

from meridian.story import briefing_text, debrief_text

if TYPE_CHECKING:
    from meridian.state import MissionState


def _progress_color(pct: float, invert: bool = False) -> str:
    """Return hex color for a percentage value."""
    if invert:
        if pct <= 20.0:
            return "#10b981"  # Good (low noise)
        elif pct <= 50.0:
            return "#f59e0b"  # Moderate
        return "#ef4444"      # High noise
    else:
        if pct >= 70.0:
            return "#10b981"  # Good
        elif pct >= 35.0:
            return "#f59e0b"  # Caution
        return "#ef4444"      # Critical


def dashboard(state: MissionState) -> HTML:
    """Render an IPython HTML mission dashboard card with inline CSS and dark space theme."""
    res = state.resources
    target = state.act1.target

    fuel_val = max(0.0, min(100.0, float(res.fuel_pct)))
    hull_val = max(0.0, min(100.0, float(res.hull_health)))
    noise_val = max(0.0, float(res.sensor_noise))
    # Normalized noise meter (0.0 to 0.5 mapped to 0-100%)
    noise_pct = min(100.0, (noise_val / 0.5) * 100.0)

    fuel_col = _progress_color(fuel_val)
    hull_col = _progress_color(hull_val)
    noise_col = _progress_color(noise_pct, invert=True)

    target_label = target.label if target.label else "unscouted world"
    star_info = f" ({target.star_id})" if target.star_id else ""

    act1_score = float(state.act1.score)
    act2_score = float(state.act2.score)
    act3_score = float(state.act3.score)
    act4_score = float(state.act4.score)
    total_score = float(state.total_score)

    card_html = f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #0b0f19;
        color: #e5e7eb;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 20px;
        max-width: 820px;
        margin: 12px 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    ">
        <!-- Header -->
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1f2937; padding-bottom: 12px; margin-bottom: 16px;">
            <div>
                <span style="background: #0369a1; color: #e0f2fe; font-size: 11px; font-weight: 700; letter-spacing: 0.08em; padding: 3px 8px; border-radius: 4px; text-transform: uppercase;">Mission Console</span>
                <h2 style="margin: 4px 0 0 0; font-size: 20px; color: #f9fafb; letter-spacing: -0.01em;">PROBE MERIDIAN // TEAM: {html.escape(str(state.team_id))}</h2>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em;">Total Score</div>
                <div style="font-size: 24px; font-weight: 800; color: #38bdf8;">{total_score:.1f} <span style="font-size: 14px; color: #6b7280; font-weight: 400;">/ 1000</span></div>
            </div>
        </div>

        <!-- Target Destination Bar -->
        <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px 16px; margin-bottom: 16px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
            <div>
                <div style="font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.06em;">Target Destination</div>
                <div style="font-size: 15px; font-weight: 700; color: #a78bfa;">{html.escape(target_label)}{html.escape(star_info)}</div>
            </div>
            <div style="display: flex; gap: 20px; font-size: 13px;">
                <div><span style="color: #9ca3af;">Gravity:</span> <strong style="color: #f3f4f6;">{target.gravity:.2f} m/s²</strong></div>
                <div><span style="color: #9ca3af;">Wind:</span> <strong style="color: #f3f4f6;">{target.wind:.2f} m/s²</strong></div>
                <div><span style="color: #9ca3af;">Cruise:</span> <strong style="color: #f3f4f6;">{target.trip_days} days</strong></div>
            </div>
        </div>

        <!-- Resources Section -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px;">
            <!-- Fuel -->
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
                    <span style="color: #9ca3af; font-weight: 600;">PROPULSION FUEL</span>
                    <strong style="color: {fuel_col};">{fuel_val:.1f}%</strong>
                </div>
                <div style="background: #1f2937; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {fuel_col}; width: {fuel_val}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>

            <!-- Hull -->
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
                    <span style="color: #9ca3af; font-weight: 600;">HULL INTEGRITY</span>
                    <strong style="color: {hull_col};">{hull_val:.1f}%</strong>
                </div>
                <div style="background: #1f2937; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {hull_col}; width: {hull_val}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>

            <!-- Sensor Noise -->
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 12px;">
                <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 6px;">
                    <span style="color: #9ca3af; font-weight: 600;">SENSOR NOISE</span>
                    <strong style="color: {noise_col};">σ = {noise_val:.3f}</strong>
                </div>
                <div style="background: #1f2937; border-radius: 4px; height: 8px; overflow: hidden;">
                    <div style="background: {noise_col}; width: {noise_pct}%; height: 100%; border-radius: 4px;"></div>
                </div>
            </div>
        </div>

        <!-- Scores per Act -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; color: #9ca3af; font-weight: 600;">ACT 1: EXOPLANET</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{act1_score:.1f}</div>
                <div style="font-size: 10px; color: #6b7280;">MAX 200</div>
            </div>

            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; color: #9ca3af; font-weight: 600;">ACT 2: CRUISE</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{act2_score:.1f}</div>
                <div style="font-size: 10px; color: #6b7280;">MAX 200</div>
            </div>

            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; color: #9ca3af; font-weight: 600;">ACT 3: GALAXY</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{act3_score:.1f}</div>
                <div style="font-size: 10px; color: #6b7280;">MAX 200</div>
            </div>

            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 8px; padding: 10px; text-align: center;">
                <div style="font-size: 11px; color: #9ca3af; font-weight: 600;">ACT 4: LANDER</div>
                <div style="font-size: 18px; font-weight: 700; color: #38bdf8; margin: 4px 0;">{act4_score:.1f}</div>
                <div style="font-size: 10px; color: #6b7280;">MAX 400</div>
            </div>
        </div>
    </div>
    """
    return HTML(card_html)


def briefing(act_no: int) -> HTML:
    """Render the mission story briefing card for the given act."""
    text = briefing_text(act_no)
    lines = text.strip().split("\n")
    header = lines[0] if lines else f"ACT {act_no}"
    body = "\n".join(lines[1:]) if len(lines) > 1 else text

    card_html = f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #0b0f19;
        color: #e5e7eb;
        border: 1px solid #0284c7;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px 20px;
        max-width: 820px;
        margin: 12px 0;
        box-shadow: 0 4px 16px rgba(2, 132, 199, 0.15);
    ">
        <div style="font-family: monospace; font-size: 12px; font-weight: 700; color: #38bdf8; letter-spacing: 0.05em; margin-bottom: 8px;">
            {html.escape(header)}
        </div>
        <div style="font-size: 14px; line-height: 1.6; color: #d1d5db;">
            {html.escape(body)}
        </div>
    </div>
    """
    return HTML(card_html)


def debrief(state: MissionState, act_no: int) -> HTML:
    """Render the mission debrief card displaying score and parameter changes for subsequent acts."""
    text = debrief_text(state, act_no)
    lines = text.strip().split("\n")
    header = lines[0] if lines else f"ACT {act_no} DEBRIEF"
    body = "<br>".join(html.escape(line) for line in lines[1:]) if len(lines) > 1 else html.escape(text)

    score_val = getattr(getattr(state, f"act{act_no}", None), "score", 0.0) if act_no in (1, 2, 3, 4) else state.total_score
    max_val = 400 if act_no == 4 else (1000 if act_no == 5 else 200)

    card_html = f"""
    <div style="
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background-color: #0b0f19;
        color: #e5e7eb;
        border: 1px solid #059669;
        border-left: 4px solid #10b981;
        border-radius: 8px;
        padding: 16px 20px;
        max-width: 820px;
        margin: 12px 0;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.15);
    ">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; border-bottom: 1px solid #1f2937; padding-bottom: 6px;">
            <div style="font-family: monospace; font-size: 13px; font-weight: 700; color: #10b981; letter-spacing: 0.05em;">
                {html.escape(header)}
            </div>
            <div style="font-size: 14px; font-weight: 700; color: #34d399;">
                Score: {score_val:.1f} / {max_val}
            </div>
        </div>
        <div style="font-size: 13.5px; line-height: 1.6; color: #d1d5db;">
            {body}
        </div>
    </div>
    """
    return HTML(card_html)


def mission_report(state: MissionState) -> plt.Figure:
    """Generate an auto mission report figure: 4-act timeline with scores and coupling summary."""
    fig = plt.figure(figsize=(11, 7), facecolor="#0b0f19")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0], hspace=0.35)

    # Subplot 1: Scores timeline and resource bars
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#111827")

    acts = ["Act 1: Exoplanet", "Act 2: Cruise", "Act 3: Galaxy", "Act 4: Lander"]
    scores = [state.act1.score, state.act2.score, state.act3.score, state.act4.score]
    max_scores = [200, 200, 200, 400]
    pcts = [(s / m) * 100.0 for s, m in zip(scores, max_scores)]

    x = np.arange(len(acts))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, scores, width, label="Achieved Score", color="#38bdf8", alpha=0.9)
    bars2 = ax1.bar(x + width / 2, max_scores, width, label="Max Possible", color="#374151", alpha=0.6)

    ax1.set_xticks(x)
    ax1.set_xticklabels(acts, color="#e5e7eb", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Score Points", color="#e5e7eb", fontsize=11)
    ax1.set_title(
        f"Project Meridian: Mission Performance Timeline (Total: {state.total_score:.1f} / 1000)",
        color="#f9fafb",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.tick_params(colors="#9ca3af")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#6b7280")
    ax1.legend(loc="upper left", facecolor="#1f2937", edgecolor="#374151", labelcolor="#e5e7eb")

    for b, pct in zip(bars1, pcts):
        h = b.get_height()
        ax1.text(b.get_x() + b.get_width() / 2, h + 5, f"{h:.0f} ({pct:.0f}%)", ha="center", va="bottom", color="#38bdf8", fontsize=9, fontweight="bold")

    # Subplot 2: Cross-act coupling consequences summary
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#111827")
    ax2.axis("off")

    target = state.act1.target
    res = state.resources

    summary_text = (
        f"CROSS-ACT MISSION COUPLING SUMMARY:\n"
        f"────────────────────────────────────────────────────────────────────────────────────\n"
        f"• Act 1 -> Act 2 & Act 4:\n"
        f"  Transit depth {target.depth:.4f} set landing gravity to {target.gravity:.2f} m/s².\n"
        f"  Orbital period {target.period_days:.2f} d determined surface wind ({target.wind:.2f} m/s²) and cruise length ({target.trip_days} d).\n\n"
        f"• Act 2 -> Act 4:\n"
        f"  Cruise monitoring caught {state.act2.caught}/{state.act2.n_faults} faults with {state.act2.false_alarms} false alarms.\n"
        f"  Missed faults raised sensor noise to σ = {res.sensor_noise:.3f}; false alarms consumed RCS fuel (remaining: {res.fuel_pct:.1f}%).\n\n"
        f"• Act 3 -> Scientific Heritage:\n"
        f"  Deep sky downlink yielded {state.act3.value_ratio:.1%} scientific efficiency across {state.act3.backend.upper()} backend.\n\n"
        f"• Act 4 -> Mission Touchdown:\n"
        f"  Descent policy achieved {state.act4.landing_rate:.1%} landing success under g={target.gravity:.1f} m/s² and w={target.wind:.1f} m/s²."
    )

    ax2.text(
        0.02,
        0.95,
        summary_text,
        color="#d1d5db",
        fontsize=10,
        fontfamily="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#0b0f19", edgecolor="#1f2937", lw=1.5),
    )

    return fig
