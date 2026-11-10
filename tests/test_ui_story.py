"""Tests for meridian/story.py, meridian/ui.py, and tools/make_notebooks.py."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import nbformat
import pytest
from IPython.display import HTML

from meridian.state import MissionState
from meridian.story import BRIEFINGS, briefing_text, debrief_text
from meridian.ui import briefing, dashboard, debrief, mission_report
from tools.make_notebooks import NOTEBOOK_FACTORIES, make_notebooks


def test_briefings_word_count_and_content():
    """Verify all act briefings exist, have mission-control tone, and are <= 120 words."""
    for act_no in [0, 1, 2, 3, 4, 5]:
        text = briefing_text(act_no)
        assert text, f"Briefing text for Act {act_no} is empty"
        words = text.split()
        assert len(words) <= 120, f"Act {act_no} briefing has {len(words)} words (> 120)"
        assert "MISSION CONTROL" in text or "FLIGHT DIRECTIVE" in text


def test_debrief_templates():
    """Verify debrief templates produce non-empty strings describing score and coupling."""
    state = MissionState(team_id="team_test")
    state.act1.done = True
    state.act1.score = 150.0
    state.act1.target.star_id = "STAR_0042"
    state.act1.target.is_real_planet = True
    state.act1.target.gravity = 6.5
    state.act1.target.wind = 2.5
    state.act1.target.trip_days = 110

    state.act2.done = True
    state.act2.score = 160.0
    state.resources.fuel_pct = 85.0
    state.resources.sensor_noise = 0.04

    state.act3.done = True
    state.act3.score = 175.0

    state.act4.done = True
    state.act4.score = 350.0
    state.act4.landing_rate = 0.95

    for act_no in [0, 1, 2, 3, 4, 5]:
        text = debrief_text(state, act_no)
        assert text, f"Debrief text for Act {act_no} is empty"
        if act_no in [1, 2, 3, 4]:
            assert "Score:" in text or "DEBRIEF" in text


def test_ui_components_html_and_styling():
    """Verify dashboard, briefing, and debrief return HTML objects with dark space styling."""
    state = MissionState(team_id="team_test")
    dash = dashboard(state)
    assert isinstance(dash, HTML)
    html_data = dash.data
    assert "#0b0f19" in html_data
    assert "PROPULSION FUEL" in html_data
    assert "HULL INTEGRITY" in html_data
    assert "SENSOR NOISE" in html_data
    assert "ACT 1: EXOPLANET" in html_data
    assert "ACT 4: LANDER" in html_data

    b_html = briefing(1)
    assert isinstance(b_html, HTML)
    assert "#0b0f19" in b_html.data

    d_html = debrief(state, 1)
    assert isinstance(d_html, HTML)
    assert "#0b0f19" in d_html.data


def test_ui_mission_report_figure():
    """Verify mission report generates a valid 2-panel matplotlib figure."""
    state = MissionState(team_id="team_test")
    fig = mission_report(state)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    plt.close(fig)


def test_make_notebooks_and_your_turn_cells(tmp_path: Path):
    """Verify make_notebooks generates all 6 notebooks, each with exactly 1 YOUR TURN cell."""
    paths = make_notebooks(tmp_path)
    assert len(paths) == 6

    expected_files = {
        "00_start.ipynb",
        "01_destination.ipynb",
        "02_cruise_watch.ipynb",
        "03_deep_field.ipynb",
        "04_descent.ipynb",
        "05_finale.ipynb",
    }
    actual_files = {p.name for p in paths}
    assert actual_files == expected_files

    for p in paths:
        nb = nbformat.read(p, as_version=4)

        # Cell 1: setup
        c1 = nb.cells[1].source
        assert "REPO_URL" in c1
        assert "# !pip install" in c1

        # Cell 2: TEAM_ID and state
        c2 = nb.cells[2].source
        assert "TEAM_ID" in c2
        assert "load_state" in c2

        # Exactly 1 YOUR TURN cell containing only knobs dict
        yt_cells = [c for c in nb.cells if c.cell_type == "code" and "YOUR TURN" in c.source]
        assert len(yt_cells) == 1, f"Expected 1 YOUR TURN cell in {p.name}, found {len(yt_cells)}"
        yt_source = yt_cells[0].source
        assert "knobs = {" in yt_source

        # Markdown sentence count <= 4
        for idx, c in enumerate(nb.cells):
            if c.cell_type == "markdown":
                text = re.sub(r"#+.*\n", "", c.source).strip()
                sentences = [s.strip() for s in re.split(r"[\.\!\?]\s+", text) if s.strip()]
                assert len(sentences) <= 4, f"Markdown cell {idx} in {p.name} exceeds 4 sentences: {sentences}"
