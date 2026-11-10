"""Leaderboard generator tool for workshop team submissions."""

from __future__ import annotations

import argparse
import csv
import html
import sys
from pathlib import Path
from typing import Any

# Ensure local meridian package is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from meridian.scoring import verify_code


def build_leaderboard(codes_file: str | Path, out_dir: str | Path = ".") -> tuple[Path, Path]:
    """Parse submission codes file, verify signatures, deduplicate teams, and generate CSV and HTML tables."""
    codes_path = Path(codes_file)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not codes_path.exists():
        raise FileNotFoundError(f"Codes file not found: {codes_path}")

    teams_best: dict[str, dict[str, Any]] = {}

    with open(codes_path, "r", encoding="utf-8") as f:
        for line in f:
            code = line.strip()
            if not code or code.startswith("#"):
                continue

            payload = verify_code(code)
            if payload is None:
                continue

            team_id = str(payload.get("team_id", "unknown"))
            total = float(payload.get("total", 0.0))

            if team_id not in teams_best or total > float(teams_best[team_id].get("total", -1e9)):
                teams_best[team_id] = payload

    sorted_teams = sorted(
        teams_best.values(),
        key=lambda entry: (-float(entry.get("total", 0.0)), str(entry.get("team_id", ""))),
    )

    csv_path = output_dir / "leaderboard.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "rank",
            "team_id",
            "total",
            "act1",
            "act2",
            "act3",
            "act4",
            "fuel_pct",
            "sensor_noise",
            "hull_health",
        ])

        for rank, entry in enumerate(sorted_teams, start=1):
            scores = entry.get("scores", {})
            resources = entry.get("resources", {})
            writer.writerow([
                rank,
                entry.get("team_id", ""),
                f"{float(entry.get('total', 0.0)):.2f}",
                f"{float(scores.get('act1', entry.get('act1', 0.0))):.2f}",
                f"{float(scores.get('act2', entry.get('act2', 0.0))):.2f}",
                f"{float(scores.get('act3', entry.get('act3', 0.0))):.2f}",
                f"{float(scores.get('act4', entry.get('act4', 0.0))):.2f}",
                f"{float(resources.get('fuel_pct', 0.0)):.2f}",
                f"{float(resources.get('sensor_noise', 0.0)):.4f}",
                f"{float(resources.get('hull_health', 0.0)):.2f}",
            ])

    html_path = output_dir / "leaderboard.html"
    rows_html = []
    for rank, entry in enumerate(sorted_teams, start=1):
        scores = entry.get("scores", {})
        resources = entry.get("resources", {})
        fuel_pct = float(resources.get("fuel_pct", 0.0))
        hull_pct = float(resources.get("hull_health", 0.0))
        noise = float(resources.get("sensor_noise", 0.0))

        rank_badge_style = "background: #1f2937; color: #9ca3af;"
        if rank == 1:
            rank_badge_style = "background: #78350f; color: #fef3c7; border: 1px solid #d97706;"
        elif rank == 2:
            rank_badge_style = "background: #374151; color: #f3f4f6; border: 1px solid #9ca3af;"
        elif rank == 3:
            rank_badge_style = "background: #451a03; color: #fed7aa; border: 1px solid #b45309;"

        rows_html.append(
            f"<tr>"
            f"<td><span style='display: inline-block; min-width: 26px; text-align: center; padding: 2px 8px; border-radius: 9999px; font-weight: 700; font-size: 12px; {rank_badge_style}'>{rank}</span></td>"
            f"<td style='font-weight: 700; color: #f9fafb;'>{html.escape(str(entry.get('team_id', '')))}</td>"
            f"<td style='font-weight: 800; color: #38bdf8; font-size: 15px;'>{float(entry.get('total', 0.0)):.2f}</td>"
            f"<td>{float(scores.get('act1', entry.get('act1', 0.0))):.2f}</td>"
            f"<td>{float(scores.get('act2', entry.get('act2', 0.0))):.2f}</td>"
            f"<td>{float(scores.get('act3', entry.get('act3', 0.0))):.2f}</td>"
            f"<td>{float(scores.get('act4', entry.get('act4', 0.0))):.2f}</td>"
            f"<td><span style='color: {'#10b981' if fuel_pct >= 60 else ('#f59e0b' if fuel_pct >= 30 else '#ef4444')}; font-weight: 600;'>{fuel_pct:.1f}%</span></td>"
            f"<td><span style='color: {'#10b981' if noise <= 0.05 else ('#f59e0b' if noise <= 0.2 else '#ef4444')}; font-family: monospace;'>σ={noise:.3f}</span></td>"
            f"<td><span style='color: {'#10b981' if hull_pct >= 70 else ('#f59e0b' if hull_pct >= 40 else '#ef4444')}; font-weight: 600;'>{hull_pct:.1f}%</span></td>"
            f"</tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Project Meridian - Mission Control Leaderboard</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin: 0;
      padding: 2.5rem 1.5rem;
      background: #0b0f19;
      color: #e5e7eb;
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      background: #111827;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1f2937;
      padding-bottom: 16px;
      margin-bottom: 20px;
    }}
    .badge {{
      background: #0369a1;
      color: #e0f2fe;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.08em;
      padding: 3px 8px;
      border-radius: 4px;
      text-transform: uppercase;
    }}
    h1 {{
      color: #f9fafb;
      margin: 6px 0 0 0;
      font-size: 22px;
      letter-spacing: -0.01em;
    }}
    p.subtitle {{
      margin: 0;
      color: #9ca3af;
      font-size: 13px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
      background: #0b0f19;
      border: 1px solid #1f2937;
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid #1f2937;
      font-size: 13.5px;
    }}
    th {{
      background: #161f30;
      color: #9ca3af;
      text-transform: uppercase;
      font-size: 11px;
      letter-spacing: 0.06em;
      font-weight: 600;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:hover td {{
      background: #131b2c;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <span class="badge">Official Standings</span>
        <h1>PROJECT MERIDIAN // MISSION LEADERBOARD</h1>
        <p class="subtitle">Verified cryptographic flight certificates recorded across all 4 acts.</p>
      </div>
    </div>
    <table>
      <thead>
        <tr>
          <th>Rank</th>
          <th>Team ID</th>
          <th>Total Score</th>
          <th>Act 1</th>
          <th>Act 2</th>
          <th>Act 3</th>
          <th>Act 4</th>
          <th>Fuel</th>
          <th>Noise</th>
          <th>Hull</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as htmlfile:
        htmlfile.write(html_content)

    return csv_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Meridian mission leaderboard from exported codes.")
    parser.add_argument("codes_file", nargs="?", default="codes.txt", help="Path to input text file containing submission codes.")
    parser.add_argument("--out-dir", default=".", help="Directory to write leaderboard.csv and leaderboard.html.")
    args = parser.parse_args()

    csv_path, html_path = build_leaderboard(args.codes_file, args.out_dir)
    print(f"Generated: {csv_path} and {html_path}")


if __name__ == "__main__":
    main()
