from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


@dataclass
class RunMetrics:
    scenario: str
    planner: str
    success: bool
    sim_time_s: float
    collision: bool
    out_of_lane: bool
    replan_count: int
    plan_ms_mean: Optional[float]
    plan_ms_p95: Optional[float]


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any) -> Optional[float]:
    try:
        if v in (None, ""):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    if lo == hi:
        return sorted_vals[lo]
    frac = k - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def compute_run_metrics(run_dir: Path) -> RunMetrics:
    meta = _read_json(run_dir / "meta.json")
    result = _read_json(run_dir / "result.json")
    rows = _read_csv_rows(run_dir / "record.csv")

    scenario = str(meta.get("scenario", "unknown"))
    planner = str(meta.get("planner", "unknown"))

    reason = str(result.get("reason", ""))
    success = reason in {"goal_reached", "success", "done"}
    collision = reason == "collision"
    out_of_lane = reason in {"lane_departure", "out_of_lane"}

    plan_ms = [
        _to_float(r.get("timing_plan_total_ms"))
        for r in rows
        if _to_float(r.get("timing_plan_total_ms")) is not None
    ]
    plan_ms_clean = sorted([v for v in plan_ms if v is not None])

    replan_count = 0
    last_sig = None
    for r in rows:
        sig = (r.get("trajN_x"), r.get("trajN_y"), r.get("traj_len"))
        if sig != last_sig:
            replan_count += 1
            last_sig = sig

    return RunMetrics(
        scenario=scenario,
        planner=planner,
        success=success,
        sim_time_s=float(result.get("sim_time_s", 0.0)),
        collision=collision,
        out_of_lane=out_of_lane,
        replan_count=replan_count,
        plan_ms_mean=mean(plan_ms_clean) if plan_ms_clean else None,
        plan_ms_p95=_percentile(plan_ms_clean, 0.95) if plan_ms_clean else None,
    )


def write_comparison_table(metrics: List[RunMetrics], output_md: Path) -> None:
    def fmt_bool(v: bool) -> str:
        return "1" if v else "0"

    def fmt_float(v: Optional[float], nd: int = 2) -> str:
        return "—" if v is None else f"{v:.{nd}f}"

    def fmt_int(v: int) -> str:
        return f"{v:d}"

    def escape_html(s: str) -> str:
        # minimal HTML escaping
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
        )

    # Optional: shorten very long planner strings for nicer table
    def planner_display(planner: str) -> str:
        # e.g., "framework.planning.local.rrt_star:RRTStarPlanner" -> "RRTStarPlanner"
        if ":" in planner:
            return planner.split(":")[-1]
        return planner.split(".")[-1]

    style = """
<style>
table.benchmark { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 14px; }
table.benchmark th, table.benchmark td { border: 1px solid #ddd; padding: 6px 8px; }
table.benchmark th { background: #f6f6f6; text-align: left; }
table.benchmark td.num, table.benchmark th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.benchmark tr:nth-child(even) { background: #fcfcfc; }
</style>
""".strip()

    header = """
<table class="benchmark">
  <thead>
    <tr>
      <th>Planner</th>
      <th>Scenario</th>
      <th class="num">Success</th>
      <th class="num">Sim Time (s)</th>
      <th class="num">Collision</th>
      <th class="num">Out of Lane</th>
      <th class="num">Replan Count</th>
      <th class="num">Plan Mean (ms)</th>
      <th class="num">Plan P95 (ms)</th>
    </tr>
  </thead>
  <tbody>
""".rstrip()

    rows_html: List[str] = []
    for m in metrics:
        rows_html.append(
            "    <tr>"
            f"<td>{escape_html(planner_display(m.planner))}</td>"
            f"<td>{escape_html(m.scenario)}</td>"
            f"<td class=\"num\">{fmt_bool(m.success)}</td>"
            f"<td class=\"num\">{fmt_float(m.sim_time_s, 2)}</td>"
            f"<td class=\"num\">{fmt_bool(m.collision)}</td>"
            f"<td class=\"num\">{fmt_bool(m.out_of_lane)}</td>"
            f"<td class=\"num\">{fmt_int(m.replan_count)}</td>"
            f"<td class=\"num\">{fmt_float(m.plan_ms_mean, 2)}</td>"
            f"<td class=\"num\">{fmt_float(m.plan_ms_p95, 2)}</td>"
            "</tr>"
        )

    footer = """
  </tbody>
</table>
""".strip()

    html = "\n".join([style, header, *rows_html, footer]) + "\n"
    output_md.write_text(html, encoding="utf-8")
