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
    lines = [
        "| Planner | Scenario | Success | Sim Time (s) | Collision | Out of Lane | Replan Count | Plan Mean (ms) | Plan P95 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        lines.append(
            f"| {m.planner} | {m.scenario} | {int(m.success)} | {m.sim_time_s:.2f} | {int(m.collision)} | {int(m.out_of_lane)} | {m.replan_count} | "
            f"{'' if m.plan_ms_mean is None else f'{m.plan_ms_mean:.2f}'} | {'' if m.plan_ms_p95 is None else f'{m.plan_ms_p95:.2f}'} |"
        )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
