from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import math

from framework.testing.lane_change_case import LaneChangeCase
from framework.testing.lane_change_runner import LaneChangeExecutionResult


@dataclass
class LaneChangeEvalResult:
    scenario_name: str
    case_id: str
    success: bool
    fail_reason: Optional[str]
    timeout: bool
    collision_count: int
    time_to_finish: Optional[float]
    route_completion: float
    replan_count: int
    planning_time_mean: Optional[float]
    planning_time_p95: Optional[float]
    min_distance_to_lead_vehicle: Optional[float]
    min_distance_to_adjacent_vehicle: Optional[float]
    max_lateral_error: Optional[float]
    max_heading_error: Optional[float]
    lane_change_started: bool
    lane_change_completed: bool
    overtake_completed: bool
    returned_to_original_lane: bool
    blocked_by_adjacent_vehicle: bool
    stuck_behind_lead_vehicle: bool
    raw_reason: Optional[str]
    run_dir: Optional[str]
    input_case: Dict[str, Any]
    planner_name: str
    scenario_impl: str
    base_config_path: str


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_record_rows(run_dir: Path) -> List[Dict[str, Any]]:
    path = run_dir / "record.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(s))) - 1
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def _normalize_fail_reason(*, timeout: bool, collision_count: int, started: bool, completed: bool, overtake: bool, returned: bool, case: LaneChangeCase, raw_reason: Optional[str], blocked: bool, stuck: bool, planner_error: bool) -> Optional[str]:
    if planner_error:
        return "planner_error"
    if collision_count > 0:
        return "collision"
    if timeout:
        return "timeout"
    if not started:
        return "lane_change_not_started"
    if not completed:
        return "lane_change_not_completed"
    if case.require_overtake_completion and not overtake:
        return "overtake_not_completed"
    if case.require_return_to_original_lane and not returned:
        return "return_to_lane_failed"
    if blocked:
        return "unsafe_gap"
    if stuck:
        return "stuck"
    if raw_reason in {"exception", "scenario_not_ready"}:
        return "scenario_error"
    return None


def evaluate_lane_change_result(case: LaneChangeCase, execution: LaneChangeExecutionResult, *, scenario_name: str = "lane_change_overtake") -> LaneChangeEvalResult:
    run_reason = str(execution.run_result.get("reason")) if isinstance(execution.run_result, dict) else None

    record_rows: List[Dict[str, Any]] = []
    metrics_summary: Dict[str, Any] = {}
    if execution.run_dir:
        run_dir = Path(execution.run_dir)
        record_rows = _load_record_rows(run_dir)
        metrics_summary = _load_json(run_dir / "metrics_summary.json")
        final_summary = _load_json(run_dir / "final_summary.json")
    else:
        final_summary = {}

    behavior_states = [str(r.get("debug_behavior_state")) for r in record_rows if r.get("debug_behavior_state")]
    behavior_reasons = [str(r.get("debug_behavior_reason")) for r in record_rows if r.get("debug_behavior_reason")]

    lane_change_started = any(s in {"LANE_CHANGE_OUT", "CRUISE_PASS_LANE", "LANE_CHANGE_BACK"} for s in behavior_states)
    lane_change_completed = ("LANE_CHANGE_OUT" in behavior_states) and ("CRUISE_PASS_LANE" in behavior_states or "LANE_CHANGE_BACK" in behavior_states)
    returned_to_original_lane = "LANE_CHANGE_BACK" in behavior_states and behavior_states[-1] == "FOLLOW"

    lead_longitudinal = [
        float(r["debug_lead_longitudinal"]) for r in record_rows if isinstance(r.get("debug_lead_longitudinal"), (int, float))
    ]
    overtake_completed = any(v < -1.0 for v in lead_longitudinal)

    plan_ms = []
    for row in record_rows:
        for key in ("timing_plan_total_ms", "timing_total_ms", "timing_plan_ms"):
            v = row.get(key)
            if isinstance(v, (int, float)):
                plan_ms.append(float(v))
                break

    min_obs = [float(r["min_obstacle_distance_m"]) for r in record_rows if isinstance(r.get("min_obstacle_distance_m"), (int, float))]

    timeout = bool(final_summary.get("timeout", False)) or run_reason in {"timeout", "max_steps_reached"}
    collision_count = int(final_summary.get("collision_count", 1 if run_reason == "collision" else 0))
    planner_error = run_reason in {"exception", "planner_error"}

    blocked_by_adjacent_vehicle = any("adjacent" in s.lower() and "block" in s.lower() for s in behavior_reasons)
    stuck_behind_lead_vehicle = bool(timeout and not lane_change_started)

    fail_reason = _normalize_fail_reason(
        timeout=timeout,
        collision_count=collision_count,
        started=lane_change_started,
        completed=lane_change_completed,
        overtake=overtake_completed,
        returned=returned_to_original_lane,
        case=case,
        raw_reason=run_reason,
        blocked=blocked_by_adjacent_vehicle,
        stuck=stuck_behind_lead_vehicle,
        planner_error=planner_error,
    )

    success = fail_reason is None

    return LaneChangeEvalResult(
        scenario_name=scenario_name,
        case_id=case.case_id,
        success=success,
        fail_reason=fail_reason,
        timeout=timeout,
        collision_count=collision_count,
        time_to_finish=float(execution.run_result.get("sim_time_s")) if isinstance(execution.run_result, dict) and execution.run_result.get("sim_time_s") is not None else final_summary.get("runtime_s"),
        route_completion=1.0 if run_reason == "reached_goal" or bool(final_summary.get("reach_goal", False)) else 0.0,
        replan_count=max(0, len(record_rows) - 1),
        planning_time_mean=(sum(plan_ms) / len(plan_ms)) if plan_ms else None,
        planning_time_p95=_percentile(plan_ms, 95.0),
        min_distance_to_lead_vehicle=min(min_obs) if min_obs else None,
        min_distance_to_adjacent_vehicle=None,
        max_lateral_error=metrics_summary.get("cte_max_abs_m"),
        max_heading_error=metrics_summary.get("heading_max_abs_rad"),
        lane_change_started=lane_change_started,
        lane_change_completed=lane_change_completed,
        overtake_completed=overtake_completed,
        returned_to_original_lane=returned_to_original_lane,
        blocked_by_adjacent_vehicle=blocked_by_adjacent_vehicle,
        stuck_behind_lead_vehicle=stuck_behind_lead_vehicle,
        raw_reason=run_reason,
        run_dir=execution.run_dir,
        input_case=asdict(case),
        planner_name=execution.planner_name,
        scenario_impl=execution.scenario_name,
        base_config_path=execution.base_config_path,
    )


def eval_result_to_dict(result: LaneChangeEvalResult) -> Dict[str, Any]:
    return asdict(result)
