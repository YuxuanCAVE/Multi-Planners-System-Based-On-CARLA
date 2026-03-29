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
    collision_count: int
    timeout: bool
    time_to_finish: Optional[float]
    route_completion: float
    planning_time_mean: Optional[float]
    planning_time_p95: Optional[float]
    min_distance_to_lead_vehicle: Optional[float]
    min_distance_to_adjacent_vehicle: Optional[float]
    lane_change_started: bool
    lane_change_completed: bool
    overtake_completed: bool
    returned_to_original_lane: bool
    run_dir: Optional[str]
    config_path: str
    run_stdout: str
    run_stderr: str
    input_case: Dict[str, Any]


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _get_first_numeric(row: Dict[str, Any], keys: List[str]) -> Optional[float]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                continue
    return None


def _percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    idx = int(math.ceil((p / 100.0) * len(s))) - 1
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def _compute_lateral_offsets(record_rows: List[Dict[str, Any]]) -> List[float]:
    """
    Estimate ego lateral displacement in a local frame aligned with the initial ego heading.
    lateral = projection onto left-normal of initial heading.
    """
    if not record_rows:
        return []

    x0 = record_rows[0].get("ego_x")
    y0 = record_rows[0].get("ego_y")
    yaw0 = record_rows[0].get("ego_yaw")
    if not all(isinstance(v, (int, float)) for v in (x0, y0, yaw0)):
        return []

    yaw0 = float(yaw0)
    nx = -math.sin(yaw0)
    ny = math.cos(yaw0)

    lats: List[float] = []
    for row in record_rows:
        x = row.get("ego_x")
        y = row.get("ego_y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue
        dx = float(x) - float(x0)
        dy = float(y) - float(y0)
        lats.append(dx * nx + dy * ny)
    return lats


def _extract_lateral_series(record_rows: List[Dict[str, Any]]) -> List[float]:
    """
    Prefer explicit lateral-related fields if present; otherwise fall back to
    initial-heading local-frame lateral displacement.
    """
    candidates = [
        "lateral_offset_m",
        "ego_lateral",
        "ego_pose_y",
        "ego_location_y",
        "pose_y",
        "y",
        "ego_y",
    ]
    series: List[float] = []
    for row in record_rows:
        v = _get_first_numeric(row, candidates)
        if v is not None:
            series.append(v)
    if len(series) >= max(5, len(record_rows) // 4):
        return series
    return _compute_lateral_offsets(record_rows)


def _extract_xy_series(record_rows: List[Dict[str, Any]], prefix: str = "ego") -> List[tuple[float, float]]:
    xs = [f"{prefix}_x", f"{prefix}_pose_x", f"{prefix}_location_x", "x"]
    ys = [f"{prefix}_y", f"{prefix}_pose_y", f"{prefix}_location_y", "y"]
    out: List[tuple[float, float]] = []
    for row in record_rows:
        x = _get_first_numeric(row, xs)
        y = _get_first_numeric(row, ys)
        if x is None or y is None:
            continue
        out.append((x, y))
    return out


def _extract_reference_route(run_dir: Optional[str], record_rows: List[Dict[str, Any]]) -> List[tuple[float, float]]:
    points: List[tuple[float, float]] = []
    if run_dir:
        ref_payload = _load_json(Path(run_dir) / "reference_path.json")
        if isinstance(ref_payload, dict):
            raw_points = ref_payload.get("points")
            if isinstance(raw_points, list):
                for p in raw_points:
                    if not isinstance(p, dict):
                        continue
                    x = _get_first_numeric(p, ["x"])
                    y = _get_first_numeric(p, ["y"])
                    if x is not None and y is not None:
                        points.append((x, y))
        if len(points) >= 2:
            return points

    # fallback: ref points in record rows (if present)
    for row in record_rows:
        x = _get_first_numeric(row, ["ref_x"])
        y = _get_first_numeric(row, ["ref_y"])
        if x is not None and y is not None:
            points.append((x, y))
    if len(points) >= 2:
        return points

    # last fallback: ego trajectory itself
    return _extract_xy_series(record_rows, prefix="ego")


def _build_cum_s(polyline_xy: List[tuple[float, float]]) -> List[float]:
    if not polyline_xy:
        return []
    out = [0.0]
    for i in range(1, len(polyline_xy)):
        x0, y0 = polyline_xy[i - 1]
        x1, y1 = polyline_xy[i]
        out.append(out[-1] + math.hypot(x1 - x0, y1 - y0))
    return out


def _resample_route_by_arclength(route_xy: List[tuple[float, float]], ds: float = 0.5) -> List[tuple[float, float]]:
    """
    Spatial resampling only (no time alignment). Uniformly sample points along arc length.
    """
    if len(route_xy) < 2:
        return route_xy
    cum_s = _build_cum_s(route_xy)
    total = cum_s[-1]
    if total <= 1e-6:
        return [route_xy[0], route_xy[-1]]
    ds = max(0.1, float(ds))
    samples = [0.0]
    cur = ds
    while cur < total:
        samples.append(cur)
        cur += ds
    if samples[-1] < total:
        samples.append(total)

    out: List[tuple[float, float]] = []
    seg = 0
    for s in samples:
        while seg < len(cum_s) - 2 and cum_s[seg + 1] < s:
            seg += 1
        s0 = cum_s[seg]
        s1 = cum_s[seg + 1]
        x0, y0 = route_xy[seg]
        x1, y1 = route_xy[seg + 1]
        if s1 <= s0 + 1e-9:
            out.append((x0, y0))
            continue
        t = (s - s0) / (s1 - s0)
        out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def _project_point_to_route_sl(x: float, y: float, route_xy: List[tuple[float, float]], cum_s: List[float]) -> tuple[float, float]:
    if len(route_xy) < 2:
        return 0.0, 0.0
    best_d2 = float("inf")
    best_s = 0.0
    best_l = 0.0
    for i in range(len(route_xy) - 1):
        x0, y0 = route_xy[i]
        x1, y1 = route_xy[i + 1]
        vx, vy = x1 - x0, y1 - y0
        seg2 = vx * vx + vy * vy
        if seg2 <= 1e-9:
            continue
        wx, wy = x - x0, y - y0
        t = max(0.0, min(1.0, (wx * vx + wy * vy) / seg2))
        px, py = x0 + t * vx, y0 + t * vy
        dx, dy = x - px, y - py
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            seg_len = math.sqrt(seg2)
            nx, ny = -vy / seg_len, vx / seg_len
            best_s = cum_s[i] + t * seg_len
            best_l = (x - px) * nx + (y - py) * ny
    return best_s, best_l


def _extract_delta_s_series_from_obstacles(
    record_rows: List[Dict[str, Any]],
    route_xy: List[tuple[float, float]],
    cum_s: List[float],
) -> List[float]:
    """
    Build ego-lead delta_s time series by tracking a likely lead obstacle ID.
    Requires record rows to contain obstacle list payloads.
    """
    if len(route_xy) < 2:
        return []

    lead_votes: Dict[int, int] = {}
    frame_candidates: List[Dict[int, float]] = []

    for row in record_rows:
        ego_x = _get_first_numeric(row, ["ego_x", "x"])
        ego_y = _get_first_numeric(row, ["ego_y", "y"])
        obs = row.get("obstacles")
        if ego_x is None or ego_y is None or not isinstance(obs, list):
            frame_candidates.append({})
            continue
        ego_s, ego_l = _project_point_to_route_sl(ego_x, ego_y, route_xy, cum_s)
        cand: Dict[int, float] = {}
        for item in obs:
            if not isinstance(item, dict):
                continue
            oid_raw = item.get("id")
            ox = _get_first_numeric(item, ["x"])
            oy = _get_first_numeric(item, ["y"])
            if ox is None or oy is None or not isinstance(oid_raw, (int, float)):
                continue
            oid = int(oid_raw)
            ob_s, ob_l = _project_point_to_route_sl(ox, oy, route_xy, cum_s)
            delta_s = ego_s - ob_s
            if delta_s < 5.0 and delta_s > -80.0 and abs(ob_l - ego_l) < 4.0:
                cand[oid] = delta_s
                if delta_s < 0.0:
                    lead_votes[oid] = lead_votes.get(oid, 0) + 1
        frame_candidates.append(cand)

    if not lead_votes:
        return []
    lead_id = max(lead_votes.items(), key=lambda kv: kv[1])[0]

    out: List[float] = []
    for cand in frame_candidates:
        if lead_id in cand:
            out.append(float(cand[lead_id]))
    return out


def _extract_delta_s_series_from_tracked_lead(
    record_rows: List[Dict[str, Any]],
    route_xy: List[tuple[float, float]],
    cum_s: List[float],
) -> List[float]:
    if len(route_xy) < 2:
        return []
    out: List[float] = []
    for row in record_rows:
        ex = _get_first_numeric(row, ["ego_x", "x"])
        ey = _get_first_numeric(row, ["ego_y", "y"])
        lx = _get_first_numeric(row, ["tracked_lead_x"])
        ly = _get_first_numeric(row, ["tracked_lead_y"])
        if ex is None or ey is None or lx is None or ly is None:
            continue
        ego_s, _ = _project_point_to_route_sl(ex, ey, route_xy, cum_s)
        lead_s, _ = _project_point_to_route_sl(lx, ly, route_xy, cum_s)
        out.append(ego_s - lead_s)
    return out


def evaluate_lane_change_result(case: LaneChangeCase, execution: LaneChangeExecutionResult, *, scenario_name: str = "lane_change_overtake") -> LaneChangeEvalResult:
    record_rows: List[Dict[str, Any]] = []
    final_summary: Dict[str, Any] = {}

    if execution.run_dir:
        record = _load_json(Path(execution.run_dir) / "record.json")
        if isinstance(record, list):
            record_rows = [r for r in record if isinstance(r, dict)]

        fs = _load_json(Path(execution.run_dir) / "final_summary.json")
        if isinstance(fs, dict):
            final_summary = fs

    plan_ms: List[float] = []
    min_obs: List[float] = []
    states: List[str] = []

    for row in record_rows:
        for key in ("timing_plan_total_ms", "timing_total_ms", "timing_plan_ms"):
            val = row.get(key)
            if isinstance(val, (int, float)):
                plan_ms.append(float(val))
                break

        d = row.get("min_obstacle_distance_m")
        if isinstance(d, (int, float)):
            min_obs.append(float(d))

        st = row.get("debug_behavior_state")
        if isinstance(st, str):
            states.append(st)

    # 1) state-based (preferred)
    state_lane_change_started = any(
        s in {"LANE_CHANGE_OUT", "CRUISE_PASS_LANE", "LANE_CHANGE_BACK"} for s in states
    )
    state_lane_change_completed = (
        "LANE_CHANGE_OUT" in states and ("CRUISE_PASS_LANE" in states or "LANE_CHANGE_BACK" in states)
    )
    state_returned_to_original_lane = "LANE_CHANGE_BACK" in states and states[-1] == "FOLLOW"

    # 2) geometry fallback based on ego lateral movement
    lateral_offsets = _extract_lateral_series(record_rows)
    y0 = lateral_offsets[0] if lateral_offsets else 0.0
    max_abs_lat = max((abs(v - y0) for v in lateral_offsets), default=0.0)
    final_abs_lat = abs((lateral_offsets[-1] - y0)) if lateral_offsets else 0.0
    # threshold defaults
    geom_lane_change_started = max_abs_lat >= 1.0
    geom_lane_change_completed = max_abs_lat >= 2.5
    geom_returned_to_original_lane = max_abs_lat >= 2.2 and final_abs_lat <= 0.8

    # Prefer state signal; if state is absent/incomplete, use geometry fallback.
    lane_change_started = state_lane_change_started or geom_lane_change_started
    lane_change_completed = state_lane_change_completed or geom_lane_change_completed
    returned_to_original_lane = state_returned_to_original_lane or geom_returned_to_original_lane

    # Route-progress-based overtake判定（优先）
    raw_route_xy = _extract_reference_route(execution.run_dir, record_rows)
    route_xy = _resample_route_by_arclength(raw_route_xy, ds=0.5)
    cum_s = _build_cum_s(route_xy)
    delta_s_series = _extract_delta_s_series_from_tracked_lead(record_rows, route_xy, cum_s)
    if not delta_s_series:
        delta_s_series = _extract_delta_s_series_from_obstacles(record_rows, route_xy, cum_s)

    initially_behind = bool(delta_s_series) and delta_s_series[0] < -2.0
    finally_ahead = bool(delta_s_series) and delta_s_series[-1] > 5.0
    stable_ahead = False
    if len(delta_s_series) >= 3:
        tail = delta_s_series[-3:]
        stable_ahead = all(v > 3.0 for v in tail)

    reach_goal = bool(final_summary.get("reach_goal", False)) or execution.run_result.get("reason") == "reached_goal"
    if delta_s_series:
        overtake_completed = lane_change_completed and initially_behind and finally_ahead and stable_ahead
    else:
        rel_long_series = [
            float(row["tracked_lead_relative_longitudinal_m"])
            for row in record_rows
            if isinstance(row.get("tracked_lead_relative_longitudinal_m"), (int, float))
        ]
        if rel_long_series:
            overtake_completed = (
                lane_change_completed
                and rel_long_series[0] < -2.0
                and rel_long_series[-1] > 5.0
            )
        else:
        # fallback: if lead-longitudinal debug exists, use sign crossing
            lead_long_series = [
                float(row["debug_lead_longitudinal"])
                for row in record_rows
                if isinstance(row.get("debug_lead_longitudinal"), (int, float))
            ]
            if lead_long_series:
                overtake_completed = (
                    lane_change_completed
                    and lead_long_series[0] > 2.0
                    and lead_long_series[-1] < -5.0
                )
            else:
                # last-resort fallback: do not fabricate overtake from goal reach alone
                overtake_completed = False

    timeout = bool(final_summary.get("timeout", False)) or execution.run_result.get("reason") in {"timeout", "max_steps_reached"}
    collision_count = int(final_summary.get("collision_count", 0))

    if not execution.success:
        fail_reason = execution.fail_reason or "scenario_error"
    elif collision_count > 0:
        fail_reason = "collision"
    elif timeout:
        fail_reason = "timeout"
    elif not lane_change_started:
        fail_reason = "lane_change_not_started"
    elif not lane_change_completed:
        fail_reason = "lane_change_not_completed"
    elif case.expected_success and not overtake_completed:
        fail_reason = "overtake_not_completed"
    else:
        fail_reason = None

    return LaneChangeEvalResult(
        scenario_name=scenario_name,
        case_id=case.case_id,
        success=fail_reason is None,
        fail_reason=fail_reason,
        collision_count=collision_count,
        timeout=timeout,
        time_to_finish=(float(final_summary.get("runtime_s")) if isinstance(final_summary.get("runtime_s"), (int, float)) else None),
        route_completion=1.0 if reach_goal else 0.0,
        planning_time_mean=(sum(plan_ms) / len(plan_ms)) if plan_ms else None,
        planning_time_p95=_percentile(plan_ms, 95.0),
        min_distance_to_lead_vehicle=min(min_obs) if min_obs else None,
        min_distance_to_adjacent_vehicle=None,
        lane_change_started=lane_change_started,
        lane_change_completed=lane_change_completed,
        overtake_completed=overtake_completed,
        returned_to_original_lane=returned_to_original_lane,
        run_dir=execution.run_dir,
        config_path=execution.config_path,
        run_stdout=execution.stdout,
        run_stderr=execution.stderr,
        input_case=asdict(case),
    )


def eval_result_to_dict(result: LaneChangeEvalResult) -> Dict[str, Any]:
    return asdict(result)
