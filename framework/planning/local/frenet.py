# framework/planning/local/frenet.py
# Frenet local planner (junction-stable, simplified):
# - windowed projection + yaw unwrap
# - stable s (jump/backtrack guard)
# - relative lateral sampling (l_target = l0 + l_off)
# - smooth lateral transition (linear ramp)
# - ALWAYS returns a trajectory (even if FAIL) for logging/controller robustness

from __future__ import annotations

import math
from dataclasses import dataclass
from bisect import bisect_right
from typing import Any, Dict, List, Optional, Tuple

from framework.core.types import (
    EgoState,
    WorldModel,
    PlanResult,
    PlanStatus,
    Route,
    Trajectory,
    TrajectoryPoint,
)
from framework.planning.base_planning import BasePlanner


# -----------------------------
# Small math helpers
# -----------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _unwrap_yaw(prev: float, cur: float) -> float:
    """unwrap cur so that it is closest to prev"""
    return prev + _wrap_pi(cur - prev)


def _hypot(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return dx * dx + dy * dy


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# -----------------------------
# Reference line built from Route
# -----------------------------
@dataclass
class RefPoint:
    x: float
    y: float
    yaw: float   # radians (UNWRAPPED)
    s: float     # cumulative arc-length


class ReferenceLine:
    """
    Reference line for Frenet planning:
    - built from Route(points=[Pose2D...])
    - project (x,y,yaw_hint) -> (s,l,ref_yaw, seg_idx, heading_err)
    - query by s -> (x,y,yaw_unwrapped)
    """

    def __init__(self, route: Route):
        if not route.points or len(route.points) < 2:
            raise ValueError("Route must contain at least 2 points.")

        pts = route.points
        self._pts: List[RefPoint] = []
        self._s_list: List[float] = []

        s = 0.0
        yaw0 = math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x)
        self._pts.append(RefPoint(pts[0].x, pts[0].y, yaw0, s))
        self._s_list.append(s)

        prev_yaw = yaw0
        for i in range(1, len(pts)):
            dx = pts[i].x - pts[i - 1].x
            dy = pts[i].y - pts[i - 1].y
            s += _hypot(dx, dy)

            if i < len(pts) - 1:
                yaw_raw = math.atan2(pts[i + 1].y - pts[i].y, pts[i + 1].x - pts[i].x)
            else:
                yaw_raw = math.atan2(pts[i].y - pts[i - 1].y, pts[i].x - pts[i - 1].x)

            yaw = _unwrap_yaw(prev_yaw, yaw_raw)
            prev_yaw = yaw
            self._pts.append(RefPoint(pts[i].x, pts[i].y, yaw, s))
            self._s_list.append(s)

        self.s_max = self._pts[-1].s

    @property
    def points(self) -> List[RefPoint]:
        return self._pts

    @staticmethod
    def _project_onto_segment(
        ax: float, ay: float, bx: float, by: float, x: float, y: float
    ) -> Tuple[float, float, float]:
        vx, vy = bx - ax, by - ay
        wx, wy = x - ax, y - ay
        vv = vx * vx + vy * vy
        if vv <= 1e-9:
            return ax, ay, 0.0
        t = (wx * vx + wy * vy) / vv
        t = _clamp(t, 0.0, 1.0)
        return ax + t * vx, ay + t * vy, t

    def _segment_proj_error(self, i: int, x: float, y: float) -> float:
        a = self._pts[i]
        b = self._pts[i + 1]
        px, py, _t = self._project_onto_segment(a.x, a.y, b.x, b.y, x, y)
        return _dist2(px, py, x, y)

    def _find_nearest_segment_window(
        self,
        x: float,
        y: float,
        *,
        hint_i: Optional[int],
        win_back: int,
        win_fwd: int,
    ) -> int:
        nseg = len(self._pts) - 1
        if nseg <= 1:
            return 0

        if hint_i is None:
            lo, hi = 0, nseg - 1
        else:
            lo = int(_clamp(hint_i - win_back, 0, nseg - 1))
            hi = int(_clamp(hint_i + win_fwd, 0, nseg - 1))

        best_i = lo
        best_err = float("inf")
        for i in range(lo, hi + 1):
            err = self._segment_proj_error(i, x, y)
            if err < best_err:
                best_err = err
                best_i = i
        return best_i

    def project_xy_to_sl(
        self,
        x: float,
        y: float,
        *,
        yaw_hint: Optional[float] = None,
        hint_i: Optional[int] = None,
        win_back: int = 20,
        win_fwd: int = 80,
        heading_gate_rad: float = math.radians(90.0),
    ) -> Tuple[float, float, float, int, float]:
        """
        Returns: (s, l, ref_yaw_unwrapped, seg_idx, heading_err_wrapped)
        """
        pts = self._pts

        # 1) windowed nearest segment
        i = self._find_nearest_segment_window(x, y, hint_i=hint_i, win_back=win_back, win_fwd=win_fwd)
        a = pts[i]
        b = pts[i + 1]
        px, py, t = self._project_onto_segment(a.x, a.y, b.x, b.y, x, y)
        s = _lerp(a.s, b.s, t)
        ref_yaw = a.yaw + t * (b.yaw - a.yaw)  # already unwrapped in pts

        heading_err = 0.0
        if yaw_hint is not None:
            heading_err = _wrap_pi(yaw_hint - ref_yaw)

            # 2) heading gate: if mismatch too big, try expanded window once
            if abs(heading_err) > heading_gate_rad and hint_i is not None:
                i2 = self._find_nearest_segment_window(
                    x, y, hint_i=hint_i, win_back=win_back * 2, win_fwd=win_fwd * 2
                )
                a2 = pts[i2]
                b2 = pts[i2 + 1]
                px2, py2, t2 = self._project_onto_segment(a2.x, a2.y, b2.x, b2.y, x, y)
                s2 = _lerp(a2.s, b2.s, t2)
                ref_yaw2 = a2.yaw + t2 * (b2.yaw - a2.yaw)
                heading_err2 = _wrap_pi(yaw_hint - ref_yaw2)

                if abs(heading_err2) < abs(heading_err):
                    i, px, py, t, s, ref_yaw, heading_err = i2, px2, py2, t2, s2, ref_yaw2, heading_err2

        # lateral offset (left positive)
        dx, dy = x - px, y - py
        nx, ny = -math.sin(ref_yaw), math.cos(ref_yaw)
        l = dx * nx + dy * ny

        return float(s), float(l), float(ref_yaw), int(i), float(heading_err)

    def query_by_s(self, s: float) -> Tuple[float, float, float]:
        """
        Binary-search by s and linear interpolate.
        Returns (x,y,yaw_unwrapped).
        """
        s = _clamp(s, 0.0, self.s_max)
        pts = self._pts
        sl = self._s_list

        if s <= sl[0]:
            p = pts[0]
            return float(p.x), float(p.y), float(p.yaw)
        if s >= sl[-1]:
            p = pts[-1]
            return float(p.x), float(p.y), float(p.yaw)

        # find j such that sl[j] <= s < sl[j+1]
        j = bisect_right(sl, s) - 1
        j = _clamp(j, 0, len(pts) - 2)

        a = pts[int(j)]
        b = pts[int(j) + 1]
        denom = (b.s - a.s)
        t = 0.0 if denom <= 1e-9 else (s - a.s) / denom
        x = _lerp(a.x, b.x, t)
        y = _lerp(a.y, b.y, t)
        yaw = a.yaw + t * (b.yaw - a.yaw)  # unwrapped
        return float(x), float(y), float(yaw)

    def frenet_to_xy(self, s: float, l: float) -> Tuple[float, float, float]:
        rx, ry, ryaw = self.query_by_s(s)
        nx, ny = -math.sin(ryaw), math.cos(ryaw)
        return float(rx + l * nx), float(ry + l * ny), float(ryaw)


# -----------------------------
# Frenet planner config
# -----------------------------
@dataclass
class FrenetConfig:
    dt: float = 0.1
    horizon_s: float = 4.0
    target_speed: float = 10.0

    lateral_offsets: Tuple[float, ...] = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
    speed_samples: Tuple[float, ...] = (0.0, 2.0, 6.0, 8.0, 10.0, 12.0)  # ✅ include low/stop

    ego_radius: float = 1.2
    min_clearance: float = 0.2
    max_curvature: float = 0.25
    max_yaw_rate: float = 1.2

    w_offset: float = 1.0
    w_speed: float = 0.5
    w_jerk: float = 0.2
    w_curvature: float = 0.5
    w_collision: float = 1000.0

    prefer_center: bool = True
    stop_at_route_end: bool = True

    # projection stability
    proj_win_back: int = 20
    proj_win_fwd: int = 80
    proj_heading_gate_deg: float = 90.0

    # s guard against wrong-branch projection
    max_s_backtrack: float = 2.0
    max_s_jump: float = 30.0

    # NEW: smooth lateral transition
    lateral_ramp_time: float = 1.0  # seconds, l0 -> l_target


# -----------------------------
# Frenet local planner
# -----------------------------
class FrenetPlanner(BasePlanner):
    name = "frenet"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = self.config or {}

        self.cfg = FrenetConfig(
            dt=float(cfg.get("dt", 0.1)),
            horizon_s=float(cfg.get("horizon_s", 4.0)),
            target_speed=float(cfg.get("target_speed", 10.0)),
            lateral_offsets=tuple(cfg.get("lateral_offsets", (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5))),
            speed_samples=tuple(cfg.get("speed_samples", (0.0, 2.0, 6.0, 8.0, 10.0, 12.0))),
            ego_radius=float(cfg.get("ego_radius", 1.2)),
            min_clearance=float(cfg.get("min_clearance", 0.2)),
            max_curvature=float(cfg.get("max_curvature", 0.25)),
            max_yaw_rate=float(cfg.get("max_yaw_rate", 1.2)),
            w_offset=float(cfg.get("w_offset", 1.0)),
            w_speed=float(cfg.get("w_speed", 0.5)),
            w_jerk=float(cfg.get("w_jerk", 0.2)),
            w_curvature=float(cfg.get("w_curvature", 0.5)),
            w_collision=float(cfg.get("w_collision", 1000.0)),
            prefer_center=bool(cfg.get("prefer_center", True)),
            stop_at_route_end=bool(cfg.get("stop_at_route_end", True)),
            proj_win_back=int(cfg.get("proj_win_back", 20)),
            proj_win_fwd=int(cfg.get("proj_win_fwd", 80)),
            proj_heading_gate_deg=float(cfg.get("proj_heading_gate_deg", 90.0)),
            max_s_backtrack=float(cfg.get("max_s_backtrack", 2.0)),
            max_s_jump=float(cfg.get("max_s_jump", 30.0)),
            lateral_ramp_time=float(cfg.get("lateral_ramp_time", 1.0)),
        )

        self._route: Optional[Route] = None
        self._ref: Optional[ReferenceLine] = None

        # projection memory
        self._ref_hint_idx: Optional[int] = None
        self._s_prev: Optional[float] = None

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route
        self._ref = ReferenceLine(route)
        self._ref_hint_idx = None
        self._s_prev = None

    # ----------------------------- stable projection -----------------------------

    def _stable_project(self, ego: EgoState) -> Tuple[float, float, float, int, float, Dict[str, Any]]:
        """
        Encapsulates:
        - window projection around previous segment
        - heading gate
        - s jump/backtrack guard
        """
        assert self._ref is not None
        ref = self._ref

        s0, l0, ref_yaw0, seg_idx, heading_err = ref.project_xy_to_sl(
            ego.pose.x,
            ego.pose.y,
            yaw_hint=ego.pose.yaw,
            hint_i=self._ref_hint_idx,
            win_back=self.cfg.proj_win_back,
            win_fwd=self.cfg.proj_win_fwd,
            heading_gate_rad=math.radians(self.cfg.proj_heading_gate_deg),
        )

        # guard s
        if self._s_prev is not None:
            ds = s0 - self._s_prev
            if ds < -self.cfg.max_s_backtrack:
                s0 = self._s_prev - self.cfg.max_s_backtrack
            if abs(ds) > self.cfg.max_s_jump:
                # likely wrong branch, do NOT jump
                s0 = self._s_prev

        self._s_prev = s0
        self._ref_hint_idx = seg_idx

        dbg = {
            "proj_seg_idx": int(seg_idx),
            "ego_s0": float(s0),
            "ego_l0": float(l0),
            "proj_heading_err": float(heading_err),
        }
        return s0, l0, ref_yaw0, seg_idx, heading_err, dbg

    # ----------------------------- planning -----------------------------

    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        if self._ref is None:
            return PlanResult(status=PlanStatus.EMPTY, trajectory=None, debug={"reason": "no_reference"})

        dt = self.cfg.dt
        steps = max(5, int(self.cfg.horizon_s / max(dt, 1e-6)))

        s0, l0, ref_yaw0, seg_idx, heading_err, proj_dbg = self._stable_project(ego)

        candidates: List[Tuple[float, Trajectory, Dict[str, Any]]] = []

        for l_off in self.cfg.lateral_offsets:
            l_target = l0 + float(l_off)  # relative sampling
            for v_target in self.cfg.speed_samples:
                traj, dbg_traj = self._build_candidate(
                    ref=self._ref,
                    s0=s0,
                    l0=l0,
                    l_target=l_target,
                    v_target=float(v_target),
                    steps=steps,
                    dt=dt,
                )
                valid, vinfo = self._validate(traj, world)
                cost, cinfo = self._score(
                    l_off=float(l_off),
                    v_target=float(v_target),
                    valid=valid,
                    vinfo=vinfo,
                )

                dbg = {
                    **proj_dbg,
                    "l_off": float(l_off),
                    "l_target": float(l_target),
                    "v_target": float(v_target),
                    **dbg_traj,
                    **vinfo,
                    **cinfo,
                    "valid": bool(valid),
                }
                candidates.append((cost, traj, dbg))

        if not candidates:
            # Should not happen because we always build a traj, but keep it safe
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={**proj_dbg, "reason": "no_candidates"},
            )

        candidates.sort(key=lambda x: x[0])
        best_cost, best_traj, best_dbg = candidates[0]

        status = PlanStatus.OK if best_dbg.get("valid", False) else PlanStatus.FAIL

        # ✅ ALWAYS return a trajectory (even if FAIL) so logs/CSV don't lose columns
        return PlanResult(
            status=status,
            trajectory=best_traj,
            debug={
                **proj_dbg,
                "best_cost": float(best_cost),
                "num_candidates": int(len(candidates)),
                "best": best_dbg,
            },
        )

    # ----------------------------- candidate generation -----------------------------

    def _l_ramp(self, l0: float, l_target: float, k: int, dt: float) -> float:
        """
        Simple linear ramp l0 -> l_target in lateral_ramp_time seconds.
        Keeps code simple but fixes the 'instant lateral jump' issue.
        """
        T = max(1e-3, float(self.cfg.lateral_ramp_time))
        t = min(k * dt, T)
        alpha = t / T
        return (1.0 - alpha) * l0 + alpha * l_target

    def _build_candidate(
        self,
        *,
        ref: ReferenceLine,
        s0: float,
        l0: float,
        l_target: float,
        v_target: float,
        steps: int,
        dt: float,
    ) -> Tuple[Trajectory, Dict[str, Any]]:
        """
        MVP+: constant v, but smooth l transition.
        """
        pts: List[TrajectoryPoint] = []

        v = max(0.0, float(v_target))
        s = float(s0)

        for k in range(steps):
            l_k = self._l_ramp(l0, l_target, k, dt)

            if k > 0:
                s_next = s + v * dt
                if self.cfg.stop_at_route_end and s_next >= ref.s_max:
                    s_next = ref.s_max
                s = s_next

            x, y, _ = ref.frenet_to_xy(s, l_k)
            pts.append(TrajectoryPoint(x=float(x), y=float(y), yaw=0.0, v=float(v)))

            if self.cfg.stop_at_route_end and abs(ref.s_max - s) < 1e-6:
                # pad with stop points
                for _ in range(k + 1, steps):
                    pts.append(TrajectoryPoint(x=float(x), y=float(y), yaw=0.0, v=0.0))
                break

        pts = self._recompute_yaw(pts)
        return Trajectory(points=pts, dt=float(dt)), {"steps": int(len(pts))}

    @staticmethod
    def _recompute_yaw(pts: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        out: List[TrajectoryPoint] = []
        prev_yaw: Optional[float] = None

        for i in range(len(pts)):
            if i == 0:
                dx = pts[1].x - pts[0].x
                dy = pts[1].y - pts[0].y
            else:
                dx = pts[i].x - pts[i - 1].x
                dy = pts[i].y - pts[i - 1].y

            yaw_raw = math.atan2(dy, dx) if (abs(dx) + abs(dy)) > 1e-9 else 0.0
            yaw_unwrapped = yaw_raw if prev_yaw is None else _unwrap_yaw(prev_yaw, yaw_raw)
            prev_yaw = yaw_unwrapped

            out.append(
                TrajectoryPoint(
                    x=float(pts[i].x),
                    y=float(pts[i].y),
                    yaw=float(_wrap_pi(yaw_unwrapped)),
                    v=float(pts[i].v),
                )
            )
        return out

    # ----------------------------- validation -----------------------------

    def _validate(self, traj: Trajectory, world: WorldModel) -> Tuple[bool, Dict[str, Any]]:
        min_dist = float("inf")
        collision = False
        ego_r = self.cfg.ego_radius + self.cfg.min_clearance


        # collision (circle-circle)

        pts = traj.points
        if pts and world.obstacles:
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            inflated_obstacles = []

            #这里加入obstacles的坐标，并取圆

            for ob in world.obstacles:
                r_sum = ego_r + ob.radius
                ox = ob.position.x
                oy = ob.position.y
                if ox < (min_x - r_sum) or ox > (max_x + r_sum) or oy < (min_y - r_sum) or oy > (max_y + r_sum):
                    continue
                inflated_obstacles.append((ox, oy, r_sum, r_sum * r_sum))

            for p in pts:
                for ox, oy, r_sum, r2 in inflated_obstacles:
                    dx = p.x - ox
                    dy = p.y - oy
                    d2 = dx * dx + dy * dy
                    if d2 <= r2:
                        min_dist = min(min_dist, _hypot(dx, dy) - r_sum)
                        collision = True
                        break

                    d = _hypot(dx, dy) - r_sum
                    if d < min_dist:
                        min_dist = d
                if collision:
                    break

                if collision:
                    break

                if collision:
                    break

                if collision:
                    break


        # kinematics checks
        max_kappa = 0.0
        max_yaw_rate = 0.0

        dt = float(traj.dt)

        for i in range(2, len(pts)):
            kappa = self._curvature_from_three_points(
                pts[i - 2].x, pts[i - 2].y,
                pts[i - 1].x, pts[i - 1].y,
                pts[i].x, pts[i].y,
            )
            if abs(kappa) > max_kappa:
                max_kappa = abs(kappa)

        for i in range(1, len(pts)):
            dyaw = _wrap_pi(pts[i].yaw - pts[i - 1].yaw)
            max_yaw_rate = max(max_yaw_rate, abs(dyaw / max(dt, 1e-6)))

        valid = (not collision) and (max_kappa <= self.cfg.max_curvature) and (max_yaw_rate <= self.cfg.max_yaw_rate)

        return valid, {
            "min_clearance_m": min_dist if min_dist != float("inf") else None,
            "collision": bool(collision),
            "max_curvature": float(max_kappa),
            "max_yaw_rate": float(max_yaw_rate),
        }

    @staticmethod
    def _curvature_from_three_points(x1: float, y1: float, x2: float, y2: float, x3: float, y3: float) -> float:
        a = _hypot(x2 - x1, y2 - y1)
        b = _hypot(x3 - x2, y3 - y2)
        c = _hypot(x3 - x1, y3 - y1)
        if a * b * c <= 1e-9:
            return 0.0

        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if area2 <= 1e-9:
            return 0.0

        return (2.0 * area2) / (a * b * c)

    # ----------------------------- scoring -----------------------------

    def _score(
        self,
        *,
        l_off: float,
        v_target: float,
        valid: bool,
        vinfo: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        # offset cost uses relative offset magnitude (good for junction stability)
        c_offset = abs(float(l_off)) if self.cfg.prefer_center else 0.0
        c_speed = abs(float(v_target) - float(self.cfg.target_speed))
        c_jerk = 0.0  # reserved for future variable-speed profiles
        c_curv = float(vinfo.get("max_curvature", 0.0) or 0.0)

        collision = bool(vinfo.get("collision", False))
        c_collision = 1.0 if collision else 0.0

        cost = (
            self.cfg.w_offset * c_offset
            + self.cfg.w_speed * c_speed
            + self.cfg.w_jerk * c_jerk
            + self.cfg.w_curvature * c_curv
            + self.cfg.w_collision * c_collision
        )

        if not valid and not collision:
            cost *= 10.0

        return float(cost), {
            "cost_total": float(cost),
            "cost_offset": float(self.cfg.w_offset * c_offset),
            "cost_speed": float(self.cfg.w_speed * c_speed),
            "cost_jerk": float(self.cfg.w_jerk * c_jerk),
            "cost_curvature": float(self.cfg.w_curvature * c_curv),
            "cost_collision": float(self.cfg.w_collision * c_collision),
        }
