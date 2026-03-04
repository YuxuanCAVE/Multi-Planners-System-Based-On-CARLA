from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from framework.core.types import (
    EgoState,
    PlanResult,
    PlanStatus,
    Route,
    Trajectory,
    TrajectoryPoint,
    WorldModel,
)
from framework.planning.base_planning import BasePlanner


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _unwrap_yaw(prev: float, cur: float) -> float:
    return prev + _wrap_pi(cur - prev)


def _hypot(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


@dataclass
class _RefPoint:
    x: float
    y: float
    yaw: float
    s: float


class _ReferenceLine:
    def __init__(self, route: Route):
        if not route.points or len(route.points) < 2:
            raise ValueError("Route must contain at least 2 points.")

        pts = route.points
        self._pts: List[_RefPoint] = []
        self._s_list: List[float] = []

        s = 0.0
        yaw0 = math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x)
        self._pts.append(_RefPoint(pts[0].x, pts[0].y, yaw0, s))
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
            self._pts.append(_RefPoint(pts[i].x, pts[i].y, yaw, s))
            self._s_list.append(s)

        self.s_max = self._pts[-1].s

    @staticmethod
    def _project_onto_segment(
        ax: float,
        ay: float,
        bx: float,
        by: float,
        x: float,
        y: float,
    ) -> Tuple[float, float, float]:
        vx, vy = bx - ax, by - ay
        wx, wy = x - ax, y - ay
        vv = vx * vx + vy * vy
        if vv <= 1e-9:
            return ax, ay, 0.0
        t = (wx * vx + wy * vy) / vv
        t = _clamp(t, 0.0, 1.0)
        return ax + t * vx, ay + t * vy, t

    def project_xy(self, x: float, y: float, hint_i: int = 0, win: int = 80) -> Tuple[float, float, float, int]:
        nseg = len(self._pts) - 1
        lo = int(_clamp(hint_i - win // 2, 0, nseg - 1))
        hi = int(_clamp(hint_i + win, 0, nseg - 1))

        best_i = lo
        best_d2 = float("inf")
        best_px = self._pts[lo].x
        best_py = self._pts[lo].y
        best_t = 0.0

        for i in range(lo, hi + 1):
            a = self._pts[i]
            b = self._pts[i + 1]
            px, py, t = self._project_onto_segment(a.x, a.y, b.x, b.y, x, y)
            dx = x - px
            dy = y - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_px, best_py, best_t = px, py, t

        a = self._pts[best_i]
        b = self._pts[best_i + 1]
        s = a.s + (b.s - a.s) * best_t
        ryaw = a.yaw + (b.yaw - a.yaw) * best_t
        nx, ny = -math.sin(ryaw), math.cos(ryaw)
        l = (x - best_px) * nx + (y - best_py) * ny
        return float(s), float(l), float(ryaw), int(best_i)

    def query_by_s(self, s: float) -> Tuple[float, float, float]:
        s = _clamp(s, 0.0, self.s_max)
        if s <= self._s_list[0]:
            p = self._pts[0]
            return p.x, p.y, p.yaw
        if s >= self._s_list[-1]:
            p = self._pts[-1]
            return p.x, p.y, p.yaw

        j = bisect_right(self._s_list, s) - 1
        j = int(_clamp(j, 0, len(self._pts) - 2))
        a = self._pts[j]
        b = self._pts[j + 1]
        t = 0.0 if (b.s - a.s) <= 1e-9 else (s - a.s) / (b.s - a.s)
        x = a.x + (b.x - a.x) * t
        y = a.y + (b.y - a.y) * t
        yaw = a.yaw + (b.yaw - a.yaw) * t
        return float(x), float(y), float(yaw)

    def frenet_to_xy(self, s: float, l: float) -> Tuple[float, float, float]:
        x, y, yaw = self.query_by_s(s)
        nx, ny = -math.sin(yaw), math.cos(yaw)
        return float(x + nx * l), float(y + ny * l), float(yaw)


@dataclass
class LatticeConfig:
    dt: float = 0.1
    horizon_s: float = 4.0
    target_speed: float = 8.0

    lateral_offsets: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    speed_samples: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)

    lateral_ramp_time: float = 1.0
    max_curvature: float = 0.35
    max_yaw_rate: float = 2.0

    ego_radius: float = 1.2
    min_clearance: float = 0.2

    w_offset: float = 1.0
    w_speed: float = 0.3
    w_curvature: float = 0.5
    w_collision: float = 1000.0


class LatticePlanner(BasePlanner):
    name = "lattice"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        cfg = self.config or {}
        self.cfg = LatticeConfig(
            dt=float(cfg.get("dt", 0.1)),
            horizon_s=float(cfg.get("horizon_s", 4.0)),
            target_speed=float(cfg.get("target_speed", 8.0)),
            lateral_offsets=tuple(cfg.get("lateral_offsets", (-1.0, -0.5, 0.0, 0.5, 1.0))),
            speed_samples=tuple(cfg.get("speed_samples", (4.0, 6.0, 8.0, 10.0))),
            lateral_ramp_time=float(cfg.get("lateral_ramp_time", 1.0)),
            max_curvature=float(cfg.get("max_curvature", 0.35)),
            max_yaw_rate=float(cfg.get("max_yaw_rate", 2.0)),
            ego_radius=float(cfg.get("ego_radius", 1.2)),
            min_clearance=float(cfg.get("min_clearance", 0.2)),
            w_offset=float(cfg.get("w_offset", 1.0)),
            w_speed=float(cfg.get("w_speed", 0.3)),
            w_curvature=float(cfg.get("w_curvature", 0.5)),
            w_collision=float(cfg.get("w_collision", 1000.0)),
        )
        self._ref: Optional[_ReferenceLine] = None
        self._last_seg_idx: int = 0

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._ref = _ReferenceLine(route)
        self._last_seg_idx = 0

    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        if self._ref is None:
            return PlanResult(status=PlanStatus.EMPTY, trajectory=None, debug={"reason": "no_route"})

        ref = self._ref
        s0, l0, _ryaw, seg_idx = ref.project_xy(ego.pose.x, ego.pose.y, hint_i=self._last_seg_idx)
        self._last_seg_idx = seg_idx

        best: Optional[Dict[str, Any]] = None
        num_candidates = 0

        for l_off in self.cfg.lateral_offsets:
            l_target = l0 + float(l_off)
            for v_target in self.cfg.speed_samples:
                traj = self._sample_candidate(
                    ref=ref,
                    s0=s0,
                    l0=l0,
                    l_target=l_target,
                    v_target=float(v_target),
                )
                if traj is None:
                    continue

                num_candidates += 1
                valid, vinfo = self._validate(traj, world)
                cost = (
                    self.cfg.w_offset * abs(float(l_off))
                    + self.cfg.w_speed * abs(float(v_target) - self.cfg.target_speed)
                    + self.cfg.w_curvature * float(vinfo["max_curvature"])
                    + self.cfg.w_collision * (1.0 if bool(vinfo["collision"]) else 0.0)
                )
                if not valid and not bool(vinfo["collision"]):
                    cost *= 10.0

                cur = {
                    "cost": float(cost),
                    "valid": bool(valid),
                    "traj": traj,
                    "l_target": float(l_target),
                    "v_target": float(v_target),
                    **vinfo,
                }
                if best is None or cur["cost"] < best["cost"]:
                    best = cur

        if best is None:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={
                    "reason": "no_candidate",
                    "num_candidates": int(num_candidates),
                    "ego_s0": float(s0),
                    "ego_l0": float(l0),
                },
            )

        status = PlanStatus.OK if bool(best["valid"]) else PlanStatus.FAIL
        return PlanResult(
            status=status,
            trajectory=best["traj"],
            debug={
                "num_candidates": int(num_candidates),
                "ego_s0": float(s0),
                "ego_l0": float(l0),
                "best": {
                    "cost_total": float(best["cost"]),
                    "l_target": float(best["l_target"]),
                    "v_target": float(best["v_target"]),
                    "valid": bool(best["valid"]),
                    "min_clearance_m": best["min_clearance_m"],
                    "collision": bool(best["collision"]),
                    "max_curvature": float(best["max_curvature"]),
                    "max_yaw_rate": float(best["max_yaw_rate"]),
                },
            },
        )

    def _sample_candidate(
        self,
        *,
        ref: _ReferenceLine,
        s0: float,
        l0: float,
        l_target: float,
        v_target: float,
    ) -> Optional[Trajectory]:
        dt = self.cfg.dt
        steps = max(2, int(self.cfg.horizon_s / max(dt, 1e-6)))
        ramp_steps = max(1, int(self.cfg.lateral_ramp_time / max(dt, 1e-6)))

        pts: List[TrajectoryPoint] = []
        s = s0
        for k in range(steps):
            if k == 0:
                v = max(0.0, v_target)
            else:
                s = min(ref.s_max, s + v_target * dt)
                v = max(0.0, v_target)

            alpha = _clamp(k / ramp_steps, 0.0, 1.0)
            l = l0 + (l_target - l0) * alpha
            x, y, _yaw = ref.frenet_to_xy(s, l)
            pts.append(TrajectoryPoint(x=float(x), y=float(y), yaw=0.0, v=float(v)))

        if len(pts) < 2:
            return None
        pts = self._recompute_yaw(pts)
        return Trajectory(points=pts, dt=float(dt))

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
            yaw_raw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-9 else 0.0
            yaw = yaw_raw if prev_yaw is None else _unwrap_yaw(prev_yaw, yaw_raw)
            prev_yaw = yaw
            out.append(TrajectoryPoint(x=pts[i].x, y=pts[i].y, yaw=_wrap_pi(yaw), v=pts[i].v))
        return out

    def _validate(self, traj: Trajectory, world: WorldModel) -> Tuple[bool, Dict[str, Any]]:
        min_dist = float("inf")
        collision = False
        ego_r = self.cfg.ego_radius + self.cfg.min_clearance

        for p in traj.points:
            for ob in world.obstacles:
                d = _hypot(p.x - ob.position.x, p.y - ob.position.y) - (ego_r + ob.radius)
                if d < min_dist:
                    min_dist = d
                if d <= 0.0:
                    collision = True
                    break
            if collision:
                break

        max_kappa = 0.0
        max_yaw_rate = 0.0
        dt = float(traj.dt)
        pts = traj.points

        for i in range(2, len(pts)):
            kappa = self._curvature_from_three_points(
                pts[i - 2].x,
                pts[i - 2].y,
                pts[i - 1].x,
                pts[i - 1].y,
                pts[i].x,
                pts[i].y,
            )
            max_kappa = max(max_kappa, abs(kappa))

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
