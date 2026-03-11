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
from framework.vehicle import VehicleModel, build_vehicle_model


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

    @classmethod
    def from_xy_points(cls, xy_points: List[Tuple[float, float]]) -> "_ReferenceLine":
        if len(xy_points) < 2:
            raise ValueError("Need at least 2 xy points to build a reference line.")

        obj = cls.__new__(cls)
        obj._pts = []
        obj._s_list = []

        s = 0.0
        yaw0 = math.atan2(
            xy_points[1][1] - xy_points[0][1],
            xy_points[1][0] - xy_points[0][0],
        )
        obj._pts.append(_RefPoint(xy_points[0][0], xy_points[0][1], yaw0, s))
        obj._s_list.append(s)

        prev_yaw = yaw0
        for i in range(1, len(xy_points)):
            dx = xy_points[i][0] - xy_points[i - 1][0]
            dy = xy_points[i][1] - xy_points[i - 1][1]
            s += _hypot(dx, dy)

            if i < len(xy_points) - 1:
                yaw_raw = math.atan2(
                    xy_points[i + 1][1] - xy_points[i][1],
                    xy_points[i + 1][0] - xy_points[i][0],
                )
            else:
                yaw_raw = math.atan2(
                    xy_points[i][1] - xy_points[i - 1][1],
                    xy_points[i][0] - xy_points[i - 1][0],
                )

            yaw = _unwrap_yaw(prev_yaw, yaw_raw)
            prev_yaw = yaw
            obj._pts.append(_RefPoint(xy_points[i][0], xy_points[i][1], yaw, s))
            obj._s_list.append(s)

        obj.s_max = obj._pts[-1].s
        return obj

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

    def project_xy(
        self,
        x: float,
        y: float,
        hint_i: int = 0,
        win: int = 80,
    ) -> Tuple[float, float, float, int]:
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
    horizon_t: float = 6.0
    target_speed: float = 4.0

    # 当前稳定版：不做局部横向偏移采样
    speed_samples: Tuple[float, ...] = (3.5, 4.0, 4.5, 5.0)

    # 默认只走当前车道；后续需要时可改成 (-3.5, 0.0, 3.5)
    virtual_lane_targets: Tuple[float, ...] = (0.0,)
    virtual_route_ds: float = 1.0
    virtual_route_preview_m: float = 35.0
    virtual_transition_length_m: float = 20.0

    # 参考线缓存：ego 前进超过该距离才重建
    virtual_rebuild_delta_s_m: float = 4.0

    w_speed: float = 0.2
    w_curvature: float = 0.2
    w_lateral_accel: float = 0.05
    w_collision: float = 1000.0
    w_clearance: float = 0.05
    w_lane_change: float = 0.3


class LatticePlanner(BasePlanner):
    name = "lattice"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        root_cfg = self.config or {}
        pcfg = root_cfg.get("planner", root_cfg)
        if not isinstance(pcfg, dict):
            pcfg = {}
        if "config" in pcfg and isinstance(pcfg["config"], dict):
            pcfg = pcfg["config"]

        self.cfg = LatticeConfig(
            dt=float(pcfg.get("dt", 0.1)),
            horizon_t=float(pcfg.get("horizon_t", pcfg.get("horizon_s", 6.0))),
            target_speed=float(pcfg.get("target_speed", 4.0)),
            speed_samples=tuple(pcfg.get("speed_samples", (3.5, 4.0, 4.5, 5.0))),
            virtual_lane_targets=tuple(pcfg.get("virtual_lane_targets", (0.0,))),
            virtual_route_ds=float(pcfg.get("virtual_route_ds", 1.0)),
            virtual_route_preview_m=float(pcfg.get("virtual_route_preview_m", 35.0)),
            virtual_transition_length_m=float(pcfg.get("virtual_transition_length_m", 20.0)),
            virtual_rebuild_delta_s_m=float(pcfg.get("virtual_rebuild_delta_s_m", 4.0)),
            w_speed=float(pcfg.get("w_speed", 0.2)),
            w_curvature=float(pcfg.get("w_curvature", 0.2)),
            w_lateral_accel=float(pcfg.get("w_lateral_accel", 0.05)),
            w_collision=float(pcfg.get("w_collision", 1000.0)),
            w_clearance=float(pcfg.get("w_clearance", 0.05)),
            w_lane_change=float(pcfg.get("w_lane_change", 0.3)),
        )

        self.vehicle: VehicleModel = build_vehicle_model(root_cfg)
        self._ref: Optional[_ReferenceLine] = None
        self._last_seg_idx: int = 0

        # 缓存当前虚拟参考线，减少每 tick 重建带来的抖动
        self._cached_lane_target_l: Optional[float] = None
        self._cached_virtual_ref: Optional[_ReferenceLine] = None
        self._cached_virtual_ref_start_s: Optional[float] = None

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._ref = _ReferenceLine(route)
        self._last_seg_idx = 0
        self._cached_lane_target_l = None
        self._cached_virtual_ref = None
        self._cached_virtual_ref_start_s = None

    @staticmethod
    def _solve_quintic_lateral(
        l0: float,
        lT: float,
        T: float,
    ) -> Tuple[float, float, float, float, float, float]:
        T = max(T, 1e-3)

        a0 = l0
        a1 = 0.0
        a2 = 0.0

        dl = lT - l0
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        a3 = 10.0 * dl / T3
        a4 = -15.0 * dl / T4
        a5 = 6.0 * dl / T5
        return a0, a1, a2, a3, a4, a5

    @staticmethod
    def _eval_quintic(
        coeffs: Tuple[float, float, float, float, float, float],
        t: float,
    ) -> float:
        a0, a1, a2, a3, a4, a5 = coeffs
        t2 = t * t
        t3 = t2 * t
        t4 = t3 * t
        t5 = t4 * t
        return a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5

    def _build_virtual_reference(
        self,
        *,
        base_ref: _ReferenceLine,
        s0: float,
        l0: float,
        target_lane_l: float,
    ) -> _ReferenceLine:
        ds = max(0.2, float(self.cfg.virtual_route_ds))
        preview_m = max(ds * 2.0, float(self.cfg.virtual_route_preview_m))
        transition_m = max(ds, float(self.cfg.virtual_transition_length_m))

        coeffs = self._solve_quintic_lateral(l0, target_lane_l, transition_m)

        xy_points: List[Tuple[float, float]] = []
        n = max(2, int(preview_m / ds) + 1)

        for k in range(n):
            s = min(base_ref.s_max, s0 + k * ds)
            s_rel = min(s - s0, transition_m)
            l_center = self._eval_quintic(coeffs, s_rel)
            x, y, _ = base_ref.frenet_to_xy(s, l_center)
            xy_points.append((float(x), float(y)))

            if s >= base_ref.s_max - 1e-6:
                break

        return _ReferenceLine.from_xy_points(xy_points)

    def _get_virtual_reference(
        self,
        *,
        base_ref: _ReferenceLine,
        s0: float,
        l0: float,
        lane_target_l: float,
    ) -> _ReferenceLine:
        need_rebuild = False

        if self._cached_virtual_ref is None:
            need_rebuild = True
        elif self._cached_lane_target_l is None:
            need_rebuild = True
        elif abs(float(self._cached_lane_target_l) - float(lane_target_l)) > 1e-6:
            need_rebuild = True
        elif self._cached_virtual_ref_start_s is None:
            need_rebuild = True
        elif abs(float(s0) - float(self._cached_virtual_ref_start_s)) >= float(self.cfg.virtual_rebuild_delta_s_m):
            need_rebuild = True

        if need_rebuild:
            self._cached_virtual_ref = self._build_virtual_reference(
                base_ref=base_ref,
                s0=s0,
                l0=l0,
                target_lane_l=lane_target_l,
            )
            self._cached_lane_target_l = float(lane_target_l)
            self._cached_virtual_ref_start_s = float(s0)

        return self._cached_virtual_ref

    def _sample_candidate(
        self,
        *,
        ref: _ReferenceLine,
        s0: float,
        v0: float,
        v_target: float,
    ) -> Optional[Trajectory]:
        dt = self.cfg.dt
        steps = max(2, int(self.cfg.horizon_t / max(dt, 1e-6)))

        pts: List[TrajectoryPoint] = []
        s = s0

        T = max(self.cfg.horizon_t, dt)
        a_cmd = (v_target - v0) / T
        a_cmd = _clamp(
            a_cmd,
            -self.vehicle.limits.max_decel_mps2,
            self.vehicle.limits.max_accel_mps2,
        )
        v = max(0.0, min(self.vehicle.limits.max_speed_mps, v0))

        for k in range(steps):
            if k > 0:
                v = max(
                    0.0,
                    min(self.vehicle.limits.max_speed_mps, v + a_cmd * dt),
                )
                s = min(ref.s_max, s + v * dt)

            x, y, yaw = ref.frenet_to_xy(s, 0.0)
            pts.append(
                TrajectoryPoint(
                    x=float(x),
                    y=float(y),
                    yaw=float(yaw),
                    v=float(v),
                )
            )

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

            out.append(
                TrajectoryPoint(
                    x=pts[i].x,
                    y=pts[i].y,
                    yaw=_wrap_pi(yaw),
                    v=pts[i].v,
                )
            )
        return out

    def _validate(self, traj: Trajectory, world: WorldModel) -> Tuple[bool, Dict[str, Any]]:
        min_dist = float("inf")
        collision = False
        ego_r = self.vehicle.ego_collision_radius

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
        max_ay = 0.0
        max_steer = 0.0
        max_steer_rate = 0.0
        max_yaw_rate = 0.0

        dt = float(traj.dt)
        pts = traj.points
        steer_list: List[float] = []

        for i in range(2, len(pts)):
            kappa = self._curvature_from_three_points(
                pts[i - 2].x, pts[i - 2].y,
                pts[i - 1].x, pts[i - 1].y,
                pts[i].x, pts[i].y,
            )
            max_kappa = max(max_kappa, abs(kappa))

            v_here = float(pts[i].v)
            ay = abs(self.vehicle.lateral_accel(v_here, kappa))
            max_ay = max(max_ay, ay)

            steer_signed = self.vehicle.steer_from_curvature(kappa)
            steer_abs = abs(steer_signed)
            max_steer = max(max_steer, steer_abs)
            steer_list.append(steer_signed)

        for i in range(1, len(pts)):
            dyaw = _wrap_pi(pts[i].yaw - pts[i - 1].yaw)
            yaw_rate = abs(dyaw / max(dt, 1e-6))
            max_yaw_rate = max(max_yaw_rate, yaw_rate)

        for i in range(1, len(steer_list)):
            dsteer = steer_list[i] - steer_list[i - 1]
            steer_rate = abs(dsteer / max(dt, 1e-6))
            max_steer_rate = max(max_steer_rate, steer_rate)

        valid = (
            (not collision)
            and (max_kappa <= self.vehicle.max_curvature + 1e-6)
        )

        fail_reason = "ok"
        if collision:
            fail_reason = "collision"
        elif max_kappa > self.vehicle.max_curvature + 1e-6:
            fail_reason = "curvature"

        return valid, {
            "min_clearance_m": min_dist if min_dist != float("inf") else None,
            "collision": bool(collision),
            "max_curvature": float(max_kappa),
            "max_lateral_accel": float(max_ay),
            "max_steer_rad": float(max_steer),
            "max_steer_rate_rad_s": float(max_steer_rate),
            "max_yaw_rate": float(max_yaw_rate),
            "fail_reason": fail_reason,
        }

    @staticmethod
    def _curvature_from_three_points(
        x1: float, y1: float,
        x2: float, y2: float,
        x3: float, y3: float,
    ) -> float:
        a = _hypot(x2 - x1, y2 - y1)
        b = _hypot(x3 - x2, y3 - y2)
        c = _hypot(x3 - x1, y3 - y1)

        if a * b * c <= 1e-9:
            return 0.0

        area2 = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        if area2 <= 1e-9:
            return 0.0

        return (2.0 * area2) / (a * b * c)

    def _compute_cost(
        self,
        *,
        lane_target_l: float,
        v_target: float,
        valid: bool,
        vinfo: Dict[str, Any],
    ) -> float:
        clearance_cost = 0.0
        if vinfo["min_clearance_m"] is not None:
            clearance_cost = 1.0 / max(vinfo["min_clearance_m"] + 1e-3, 1e-3)

        cost = (
            self.cfg.w_speed * abs(float(v_target) - float(self.cfg.target_speed))
            + self.cfg.w_curvature * float(vinfo["max_curvature"])
            + self.cfg.w_lateral_accel * float(vinfo["max_lateral_accel"])
            + self.cfg.w_clearance * float(clearance_cost)
            + self.cfg.w_lane_change * abs(float(lane_target_l))
            + self.cfg.w_collision * (1.0 if bool(vinfo["collision"]) else 0.0)
        )

        if not valid and not bool(vinfo["collision"]):
            cost *= 10.0

        return float(cost)

    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        if self._ref is None:
            return PlanResult(
                status=PlanStatus.EMPTY,
                trajectory=None,
                debug={"reason": "no_route"},
            )

        base_ref = self._ref
        s0_base, l0_base, _ryaw, seg_idx = base_ref.project_xy(
            ego.pose.x,
            ego.pose.y,
            hint_i=self._last_seg_idx,
        )
        self._last_seg_idx = seg_idx

        best_valid: Optional[Dict[str, Any]] = None
        best_any: Optional[Dict[str, Any]] = None
        num_candidates = 0

        for lane_target_l in self.cfg.virtual_lane_targets:
            lane_target_l = float(lane_target_l)

            virtual_ref = self._get_virtual_reference(
                base_ref=base_ref,
                s0=s0_base,
                l0=l0_base,
                lane_target_l=lane_target_l,
            )

            s0_v, _l0_v, _ryaw_v, _seg_idx_v = virtual_ref.project_xy(
                ego.pose.x,
                ego.pose.y,
                hint_i=0,
            )

            for v_target in self.cfg.speed_samples:
                v_target = float(v_target)
                v_target = _clamp(v_target, 0.0, self.vehicle.limits.max_speed_mps)

                traj = self._sample_candidate(
                    ref=virtual_ref,
                    s0=s0_v,
                    v0=float(max(0.0, ego.speed)),
                    v_target=v_target,
                )
                if traj is None:
                    continue

                num_candidates += 1
                valid, vinfo = self._validate(traj, world)

                cost = self._compute_cost(
                    lane_target_l=lane_target_l,
                    v_target=v_target,
                    valid=valid,
                    vinfo=vinfo,
                )

                cur = {
                    "cost": float(cost),
                    "valid": bool(valid),
                    "traj": traj,
                    "lane_target_l": float(lane_target_l),
                    "v_target": float(v_target),
                    **vinfo,
                }

                if best_any is None or cur["cost"] < best_any["cost"]:
                    best_any = cur

                if cur["valid"]:
                    if best_valid is None or cur["cost"] < best_valid["cost"]:
                        best_valid = cur

        chosen = best_valid if best_valid is not None else best_any

        if chosen is None:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={
                    "reason": "no_candidate",
                    "num_candidates": int(num_candidates),
                    "ego_s0": float(s0_base),
                    "ego_l0": float(l0_base),
                },
            )

        status = PlanStatus.OK if bool(chosen["valid"]) else PlanStatus.FAIL

        return PlanResult(
            status=status,
            trajectory=chosen["traj"],
            debug={
                "num_candidates": int(num_candidates),
                "ego_s0": float(s0_base),
                "ego_l0": float(l0_base),
                "vehicle": {
                    "wheelbase_m": float(self.vehicle.wheelbase),
                    "max_steer_deg": float(self.vehicle.limits.max_steer_deg),
                    "max_curvature": float(self.vehicle.max_curvature),
                },
                "selection": {
                    "used_best_valid": bool(best_valid is not None),
                    "status": status.name if hasattr(status, "name") else str(status),
                },
                "best": {
                    "cost_total": float(chosen["cost"]),
                    "lane_target_l": float(chosen["lane_target_l"]),
                    "v_target": float(chosen["v_target"]),
                    "valid": bool(chosen["valid"]),
                    "min_clearance_m": chosen["min_clearance_m"],
                    "collision": bool(chosen["collision"]),
                    "max_curvature": float(chosen["max_curvature"]),
                    "max_lateral_accel": float(chosen["max_lateral_accel"]),
                    "max_steer_rad": float(chosen["max_steer_rad"]),
                    "max_steer_rate_rad_s": float(chosen["max_steer_rate_rad_s"]),
                    "max_yaw_rate": float(chosen["max_yaw_rate"]),
                    "fail_reason": chosen["fail_reason"],
                },
                "virtual_ref": {
                    "cached_lane_target_l": self._cached_lane_target_l,
                    "cached_start_s": self._cached_virtual_ref_start_s,
                },
            },
        )