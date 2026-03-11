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
from framework.planning.mode_decider import ModeDecision

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
    horizon_t: float = 4.0
    target_speed: float = 8.0

    lateral_offsets: Tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
    speed_samples: Tuple[float, ...] = (4.0, 6.0, 8.0, 10.0)

    lateral_ramp_time: float = 1.0

    w_offset: float = 1.0
    w_speed: float = 0.3
    w_curvature: float = 0.5
    w_lateral_accel: float = 0.2
    w_collision: float = 1000.0
    w_clearance: float = 0.2


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
            horizon_t=float(pcfg.get("horizon_t", pcfg.get("horizon_s", 4.0))),
            target_speed=float(pcfg.get("target_speed", 8.0)),
            lateral_offsets=tuple(pcfg.get("lateral_offsets", (-1.0, -0.5, 0.0, 0.5, 1.0))),
            speed_samples=tuple(pcfg.get("speed_samples", (4.0, 6.0, 8.0, 10.0))),
            lateral_ramp_time=float(pcfg.get("lateral_ramp_time", 1.0)),
            w_offset=float(pcfg.get("w_offset", 1.0)),
            w_speed=float(pcfg.get("w_speed", 0.3)),
            w_curvature=float(pcfg.get("w_curvature", 0.5)),
            w_lateral_accel=float(pcfg.get("w_lateral_accel", 0.2)),
            w_collision=float(pcfg.get("w_collision", 1000.0)),
            w_clearance=float(pcfg.get("w_clearance", 0.2)),
        )

        self.vehicle: VehicleModel = build_vehicle_model(root_cfg)
        self._ref: Optional[_ReferenceLine] = None
        self._last_seg_idx: int = 0

        self._mode_decision: Optional[ModeDecision] = None

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._ref = _ReferenceLine(route)
        self._last_seg_idx = 0

    def set_mode_decision(self, decision: ModeDecision) -> None:
        self._mode_decision = decision

    def _get_mode_decision_or_default(self) -> ModeDecision:
        if self._mode_decision is not None:
            return self._mode_decision
        return ModeDecision(
            mode="KEEP_LANE",
            target_lane_l=0.0,
            blocking_obstacle_index=None,
            reason="default_keep_lane",
        )

    def _get_sampling_policy(self, decision: ModeDecision) -> Tuple[float, Tuple[float, ...], Tuple[float, ...]]:
        """
        Returns:
            l_center, lateral_samples, speed_samples
        """
        mode = decision.mode

        if mode == "KEEP_LANE":
            # 当前 GRP 就是当前车道中心线
            l_center = 0.0
            lateral_samples = self.cfg.lateral_offsets
            speed_samples = self.cfg.speed_samples

        elif mode in ("CHANGE_LEFT", "CHANGE_RIGHT"):
            # 关键：围绕目标车道中心采样，而不是围绕当前 l0
            l_center = float(decision.target_lane_l)
            lateral_samples = (-0.3, 0.0, 0.3)
            speed_samples = self.cfg.speed_samples

        elif mode == "FOLLOW_OR_STOP":
       
            l_center = 0.0
            lateral_samples = self.cfg.lateral_offsets
            speed_samples = self.cfg.speed_samples

        else:
            l_center = 0.0
            lateral_samples = self.cfg.lateral_offsets
            speed_samples = self.cfg.speed_samples

        return l_center, lateral_samples, speed_samples
    
    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        if self._ref is None:
            return PlanResult(
                status=PlanStatus.EMPTY,
                trajectory=None,
                debug={"reason": "no_route"},
            )

        ref = self._ref
        s0, l0, _ryaw, seg_idx = ref.project_xy(
            ego.pose.x,
            ego.pose.y,
            hint_i=self._last_seg_idx,
        )
        self._last_seg_idx = seg_idx

        decision = self._get_mode_decision_or_default()
        l_center, lateral_samples, speed_samples = self._get_sampling_policy(decision)

        best_valid: Optional[Dict[str, Any]] = None
        best_any: Optional[Dict[str, Any]] = None
        num_candidates = 0

        for l_off in lateral_samples:
            # 关键改动：不再用 l0 + l_off
            l_target = float(l_center) + float(l_off)

            for v_target in speed_samples:
                v_target = float(v_target)
                v_target = _clamp(v_target, 0.0, self.vehicle.limits.max_speed_mps)

                traj = self._sample_candidate(
                    ref=ref,
                    s0=s0,
                    l0=l0,
                    l_target=l_target,
                    v0=float(max(0.0, ego.speed)),
                    v_target=v_target,
                )
                if traj is None:
                    continue

                num_candidates += 1
                valid, vinfo = self._validate(traj, world)

                cost = self._compute_cost(
                    decision=decision,
                    l_center=l_center,
                    l_target=l_target,
                    v_target=v_target,
                    valid=valid,
                    vinfo=vinfo,
                )

                cur = {
                    "cost": float(cost),
                    "valid": bool(valid),
                    "traj": traj,
                    "l_target": float(l_target),
                    "l_center": float(l_center),
                    "v_target": float(v_target),
                    "mode": decision.mode,
                    "target_lane_l": float(decision.target_lane_l),
                    "blocking_obstacle_index": decision.blocking_obstacle_index,
                    "reason": decision.reason,
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
                    "ego_s0": float(s0),
                    "ego_l0": float(l0),
                    "mode_decision": {
                        "mode": decision.mode,
                        "target_lane_l": float(decision.target_lane_l),
                        "blocking_obstacle_index": decision.blocking_obstacle_index,
                        "reason": decision.reason,
                    },
                },
            )

        status = PlanStatus.OK if bool(chosen["valid"]) else PlanStatus.FAIL

        return PlanResult(
            status=status,
            trajectory=chosen["traj"],
            debug={
                "num_candidates": int(num_candidates),
                "ego_s0": float(s0),
                "ego_l0": float(l0),
                "vehicle": {
                    "wheelbase_m": float(self.vehicle.wheelbase),
                    "max_steer_deg": float(self.vehicle.limits.max_steer_deg),
                    "max_curvature": float(self.vehicle.max_curvature),
                },
                "mode_decision": {
                    "mode": decision.mode,
                    "target_lane_l": float(decision.target_lane_l),
                    "blocking_obstacle_index": decision.blocking_obstacle_index,
                    "reason": decision.reason,
                },
                "selection": {
                    "used_best_valid": bool(best_valid is not None),
                    "status": status.name if hasattr(status, "name") else str(status),
                },
                "best": {
                    "cost_total": float(chosen["cost"]),
                    "l_center": float(chosen["l_center"]),
                    "l_target": float(chosen["l_target"]),
                    "v_target": float(chosen["v_target"]),
                    "valid": bool(chosen["valid"]),
                    "mode": chosen["mode"],
                    "target_lane_l": float(chosen["target_lane_l"]),
                    "min_clearance_m": chosen["min_clearance_m"],
                    "collision": bool(chosen["collision"]),
                    "max_curvature": float(chosen["max_curvature"]),
                    "max_lateral_accel": float(chosen["max_lateral_accel"]),
                    "max_steer_rad": float(chosen["max_steer_rad"]),
                    "max_steer_rate_rad_s": float(chosen["max_steer_rate_rad_s"]),
                    "max_yaw_rate": float(chosen["max_yaw_rate"]),
                },
            },
        )
    @staticmethod
    def _solve_quintic_lateral(
        l0: float,
        lT: float,
        T: float,
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Quintic polynomial:
            l(t) = a0 + a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5

        Boundary conditions:
            l(0)   = l0
            l'(0)  = 0
            l''(0) = 0
            l(T)   = lT
            l'(T)  = 0
            l''(T) = 0
        """
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
    
    def _sample_candidate(
        self,
        *,
        ref: _ReferenceLine,
        s0: float,
        l0: float,
        l_target: float,
        v0: float,
        v_target: float,
    ) -> Optional[Trajectory]:
        dt = self.cfg.dt
        steps = max(2, int(self.cfg.horizon_t / max(dt, 1e-6)))

        pts: List[TrajectoryPoint] = []
        s = s0

        # longitudinal: constant-acceleration rollout toward target speed
        T = max(self.cfg.horizon_t, dt)
        a_cmd = (v_target - v0) / T
        a_cmd = _clamp(
            a_cmd,
            -self.vehicle.limits.max_decel_mps2,
            self.vehicle.limits.max_accel_mps2,
        )
        v = max(0.0, min(self.vehicle.limits.max_speed_mps, v0))

        # lateral: quintic polynomial instead of linear ramp
        T_lat = max(self.cfg.lateral_ramp_time, dt)
        lat_coeffs = self._solve_quintic_lateral(l0, l_target, T_lat)

        for k in range(steps):
            tk = k * dt

            if k > 0:
                v = max(
                    0.0,
                    min(self.vehicle.limits.max_speed_mps, v + a_cmd * dt),
                )
                s = min(ref.s_max, s + v * dt)

            # after T_lat, hold the target lateral offset
            t_lat = min(tk, T_lat)
            l = self._eval_quintic(lat_coeffs, t_lat)

            x, y, _yaw = ref.frenet_to_xy(s, l)
            pts.append(
                TrajectoryPoint(
                    x=float(x),
                    y=float(y),
                    yaw=0.0,
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
            and (max_ay <= self.vehicle.limits.max_lateral_accel_mps2 + 1e-6)
            and (max_steer <= self.vehicle.limits.max_steer_rad + 1e-6)
            and (max_steer_rate <= self.vehicle.limits.max_steer_rate_rad_s + 1e-6)
        )

        return valid, {
            "min_clearance_m": min_dist if min_dist != float("inf") else None,
            "collision": bool(collision),
            "max_curvature": float(max_kappa),
            "max_lateral_accel": float(max_ay),
            "max_steer_rad": float(max_steer),
            "max_steer_rate_rad_s": float(max_steer_rate),
            "max_yaw_rate": float(max_yaw_rate),
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
        decision: ModeDecision,
        l_center: float,
        l_target: float,
        v_target: float,
        valid: bool,
        vinfo: Dict[str, Any],
    ) -> float:
        clearance_cost = 0.0
        if vinfo["min_clearance_m"] is not None:
            clearance_cost = 1.0 / max(vinfo["min_clearance_m"] + 1e-3, 1e-3)

        # 这里的 offset 不再是相对 ego 当前 l0，而是相对 mode 目标中心
        l_off_center = abs(float(l_target) - float(l_center))

        # 变道模式下降低 offset 惩罚，否则它会被自己压回去
        if decision.mode in ("CHANGE_LEFT", "CHANGE_RIGHT"):
            w_offset = 0.2 * self.cfg.w_offset
        else:
            w_offset = self.cfg.w_offset

        # FOLLOW/STOP 模式更强地惩罚高速度
        if decision.mode == "FOLLOW_OR_STOP":
            speed_ref = 0.0
        else:
            speed_ref = self.cfg.target_speed

        cost = (
            w_offset * l_off_center
            + self.cfg.w_speed * abs(float(v_target) - float(speed_ref))
            + self.cfg.w_curvature * float(vinfo["max_curvature"])
            + self.cfg.w_lateral_accel * float(vinfo["max_lateral_accel"])
            + self.cfg.w_clearance * float(clearance_cost)
            + self.cfg.w_collision * (1.0 if bool(vinfo["collision"]) else 0.0)
        )

        if not valid and not bool(vinfo["collision"]):
            cost *= 10.0

        return float(cost)