# framework/control/pure_pursuit.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import carla

from framework.core.types import EgoState, Trajectory, TrajectoryPoint


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _hypot(x: float, y: float) -> float:
    return math.sqrt(x * x + y * y)


def _isfinite(x: float) -> bool:
    return not (math.isinf(x) or math.isnan(x))


@dataclass
class PurePursuitConfig:
    # speed
    target_speed: float = 10.0
    enable_longitudinal: bool = True
    kp_speed: float = 0.4
    ki_speed: float = 0.0
    kd_speed: float = 0.0
    max_throttle: float = 0.6
    max_brake: float = 0.6

    # lateral
    wheel_base: float = 2.7
    lookahead_base: float = 4.0
    lookahead_gain: float = 0.35
    max_steer_deg: float = 35.0

    # safety / behavior
    stop_if_no_traj: bool = True
    min_lookahead: float = 2.0
    max_lookahead: float = 20.0


class PurePursuitController:
    """
    Controller that matches the planning interface:
      input: EgoState + Trajectory (list of TrajectoryPoint) + carla.Vehicle
      output: carla.VehicleControl

    Works with any planner as long as it outputs Trajectory(points=[...]) in world frame.

    - Lateral: pure pursuit
    - Longitudinal (optional): PID on speed tracking.
      Uses trajectory point v if available; otherwise uses config.target_speed.
    """

    name = "pure_pursuit"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        c = config or {}
        self.cfg = PurePursuitConfig(
            target_speed=float(c.get("target_speed", 10.0)),
            enable_longitudinal=bool(c.get("enable_longitudinal", True)),
            kp_speed=float(c.get("kp_speed", 0.4)),
            ki_speed=float(c.get("ki_speed", 0.0)),
            kd_speed=float(c.get("kd_speed", 0.0)),
            max_throttle=float(c.get("max_throttle", 0.6)),
            max_brake=float(c.get("max_brake", 0.6)),
            wheel_base=float(c.get("wheel_base", 2.7)),
            lookahead_base=float(c.get("lookahead_base", 4.0)),
            lookahead_gain=float(c.get("lookahead_gain", 0.35)),
            max_steer_deg=float(c.get("max_steer_deg", 35.0)),
            stop_if_no_traj=bool(c.get("stop_if_no_traj", True)),
            min_lookahead=float(c.get("min_lookahead", 2.0)),
            max_lookahead=float(c.get("max_lookahead", 20.0)),
        )

        # longitudinal PID state
        self._int_err: float = 0.0
        self._prev_err: Optional[float] = None

    def reset(self) -> None:
        self._int_err = 0.0
        self._prev_err = None

    def compute_control(self, *, vehicle: carla.Vehicle, ego: EgoState, traj: Trajectory) -> carla.VehicleControl:
        # Basic validation
        if traj is None or not getattr(traj, "points", None):
            return self._stop_control()

        pts: List[TrajectoryPoint] = traj.points
        if len(pts) < 2:
            return self._stop_control()

        # Determine dt (if available); used for longitudinal PID only
        dt = float(getattr(traj, "dt", 0.05) or 0.05)
        if dt <= 1e-6:
            dt = 0.05

        # 1) Lateral steer
        steer = self._compute_steer_pure_pursuit(ego=ego, pts=pts)

        # 2) Longitudinal throttle/brake
        throttle, brake = 0.0, 0.0
        if self.cfg.enable_longitudinal:
            v_ref = self._pick_reference_speed(ego=ego, pts=pts)
            throttle, brake = self._compute_throttle_brake_pid(v_ref=v_ref, v=ego.speed, dt=dt)

        return carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False,
        )

    # -----------------------------
    # Lateral: Pure Pursuit
    # -----------------------------
    def _compute_steer_pure_pursuit(self, *, ego: EgoState, pts: List[TrajectoryPoint]) -> float:
        # Dynamic lookahead based on speed
        Ld = self.cfg.lookahead_base + self.cfg.lookahead_gain * max(0.0, float(ego.speed))
        Ld = _clamp(Ld, self.cfg.min_lookahead, self.cfg.max_lookahead)

        # Find target point at distance >= Ld along the path from ego position
        tx, ty = self._find_lookahead_point(ego_x=ego.pose.x, ego_y=ego.pose.y, pts=pts, lookahead=Ld)

        # transform target point into ego frame
        dx = tx - ego.pose.x
        dy = ty - ego.pose.y
        yaw = float(ego.pose.yaw)

        # rotate into vehicle frame: x' forward, y' left
        x_r = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        y_r = math.sin(-yaw) * dx + math.cos(-yaw) * dy

        # If lookahead point is behind (x_r <= 0), reduce aggressiveness
        if x_r <= 1e-3:
            # steer towards the point by sign of y_r
            raw = _clamp(y_r, -1.0, 1.0)
            return float(raw)

        # curvature kappa = 2*y / Ld^2 in vehicle frame
        kappa = 2.0 * y_r / (Ld * Ld)

        # steering angle delta = atan(L * kappa)
        delta = math.atan(self.cfg.wheel_base * kappa)

        # map steering angle to [-1,1] via max steer
        max_steer_rad = math.radians(self.cfg.max_steer_deg)
        steer = _clamp(delta / max_steer_rad, -1.0, 1.0)
        return float(steer)

    @staticmethod
    def _find_lookahead_point(*, ego_x: float, ego_y: float, pts: List[TrajectoryPoint], lookahead: float) -> Tuple[float, float]:
        """
        Pick the first point whose distance from ego >= lookahead.
        If none, use the last point.
        """
        for p in pts:
            d = _hypot(p.x - ego_x, p.y - ego_y)
            if d >= lookahead:
                return float(p.x), float(p.y)
        last = pts[-1]
        return float(last.x), float(last.y)

    # -----------------------------
    # Longitudinal: PID speed tracking
    # -----------------------------
    def _pick_reference_speed(self, *, ego: EgoState, pts: List[TrajectoryPoint]) -> float:
        """
        Use v from trajectory point if available and valid; otherwise cfg.target_speed.
        We pick a "near-future" point (e.g. 3rd or 5th) to reduce jitter.
        """
        idx = min(5, len(pts) - 1)
        p = pts[idx]

        v = getattr(p, "v", None)
        if isinstance(v, (int, float)) and _isfinite(float(v)):
            return max(0.0, float(v))
        return max(0.0, float(self.cfg.target_speed))

    def _compute_throttle_brake_pid(self, *, v_ref: float, v: float, dt: float) -> Tuple[float, float]:
        err = float(v_ref - v)

        # integrate
        self._int_err += err * dt

        # derivative
        derr = 0.0
        if self._prev_err is not None:
            derr = (err - self._prev_err) / dt
        self._prev_err = err

        u = self.cfg.kp_speed * err + self.cfg.ki_speed * self._int_err + self.cfg.kd_speed * derr

        # Convert to throttle/brake
        if u >= 0.0:
            throttle = _clamp(u, 0.0, self.cfg.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(-u, 0.0, self.cfg.max_brake)

        return float(throttle), float(brake)

    # -----------------------------
    # Safety
    # -----------------------------
    def _stop_control(self) -> carla.VehicleControl:
        if not self.cfg.stop_if_no_traj:
            return carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0)
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
