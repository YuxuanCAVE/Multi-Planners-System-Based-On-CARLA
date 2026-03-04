# framework/control/stanley.py
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import carla
from framework.core.types import EgoState, Trajectory


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class StanleyController:
    """
    Lateral: Stanley
    Longitudinal: same simple P speed control as PurePursuitController
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}

        # Vehicle / steering params
        self.wheel_base = float(cfg.get("wheel_base", 2.7))
        self.max_steer_deg = float(cfg.get("max_steer_deg", 35.0))

        # Stanley params
        self.k_stanley = float(cfg.get("k_stanley", 1.2))      # lateral gain
        self.v_soft = float(cfg.get("v_soft", 1.0))            # low-speed softening term (m/s)

        # Optional lookahead for picking a target point ahead of nearest projection
        self.use_lookahead = bool(cfg.get("use_lookahead", True))
        self.lookahead_base = float(cfg.get("lookahead_base", 2.0))
        self.lookahead_gain = float(cfg.get("lookahead_gain", 0.2))

        # Steering rate limit (helps jitter)
        self.enable_steer_rate_limit = bool(cfg.get("enable_steer_rate_limit", True))
        self.max_steer_rate_deg_s = float(cfg.get("max_steer_rate_deg_s", 120.0))  # deg/s
        self._prev_steer = 0.0
        self._prev_time_s: Optional[float] = None

        # Longitudinal control (copy from PP)
        self.target_speed = float(cfg.get("target_speed", 10.0))
        self.enable_longitudinal = bool(cfg.get("enable_longitudinal", True))
        self.kp_speed = float(cfg.get("kp_speed", 0.4))
        self.max_throttle = float(cfg.get("max_throttle", 0.6))
        self.max_brake = float(cfg.get("max_brake", 0.6))

    def compute_control(self, *, vehicle: carla.Vehicle, ego: EgoState, traj: Trajectory) -> carla.VehicleControl:
        control = carla.VehicleControl()
        if not traj.points:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        steer = self._stanley_steer(vehicle=vehicle, ego=ego, traj=traj)
        steer = float(_clamp(steer, -1.0, 1.0))

        # Optional steer rate limiting based on sim time
        if self.enable_steer_rate_limit:
            steer = self._apply_steer_rate_limit(steer, vehicle)

        control.steer = steer

        if self.enable_longitudinal:
            throttle, brake = self._simple_speed_control(speed=ego.speed, target=self.target_speed)
            control.throttle = float(throttle)
            control.brake = float(brake)
        else:
            control.throttle = 0.35
            control.brake = 0.0

        return control

    # -------------------------
    # Stanley lateral control
    # -------------------------
    def _stanley_steer(self, *, vehicle: carla.Vehicle, ego: EgoState, traj: Trajectory) -> float:
        tf = vehicle.get_transform()
        ex = tf.location.x
        ey = tf.location.y
        ego_yaw = math.radians(tf.rotation.yaw)  # world yaw

        # 1) Choose reference point on path:
        #    - find nearest point index (simple, cheap)
        nearest_i = 0
        best_d2 = float("inf")
        for i, p in enumerate(traj.points):
            dx = p.x - ex
            dy = p.y - ey
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                nearest_i = i

        ref_i = nearest_i
        if self.use_lookahead:
            # pick a point ahead to reduce noise
            Ld = self.lookahead_base + self.lookahead_gain * ego.speed
            Ld = max(0.5, Ld)
            for i in range(nearest_i, len(traj.points)):
                p = traj.points[i]
                dx, dy = p.x - ex, p.y - ey
                if math.hypot(dx, dy) >= Ld:
                    ref_i = i
                    break

        # 2) Path heading at reference point
        # Prefer using traj yaw if available; otherwise infer from neighbor points.
        ref_p = traj.points[ref_i]
        path_yaw = self._get_path_yaw(traj, ref_i)

        # 3) Heading error
        e_psi = _wrap_pi(path_yaw - ego_yaw)

        # 4) Signed cross-track error:
        #    sign = sign( n · (ego - path_point) ), where n is left normal of path tangent.
        #    tangent t = [cos(path_yaw), sin(path_yaw)]
        #    left normal n = [-sin, cos]
        dx = ex - ref_p.x
        dy = ey - ref_p.y
        n_x = -math.sin(path_yaw)
        n_y = math.cos(path_yaw)
        e_y = dx * n_x + dy * n_y  # positive means ego is left of path (under this convention)

        # 5) Stanley control law
        # delta = heading_error + atan2(k * e_y, v + v_soft)
        v = max(0.0, float(ego.speed))
        stanley_term = math.atan2(self.k_stanley * e_y, v + self.v_soft)
        delta = e_psi + stanley_term

        # Convert to normalized steer [-1, 1]
        max_steer = math.radians(self.max_steer_deg)
        steer_cmd = float(delta / max_steer)
        return _clamp(steer_cmd, -1.0, 1.0)

    def _get_path_yaw(self, traj: Trajectory, i: int) -> float:
        """
        Get path yaw at index i.
        If TrajectoryPoint has yaw attribute use it (assumed in radians).
        Otherwise infer from neighbors.
        """
        p = traj.points[i]
        if hasattr(p, "yaw") and p.yaw is not None:
            # assume radians; if your project stores degrees, convert here.
            return float(p.yaw)

        # Infer from forward/backward difference
        if i < len(traj.points) - 1:
            p2 = traj.points[i + 1]
            return math.atan2(p2.y - p.y, p2.x - p.x)
        elif i > 0:
            p0 = traj.points[i - 1]
            return math.atan2(p.y - p0.y, p.x - p0.x)
        else:
            return 0.0

    def _apply_steer_rate_limit(self, steer: float, vehicle: carla.Vehicle) -> float:
        now = vehicle.get_world().get_snapshot().timestamp.elapsed_seconds
        if self._prev_time_s is None:
            self._prev_time_s = now
            self._prev_steer = steer
            return steer

        dt = max(1e-3, now - self._prev_time_s)
        max_delta = math.radians(self.max_steer_rate_deg_s) * dt / math.radians(self.max_steer_deg)
        # max_delta is in normalized steer units
        steer_limited = _clamp(steer, self._prev_steer - max_delta, self._prev_steer + max_delta)

        self._prev_time_s = now
        self._prev_steer = steer_limited
        return steer_limited

    # -------------------------
    # Longitudinal (same as PP)
    # -------------------------
    def _simple_speed_control(self, *, speed: float, target: float) -> Tuple[float, float]:
        err = target - speed
        if err >= 0:
            throttle = _clamp(self.kp_speed * err, 0.0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(-self.kp_speed * err, 0.0, self.max_brake)
        return throttle, brake