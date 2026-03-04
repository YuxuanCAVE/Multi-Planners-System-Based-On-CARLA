# framework/control/pure_pursuit.py

"""
逻辑：
1.Planner 输出 Trajectory
2.Controller 把轨迹变成 Carla的 VehicleControl
(Controller不负责规划，只负责追踪)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import carla
from framework.core.types import EgoState, Trajectory

#把数值限制在区间内，防止输出超范围
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

#把角度规范到[-pi,pi]
def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


class PurePursuitController:
    """
    先跑通为主：
    - 横向：Pure Pursuit
    - 纵向：简单 P 控速（可关掉用恒定 throttle）
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        #初始化参数
        self.wheel_base = float(cfg.get("wheel_base", 2.7))  # 车型不同可调
        self.lookahead_base = float(cfg.get("lookahead_base", 4.0))
        self.lookahead_gain = float(cfg.get("lookahead_gain", 0.3))  # 随速度增长
        self.max_steer_deg = float(cfg.get("max_steer_deg", 35.0))   # 近似
        self.target_speed = float(cfg.get("target_speed", 10.0))

        # 纵向控制相关参数（极简）
        self.enable_longitudinal = bool(cfg.get("enable_longitudinal", True))
        self.kp_speed = float(cfg.get("kp_speed", 0.4))
        self.max_throttle = float(cfg.get("max_throttle", 0.6))
        self.max_brake = float(cfg.get("max_brake", 0.6))

        # 如果你只想先做横向，让车动起来：设 enable_longitudinal=False，然后 Runner 里给恒定 throttle

    def compute_control(self, *, vehicle: carla.Vehicle, ego: EgoState, traj: Trajectory) -> carla.VehicleControl:
        #vehicle, ego , traj
        control = carla.VehicleControl()
        if not traj.points:
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            return control

        steer = self._pure_pursuit_steer(vehicle=vehicle, ego=ego, traj=traj)
        control.steer = float(_clamp(steer, -1.0, 1.0))

        #如果没有轨迹点：直接刹停
        if self.enable_longitudinal:
            throttle, brake = self._simple_speed_control(speed=ego.speed, target=self.target_speed)
            control.throttle = float(throttle)
            control.brake = float(brake)
        else:
            control.throttle = 0.35
            control.brake = 0.0

        return control
    #首先是横向控制，
    #核心思想是 在轨迹上找一个前视目标点，让车去追那个点，算出需要的曲率，再转换成转角
    def _pure_pursuit_steer(self, *, vehicle: carla.Vehicle, ego: EgoState, traj: Trajectory) -> float:
        # lookahead 随速度变化
        Ld = self.lookahead_base + self.lookahead_gain * ego.speed
        Ld = max(2.0, Ld)

        # 在轨迹上找距离 ego 最近点之后的 lookahead 目标点
        ex, ey = ego.pose.x, ego.pose.y
        best_i = 0
        best_d2 = float("inf")
        #寻找最近的best_i point
        for i, p in enumerate(traj.points):
            dx = p.x - ex
            dy = p.y - ey
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        target = traj.points[-1]
        # 从 best_i 开始往前找距离 >= Ld 的点 作为target point
        for i in range(best_i, len(traj.points)):
            p = traj.points[i]
            dx, dy = p.x - ex, p.y - ey
            if math.sqrt(dx * dx + dy * dy) >= Ld:
                target = p
                break

        # 把 target 转到车体坐标系（vehicle transform 更可靠）
        tf = vehicle.get_transform()
  

        # ego 航向
        ego_yaw = math.radians(tf.rotation.yaw)

        # 目标点方向角（世界系）
        target_yaw = math.atan2(
            target.y - tf.location.y,
            target.x - tf.location.x
        )

        # 航向误差（wrap 到 [-pi, pi]）
        alpha = _wrap_pi(target_yaw - ego_yaw)

        # Pure Pursuit 曲率 -> 转角
        # kappa = 2*y / Ld^2 (近似)，delta = atan(L * kappa)
        kappa = 2.0 * math.sin(alpha) / Ld
        delta = math.atan(self.wheel_base * kappa)

        max_steer = math.radians(self.max_steer_deg)
        steer_cmd = float(delta / max_steer)
        return _clamp(steer_cmd, -1.0, 1.0)

    #简单的 速度误差P控制 如果err >= 0 , 速度不够则给油门，反之则刹车
    def _simple_speed_control(self, *, speed: float, target: float) -> Tuple[float, float]:
        err = target - speed
        if err >= 0:
            throttle = _clamp(self.kp_speed * err, 0.0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = _clamp(-self.kp_speed * err, 0.0, self.max_brake)
        return throttle, brake
