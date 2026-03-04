# framework/vehicle/kinematics.py
from __future__ import annotations

import math
from typing import List, Tuple


def wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def bicycle_rollout(
    *,
    x: float,
    y: float,
    yaw: float,
    v: float,
    steer: float,
    wheelbase: float,
    dt: float,
    steps: int,
) -> List[Tuple[float, float, float]]:
    """
    Kinematic bicycle rollout (forward only if v>0; caller can set v negative for reverse).
    Returns a list of (x,y,yaw) points of length == steps.
    """
    seg: List[Tuple[float, float, float]] = []
    cx, cy, cyaw = float(x), float(y), float(yaw)
    L = float(wheelbase)

    for _ in range(int(steps)):
        cx += v * math.cos(cyaw) * dt
        cy += v * math.sin(cyaw) * dt
        cyaw = wrap_pi(cyaw + (v / L) * math.tan(steer) * dt)
        seg.append((cx, cy, cyaw))

    return seg