from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class VehicleGeometry:
    wheelbase_m: float = 2.875
    length_m: float = 4.720
    width_m: float = 1.850
    front_overhang_m: float = 0.868
    rear_overhang_m: float = 0.977


@dataclass
class VehicleFootprint:
    collision_radius_m: float = 1.20
    safety_margin_m: float = 0.20


@dataclass
class VehicleLimits:
    max_speed_mps: float = 16.0
    max_accel_mps2: float = 3.0
    max_decel_mps2: float = 5.0
    max_steer_deg: float = 27.0
    max_steer_rate_deg_s: float = 120.0
    max_lateral_accel_mps2: float = 3.5

    @property
    def max_steer_rad(self) -> float:
        return math.radians(self.max_steer_deg)

    @property
    def max_steer_rate_rad_s(self) -> float:
        return math.radians(self.max_steer_rate_deg_s)


@dataclass
class VehicleModel:
    geometry: VehicleGeometry
    footprint: VehicleFootprint
    limits: VehicleLimits

    @property
    def wheelbase(self) -> float:
        return self.geometry.wheelbase_m

    @property
    def max_curvature(self) -> float:
        return math.tan(self.limits.max_steer_rad) / max(self.wheelbase, 1e-6)

    @property
    def ego_collision_radius(self) -> float:
        return self.footprint.collision_radius_m + self.footprint.safety_margin_m

    def steer_from_curvature(self, kappa: float) -> float:
        return math.atan(self.wheelbase * kappa)

    def curvature_from_steer(self, steer: float) -> float:
        return math.tan(steer) / max(self.wheelbase, 1e-6)

    def lateral_accel(self, v: float, kappa: float) -> float:
        return v * v * kappa


def build_vehicle_model(cfg: Dict[str, Any]) -> VehicleModel:
    """
    支持两种输入：
    1. root config: {"planner": {...}, "vehicle": {...}, ...}
    2. vehicle config: {"geometry": {...}, "limits": {...}, ...}
    """
    if not isinstance(cfg, dict):
        cfg = {}

    vcfg = cfg.get("vehicle", cfg)
    if not isinstance(vcfg, dict):
        vcfg = {}

    gcfg = vcfg.get("geometry", {})
    fcfg = vcfg.get("footprint", {})
    lcfg = vcfg.get("limits", {})

    geometry = VehicleGeometry(
        wheelbase_m=float(gcfg.get("wheelbase_m", 2.875)),
        length_m=float(gcfg.get("length_m", 4.720)),
        width_m=float(gcfg.get("width_m", 1.850)),
        front_overhang_m=float(gcfg.get("front_overhang_m", 0.868)),
        rear_overhang_m=float(gcfg.get("rear_overhang_m", 0.977)),
    )
    footprint = VehicleFootprint(
        collision_radius_m=float(fcfg.get("collision_radius_m", 1.20)),
        safety_margin_m=float(fcfg.get("safety_margin_m", 0.20)),
    )
    limits = VehicleLimits(
        max_speed_mps=float(lcfg.get("max_speed_mps", 16.0)),
        max_accel_mps2=float(lcfg.get("max_accel_mps2", 3.0)),
        max_decel_mps2=float(lcfg.get("max_decel_mps2", 5.0)),
        max_steer_deg=float(lcfg.get("max_steer_deg", 27.0)),
        max_steer_rate_deg_s=float(lcfg.get("max_steer_rate_deg_s", 120.0)),
        max_lateral_accel_mps2=float(lcfg.get("max_lateral_accel_mps2", 3.5)),
    )

    return VehicleModel(
        geometry=geometry,
        footprint=footprint,
        limits=limits,
    )