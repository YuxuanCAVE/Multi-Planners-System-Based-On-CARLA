from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BoxGeom:
    actor_id: Any
    kind: str
    x: float
    y: float
    yaw: float
    length_m: float
    width_m: float
    speed: float


@dataclass
class EllipticSafetyField:
    actor_id: Any
    x: float
    y: float
    yaw: float
    speed: float

    # semi-axes in actor local frame
    front_a_m: float
    rear_a_m: float
    lateral_b_m: float

    # cost shaping
    weight: float
    power: float

    # optional center shift along local x
    center_shift_x_m: float = 0.0

    @property
    def max_longitudinal_extent_m(self) -> float:
        return max(self.front_a_m, self.rear_a_m)

    @property
    def max_lateral_extent_m(self) -> float:
        return self.lateral_b_m


@dataclass
class ActorModel:
    actor_id: Any
    x: float
    y: float
    yaw: float
    speed: float

    raw_length_m: float
    raw_width_m: float

    front_safe_m: float
    rear_safe_m: float
    lateral_safe_m: float

    body: BoxGeom
    safety_field: EllipticSafetyField


class ActorModelAdapter:
    """
    Actor geometry adapter:
    - body: hard collision box
    - safety_field: asymmetric elliptic soft-cost field
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.config = cfg

        # default size fallback
        self.default_actor_length_m = float(cfg.get("default_actor_length_m", 4.5))
        self.default_actor_width_m = float(cfg.get("default_actor_width_m", 1.8))

        # body margins
        self.body_front_margin_m = float(cfg.get("body_front_margin_m", 0.0))
        self.body_rear_margin_m = float(cfg.get("body_rear_margin_m", 0.0))
        self.body_lateral_margin_m = float(cfg.get("body_lateral_margin_m", 0.0))

        # safety extents
        self.safety_front_m = float(cfg.get("safety_front_m", 6.0))
        self.safety_rear_m = float(cfg.get("safety_rear_m", 2.0))
        self.safety_lateral_m = float(cfg.get("safety_lateral_m", 0.6))

        # dynamic safety
        self.enable_dynamic_safety = bool(cfg.get("enable_dynamic_safety", False))
        self.safety_front_speed_coeff = float(cfg.get("safety_front_speed_coeff", 0.0))
        self.safety_rear_speed_coeff = float(cfg.get("safety_rear_speed_coeff", 0.0))
        self.safety_lateral_speed_coeff = float(cfg.get("safety_lateral_speed_coeff", 0.0))

        self.max_safety_front_m = float(cfg.get("max_safety_front_m", 12.0))
        self.max_safety_rear_m = float(cfg.get("max_safety_rear_m", 5.0))
        self.max_safety_lateral_m = float(cfg.get("max_safety_lateral_m", 1.5))

        # elliptic field params
        self.safety_field_weight = float(cfg.get("safety_field_weight", 1.0))
        self.safety_field_power = float(cfg.get("safety_field_power", 2.0))

        # optional shift to emphasize front region
        self.safety_field_center_shift_x_m = float(
            cfg.get("safety_field_center_shift_x_m", 0.0)
        )

        self.min_speed_for_velocity_yaw_mps = float(
            cfg.get("min_speed_for_velocity_yaw_mps", 0.2)
        )

        self.ignore_actor_ids = set(cfg.get("ignore_actor_ids", []))

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def build_actor_model(self, obs: Any) -> Optional[ActorModel]:
        actor_id = getattr(obs, "id", None)
        if actor_id in self.ignore_actor_ids:
            return None

        pos = getattr(obs, "position", None)
        if pos is None:
            return None

        x = getattr(pos, "x", None)
        y = getattr(pos, "y", None)
        if x is None or y is None:
            return None

        x = float(x)
        y = float(y)

        speed = self._obs_speed(obs)
        yaw = self._obs_yaw(obs, speed=speed)

        raw_length_m = self._obs_length(obs)
        raw_width_m = self._obs_width(obs)

        body = self._build_body_box(
            actor_id=actor_id,
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            raw_length_m=raw_length_m,
            raw_width_m=raw_width_m,
        )

        front_safe_m, rear_safe_m, lateral_safe_m = self._compute_safety_extensions(speed)

        # semi-axes of safety ellipse are based on body half-size + safety extension
        body_half_l = 0.5 * body.length_m
        body_half_w = 0.5 * body.width_m

        safety_field = EllipticSafetyField(
            actor_id=actor_id,
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            front_a_m=body_half_l + front_safe_m,
            rear_a_m=body_half_l + rear_safe_m,
            lateral_b_m=body_half_w + lateral_safe_m,
            weight=self.safety_field_weight,
            power=self.safety_field_power,
            center_shift_x_m=self.safety_field_center_shift_x_m,
        )

        return ActorModel(
            actor_id=actor_id,
            x=x,
            y=y,
            yaw=yaw,
            speed=speed,
            raw_length_m=raw_length_m,
            raw_width_m=raw_width_m,
            front_safe_m=front_safe_m,
            rear_safe_m=rear_safe_m,
            lateral_safe_m=lateral_safe_m,
            body=body,
            safety_field=safety_field,
        )

    def build_all(self, world: Any) -> List[ActorModel]:
        out: List[ActorModel] = []
        for obs in getattr(world, "obstacles", []) or []:
            model = self.build_actor_model(obs)
            if model is not None:
                out.append(model)
        return out

    def get_box_corners(self, box: BoxGeom) -> List[Tuple[float, float]]:
        c = math.cos(box.yaw)
        s = math.sin(box.yaw)

        half_l = 0.5 * box.length_m
        half_w = 0.5 * box.width_m

        local_pts = [
            (+half_l, +half_w),
            (+half_l, -half_w),
            (-half_l, -half_w),
            (-half_l, +half_w),
        ]

        world_pts: List[Tuple[float, float]] = []
        for lx, ly in local_pts:
            wx = box.x + lx * c - ly * s
            wy = box.y + lx * s + ly * c
            world_pts.append((wx, wy))
        return world_pts

    # ------------------------------------------------------------------
    # low-cost safety field evaluation
    # ------------------------------------------------------------------
    def coarse_filter_circle_vs_field(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        actor: ActorModel,
        coarse_margin_m: float = 2.0,
    ) -> bool:
        """
        Cheap coarse filter before precise safety evaluation.
        Returns True if detailed eval is needed.
        Uses actor-local AABB around the elliptic field.
        """
        field = actor.safety_field

        dx = cx - field.x
        dy = cy - field.y

        c = math.cos(field.yaw)
        s = math.sin(field.yaw)

        lx = dx * c + dy * s - field.center_shift_x_m
        ly = -dx * s + dy * c

        max_half_x = max(field.front_a_m, field.rear_a_m) + r + coarse_margin_m
        max_half_y = field.lateral_b_m + r + coarse_margin_m

        return (abs(lx) <= max_half_x) and (abs(ly) <= max_half_y)

    def safety_cost_circle(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        actor: ActorModel,
    ) -> float:
        """
        Elliptic safety field cost for one ego circle center against one actor.
        Complexity: O(1)
        """
        field = actor.safety_field

        dx = cx - field.x
        dy = cy - field.y

        c = math.cos(field.yaw)
        s = math.sin(field.yaw)

        # transform to actor local frame
        lx = dx * c + dy * s - field.center_shift_x_m
        ly = -dx * s + dy * c

        # choose asymmetric longitudinal semi-axis
        a = field.front_a_m if lx >= 0.0 else field.rear_a_m
        b = field.lateral_b_m

        # ego circle inflation: subtract r from axes to approximate disc penetration cheaply
        a_eff = max(1e-3, a + r)
        b_eff = max(1e-3, b + r)

        rho2 = (lx / a_eff) * (lx / a_eff) + (ly / b_eff) * (ly / b_eff)
        if rho2 >= 1.0:
            return 0.0

        # no sqrt: use rho2 directly for lower constant cost
        # inside ellipse => smaller rho2 => higher cost
        ratio = 1.0 - rho2
        return field.weight * (ratio ** field.power)

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _compute_safety_extensions(self, speed: float) -> Tuple[float, float, float]:
        front_m = self.safety_front_m
        rear_m = self.safety_rear_m
        lateral_m = self.safety_lateral_m

        if self.enable_dynamic_safety:
            front_m = min(
                self.max_safety_front_m,
                self.safety_front_m + self.safety_front_speed_coeff * speed,
            )
            rear_m = min(
                self.max_safety_rear_m,
                self.safety_rear_m + self.safety_rear_speed_coeff * speed,
            )
            lateral_m = min(
                self.max_safety_lateral_m,
                self.safety_lateral_m + self.safety_lateral_speed_coeff * speed,
            )

        return front_m, rear_m, lateral_m

    def _build_body_box(
        self,
        *,
        actor_id: Any,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        raw_length_m: float,
        raw_width_m: float,
    ) -> BoxGeom:
        total_length_m = (
            raw_length_m + self.body_front_margin_m + self.body_rear_margin_m
        )
        total_width_m = raw_width_m + 2.0 * self.body_lateral_margin_m

        shift_x_local = 0.5 * (self.body_front_margin_m - self.body_rear_margin_m)
        c = math.cos(yaw)
        s = math.sin(yaw)

        cx = x + shift_x_local * c
        cy = y + shift_x_local * s

        return BoxGeom(
            actor_id=actor_id,
            kind="body",
            x=cx,
            y=cy,
            yaw=yaw,
            length_m=total_length_m,
            width_m=total_width_m,
            speed=speed,
        )

    def _obs_speed(self, obs: Any) -> float:
        vel = getattr(obs, "velocity", None)
        if vel is None:
            return 0.0

        vx = float(getattr(vel, "x", 0.0))
        vy = float(getattr(vel, "y", 0.0))
        vz = float(getattr(vel, "z", 0.0))
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def _obs_yaw(self, obs: Any, speed: float) -> float:
        for name in ("yaw", "heading", "theta"):
            val = getattr(obs, name, None)
            if val is not None:
                return float(val)

        pose = getattr(obs, "pose", None)
        if pose is not None:
            for name in ("yaw", "heading", "theta"):
                val = getattr(pose, name, None)
                if val is not None:
                    return float(val)

        orientation = getattr(obs, "orientation", None)
        if orientation is not None:
            val = getattr(orientation, "yaw", None)
            if val is not None:
                return float(val)

        if speed > self.min_speed_for_velocity_yaw_mps:
            vel = getattr(obs, "velocity", None)
            if vel is not None:
                vx = float(getattr(vel, "x", 0.0))
                vy = float(getattr(vel, "y", 0.0))
                if math.hypot(vx, vy) > 1e-6:
                    return math.atan2(vy, vx)

        return 0.0

    def _obs_length(self, obs: Any) -> float:
        for name in ("length", "length_m", "size_x"):
            val = getattr(obs, name, None)
            if val is not None:
                return max(0.1, float(val))

        dims = getattr(obs, "dimensions", None)
        if dims is not None:
            for name in ("length", "x"):
                val = getattr(dims, name, None)
                if val is not None:
                    return max(0.1, float(val))

        for bbox_name in ("bbox", "bounding_box"):
            bbox = getattr(obs, bbox_name, None)
            if bbox is None:
                continue
            extent = getattr(bbox, "extent", None)
            if extent is not None:
                ex = getattr(extent, "x", None)
                if ex is not None:
                    return max(0.1, 2.0 * float(ex))

        return self.default_actor_length_m

    def _obs_width(self, obs: Any) -> float:
        for name in ("width", "width_m", "size_y"):
            val = getattr(obs, name, None)
            if val is not None:
                return max(0.1, float(val))

        dims = getattr(obs, "dimensions", None)
        if dims is not None:
            for name in ("width", "y"):
                val = getattr(dims, name, None)
                if val is not None:
                    return max(0.1, float(val))

        for bbox_name in ("bbox", "bounding_box"):
            bbox = getattr(obs, bbox_name, None)
            if bbox is None:
                continue
            extent = getattr(bbox, "extent", None)
            if extent is not None:
                ey = getattr(extent, "y", None)
                if ey is not None:
                    return max(0.1, 2.0 * float(ey))

        return self.default_actor_width_m