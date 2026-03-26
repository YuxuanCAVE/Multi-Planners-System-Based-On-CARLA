from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BoxGeom:
    actor_id: Any
    kind: str  # "body" or "safety"
    x: float
    y: float
    yaw: float
    length_m: float
    width_m: float
    speed: float


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
    safety: BoxGeom


class ActorModelAdapter:
    """
    Read raw obstacle objects from world.obstacles and convert them into:
    - body box   : real actor body, for hard collision
    - safety box : extended safety region, for soft cost / behavior bias

    Expected obstacle style is aligned with your existing scripts:
    - obs.id
    - obs.position.x / obs.position.y
    - obs.velocity.x / obs.velocity.y / obs.velocity.z

    Optional fields supported:
    - yaw / heading / theta
    - pose.yaw / orientation.yaw
    - length / width
    - bbox.extent.x / bbox.extent.y
    - bounding_box.extent.x / bounding_box.extent.y
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.config = cfg

        # ------------------------------------------------------------------
        # default raw actor size
        # ------------------------------------------------------------------
        self.default_actor_length_m = float(cfg.get("default_actor_length_m", 4.5))
        self.default_actor_width_m = float(cfg.get("default_actor_width_m", 1.8))

        # ------------------------------------------------------------------
        # tiny body inflation for robust hard collision
        # ------------------------------------------------------------------
        self.body_front_margin_m = float(cfg.get("body_front_margin_m", 0.0))
        self.body_rear_margin_m = float(cfg.get("body_rear_margin_m", 0.0))
        self.body_lateral_margin_m = float(cfg.get("body_lateral_margin_m", 0.0))

        # ------------------------------------------------------------------
        # safety extension
        # ------------------------------------------------------------------
        self.safety_front_m = float(cfg.get("safety_front_m", 6.0))
        self.safety_rear_m = float(cfg.get("safety_rear_m", 2.0))
        self.safety_lateral_m = float(cfg.get("safety_lateral_m", 0.4))

        # optional dynamic safety extension
        self.enable_dynamic_safety = bool(cfg.get("enable_dynamic_safety", False))
        self.safety_front_speed_coeff = float(cfg.get("safety_front_speed_coeff", 0.0))
        self.safety_rear_speed_coeff = float(cfg.get("safety_rear_speed_coeff", 0.0))
        self.safety_lateral_speed_coeff = float(cfg.get("safety_lateral_speed_coeff", 0.0))

        self.max_safety_front_m = float(cfg.get("max_safety_front_m", 12.0))
        self.max_safety_rear_m = float(cfg.get("max_safety_rear_m", 5.0))
        self.max_safety_lateral_m = float(cfg.get("max_safety_lateral_m", 1.2))

        # yaw fallback
        self.min_speed_for_velocity_yaw_mps = float(
            cfg.get("min_speed_for_velocity_yaw_mps", 0.2)
        )

        # filtering
        self.ignore_actor_ids = set(cfg.get("ignore_actor_ids", []))

    # ----------------------------------------------------------------------
    # public
    # ----------------------------------------------------------------------
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

        # -------------------------
        # body box
        # -------------------------
        body_front = self.body_front_margin_m
        body_rear = self.body_rear_margin_m
        body_lat = self.body_lateral_margin_m

        body_box = self._build_shifted_box(
            actor_id=actor_id,
            kind="body",
            ref_x=x,
            ref_y=y,
            yaw=yaw,
            speed=speed,
            base_length_m=raw_length_m,
            base_width_m=raw_width_m,
            front_ext_m=body_front,
            rear_ext_m=body_rear,
            lateral_ext_m=body_lat,
        )

        # -------------------------
        # safety box
        # -------------------------
        front_safe_m, rear_safe_m, lateral_safe_m = self._compute_safety_extensions(speed)

        safety_box = self._build_shifted_box(
            actor_id=actor_id,
            kind="safety",
            ref_x=x,
            ref_y=y,
            yaw=yaw,
            speed=speed,
            base_length_m=raw_length_m,
            base_width_m=raw_width_m,
            front_ext_m=front_safe_m,
            rear_ext_m=rear_safe_m,
            lateral_ext_m=lateral_safe_m,
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
            body=body_box,
            safety=safety_box,
        )

    def build_all(self, world: Any) -> List[ActorModel]:
        out: List[ActorModel] = []
        for obs in getattr(world, "obstacles", []) or []:
            model = self.build_actor_model(obs)
            if model is not None:
                out.append(model)
        return out

    def get_box_corners(self, box: BoxGeom) -> List[Tuple[float, float]]:
        """
        Return 4 corners of the oriented box in world frame.
        Order:
            front-left, front-right, rear-right, rear-left
        """
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

    def sample_box_points(
        self,
        box: BoxGeom,
        longitudinal_step_m: float = 0.5,
        lateral_step_m: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """
        Useful when your planner writes obstacles into occupancy grid by sampled points.
        """
        length_m = max(0.1, float(box.length_m))
        width_m = max(0.1, float(box.width_m))
        longitudinal_step_m = max(0.05, float(longitudinal_step_m))
        lateral_step_m = max(0.05, float(lateral_step_m))

        nx = max(1, int(math.ceil(length_m / longitudinal_step_m)))
        ny = max(1, int(math.ceil(width_m / lateral_step_m)))

        c = math.cos(box.yaw)
        s = math.sin(box.yaw)

        half_l = 0.5 * length_m
        half_w = 0.5 * width_m

        pts: List[Tuple[float, float]] = []
        for ix in range(nx + 1):
            lx = -half_l + (length_m * ix / nx)
            for iy in range(ny + 1):
                ly = -half_w + (width_m * iy / ny)
                wx = box.x + lx * c - ly * s
                wy = box.y + lx * s + ly * c
                pts.append((wx, wy))
        return pts

    # ----------------------------------------------------------------------
    # internal
    # ----------------------------------------------------------------------
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

    def _build_shifted_box(
        self,
        *,
        actor_id: Any,
        kind: str,
        ref_x: float,
        ref_y: float,
        yaw: float,
        speed: float,
        base_length_m: float,
        base_width_m: float,
        front_ext_m: float,
        rear_ext_m: float,
        lateral_ext_m: float,
    ) -> BoxGeom:
        """
        For asymmetric front/rear extension, box center must shift along heading direction.
        """
        total_length_m = base_length_m + front_ext_m + rear_ext_m
        total_width_m = base_width_m + 2.0 * lateral_ext_m

        shift_x_local = 0.5 * (front_ext_m - rear_ext_m)
        c = math.cos(yaw)
        s = math.sin(yaw)

        cx = ref_x + shift_x_local * c
        cy = ref_y + shift_x_local * s

        return BoxGeom(
            actor_id=actor_id,
            kind=kind,
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
        # direct fields
        for name in ("yaw", "heading", "theta"):
            val = getattr(obs, name, None)
            if val is not None:
                return float(val)

        # nested pose
        pose = getattr(obs, "pose", None)
        if pose is not None:
            for name in ("yaw", "heading", "theta"):
                val = getattr(pose, name, None)
                if val is not None:
                    return float(val)

        # nested orientation
        orientation = getattr(obs, "orientation", None)
        if orientation is not None:
            val = getattr(orientation, "yaw", None)
            if val is not None:
                return float(val)

        # velocity direction fallback
        if speed > self.min_speed_for_velocity_yaw_mps:
            vel = getattr(obs, "velocity", None)
            if vel is not None:
                vx = float(getattr(vel, "x", 0.0))
                vy = float(getattr(vel, "y", 0.0))
                if math.hypot(vx, vy) > 1e-6:
                    return math.atan2(vy, vx)

        return 0.0

    def _obs_length(self, obs: Any) -> float:
        # direct fields
        for name in ("length", "length_m", "size_x"):
            val = getattr(obs, name, None)
            if val is not None:
                return max(0.1, float(val))

        # nested dimensions
        dims = getattr(obs, "dimensions", None)
        if dims is not None:
            for name in ("length", "x"):
                val = getattr(dims, name, None)
                if val is not None:
                    return max(0.1, float(val))

        # bbox.extent.x -> half length
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
        # direct fields
        for name in ("width", "width_m", "size_y"):
            val = getattr(obs, name, None)
            if val is not None:
                return max(0.1, float(val))

        # nested dimensions
        dims = getattr(obs, "dimensions", None)
        if dims is not None:
            for name in ("width", "y"):
                val = getattr(dims, name, None)
                if val is not None:
                    return max(0.1, float(val))

        # bbox.extent.y -> half width
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