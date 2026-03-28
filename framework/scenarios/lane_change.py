from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import carla

from framework.core.types import Pose2D, Route
from framework.scenarios.base_scenario import BaseScenario
from framework.carla_io.sensor import SensorSuite


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _yaw_rad_from_deg(yaw_deg: float) -> float:
    return math.radians(yaw_deg)


@dataclass
class ScenarioConfig:
    map_name: str = "Town10HD"
    weather: str = "ClearNoon"

    ego_blueprint: str = "vehicle.tesla.model3"
    ego_spawn: Dict[str, Any] = None

    goal: Dict[str, Any] = None
    goal_radius_m: float = 4.0
    timeout_s: float = 200.0

    route: Dict[str, Any] = None
    actors: list[Dict[str, Any]] = None
    ego_initial_speed_mps: float = 0.0
    random_seed: int = 0


class ConfigurableRouteScenario(BaseScenario):
    name = "configurable_route"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        cfg = self.config or {}
        self.cfg = ScenarioConfig(
            map_name=str(cfg.get("map_name", "Town10HD")),
            weather=str(cfg.get("weather", "ClearNoon")),
            ego_blueprint=str(cfg.get("ego_blueprint", "vehicle.tesla.model3")),
            ego_spawn=cfg.get("ego_spawn") or {"spawn_point_index": int(cfg.get("ego_spawn_index", 0))},
            goal=cfg.get("goal") or {"spawn_point_index": int(cfg.get("goal_index", 20))},
            goal_radius_m=float(cfg.get("goal_radius_m", 4.0)),
            timeout_s=float(cfg.get("timeout_s", 200.0)),
            route=cfg.get("route") or {"source": "grp", "sampling_resolution_m": float(cfg.get("sampling_resolution_m", 2.0))},
            actors=list(cfg.get("actors") or []),
            ego_initial_speed_mps=float(cfg.get("ego_initial_speed_mps", 0.0)),
            random_seed=int(cfg.get("random_seed", 0)),
        )

        self._route: Optional[Route] = None
        self._goal_transform: Optional[carla.Transform] = None
        self._start_transform: Optional[carla.Transform] = None
        self._t0_sim: Optional[float] = None

        self.lead_vehicle_cfg = dict(cfg.get("lead_vehicle") or {})
        self.lead_vehicle: Optional[carla.Vehicle] = None
        self.adjacent_vehicle_cfg = dict(cfg.get("adjacent_vehicle") or {})
        self.adjacent_vehicle: Optional[carla.Vehicle] = None
        self._ego_speed_initialized: bool = False

    # ---------------------------
    # BaseScenario API
    # ---------------------------
    def setup(self, client: carla.Client) -> carla.World:
        random.seed(self.cfg.random_seed)
        world = client.load_world(self.cfg.map_name)
        self.world = world

        world.set_weather(self._resolve_weather(self.cfg.weather))

        carla_map = world.get_map()
        spawns = carla_map.get_spawn_points()
        if not spawns:
            raise RuntimeError("No spawn points available in this map.")

        self._start_transform = self._resolve_spawn(spawns, self.cfg.ego_spawn, z_lift=0.3)
        self._goal_transform = self._resolve_spawn(spawns, self.cfg.goal, z_lift=0.0)

        bp_lib = world.get_blueprint_library()
        ego_bp = bp_lib.find(self.cfg.ego_blueprint)
        ego = world.try_spawn_actor(ego_bp, self._start_transform)
        if ego is None:
            ego = self._retry_spawn_vehicle(world, ego_bp, spawns, preferred=self.cfg.ego_spawn, tries=10)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.ego_vehicle = ego
        #self._start_transform = ego.get_transform()

        s_cfg = dict((self.config or {}).get("sensors") or {})
        camera_cfg = dict(s_cfg.get("camera") or {})
        radar_cfg = dict(s_cfg.get("radar") or {})

        enable_camera = bool(s_cfg.get("enable_camera", True))
        enable_radar = bool(s_cfg.get("enable_radar", True))

        self.sensor_suite = SensorSuite(
            world,
            self.ego_vehicle,
            camera_cfg=camera_cfg,
            radar_cfg=radar_cfg,
            enable_camera=enable_camera,
            enable_radar=enable_radar,
        )

        self._spawn_configured_actors(world, spawns, self.cfg.actors)

        self._route = self._build_route(world, self._start_transform, self._goal_transform, self.cfg.route)

        self._done_info = {}
        self._t0_sim = None
        self._ego_speed_initialized = False

        if self.lead_vehicle_cfg.get("enable", False):
            self._spawn_lead_vehicle_ahead(world)
        if self.adjacent_vehicle_cfg.get("enable", False):
            self._spawn_adjacent_rear_vehicle(world)


        return world
    

    def get_route(self) -> Route:
        if self._route is None:
            raise RuntimeError("Route not built yet. Did you call setup()?")
        return self._route

    def tick(self, t_sim: float) -> None:
        if self._t0_sim is None:
            self._t0_sim = float(t_sim)

        if (
            not self._ego_speed_initialized
            and self.ego_vehicle is not None
            and self.cfg.ego_initial_speed_mps > 0.0
        ):
            try:
                tf = self.ego_vehicle.get_transform()
                yaw = math.radians(float(tf.rotation.yaw))
                v = float(self.cfg.ego_initial_speed_mps)
                self.ego_vehicle.set_target_velocity(
                    carla.Vector3D(
                        x=v * math.cos(yaw),
                        y=v * math.sin(yaw),
                        z=0.0,
                    )
                )
            except Exception:
                pass
            self._ego_speed_initialized = True

        if self.lead_vehicle is not None:
            try:
                tf = self.lead_vehicle.get_transform()
                yaw = math.radians(float(tf.rotation.yaw))
                v = float(self.lead_vehicle_cfg.get("target_speed_mps", 2.0))
                self.lead_vehicle.set_target_velocity(
                    carla.Vector3D(
                        x=v * math.cos(yaw),
                        y=v * math.sin(yaw),
                        z=0.0,
                    )
                )
            except Exception:
                pass

        if self.adjacent_vehicle is not None:
            try:
                tf = self.adjacent_vehicle.get_transform()
                yaw = math.radians(float(tf.rotation.yaw))
                v = float(self.adjacent_vehicle_cfg.get("target_speed_mps", 2.0))
                self.adjacent_vehicle.set_target_velocity(
                    carla.Vector3D(
                        x=v * math.cos(yaw),
                        y=v * math.sin(yaw),
                        z=0.0,
                    )
                )
            except Exception:
                pass

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        if self.world is None or self.ego_vehicle is None or self._goal_transform is None:
            return True, {"reason": "scenario_not_ready"}

        ego_loc = self.ego_vehicle.get_transform().location
        goal_loc = self._goal_transform.location
        dx = ego_loc.x - goal_loc.x
        dy = ego_loc.y - goal_loc.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= self.cfg.goal_radius_m:
            return True, {"reason": "reached_goal", "dist_to_goal": float(dist)}

        if self._t0_sim is not None:
            snap = self.world.get_snapshot()
            now = float(snap.timestamp.elapsed_seconds) if snap is not None else None
            if now is not None and now >= self.cfg.timeout_s:
                return True, {"reason": "timeout", "elapsed_s": float(now), "dist_to_goal": float(dist)}

        return False, {"reason": "running", "dist_to_goal": float(dist)}

    def destroy(self) -> None:
        super().destroy()
        self._route = None
        self._goal_transform = None
        self._start_transform = None
        self._t0_sim = None
        self.lead_vehicle = None
        self.adjacent_vehicle = None

    # ---------------------------
    # Helpers
    # ---------------------------
    def _resolve_weather(self, name: str) -> carla.WeatherParameters:
        preset = getattr(carla.WeatherParameters, name, None)
        if preset is None:
            preset = carla.WeatherParameters.ClearNoon
        return preset

    def _resolve_spawn(
        self,
        spawns: list[carla.Transform],
        spec: Dict[str, Any],
        *,
        z_lift: float = 0.0,
    ) -> carla.Transform:
        if "spawn_point_index" in spec:
            idx = int(spec["spawn_point_index"])
            idx = max(0, min(idx, len(spawns) - 1))
            tf = spawns[idx]
            return carla.Transform(
                carla.Location(
                    x=tf.location.x,
                    y=tf.location.y,
                    z=tf.location.z + float(z_lift),
                ),
                tf.rotation,
            )

        if "transform" in spec:
            t = spec["transform"]
            loc = carla.Location(
                x=float(t.get("x", 0.0)),
                y=float(t.get("y", 0.0)),
                z=float(t.get("z", 0.0)) + float(z_lift),
            )
            rot = carla.Rotation(
                yaw=float(t.get("yaw", 0.0)),
                pitch=float(t.get("pitch", 0.0)),
                roll=float(t.get("roll", 0.0)),
            )
            return carla.Transform(loc, rot)

        return spawns[0]

    def _retry_spawn_vehicle(
        self,
        world: carla.World,
        bp: carla.ActorBlueprint,
        spawns: list[carla.Transform],
        *,
        preferred: Dict[str, Any],
        tries: int = 10,
    ) -> Optional[carla.Vehicle]:
        start_idx = 0
        if "spawn_point_index" in preferred:
            start_idx = int(preferred["spawn_point_index"]) % len(spawns)

        for k in range(min(tries, len(spawns))):
            tf = spawns[(start_idx + k) % len(spawns)]
            tf = carla.Transform(
                carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.3),
                tf.rotation,
            )
            a = world.try_spawn_actor(bp, tf)
            if a is not None:
                return a
        return None

    def _spawn_configured_actors(
        self,
        world: carla.World,
        spawns: list[carla.Transform],
        actors_cfg: list[Dict[str, Any]],
    ) -> None:
        if not actors_cfg:
            return

        bp_lib = world.get_blueprint_library()

        for spec in actors_cfg:
            a_type = str(spec.get("type", "vehicle"))
            blueprint = str(spec.get("blueprint", "vehicle.audi.tt"))
            spawn_spec = spec.get("spawn", {"spawn_point_index": 1})

            if a_type == "vehicle":
                bp = bp_lib.find(blueprint)
                tf = self._resolve_spawn(spawns, spawn_spec, z_lift=0.3)
                actor = world.try_spawn_actor(bp, tf)
                if actor is None:
                    continue
                self.actors.append(actor)

                if bool(spec.get("autopilot", False)):
                    actor.set_autopilot(True)

            elif a_type == "walker":
                bp = bp_lib.find(blueprint)
                tf = self._resolve_spawn(spawns, spawn_spec, z_lift=0.0)
                actor = world.try_spawn_actor(bp, tf)
                if actor is None:
                    continue
                self.actors.append(actor)

            elif a_type == "prop":
                bp = bp_lib.find(blueprint)
                tf = self._resolve_spawn(spawns, spawn_spec, z_lift=0.0)
                actor = world.try_spawn_actor(bp, tf)
                if actor is None:
                    continue
                self.actors.append(actor)

    def _build_route(self, world: carla.World, start: carla.Transform, goal: carla.Transform, route_cfg: Dict[str, Any]) -> Route:
        source = str(route_cfg.get("source", "grp"))
        if source != "grp":
            raise ValueError(f"Unsupported route source: {source}. Use 'grp' for now.")

        sampling = float(route_cfg.get("sampling_resolution_m", 2.0))

        from agents.navigation.global_route_planner import GlobalRoutePlanner

        carla_map = world.get_map()
    
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=sampling)

        route = grp.trace_route(start.location, goal.location)
        points: list[Pose2D] = []
        for wp, _opt in route:
            tf = wp.transform
            points.append(
                Pose2D(
                    x=float(tf.location.x),
                    y=float(tf.location.y),
                    yaw=_wrap_pi(_yaw_rad_from_deg(float(tf.rotation.yaw))),
                )
            )

        if not points:
            raise RuntimeError("GRP produced an empty route.")

        return Route(points=points)

    # optional for metrics/debug
    def get_goal(self) -> Optional[Any]:
        return self._goal_transform

    def get_meta(self) -> Dict[str, Any]:
        return {
            "scenario": self.name,
            "map_name": self.cfg.map_name,
            "weather": self.cfg.weather,
            "ego_spawn": self.cfg.ego_spawn,
            "goal": self.cfg.goal,
            "goal_radius_m": self.cfg.goal_radius_m,
            "timeout_s": self.cfg.timeout_s,
            "route": self.cfg.route,
            "actors": self.cfg.actors,
        }


    def _advance_same_lane(
        self,
        wp: carla.Waypoint,
        dist_m: float,
        *,
        step_m: float = 2.0,
    ) -> Optional[carla.Waypoint]:
        cur = wp
        traveled = 0.0

        while traveled < dist_m:
            nxts = cur.next(step_m)
            if not nxts:
                return None

            nxt = None
            for cand in nxts:
                if cand.road_id == cur.road_id and cand.lane_id == cur.lane_id:
                    nxt = cand
                    break

            if nxt is None:
                return None

            cur = nxt
            traveled += step_m

        return cur

    def _spawn_lead_vehicle_ahead(self, world: carla.World) -> None:
        if self.ego_vehicle is None:
            return

        cfg = self.lead_vehicle_cfg
        blueprint_id = str(cfg.get("blueprint", "vehicle.audi.tt"))
        distance_m = float(cfg.get("distance_m", 20.0))
        step_m = float(cfg.get("step_m", 2.0))
        lateral_offset_m = float(cfg.get("lateral_offset_m", 0.0))
        z_lift = float(cfg.get("z_lift", 0.5))

        carla_map = world.get_map()
        bp_lib = world.get_blueprint_library()

        ego_tf = self.ego_vehicle.get_transform()
        ego_wp = carla_map.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            return

        distance_candidates = [
            distance_m,
            distance_m + 5.0,
            distance_m + 10.0,
            max(8.0, distance_m - 5.0),
        ]

        bp = bp_lib.find(blueprint_id)
        actor = None

        for dist in distance_candidates:
            lead_wp = self._advance_same_lane(ego_wp, dist, step_m=step_m)
            if lead_wp is None:
                continue

            tf0 = lead_wp.transform
            yaw_rad = math.radians(float(tf0.rotation.yaw))

            left_x = -math.sin(yaw_rad)
            left_y = math.cos(yaw_rad)

            base_x = tf0.location.x + lateral_offset_m * left_x
            base_y = tf0.location.y + lateral_offset_m * left_y
            base_z = tf0.location.z

            offsets_s = [0.0, 1.0, -1.0, 2.0, -2.0]
            z_lifts = [z_lift, z_lift + 0.3, z_lift + 0.6]

            for s in offsets_s:
                for dz in z_lifts:
                    x = base_x + s * math.cos(yaw_rad)
                    y = base_y + s * math.sin(yaw_rad)
                    z = base_z + dz

                    tf = carla.Transform(
                        carla.Location(x=x, y=y, z=z),
                        tf0.rotation,
                    )

                    actor = world.try_spawn_actor(bp, tf)
                    if actor is not None:
                        break
                if actor is not None:
                    break

            if actor is not None:
                break

        if actor is None:
            return

        self.actors.append(actor)
        self.lead_vehicle = actor

    def _spawn_adjacent_rear_vehicle(self, world: carla.World) -> None:
        if self.ego_vehicle is None:
            return
        cfg = self.adjacent_vehicle_cfg
        side = str(cfg.get("side", "left")).lower()
        blueprint_id = str(cfg.get("blueprint", "vehicle.audi.tt"))
        rear_distance_m = float(cfg.get("rear_distance_m", 18.0))
        step_m = float(cfg.get("step_m", 2.0))
        z_lift = float(cfg.get("z_lift", 0.5))

        carla_map = world.get_map()
        bp_lib = world.get_blueprint_library()

        ego_tf = self.ego_vehicle.get_transform()
        ego_wp = carla_map.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if ego_wp is None:
            return

        adj_wp = ego_wp.get_left_lane() if side == "left" else ego_wp.get_right_lane()
        if adj_wp is None or adj_wp.lane_type != carla.LaneType.Driving:
            return

        cur = adj_wp
        traveled = 0.0
        while traveled < rear_distance_m:
            prevs = cur.previous(step_m)
            if not prevs:
                break
            cur = prevs[0]
            traveled += step_m

        bp = bp_lib.find(blueprint_id)
        tf = cur.transform
        spawn_tf = carla.Transform(
            carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + z_lift),
            tf.rotation,
        )
        actor = world.try_spawn_actor(bp, spawn_tf)
        if actor is None:
            return

        self.actors.append(actor)
        self.adjacent_vehicle = actor
