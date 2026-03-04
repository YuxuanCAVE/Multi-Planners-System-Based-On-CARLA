# framework/scenarios/scenario.py
from __future__ import annotations

import math
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
    # world
    map_name: str = "Town10HD"
    weather: str = "ClearNoon"

    # ego
    ego_blueprint: str = "vehicle.tesla.model3"
    ego_spawn: Dict[str, Any] = None  # {"spawn_point_index": 0} OR {"transform": {...}}

    # goal
    goal: Dict[str, Any] = None  # {"spawn_point_index": 20} OR {"transform": {...}}
    goal_radius_m: float = 4.0
    timeout_s: float = 200.0

    # route
    route: Dict[str, Any] = None  # {"source": "grp", "sampling_resolution_m": 2.0}

    # actors (optional)
    actors: list[Dict[str, Any]] = None  # list of {type, blueprint, spawn, autopilot, ...}


class ConfigurableRouteScenario(BaseScenario):
    """
    A YAML-configurable scenario based on BaseScenario.

    Supports:
    - map selection
    - ego spawn by spawn_point_index or transform
    - goal by spawn_point_index or transform
    - route generation via CARLA GlobalRoutePlanner (GRP)
    - optional extra actors (vehicles/walkers) spawned from config
    - termination: reached_goal OR timeout (collision handled by Runner)

    Example YAML (single file):
      scenario:
        type: configurable_route
        map_name: Town10HD
        weather: ClearNoon
        ego_spawn: {spawn_point_index: 0}
        goal: {spawn_point_index: 20}
        goal_radius_m: 4.0
        timeout_s: 200.0
        route: {source: grp, sampling_resolution_m: 2.0}
        actors:
          - type: vehicle
            blueprint: vehicle.audi.tt
            spawn: {spawn_point_index: 10}
            autopilot: true
    """

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
        )

        self._route: Optional[Route] = None
        self._goal_transform: Optional[carla.Transform] = None
        self._start_transform: Optional[carla.Transform] = None
        self._t0_sim: Optional[float] = None  # set on first tick or in runner meta

    # ---------------------------
    # BaseScenario API
    # ---------------------------
    def setup(self, client: carla.Client) -> carla.World:
        world = client.load_world(self.cfg.map_name)
        self.world = world

        # weather
        world.set_weather(self._resolve_weather(self.cfg.weather))

        # spawn ego
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
            # retry a few spawn points if failed
            ego = self._retry_spawn_vehicle(world, ego_bp, spawns, preferred=self.cfg.ego_spawn, tries=10)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.ego_vehicle = ego  # BaseScenario expects this

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


        # optional extra actors
        self._spawn_configured_actors(world, spawns, self.cfg.actors)

        # build route
        self._route = self._build_route(world, self._start_transform, self._goal_transform, self.cfg.route)

        # reset done info
        self._done_info = {}
        self._t0_sim = None
        return world

    def get_route(self) -> Route:
        if self._route is None:
            raise RuntimeError("Route not built yet. Did you call setup()?")
        return self._route

    def tick(self, t_sim: float) -> None:
        # store episode start time reference (for timeout)
        if self._t0_sim is None:
            self._t0_sim = float(t_sim)

        # (optional) scripted events can be implemented here later
        # e.g., event_engine.update(t_sim)
        return

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        if self.world is None or self.ego_vehicle is None or self._goal_transform is None:
            return True, {"reason": "scenario_not_ready"}

        # goal reached
        ego_loc = self.ego_vehicle.get_transform().location
        goal_loc = self._goal_transform.location
        dx = ego_loc.x - goal_loc.x
        dy = ego_loc.y - goal_loc.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist <= self.cfg.goal_radius_m:
            return True, {"reason": "reached_goal", "dist_to_goal": float(dist)}

        # timeout (if we have t0)
        if self._t0_sim is not None:
            # Runner passes sim time to tick; we don't store current sim time here
            # so we approximate via world snapshot timestamp if available.
            snap = self.world.get_snapshot()
            now = float(snap.timestamp.elapsed_seconds) if snap is not None else None
            # If snapshot not reliable, you can instead pass t_sim into is_done in Runner
            if now is not None and (now - (now - 0.0)) is not None:
                # fallback: use snapshot elapsed_seconds directly against timeout,
                # not perfect but works if episodes start near 0.
                if now >= self.cfg.timeout_s:
                    return True, {"reason": "timeout", "elapsed_s": float(now), "dist_to_goal": float(dist)}

        return False, {"reason": "running", "dist_to_goal": float(dist)}

    def destroy(self) -> None:
        # BaseScenario already destroys sensors/actors/ego safely
        super().destroy()
        self._route = None
        self._goal_transform = None
        self._start_transform = None
        self._t0_sim = None
        self.sensor_suite.destroy()

    # ---------------------------
    # Helpers
    # ---------------------------
    def _resolve_weather(self, name: str) -> carla.WeatherParameters:
        # Map common names to CARLA presets; default to ClearNoon
        preset = getattr(carla.WeatherParameters, name, None)
        if preset is None:
            preset = carla.WeatherParameters.ClearNoon
        return preset

    def _resolve_spawn(self, spawns: list[carla.Transform], spec: Dict[str, Any], *, z_lift: float = 0.0) -> carla.Transform:
        """
        spec:
          - {"spawn_point_index": int}
          - {"transform": {"x":..,"y":..,"z":..,"yaw":..,"pitch":..,"roll":..}}
        """
        if "spawn_point_index" in spec:
            idx = int(spec["spawn_point_index"])
            idx = max(0, min(idx, len(spawns) - 1))
            tf = spawns[idx]
            tf = carla.Transform(
                carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + float(z_lift)),
                tf.rotation,
            )
            return tf

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

        # fallback
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
        # try preferred index first, then scan
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

    def _spawn_configured_actors(self, world: carla.World, spawns: list[carla.Transform], actors_cfg: list[Dict[str, Any]]) -> None:
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

                # autopilot (TrafficManager)
                if bool(spec.get("autopilot", False)):
                    actor.set_autopilot(True)

            elif a_type == "walker":
                # basic walker spawn (no AI controller here; can extend later)
                bp = bp_lib.find(blueprint)  # e.g., "walker.pedestrian.0001"
                tf = self._resolve_spawn(spawns, spawn_spec, z_lift=0.0)
                actor = world.try_spawn_actor(bp, tf)
                if actor is None:
                    continue
                self.actors.append(actor)

            elif a_type == "prop":
                # props depend on available blueprints in your CARLA build; keep as placeholder
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

        # CARLA GlobalRoutePlanner
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

        carla_map = world.get_map()
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=sampling)
        #grp.setup()

        route = grp.trace_route(start.location, goal.location)  # List[(Waypoint, RoadOption)]
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


    #做选择sensor
    def get_sensor_snapshot(self) -> Dict[str, Any]:
        
        out:Dict[str, Any] = {}

        suite = getattr(self, "sensor_suite", None)
        if suite is None:
            return out
        
        try:
            radar_summary, radar_ts = suite.get_radar_summary()
            if radar_summary is not None:
                out["radar"] = {
                    "ts": float(radar_ts),
                    "count": int(radar_summary.get("count", 0)),
                    "min_depth_m": radar_summary.get("min_depth_m", None),
                    "min_depth_radial_vel_mps": radar_summary.get("min_depth_radial_vel_mps", None),
                }
        except Exception:
            # keep robust: sensor failure shouldn't crash simulation
            pass

        # ---- Camera timestamp (lightweight) ----
        # SensorSuite.get_front_image() returns (carla.Image, ts)
        # We only store ts (and maybe frame if available), not the image data.
        try:
            img, cam_ts = suite.get_front_image()
            if img is not None:
                # carla.Image usually has .frame, and timestamp is cam_ts
                out["camera"] = {
                    "ts": float(cam_ts),
                    "frame": int(getattr(img, "frame", -1)),
                    "width": int(getattr(img, "width", 0)),
                    "height": int(getattr(img, "height", 0)),
                }
        except Exception:
            pass

        return out