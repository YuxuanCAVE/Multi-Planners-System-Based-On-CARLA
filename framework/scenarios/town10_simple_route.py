# framework/scenarios/town10_simple_route.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import carla

from framework.core.types import Pose2D, Route


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def _yaw_rad_from_carla_yaw_deg(yaw_deg: float) -> float:
    return math.radians(yaw_deg)


@dataclass
class Town10SimpleRouteConfig:
    map_name: str = "Town10HD"
    spawn_index: int = 0
    goal_index: int = 20
    # 可选：到达判定距离
    goal_radius_m: float = 4.0


class Town10SimpleRouteScenario:
    """
    最简单场景：
    - 加载 Town10HD
    - spawn ego
    - 设置 goal（另一个 spawn point）
    - 生成 GRP route（全局参考线）
    """
    name = "town10_simple_route"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.cfg = Town10SimpleRouteConfig(
            map_name=str(cfg.get("map_name", "Town10HD")),
            spawn_index=int(cfg.get("spawn_index", 0)),
            goal_index=int(cfg.get("goal_index", 20)),
            goal_radius_m=float(cfg.get("goal_radius_m", 4.0)),
        )
        self.ego_vehicle: Optional[carla.Vehicle] = None
        self.goal_transform: Optional[carla.Transform] = None
        self.route: Optional[Route] = None
    #加载地图并spawn 车辆与route
    def setup(self, client: carla.Client) -> carla.World:
        world = client.load_world(self.cfg.map_name)

        # 基础设置：你也可以在 runner 里统一设置
        world.set_weather(carla.WeatherParameters.ClearNoon)

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points in this map.")
        
        #设置初始点
        si = max(0, min(self.cfg.spawn_index, len(spawn_points) - 1))
        #设置终点
        gi = max(0, min(self.cfg.goal_index, len(spawn_points) - 1))
        spawn_tf = spawn_points[si]
        self.goal_transform = spawn_points[gi]

        blueprint_library = world.get_blueprint_library()
        
        #选择车量blueprint并spawn
        bp = blueprint_library.filter("vehicle.tesla.model3")[0]
        # 更稳定的 spawn（抬高一点点避免碰撞地面）
        spawn_tf.location.z += 0.3

        ego = world.try_spawn_actor(bp, spawn_tf)
        if ego is None:
            # 失败就尝试多几个 spawn point
            for k in range(min(10, len(spawn_points))):
                tf = spawn_points[(si + k) % len(spawn_points)]
                tf.location.z += 0.3
                ego = world.try_spawn_actor(bp, tf)
                if ego is not None:
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        self.ego_vehicle = ego

        # 生成 route
        self.route = self._build_route_with_grp(world, spawn_tf, self.goal_transform)

        if self.goal_transform is not None:
            g = self.goal_transform.location
            world.debug.draw_point(g + carla.Location(z=1.0), size=0.25, color=carla.Color(0, 0, 255), life_time=30.0)
            world.debug.draw_string(g + carla.Location(z=1.8), "GOAL", color=carla.Color(0, 0, 255), life_time=30.0)

        return world

    #用CarlaGlobalRoutePlanner生成全局路线
    def _build_route_with_grp(self, world: carla.World, start: carla.Transform, goal: carla.Transform) -> Route:
        # 需要 CARLA PythonAPI 的 agents 模块
        # 最好在run.py 里把Carla PythonAPI加入 sys.path
        from agents.navigation.global_route_planner import GlobalRoutePlanner  # type: ignore

        carla_map = world.get_map()
        # sampling_resolution 越小点越密；先用 2.0m 足够
        grp = GlobalRoutePlanner(carla_map, sampling_resolution=2.0)
        #grp.setup()

        route = grp.trace_route(start.location, goal.location)  # List[(Waypoint, RoadOption)]
        points: list[Pose2D] = []
        for wp, _ in route:
            tf = wp.transform
            points.append(
                Pose2D(
                    x=float(tf.location.x),
                    y=float(tf.location.y),
                    yaw=_wrap_pi(_yaw_rad_from_carla_yaw_deg(float(tf.rotation.yaw))),
                )
            )
        if not points:
            raise RuntimeError("GRP produced empty route.")
        return Route(points=points)

    def get_route(self) -> Route:
        if self.route is None:
            raise RuntimeError("Scenario route is not built yet.")
        return self.route

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        最小结束条件：到 goal 附近
        """
        if self.ego_vehicle is None or self.goal_transform is None:
            return True, {"reason": "ego_or_goal_missing"}

        ego_loc = self.ego_vehicle.get_transform().location
        goal_loc = self.goal_transform.location
        dx = ego_loc.x - goal_loc.x
        dy = ego_loc.y - goal_loc.y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= self.cfg.goal_radius_m:
            return True, {"reason": "reached_goal", "dist_to_goal": dist}
        return False, {"reason": "running", "dist_to_goal": dist}

    def destroy(self) -> None:
        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
