from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from framework.core.types import (
    EgoState,
    PlanResult,
    PlanStatus,
    Pose2D,
    Route,
    Trajectory,
    TrajectoryPoint,
    WorldModel,
)
from framework.planning.base_planning import BasePlanner


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _hypot(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


@dataclass
class _Node:
    x: float
    y: float
    parent: Optional[int]
    cost: float


class _OccGrid:
    def __init__(self, *, ego_x: float, ego_y: float, size_m: float, res_m: float):
        self.size_m = float(size_m)
        self.res_m = float(res_m)
        self.w = int(math.ceil(self.size_m / self.res_m))
        self.h = int(math.ceil(self.size_m / self.res_m))
        half = 0.5 * self.size_m
        self.min_x = float(ego_x) - half
        self.min_y = float(ego_y) - half
        self.max_x = self.min_x + self.w * self.res_m
        self.max_y = self.min_y + self.h * self.res_m
        self.occ = [[False] * self.w for _ in range(self.h)]

    def world_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        eps = 1e-9
        if x < self.min_x - eps or x > self.max_x + eps or y < self.min_y - eps or y > self.max_y + eps:
            return None
        j = int((min(max(x, self.min_x), self.max_x - eps) - self.min_x) / self.res_m)
        i = int((min(max(y, self.min_y), self.max_y - eps) - self.min_y) / self.res_m)
        if 0 <= i < self.h and 0 <= j < self.w:
            return i, j
        return None

    def is_occupied(self, x: float, y: float) -> bool:
        ij = self.world_to_ij(x, y)
        if ij is None:
            return True
        i, j = ij
        return self.occ[i][j]

    def set_occupied_disc(self, x: float, y: float, r: float) -> None:
        c = self.world_to_ij(x, y)
        if c is None:
            return
        ci, cj = c
        ri = int(math.ceil(r / self.res_m))
        r2 = r * r
        for di in range(-ri, ri + 1):
            ii = ci + di
            if not (0 <= ii < self.h):
                continue
            dy = di * self.res_m
            row = self.occ[ii]
            for dj in range(-ri, ri + 1):
                jj = cj + dj
                if not (0 <= jj < self.w):
                    continue
                dx = dj * self.res_m
                if dx * dx + dy * dy <= r2:
                    row[jj] = True


class RRTStarPlanner(BasePlanner):
    name = "rrt_star"

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route
        self._last_goal_idx = 0

    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        cfg = self.config
        dt_out = float(cfg.get("dt", 0.1))
        horizon_s = float(cfg.get("horizon_s", 5.0))
        target_speed = float(cfg.get("target_speed", 6.0))

        grid = _OccGrid(
            ego_x=ego.pose.x,
            ego_y=ego.pose.y,
            size_m=float(cfg.get("grid_size_m", 60.0)),
            res_m=float(cfg.get("grid_res_m", 0.3)),
        )
        inflation = float(cfg.get("inflation_m", 1.0))
        for obs in world.obstacles:
            r = max(0.0, float(obs.radius) + inflation)
            grid.set_occupied_disc(obs.position.x, obs.position.y, r)

        goal = self._pick_goal(ego=ego, lookahead_m=float(cfg.get("lookahead_m", 25.0)))
        if goal is None:
            return PlanResult(status=PlanStatus.EMPTY, trajectory=None, debug={"reason": "no_route"})

        t0 = time.time()
        path_xy = self._rrt_star_search(
            start=(ego.pose.x, ego.pose.y),
            goal=(goal.x, goal.y),
            grid=grid,
            max_iter=int(cfg.get("max_iter", 1200)),
            max_time_ms=float(cfg.get("max_time_ms", 50.0)),
            step_size_m=float(cfg.get("step_size_m", 2.0)),
            goal_sample_rate=float(cfg.get("goal_sample_rate", 0.2)),
            goal_tolerance_m=float(cfg.get("goal_tolerance_m", 2.0)),
            rewire_radius_m=float(cfg.get("rewire_radius_m", 4.0)),
            edge_sample_res_m=float(cfg.get("edge_sample_res_m", 0.5)),
            rng_seed=cfg.get("rng_seed", None),
        )
        ms = (time.time() - t0) * 1000.0

        if not path_xy:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={"planner": "rrt_star", "reason": "no_path", "ms": float(ms)},
            )

        path_xyz = self._xy_to_xyyaw(path_xy, ego.pose.yaw)
        traj = self._path_to_trajectory(path_xyz=path_xyz, dt=dt_out, horizon_s=horizon_s, v=target_speed)

        return PlanResult(
            status=PlanStatus.OK,
            trajectory=traj,
            debug={
                "planner": "rrt_star",
                "ms": float(ms),
                "path_len": int(len(path_xyz)),
                "goal": (float(goal.x), float(goal.y), float(goal.yaw)),
            },
        )

    def _pick_goal(self, *, ego: EgoState, lookahead_m: float) -> Optional[Pose2D]:
        pts = self._route.points if hasattr(self, "_route") and self._route is not None else []
        if not pts:
            return None

        ex, ey = ego.pose.x, ego.pose.y
        best_i = 0
        best_d2 = float("inf")
        for i, p in enumerate(pts):
            d2 = _dist2(p.x, p.y, ex, ey)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        dist = 0.0
        j = best_i
        while j + 1 < len(pts) and dist < lookahead_m:
            dx = pts[j + 1].x - pts[j].x
            dy = pts[j + 1].y - pts[j].y
            dist += _hypot(dx, dy)
            j += 1

        self._last_goal_idx = j
        return pts[j]

    def _rrt_star_search(
        self,
        *,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        grid: _OccGrid,
        max_iter: int,
        max_time_ms: float,
        step_size_m: float,
        goal_sample_rate: float,
        goal_tolerance_m: float,
        rewire_radius_m: float,
        edge_sample_res_m: float,
        rng_seed: Optional[int],
    ) -> Optional[List[Tuple[float, float]]]:
        rng = random.Random(rng_seed)

        sx, sy = start
        gx, gy = goal

        if grid.is_occupied(sx, sy) or grid.is_occupied(gx, gy):
            return None

        nodes: List[_Node] = [_Node(x=sx, y=sy, parent=None, cost=0.0)]
        goal_idx: Optional[int] = None
        t0 = time.time()

        for _ in range(max_iter):
            if (time.time() - t0) * 1000.0 >= max_time_ms:
                break

            if rng.random() < goal_sample_rate:
                rx, ry = gx, gy
            else:
                rx = rng.uniform(grid.min_x, grid.max_x)
                ry = rng.uniform(grid.min_y, grid.max_y)

            near_idx = self._nearest_node(nodes, rx, ry)
            nx, ny = self._steer(nodes[near_idx].x, nodes[near_idx].y, rx, ry, step_size_m)

            if not self._collision_free_segment(nodes[near_idx].x, nodes[near_idx].y, nx, ny, grid, edge_sample_res_m):
                continue

            near_ids = self._near_nodes(nodes, nx, ny, rewire_radius_m)
            parent_idx = near_idx
            new_cost = nodes[near_idx].cost + _hypot(nx - nodes[near_idx].x, ny - nodes[near_idx].y)

            for i in near_ids:
                if not self._collision_free_segment(nodes[i].x, nodes[i].y, nx, ny, grid, edge_sample_res_m):
                    continue
                c = nodes[i].cost + _hypot(nx - nodes[i].x, ny - nodes[i].y)
                if c < new_cost:
                    parent_idx = i
                    new_cost = c

            new_idx = len(nodes)
            nodes.append(_Node(x=nx, y=ny, parent=parent_idx, cost=new_cost))

            for i in near_ids:
                c_through_new = new_cost + _hypot(nodes[i].x - nx, nodes[i].y - ny)
                if c_through_new >= nodes[i].cost:
                    continue
                if not self._collision_free_segment(nx, ny, nodes[i].x, nodes[i].y, grid, edge_sample_res_m):
                    continue
                nodes[i].parent = new_idx
                nodes[i].cost = c_through_new

            if _dist2(nx, ny, gx, gy) <= goal_tolerance_m * goal_tolerance_m:
                if self._collision_free_segment(nx, ny, gx, gy, grid, edge_sample_res_m):
                    goal_idx = len(nodes)
                    total_cost = new_cost + _hypot(gx - nx, gy - ny)
                    nodes.append(_Node(x=gx, y=gy, parent=new_idx, cost=total_cost))
                    break

        if goal_idx is None:
            best = None
            best_d2 = float("inf")
            for i, n in enumerate(nodes):
                d2 = _dist2(n.x, n.y, gx, gy)
                if d2 < best_d2 and self._collision_free_segment(n.x, n.y, gx, gy, grid, edge_sample_res_m):
                    best_d2 = d2
                    best = i
            if best is None:
                return None
            goal_idx = len(nodes)
            nodes.append(_Node(x=gx, y=gy, parent=best, cost=nodes[best].cost + _hypot(gx - nodes[best].x, gy - nodes[best].y)))

        return self._backtrack_path(nodes, goal_idx)

    @staticmethod
    def _nearest_node(nodes: List[_Node], x: float, y: float) -> int:
        best_i = 0
        best_d2 = float("inf")
        for i, n in enumerate(nodes):
            d2 = _dist2(n.x, n.y, x, y)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return best_i

    @staticmethod
    def _near_nodes(nodes: List[_Node], x: float, y: float, r: float) -> List[int]:
        r2 = r * r
        out: List[int] = []
        for i, n in enumerate(nodes):
            if _dist2(n.x, n.y, x, y) <= r2:
                out.append(i)
        return out

    @staticmethod
    def _steer(x0: float, y0: float, x1: float, y1: float, step: float) -> Tuple[float, float]:
        dx = x1 - x0
        dy = y1 - y0
        d = _hypot(dx, dy)
        if d <= step:
            return x1, y1
        ux = dx / max(d, 1e-9)
        uy = dy / max(d, 1e-9)
        return x0 + ux * step, y0 + uy * step

    @staticmethod
    def _collision_free_segment(x0: float, y0: float, x1: float, y1: float, grid: _OccGrid, ds: float) -> bool:
        dist = _hypot(x1 - x0, y1 - y0)
        n = max(2, int(dist / max(ds, 1e-6)) + 1)
        for k in range(n):
            t = k / (n - 1)
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            if grid.is_occupied(x, y):
                return False
        return True

    @staticmethod
    def _backtrack_path(nodes: List[_Node], goal_idx: int) -> List[Tuple[float, float]]:
        path_rev: List[Tuple[float, float]] = []
        i: Optional[int] = goal_idx
        while i is not None:
            n = nodes[i]
            path_rev.append((n.x, n.y))
            i = n.parent
        path_rev.reverse()
        return path_rev

    @staticmethod
    def _xy_to_xyyaw(path_xy: List[Tuple[float, float]], yaw0: float) -> List[Tuple[float, float, float]]:
        if not path_xy:
            return []
        out: List[Tuple[float, float, float]] = []
        prev_yaw = yaw0
        for i in range(len(path_xy)):
            x, y = path_xy[i]
            if i + 1 < len(path_xy):
                nx, ny = path_xy[i + 1]
                yaw_raw = math.atan2(ny - y, nx - x)
            else:
                yaw_raw = prev_yaw
            yaw = prev_yaw + _wrap_pi(yaw_raw - prev_yaw)
            prev_yaw = yaw
            out.append((float(x), float(y), float(_wrap_pi(yaw))))
        return out

    @staticmethod
    def _path_to_trajectory(path_xyz: List[Tuple[float, float, float]], dt: float, horizon_s: float, v: float) -> Trajectory:
        if not path_xyz:
            return Trajectory(points=[], dt=dt)

        steps = max(2, int(horizon_s / max(dt, 1e-6)))

        arclen = [0.0]
        for i in range(1, len(path_xyz)):
            dx = path_xyz[i][0] - path_xyz[i - 1][0]
            dy = path_xyz[i][1] - path_xyz[i - 1][1]
            arclen.append(arclen[-1] + _hypot(dx, dy))

        total = arclen[-1]
        if total <= 1e-6:
            p = path_xyz[0]
            return Trajectory(points=[TrajectoryPoint(x=p[0], y=p[1], yaw=p[2], v=0.0) for _ in range(steps)], dt=dt)

        out: List[TrajectoryPoint] = []
        for k in range(steps):
            s = min(total, k * max(v, 0.0) * dt)
            j = 0
            while j + 1 < len(arclen) and arclen[j + 1] < s:
                j += 1
            if j + 1 >= len(path_xyz):
                x, y, yaw = path_xyz[-1]
            else:
                s0 = arclen[j]
                s1 = arclen[j + 1]
                u = 0.0 if (s1 - s0) <= 1e-9 else (s - s0) / (s1 - s0)
                x0, y0, yaw0 = path_xyz[j]
                x1, y1, yaw1 = path_xyz[j + 1]
                x = x0 + (x1 - x0) * u
                y = y0 + (y1 - y0) * u
                yaw = yaw0 + _wrap_pi(yaw1 - yaw0) * u
            out.append(TrajectoryPoint(x=float(x), y=float(y), yaw=float(_wrap_pi(yaw)), v=float(max(v, 0.0))))

        return Trajectory(points=out, dt=dt)
