# framework/planning/local/hybrid_astar_a.py
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from framework.planning.base_planning import BasePlanner
from framework.core.types import (
    EgoState,
    WorldModel,
    Route,
    PlanResult,
    PlanStatus,
    Trajectory,
    TrajectoryPoint,
    Pose2D,
)

from framework.control.vehicle.kinematics import bicycle_rollout, wrap_pi


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def deg2rad(d: float) -> float:
    return d * math.pi / 180.0


def dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx = ax - bx
    dy = ay - by
    return dx * dx + dy * dy


def point_to_segment_dist2(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay
    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        return dist2(px, py, ax, ay)
    t = (apx * abx + apy * aby) / ab2
    t = clamp(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return dist2(px, py, cx, cy)


class OccGrid:
    """Ego-centered occupancy grid in world coordinates."""
    def __init__(self, *, ego_x: float, ego_y: float, size_m: float, res_m: float):
        self.size_m = float(size_m)
        self.res_m = float(res_m)
        self.w = int(math.ceil(self.size_m / self.res_m))
        self.h = int(math.ceil(self.size_m / self.res_m))
        half = 0.5 * self.size_m
        self.min_x = float(ego_x) - half
        self.min_y = float(ego_y) - half
        self.occ = [[False] * self.w for _ in range(self.h)]

    def ij_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.min_x + (j + 0.5) * self.res_m
        y = self.min_y + (i + 0.5) * self.res_m
        return x, y

    def world_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        i = int((y - self.min_y) / self.res_m)
        j = int((x - self.min_x) / self.res_m)
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
        center = self.world_to_ij(x, y)
        if center is None:
            return
        ci, cj = center
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


@dataclass
class Node:
    x: float
    y: float
    yaw: float
    g: float
    h: float
    parent: Optional[Tuple[int, int, int]]
    steer: float
    seg: Optional[List[Tuple[float, float, float]]] = None


class HybridAStarMapPlanner(BasePlanner):
    """
    Minimal Hybrid A* local planner.
    Hard constraint: drivable area from CARLA map (Driving lane), else occupied.
    Dynamic obstacles are added on top.
    """
    name: str = "hybrid_astar_map"

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route
        self._carla_map = (map_info or {}).get("carla_map", None)
        self._last_nearest_idx = 0

        #添加缓存成员
        self._mask_tick: int = 0
        self._mask_center_xy: Optional[Tuple[float, float]] = None
        self._mask_grid_params: Optional[Tuple[int, int, float, float, float]] = None  # (w,h,res,min_x,min_y)
        self._mask_occ: Optional[List[List[bool]]] = None  # True = occupied (non-drivable)

    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        cfg = self.config

        if self._carla_map is None:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={"reason": "carla_map_missing_in_map_info"},
            )

        # --- config ---
        dt_out = float(cfg.get("dt", 0.1))
        horizon_s = float(cfg.get("horizon_s", 5.0))
        target_speed = float(cfg.get("target_speed", 6.0))

        grid_size_m = float(cfg.get("grid_size_m", 60.0))
        grid_res_m = float(cfg.get("grid_res_m", 0.8))

        inflation_m = float(cfg.get("inflation_m", 1.0))

        lookahead_m = float(cfg.get("lookahead_m", 25.0))

        wheelbase_m = float(cfg.get("wheelbase_m", 2.8))
        steer_max = deg2rad(float(cfg.get("steer_max_deg", 30.0)))
        steer_samples = int(cfg.get("steer_samples", 7))
        prim_dt = float(cfg.get("primitive_dt", 0.1))
        prim_steps = int(cfg.get("primitive_steps", 10))
        sim_speed = float(cfg.get("sim_speed_mps", 6.0))

        heading_bins = int(cfg.get("heading_bins", 72))

        max_expansions = int(cfg.get("max_expansions", 6000))
        max_time_ms = float(cfg.get("max_time_ms", 60.0))

        # soft preference (optional)
        w_ref = float(cfg.get("w_ref", 0.0))

        # --- goal on route ---
        route_pts = self._route.points
        goal = self._pick_goal(ego=ego, route_pts=route_pts, lookahead_m=lookahead_m)

        # --- build grid ---
        grid = OccGrid(ego_x=ego.pose.x, ego_y=ego.pose.y, size_m=grid_size_m, res_m=grid_res_m)

        # Hard constraint: drivable area mask (Driving lanes only)
        self._apply_drivable_mask_cached(grid, ego_x=ego.pose.x, ego_y=ego.pose.y)

        # Add obstacles
        for obs in world.obstacles:
            r = max(0.0, float(obs.radius) + inflation_m)
            grid.set_occupied_disc(obs.position.x, obs.position.y, r)

        # --- search ---
        t0 = time.time()
        path = self._search(
            ego=ego,
            goal=goal,
            grid=grid,
            route_pts=route_pts,
            wheelbase_m=wheelbase_m,
            steer_max=steer_max,
            steer_samples=steer_samples,
            prim_dt=prim_dt,
            prim_steps=prim_steps,
            sim_speed=sim_speed,
            heading_bins=heading_bins,
            max_expansions=max_expansions,
            max_time_ms=max_time_ms,
            w_ref=w_ref,
        )
        ms = (time.time() - t0) * 1000.0

        if not path:
            return PlanResult(status=PlanStatus.FAIL, trajectory=None, debug={"ms": ms, "reason": "no_path"})

        traj = self._path_to_trajectory(path_xyz=path, dt=dt_out, horizon_s=horizon_s, v=target_speed)
        return PlanResult(
            status=PlanStatus.OK,
            trajectory=traj,
            debug={"ms": ms, "path_len": len(path), "goal": (goal.x, goal.y, goal.yaw)},
        )

    # ---------------------------
    # Drivable mask from CARLA map
    # ---------------------------
    def _apply_drivable_mask(self, grid: OccGrid) -> None:
        import carla  # local import

        driving = carla.LaneType.Driving
        # For simplicity: mark non-driving as occupied, driving as free.
        for i in range(grid.h):
            row = grid.occ[i]
            for j in range(grid.w):
                x, y = grid.ij_to_world(i, j)
                loc = carla.Location(x=float(x), y=float(y), z=0.0)
                wp = self._carla_map.get_waypoint(loc, project_to_road=False, lane_type=driving)
                row[j] = (wp is None)

    def _apply_drivable_mask_cached(self, grid: OccGrid, *, ego_x: float, ego_y: float) -> None:
        """
        Cache the drivable-area occupancy (non-drivable => occupied).
        Rebuild only every N ticks OR if ego moved more than a threshold.

        This avoids calling carla_map.get_waypoint for every cell on every tick.
        """
        cfg = self.config
        every_n = int(cfg.get("mask_update_every_n_ticks", 8))          # e.g. 5~10
        min_move = float(cfg.get("mask_update_min_move_m", 1.0))       # e.g. 0.5~2.0

        self._mask_tick += 1

        # cache validity: same grid shape/res and still roughly same center region
        same_grid = False
        if self._mask_grid_params is not None:
            w, h, res, min_x, min_y = self._mask_grid_params
            same_grid = (w == grid.w and h == grid.h and abs(res - grid.res_m) < 1e-9)

            # If grid origin changed a lot, cache becomes invalid (because occ is aligned to world via min_x/min_y)
            # Since our grid is ego-centered, origin changes with ego. We'll rely mainly on move threshold below.
            # But if someone changes grid_size/res at runtime, we force rebuild.
            if not same_grid:
                pass

        moved = True
        if self._mask_center_xy is not None:
            dx = float(ego_x) - self._mask_center_xy[0]
            dy = float(ego_y) - self._mask_center_xy[1]
            moved = (dx * dx + dy * dy) >= (min_move * min_move)

        due = (every_n > 0 and (self._mask_tick % every_n == 0))

        need_rebuild = (self._mask_occ is None) or (not same_grid) or moved or due

        if need_rebuild:
            # Build mask into current grid, then cache
            self._apply_drivable_mask(grid)

            # Deep-copy occ into cache (do NOT alias, because we'll later modify grid.occ by adding obstacles)
            self._mask_occ = [row[:] for row in grid.occ]
            self._mask_center_xy = (float(ego_x), float(ego_y))
            self._mask_grid_params = (grid.w, grid.h, grid.res_m, grid.min_x, grid.min_y)
            return

        # Reuse cached mask: copy cached occupancy into current grid
        # (Still deep copy row by row to avoid aliasing)
        assert self._mask_occ is not None
        for i in range(grid.h):
            grid.occ[i] = self._mask_occ[i][:]

    # ---------------------------
    # Goal selection on route
    # ---------------------------
    def _pick_goal(self, *, ego: EgoState, route_pts: List[Pose2D], lookahead_m: float) -> Pose2D:
        if not route_pts:
            return Pose2D(x=ego.pose.x, y=ego.pose.y, yaw=ego.pose.yaw)

        ex, ey = ego.pose.x, ego.pose.y
        # nearest search near last idx for stability
        win = int(self.config.get("nearest_search_window", 600))
        lo = max(0, self._last_nearest_idx - win)
        hi = min(len(route_pts) - 1, self._last_nearest_idx + win)

        best_i = self._last_nearest_idx
        best_d2 = 1e18
        for i in range(lo, hi + 1):
            p = route_pts[i]
            d2 = dist2(p.x, p.y, ex, ey)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        self._last_nearest_idx = best_i

        # move forward by arc length
        dist = 0.0
        j = best_i
        while j + 1 < len(route_pts) and dist < lookahead_m:
            dx = route_pts[j + 1].x - route_pts[j].x
            dy = route_pts[j + 1].y - route_pts[j].y
            dist += math.hypot(dx, dy)
            j += 1

        return route_pts[j]

    # ---------------------------
    # Soft distance-to-route (optional)
    # ---------------------------
    def _dist_to_route(self, *, x: float, y: float, route_pts: List[Pose2D]) -> float:
        if len(route_pts) < 2:
            return math.hypot(x - route_pts[0].x, y - route_pts[0].y) if route_pts else 0.0
        best = 1e18
        ax, ay = route_pts[0].x, route_pts[0].y
        for i in range(1, len(route_pts)):
            bx, by = route_pts[i].x, route_pts[i].y
            d2 = point_to_segment_dist2(x, y, ax, ay, bx, by)
            if d2 < best:
                best = d2
            ax, ay = bx, by
        return math.sqrt(best)

    # ---------------------------
    # Hybrid A*
    # ---------------------------
    def _search(
        self,
        *,
        ego: EgoState,
        goal: Pose2D,
        grid: OccGrid,
        route_pts: List[Pose2D],
        wheelbase_m: float,
        steer_max: float,
        steer_samples: int,
        prim_dt: float,
        prim_steps: int,
        sim_speed: float,
        heading_bins: int,
        max_expansions: int,
        max_time_ms: float,
        w_ref: float,
    ) -> Optional[List[Tuple[float, float, float]]]:
        def key_of(x: float, y: float, yaw: float) -> Tuple[int, int, int]:
            ij = grid.world_to_ij(x, y)
            if ij is None:
                return (-999999, -999999, 0)
            i, j = ij
            yaw = wrap_pi(yaw)
            k = int((yaw + math.pi) / (2.0 * math.pi) * heading_bins) % heading_bins
            return i, j, k

        def heuristic(x: float, y: float) -> float:
            return math.hypot(goal.x - x, goal.y - y)

        def goal_reached(x: float, y: float, yaw: float) -> bool:
            if (goal.x - x) ** 2 + (goal.y - y) ** 2 > (2.0 ** 2):
                return False
            if abs(wrap_pi(goal.yaw - yaw)) > deg2rad(20.0):
                return False
            return True

        def collision_free(seg: List[Tuple[float, float, float]]) -> bool:
            for sx, sy, _ in seg:
                if grid.is_occupied(sx, sy):
                    return False
            return True

        # steering set
        if steer_samples <= 1:
            steers = [0.0]
        else:
            steers = [(-1.0 + 2.0 * i / (steer_samples - 1)) * steer_max for i in range(steer_samples)]

        sx, sy, syaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        skey = key_of(sx, sy, syaw)

        start = Node(x=sx, y=sy, yaw=syaw, g=0.0, h=heuristic(sx, sy), parent=None, steer=0.0, seg=None)

        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (start.g + start.h, 0, skey))
        nodes: Dict[Tuple[int, int, int], Node] = {skey: start}
        closed: set[Tuple[int, int, int]] = set()

        best_key = skey
        best_f = start.g + start.h

        push_id = 1
        expansions = 0
        t0 = time.time()

        while open_heap and expansions < max_expansions:
            if (time.time() - t0) * 1000.0 > max_time_ms:
                break

            f, _pid, k = heapq.heappop(open_heap)
            if k in closed:
                continue
            closed.add(k)

            cur = nodes[k]
            expansions += 1

            if f < best_f:
                best_f = f
                best_key = k

            if goal_reached(cur.x, cur.y, cur.yaw):
                return self._reconstruct(nodes, k)

            for steer in steers:
                seg = bicycle_rollout(
                    x=cur.x,
                    y=cur.y,
                    yaw=cur.yaw,
                    v=sim_speed,
                    steer=steer,
                    wheelbase=wheelbase_m,
                    dt=prim_dt,
                    steps=prim_steps,
                )
                if not collision_free(seg):
                    continue

                nx, ny, nyaw = seg[-1]
                nk = key_of(nx, ny, nyaw)

                # simple cost: length + small steering penalty
                length = abs(sim_speed) * prim_dt * prim_steps
                ng = cur.g + length + 0.1 * abs(steer - cur.steer)

                if w_ref > 0.0 and route_pts:
                    d = self._dist_to_route(x=nx, y=ny, route_pts=route_pts)
                    ng += w_ref * (d * d)

                nh = heuristic(nx, ny)
                nf = ng + nh

                prev = nodes.get(nk)
                if prev is None or ng < prev.g:
                    nodes[nk] = Node(x=nx, y=ny, yaw=nyaw, g=ng, h=nh, parent=k, steer=steer, seg=seg)
                    heapq.heappush(open_heap, (nf, push_id, nk))
                    push_id += 1

        # best-effort path
        return self._reconstruct(nodes, best_key) if best_key in nodes else None

    def _reconstruct(self, nodes: Dict[Tuple[int, int, int], Node], last_key: Tuple[int, int, int]) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        k = last_key
        while True:
            n = nodes[k]
            if n.seg is not None:
                out.extend(reversed(n.seg))
            else:
                out.append((n.x, n.y, n.yaw))
            if n.parent is None:
                break
            k = n.parent
        out.reverse()
        # remove near-duplicates
        filtered: List[Tuple[float, float, float]] = []
        for p in out:
            if not filtered:
                filtered.append(p)
                continue
            dx = p[0] - filtered[-1][0]
            dy = p[1] - filtered[-1][1]
            if dx * dx + dy * dy > 1e-4:
                filtered.append(p)
        return filtered

    # ---------------------------
    # Path -> Trajectory (simple resampling)
    # ---------------------------
    def _path_to_trajectory(self, *, path_xyz: List[Tuple[float, float, float]], dt: float, horizon_s: float, v: float) -> Trajectory:
        if len(path_xyz) < 2:
            p = path_xyz[0] if path_xyz else (0.0, 0.0, 0.0)
            return Trajectory(points=[TrajectoryPoint(x=p[0], y=p[1], yaw=p[2], v=v)], dt=dt)

        xs = [p[0] for p in path_xyz]
        ys = [p[1] for p in path_xyz]
        yaws = [p[2] for p in path_xyz]

        s = [0.0]
        for i in range(1, len(path_xyz)):
            s.append(s[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
        total = s[-1]
        if total < 1e-3:
            p = path_xyz[-1]
            return Trajectory(points=[TrajectoryPoint(x=p[0], y=p[1], yaw=p[2], v=v)], dt=dt)

        n_out = max(2, int(math.ceil(horizon_s / dt)) + 1)
        ds = max(0.3, abs(v) * dt)  # minimum step
        points: List[TrajectoryPoint] = []

        sk = 0.0
        for _ in range(n_out):
            sk = min(sk, total)

            idx = 0
            while idx + 1 < len(s) and s[idx + 1] < sk:
                idx += 1
            if idx + 1 >= len(s):
                idx = len(s) - 2

            s0, s1 = s[idx], s[idx + 1]
            r = 0.0 if s1 <= s0 else (sk - s0) / (s1 - s0)

            x = xs[idx] + r * (xs[idx + 1] - xs[idx])
            y = ys[idx] + r * (ys[idx + 1] - ys[idx])

            yaw0, yaw1 = yaws[idx], yaws[idx + 1]
            dyaw = wrap_pi(yaw1 - yaw0)
            yaw = wrap_pi(yaw0 + r * dyaw)

            points.append(TrajectoryPoint(x=x, y=y, yaw=yaw, v=v))
            sk += ds

        return Trajectory(points=points, dt=dt)