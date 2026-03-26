from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from framework.planning.base_planning import BasePlanner
from framework.planning.mapping import OccupancyMapProvider, LocalOccPatch
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
from framework.planning.actor_model import ActorModelAdapter, ActorModel
from framework.planning.lane_selector import LaneSelector


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
    Lane-corridor Hybrid A* local planner.

    Core idea:
    - Run ONE Hybrid A* search per planning tick.
    - Search is guided by a single target lane / corridor, not by multiple lateral-offset candidates.
    - Static map is handled by occupancy patch.
    - Dynamic actors are handled separately:
        * actor.body         -> hard collision
        * rectangular margin -> soft cost

    Today, the target corridor is resolved from config (and optionally from map_info hints).
    Later, the global planner can provide the same lane/corridor hints without increasing
    search multiplicity, so complexity stays close to the current single-search version.
    """

    name: str = "hybrid_astar_lane_corridor"

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route
        self._map_info = map_info or {}
        self._carla_map = self._map_info.get("carla_map", None)
        self._carla_world = self._map_info.get("carla_world", None)
        self._last_nearest_idx = 0

        static_res_m = float(self.config.get("static_map_res_m", 0.5))
        static_margin_m = float(self.config.get("static_map_margin_m", 30.0))

        self._occ_provider = OccupancyMapProvider(static_res_m=static_res_m)
        self._occ_provider.build_static_from_carla_map(
            carla_map=self._carla_map,
            route_points=self._route.points,
            margin_m=static_margin_m,
            free_space_relax_m=1.0,
        )

        self._actor_model_adapter = ActorModelAdapter(
            config=self.config.get("actor_model", {})
        )

        self._last_selector = LaneSelector(
            config=self.config.get("lane_selector",{})
        )

        self._lane_selector = LaneSelector(
            config=self.config.get("lane_selector", {})
        )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        cfg = self.config

        if self._occ_provider is None:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={"reason": "occupancy_provider_not_initialized"},
            )

        dt_out = float(cfg.get("dt", 0.1))
        horizon_s = float(cfg.get("horizon_s", 5.0))
        default_target_speed = float(cfg.get("target_speed", 6.0))

        grid_size_m = float(cfg.get("grid_size_m", 60.0))
        inflation_m = float(cfg.get("inflation_m", 1.0))

        wheelbase_m = float(cfg.get("wheelbase_m", 2.8))
        steer_max = deg2rad(float(cfg.get("steer_max_deg", 30.0)))
        steer_samples = int(cfg.get("steer_samples", 7))
        prim_dt = float(cfg.get("primitive_dt", 0.12))
        prim_steps = int(cfg.get("primitive_steps", 8))
        sim_speed = float(cfg.get("sim_speed_mps", 6.0))

        heading_bins = int(cfg.get("heading_bins", 72))
        max_expansions = int(cfg.get("max_expansions", 6000))
        max_time_ms = float(cfg.get("max_time_ms", 60.0))

        local_size_x_m = float(cfg.get("local_patch_size_x_m", grid_size_m))
        local_size_y_m = float(cfg.get("local_patch_size_y_m", grid_size_m))
        actor_filter_radius_m = float(
            cfg.get("actor_filter_radius_m", max(local_size_x_m, local_size_y_m) * 0.75)
        )

        lookahead_m = float(cfg.get("lookahead_m", 40.0))
        target_speed = float(cfg.get("planner_target_speed", default_target_speed))

        route_pts = self._route.points
        if not route_pts:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={"reason": "empty_route"},
            )

        # target_offset_m, corridor_soft_half_width_m, corridor_hard_half_width_m = (
        #     self._resolve_target_corridor()
        # )

        selector_out = self._lane_selector.update(
            ego=ego,
            world=world,
            route_pts=route_pts,
            t=t,
        )

        target_offset_m = selector_out.target_lane_offset_m

        corridor_soft_half_width_m, corridor_hard_half_width_m = (
            self._resolve_target_corridor()
)

        goal = self._offset_goal_lateral(
            self._pick_goal(ego=ego, route_pts=route_pts, lookahead_m=lookahead_m),
            target_offset_m,
        )

        grid = self._build_local_patch(
            ego_x=ego.pose.x,
            ego_y=ego.pose.y,
            world=world,
            size_x_m=local_size_x_m,
            size_y_m=local_size_y_m,
            obstacle_inflation_m=inflation_m,
            actor_filter_radius_m=actor_filter_radius_m,
        )

        actor_models = self._actor_model_adapter.build_all(world)

        t0 = time.time()
        path, debug = self._search(
            ego=ego,
            goal=goal,
            grid=grid,
            route_pts=route_pts,
            actor_models=actor_models,
            target_offset_m=target_offset_m,
            corridor_soft_half_width_m=corridor_soft_half_width_m,
            corridor_hard_half_width_m=corridor_hard_half_width_m,
            wheelbase_m=wheelbase_m,
            steer_max=steer_max,
            steer_samples=steer_samples,
            prim_dt=prim_dt,
            prim_steps=prim_steps,
            sim_speed=sim_speed,
            heading_bins=heading_bins,
            max_expansions=max_expansions,
            max_time_ms=max_time_ms,
        )
        debug["ms"] = (time.time() - t0) * 1000.0
        debug["target_offset_m"] = target_offset_m
        debug["corridor_soft_half_width_m"] = corridor_soft_half_width_m
        debug["corridor_hard_half_width_m"] = corridor_hard_half_width_m
        debug["goal"] = (goal.x, goal.y, goal.yaw)

        debug["lane_selector_reason"] = selector_out.reason
        debug["lane_selector_switched"] = selector_out.switched
        debug["tracked_lead_id"] = selector_out.tracked_lead_id
        debug["keep_cost"] = selector_out.keep_cost
        debug["pass_cost"] = selector_out.pass_cost

        if path is None:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug=debug,
            )

        if bool(self.config.get("enable_path_smoothing", True)):
            smoothed_path = self._smooth_path(
                path_xyz=path,
                route_pts=route_pts,
                target_offset_m=target_offset_m,
                corridor_hard_half_width_m=corridor_hard_half_width_m,
                grid=grid,
                actor_models=actor_models,
            )
            debug["path_len_raw"] = len(path)
            debug["path_len_smoothed"] = len(smoothed_path)
            path = smoothed_path

        traj = self._path_to_trajectory(
            path_xyz=path,
            dt=dt_out,
            horizon_s=horizon_s,
            v=target_speed,
        )
        
        return PlanResult(
            status=PlanStatus.OK,
            trajectory=traj,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Corridor resolution
    # ------------------------------------------------------------------
    def _resolve_target_corridor(self) -> Tuple[float, float]:
        
        hints = self._map_info.get("planner_hints", {}) if isinstance(self._map_info, dict) else {}

        corridor_soft_half_width_m = float(
            hints.get(
                "target_corridor_soft_half_width_m",
                self.config.get("target_corridor_soft_half_width_m", 1.6),
            )
        )
        corridor_hard_half_width_m = float(
            hints.get(
                "target_corridor_hard_half_width_m",
                self.config.get("target_corridor_hard_half_width_m", 2.2),
            )
        )
        return corridor_soft_half_width_m, corridor_hard_half_width_m
    # ------------------------------------------------------------------
    # Patch build (prefer static-only patch if provider supports it)
    # ------------------------------------------------------------------
    def _build_local_patch(
        self,
        *,
        ego_x: float,
        ego_y: float,
        world: Any,
        size_x_m: float,
        size_y_m: float,
        obstacle_inflation_m: float,
        actor_filter_radius_m: Optional[float],
    ) -> LocalOccPatch:
        kwargs = dict(
            ego_x=ego_x,
            ego_y=ego_y,
            world=world,
            size_x_m=size_x_m,
            size_y_m=size_y_m,
            obstacle_inflation_m=obstacle_inflation_m,
            actor_filter_radius_m=actor_filter_radius_m,
        )
        try:
            return self._occ_provider.get_local_patch(
                **kwargs,
                include_dynamic_obstacles=False,
            )
        except TypeError:
            return self._occ_provider.get_local_patch(**kwargs)

    # ------------------------------------------------------------------
    # Goal and route geometry
    # ------------------------------------------------------------------
    def _pick_goal(self, *, ego: EgoState, route_pts: List[Pose2D], lookahead_m: float) -> Pose2D:
        if not route_pts:
            return Pose2D(x=ego.pose.x, y=ego.pose.y, yaw=ego.pose.yaw)

        ex, ey = ego.pose.x, ego.pose.y
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

        dist = 0.0
        j = best_i
        while j + 1 < len(route_pts) and dist < lookahead_m:
            dx = route_pts[j + 1].x - route_pts[j].x
            dy = route_pts[j + 1].y - route_pts[j].y
            dist += math.hypot(dx, dy)
            j += 1

        return route_pts[j]

    def _offset_goal_lateral(self, base_goal: Pose2D, offset_m: float) -> Pose2D:
        nx = -math.sin(base_goal.yaw)
        ny = math.cos(base_goal.yaw)
        return Pose2D(
            x=base_goal.x + offset_m * nx,
            y=base_goal.y + offset_m * ny,
            yaw=base_goal.yaw,
        )

    def _project_to_route_sl(self, *, x: float, y: float, route_pts: List[Pose2D]) -> Tuple[float, float]:
        if len(route_pts) < 2:
            return 0.0, 0.0

        best_s = 0.0
        best_l = 0.0
        best_d2 = 1e18
        accum_s = 0.0

        for i in range(len(route_pts) - 1):
            ax, ay = route_pts[i].x, route_pts[i].y
            bx, by = route_pts[i + 1].x, route_pts[i + 1].y

            abx = bx - ax
            aby = by - ay
            ab2 = abx * abx + aby * aby
            seg_len = math.hypot(abx, aby)
            if ab2 <= 1e-9 or seg_len <= 1e-9:
                accum_s += seg_len
                continue

            apx = x - ax
            apy = y - ay
            t = clamp((apx * abx + apy * aby) / ab2, 0.0, 1.0)

            px = ax + t * abx
            py = ay + t * aby

            dx = x - px
            dy = y - py
            d2 = dx * dx + dy * dy

            if d2 < best_d2:
                yaw = math.atan2(aby, abx)
                left_nx = -math.sin(yaw)
                left_ny = math.cos(yaw)
                l_signed = (x - px) * left_nx + (y - py) * left_ny

                best_d2 = d2
                best_s = accum_s + t * seg_len
                best_l = l_signed

            accum_s += seg_len

        return best_s, best_l

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _search(
        self,
        *,
        ego: EgoState,
        goal: Pose2D,
        grid: LocalOccPatch,
        route_pts: List[Pose2D],
        actor_models: List[ActorModel],
        target_offset_m: float,
        corridor_soft_half_width_m: float,
        corridor_hard_half_width_m: float,
        wheelbase_m: float,
        steer_max: float,
        steer_samples: int,
        prim_dt: float,
        prim_steps: int,
        sim_speed: float,
        heading_bins: int,
        max_expansions: int,
        max_time_ms: float,
    ) -> Tuple[Optional[List[Tuple[float, float, float]]], Dict[str, Any]]:
        goal_tol_xy_m = float(self.config.get("goal_tol_xy_m", 2.0))
        goal_tol_yaw_rad = deg2rad(float(self.config.get("goal_tol_yaw_deg", 20.0)))

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
            if (goal.x - x) ** 2 + (goal.y - y) ** 2 > (goal_tol_xy_m ** 2):
                return False
            if abs(wrap_pi(goal.yaw - yaw)) > goal_tol_yaw_rad:
                return False
            return True

        def eval_segment(seg) -> Tuple[bool, float, float]:
            total_corridor_cost = 0.0
            total_safety_cost = 0.0

            w_target_center = float(self.config.get("w_target_center", 3.0))
            w_target_corridor = float(self.config.get("w_target_corridor", 2.0))
            corridor_soft_tol_m = float(self.config.get("target_center_soft_tol_m", 0.35))
            forbid_outside_hard_corridor = bool(
                self.config.get("forbid_outside_hard_corridor", False)
            )

            for sx, sy, syaw in seg:
                # static + dynamic hard collision and actor safety cost
                hard_collision, pose_safety_cost = self._ego_two_circles_eval(
                    x=sx,
                    y=sy,
                    yaw=syaw,
                    grid=grid,
                    actor_models=actor_models,
                )
                if hard_collision:
                    return False, float("inf"), float("inf")

                total_safety_cost += pose_safety_cost

                # target lane / corridor cost
                _, l = self._project_to_route_sl(x=sx, y=sy, route_pts=route_pts)
                dl = l - target_offset_m

                if abs(dl) > corridor_hard_half_width_m and forbid_outside_hard_corridor:
                    return False, float("inf"), float("inf")

                target_center_cost = dl * dl
                excess_soft = max(0.0, abs(dl) - corridor_soft_tol_m)
                target_corridor_cost = excess_soft * excess_soft
                excess_hard = max(0.0, abs(dl) - corridor_soft_half_width_m)
                hard_corridor_cost = excess_hard * excess_hard

                total_corridor_cost += (
                    w_target_center * target_center_cost
                    + w_target_corridor * target_corridor_cost
                    + 4.0 * w_target_corridor * hard_corridor_cost
                )

            return True, total_corridor_cost, total_safety_cost

        if steer_samples <= 1:
            steers = [0.0]
        else:
            steers = [(-1.0 + 2.0 * i / (steer_samples - 1)) * steer_max for i in range(steer_samples)]

        sx, sy, syaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        hard_collision, _ = self._ego_two_circles_eval(
            x=sx,
            y=sy,
            yaw=syaw,
            grid=grid,
            actor_models=actor_models,
        )
        if hard_collision:
            return None, {"reason": "start_in_collision"}

        skey = key_of(sx, sy, syaw)
        start = Node(
            x=sx,
            y=sy,
            yaw=syaw,
            g=0.0,
            h=heuristic(sx, sy),
            parent=None,
            steer=0.0,
            seg=None,
        )

        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (start.g + start.h, 0, skey))

        nodes: Dict[Tuple[int, int, int], Node] = {skey: start}
        closed: set[Tuple[int, int, int]] = set()

        best_key = skey
        best_h = start.h

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

            if cur.h < best_h:
                best_h = cur.h
                best_key = k

            if goal_reached(cur.x, cur.y, cur.yaw):
                return self._reconstruct(nodes, k), {
                    "reason": "goal_reached",
                    "expansions": expansions,
                    "best_cost": cur.g + cur.h,
                }

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

                ok, corridor_cost, safety_cost = eval_segment(seg)
                if not ok:
                    continue

                nx, ny, nyaw = seg[-1]
                nk = key_of(nx, ny, nyaw)

                length_cost = abs(sim_speed) * prim_dt * prim_steps
                steer_smooth_cost = 0.1 * abs(steer - cur.steer)
                ng = cur.g + length_cost + steer_smooth_cost

                w_corridor = float(self.config.get("w_corridor", 2.5))
                w_safety = float(self.config.get("w_safety", 1.5))
                ng += w_corridor * (corridor_cost / max(1, len(seg)))
                ng += w_safety * (safety_cost / max(1, len(seg)))

                nh = heuristic(nx, ny)
                nf = ng + nh

                prev = nodes.get(nk)
                if prev is None or ng < prev.g:
                    nodes[nk] = Node(
                        x=nx,
                        y=ny,
                        yaw=nyaw,
                        g=ng,
                        h=nh,
                        parent=k,
                        steer=steer,
                        seg=seg,
                    )
                    heapq.heappush(open_heap, (nf, push_id, nk))
                    push_id += 1

        if best_key in nodes:
            best_node = nodes[best_key]
            return self._reconstruct(nodes, best_key), {
                "reason": "best_effort",
                "expansions": expansions,
                "best_cost": best_node.g + best_node.h,
            }

        return None, {"reason": "search_failed", "expansions": expansions}

    def _reconstruct(
        self,
        nodes: Dict[Tuple[int, int, int], Node],
        last_key: Tuple[int, int, int],
    ) -> List[Tuple[float, float, float]]:
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

    
    def _smooth_path(
        self,
        *,
        path_xyz: List[Tuple[float, float, float]],
        route_pts: List[Pose2D],
        target_offset_m: float,
        corridor_hard_half_width_m: float,
        grid: LocalOccPatch,
        actor_models: List[ActorModel],
    ) -> List[Tuple[float, float, float]]:
        if len(path_xyz) < 5:
            return path_xyz

        iterations = int(self.config.get("path_smoothing_iterations", 15))
        alpha = float(self.config.get("path_smoothing_alpha", 0.30))
        beta = float(self.config.get("path_smoothing_beta", 0.20))
        max_dev_m = float(self.config.get("path_smoothing_max_dev_m", 0.80))

        # work on xy only, yaw recomputed later
        orig_xy = [(p[0], p[1]) for p in path_xyz]
        smooth_xy = [(p[0], p[1]) for p in path_xyz]

        n = len(smooth_xy)

        for _ in range(iterations):
            new_xy = smooth_xy[:]

            # keep endpoints fixed
            for i in range(1, n - 1):
                x, y = smooth_xy[i]
                x_prev, y_prev = smooth_xy[i - 1]
                x_next, y_next = smooth_xy[i + 1]
                x0, y0 = orig_xy[i]

                # Laplacian smooth + pull back to original
                x_new = x + alpha * (0.5 * (x_prev + x_next) - x) + beta * (x0 - x)
                y_new = y + alpha * (0.5 * (y_prev + y_next) - y) + beta * (y0 - y)

                # limit deviation from original point
                dx0 = x_new - x0
                dy0 = y_new - y0
                d0 = math.hypot(dx0, dy0)
                if d0 > max_dev_m and d0 > 1e-6:
                    scale = max_dev_m / d0
                    x_new = x0 + dx0 * scale
                    y_new = y0 + dy0 * scale

                # keep point near target corridor
                _, l = self._project_to_route_sl(x=x_new, y=y_new, route_pts=route_pts)
                dl = l - target_offset_m
                if abs(dl) > corridor_hard_half_width_m:
                    # reject this local smoothing move if outside hard corridor
                    x_new, y_new = x, y

                new_xy[i] = (x_new, y_new)

            smooth_xy = new_xy

        smoothed_path = self._recompute_yaw_from_xy(smooth_xy)

        # final validity check; fallback to raw path if smoothed path becomes invalid
        if not self._path_is_valid(
            path_xyz=smoothed_path,
            route_pts=route_pts,
            target_offset_m=target_offset_m,
            corridor_hard_half_width_m=corridor_hard_half_width_m,
            grid=grid,
            actor_models=actor_models,
        ):
            return path_xyz

        return smoothed_path


    def _recompute_yaw_from_xy(
        self,
        xy: List[Tuple[float, float]],
    ) -> List[Tuple[float, float, float]]:
        n = len(xy)
        if n == 0:
            return []

        out: List[Tuple[float, float, float]] = []
        for i in range(n):
            x, y = xy[i]
            if i == 0:
                x2, y2 = xy[min(1, n - 1)]
                yaw = math.atan2(y2 - y, x2 - x)
            elif i == n - 1:
                x1, y1 = xy[n - 2]
                yaw = math.atan2(y - y1, x - x1)
            else:
                x1, y1 = xy[i - 1]
                x2, y2 = xy[i + 1]
                yaw = math.atan2(y2 - y1, x2 - x1)

            out.append((x, y, wrap_pi(yaw)))
        return out


    def _path_is_valid(
        self,
        *,
        path_xyz: List[Tuple[float, float, float]],
        route_pts: List[Pose2D],
        target_offset_m: float,
        corridor_hard_half_width_m: float,
        grid: LocalOccPatch,
        actor_models: List[ActorModel],
    ) -> bool:
        # sparse check is enough for post-smoothing
        stride = max(1, int(len(path_xyz) / 30))

        for i in range(0, len(path_xyz), stride):
            x, y, yaw = path_xyz[i]

            hard_collision, _ = self._ego_two_circles_eval(
                x=x,
                y=y,
                yaw=yaw,
                grid=grid,
                actor_models=actor_models,
            )
            if hard_collision:
                return False

            _, l = self._project_to_route_sl(x=x, y=y, route_pts=route_pts)
            if abs(l - target_offset_m) > corridor_hard_half_width_m:
                return False

        # ensure last point also checked
        if path_xyz:
            x, y, yaw = path_xyz[-1]
            hard_collision, _ = self._ego_two_circles_eval(
                x=x,
                y=y,
                yaw=yaw,
                grid=grid,
                actor_models=actor_models,
            )
            if hard_collision:
                return False

            _, l = self._project_to_route_sl(x=x, y=y, route_pts=route_pts)
            if abs(l - target_offset_m) > corridor_hard_half_width_m:
                return False

        return True
    
    # ------------------------------------------------------------------
    # Path -> Trajectory
    # ------------------------------------------------------------------
    def _path_to_trajectory(
        self,
        *,
        path_xyz: List[Tuple[float, float, float]],
        dt: float,
        horizon_s: float,
        v: float,
    ) -> Trajectory:
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
        ds = max(0.3, abs(v) * dt)
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

    # ------------------------------------------------------------------
    # Ego geometry: two-circle model
    # ------------------------------------------------------------------
    def _get_two_circle_params(self) -> Tuple[float, float, float]:
        wheelbase_m = float(self.config.get("ego_wheelbase_m", 2.875))
        width_m = float(self.config.get("ego_width_m", 1.85))
        front_overhang_m = float(self.config.get("ego_front_overhang_m", 0.868))
        rear_overhang_m = float(self.config.get("ego_rear_overhang_m", 0.977))
        safety_margin_m = float(self.config.get("ego_safety_margin_m", 0.05))

        radius_scale = float(self.config.get("ego_circle_radius_scale", 0.85))
        center_scale = float(self.config.get("ego_circle_center_scale", 1.0))

        x_min = -rear_overhang_m
        x_max = wheelbase_m + front_overhang_m
        length = x_max - x_min
        half_w = 0.5 * width_m

        rear_circle_x = x_min + 0.15 * length * center_scale
        front_circle_x = x_min + 0.7 * length * center_scale

        quarter_len = 0.25 * length
        base_radius = math.hypot(quarter_len, half_w)
        radius = base_radius * radius_scale + safety_margin_m

        return front_circle_x, rear_circle_x, radius

    def _disc_in_collision(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        grid: LocalOccPatch,
    ) -> bool:
        radial_step = float(self.config.get("circle_sample_radial_step_m", 0.35))
        radial_step = max(0.15, radial_step)

        if grid.is_occupied(cx, cy):
            return True

        nr = max(1, int(math.ceil(r / radial_step)))
        for ir in range(1, nr + 1):
            rr = r * ir / nr
            ntheta = max(8, int(math.ceil(2.0 * math.pi * rr / radial_step)))
            for k in range(ntheta):
                th = 2.0 * math.pi * k / ntheta
                px = cx + rr * math.cos(th)
                py = cy + rr * math.sin(th)
                if grid.is_occupied(px, py):
                    return True
        return False

    def _ego_two_circles_eval(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        grid: LocalOccPatch,
        actor_models: List[ActorModel],
    ) -> Tuple[bool, float]:
        front_x, rear_x, radius = self._get_two_circle_params()

        c = math.cos(yaw)
        s = math.sin(yaw)

        fx = x + c * front_x
        fy = y + s * front_x
        rx = x + c * rear_x
        ry = y + s * rear_x

        # static hard collision
        if self._disc_in_collision(cx=fx, cy=fy, r=radius, grid=grid):
            return True, float("inf")
        if self._disc_in_collision(cx=rx, cy=ry, r=radius, grid=grid):
            return True, float("inf")

        front_hard, front_cost = self._circle_vs_actors_eval(
            cx=fx, cy=fy, r=radius, actor_models=actor_models
        )
        if front_hard:
            return True, float("inf")

        rear_hard, rear_cost = self._circle_vs_actors_eval(
            cx=rx, cy=ry, r=radius, actor_models=actor_models
        )
        if rear_hard:
            return True, float("inf")

        return False, max(front_cost, rear_cost)

    # ------------------------------------------------------------------
    # Dynamic actor: body hard collision + rectangular safety box soft cost
    # ------------------------------------------------------------------
    def _circle_vs_actors_eval(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        actor_models: List[ActorModel],
    ) -> Tuple[bool, float]:
        total_cost = 0.0

        safety_actor_eval_radius_m = float(
            self.config.get("safety_actor_eval_radius_m", 20.0)
        )
        coarse_margin_m = float(
            self.config.get("actor_coarse_filter_margin_m", 3.0)
        )

        eval_r = max(0.0, safety_actor_eval_radius_m) + max(0.0, coarse_margin_m) + r
        eval_r2 = eval_r * eval_r

        for actor in actor_models:
            dx = actor.x - cx
            dy = actor.y - cy
            if dx * dx + dy * dy > eval_r2:
                continue

            hard_collision, cost = self._circle_vs_single_actor_eval(
                cx=cx, cy=cy, r=r, actor=actor
            )
            if hard_collision:
                return True, float("inf")
            total_cost += cost

        return False, total_cost

    def _circle_vs_single_actor_eval(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        actor: ActorModel,
    ) -> Tuple[bool, float]:
        # 1) hard collision against actor body
        d_body = self._point_to_oriented_box_distance(
            px=cx,
            py=cy,
            box_x=actor.body.x,
            box_y=actor.body.y,
            box_yaw=actor.body.yaw,
            box_length=actor.body.length_m,
            box_width=actor.body.width_m,
        ) - r
        if d_body <= 0.0:
            return True, float("inf")

        # 2) cheap coarse filter against expanded rectangular safety shell
        coarse_margin_m = float(self.config.get("actor_coarse_filter_margin_m", 3.0))
        need_eval = self._rect_safety_coarse_filter(
            cx=cx,
            cy=cy,
            r=r,
            actor=actor,
            coarse_margin_m=coarse_margin_m,
        )
        if not need_eval:
            return False, 0.0

        # 3) rectangular safety cost (soft)
        cost = self._rect_safety_cost(
            cx=cx,
            cy=cy,
            r=r,
            actor=actor,
        )
        return False, cost

    def _rect_safety_coarse_filter(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        actor: ActorModel,
        coarse_margin_m: float,
    ) -> bool:
        dx = cx - actor.body.x
        dy = cy - actor.body.y

        c = math.cos(actor.body.yaw)
        s = math.sin(actor.body.yaw)

        lx = dx * c + dy * s
        ly = -dx * s + dy * c

        half_l = 0.5 * actor.body.length_m
        half_w = 0.5 * actor.body.width_m

        front_extent = half_l + actor.front_safe_m + r + coarse_margin_m
        rear_extent = half_l + actor.rear_safe_m + r + coarse_margin_m
        lat_extent = half_w + actor.lateral_safe_m + r + coarse_margin_m

        x_ok = (lx <= front_extent) and (lx >= -rear_extent)
        y_ok = abs(ly) <= lat_extent
        return x_ok and y_ok

    def _rect_safety_cost(
        self,
        *,
        cx: float,
        cy: float,
        r: float,
        actor: ActorModel,
    ) -> float:
        dx = cx - actor.body.x
        dy = cy - actor.body.y

        c = math.cos(actor.body.yaw)
        s = math.sin(actor.body.yaw)

        lx = dx * c + dy * s
        ly = -dx * s + dy * c

        half_l = 0.5 * actor.body.length_m
        half_w = 0.5 * actor.body.width_m

        # normalized rectangular shell coordinates (Chebyshev-like, O(1), no sqrt)
        x_scale = (half_l + actor.front_safe_m + r) if lx >= 0.0 else (half_l + actor.rear_safe_m + r)
        y_scale = half_w + actor.lateral_safe_m + r

        x_scale = max(1e-3, x_scale)
        y_scale = max(1e-3, y_scale)

        rho = max(abs(lx) / x_scale, abs(ly) / y_scale)
        if rho >= 1.0:
            return 0.0

        safety_weight = float(self.config.get("rect_safety_weight", 3.0))
        safety_power = float(self.config.get("rect_safety_power", 3.0))
        ratio = 1.0 - rho
        return safety_weight * (ratio ** safety_power)

    def _point_to_oriented_box_distance(
        self,
        *,
        px: float,
        py: float,
        box_x: float,
        box_y: float,
        box_yaw: float,
        box_length: float,
        box_width: float,
    ) -> float:
        dx = px - box_x
        dy = py - box_y

        c = math.cos(box_yaw)
        s = math.sin(box_yaw)

        lx = dx * c + dy * s
        ly = -dx * s + dy * c

        half_l = 0.5 * box_length
        half_w = 0.5 * box_width

        qx = abs(lx) - half_l
        qy = abs(ly) - half_w

        ox = max(qx, 0.0)
        oy = max(qy, 0.0)
        outside_dist = math.hypot(ox, oy)

        if qx <= 0.0 and qy <= 0.0:
            return max(qx, qy)
        return outside_dist
