# framework/planning/local/hybrid_astar_behavior.py
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
from framework.planning.overtake_sm import OvertakeStateMachine


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


class HybridAStarBehaviorPlanner(BasePlanner):
    """
    Hybrid A* planner with behavior-aware search constraints.

    Behavior states:
      - FOLLOW
      - LANE_CHANGE_OUT
      - CRUISE_PASS_LANE
      - LANE_CHANGE_BACK
    """
    name: str = "hybrid_astar_behavior"

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route
        self._carla_map = (map_info or {}).get("carla_map", None)
        self._carla_world = (map_info or {}).get("carla_world", None)
        self._last_nearest_idx = 0
        self._last_print_t = 0.0

        static_res_m = float(self.config.get("static_map_res_m", 0.5))
        static_margin_m = float(self.config.get("static_map_margin_m", 30.0))

        self._occ_provider = OccupancyMapProvider(static_res_m=static_res_m)
        self._occ_provider.build_static_from_carla_map(
            carla_map=self._carla_map,
            route_points=self._route.points,
            margin_m=static_margin_m,
            free_space_relax_m=1.0,
        )

        self._behavior_sm = OvertakeStateMachine(config=self.config)

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
        inflation_m = float(cfg.get("inflation_m", 0.3))

        wheelbase_m = float(cfg.get("wheelbase_m", 2.875))
        steer_max = deg2rad(float(cfg.get("steer_max_deg", 30.0)))
        steer_samples = int(cfg.get("steer_samples", 5))
        prim_dt = float(cfg.get("primitive_dt", 0.1))
        prim_steps = int(cfg.get("primitive_steps", 6))
        sim_speed = float(cfg.get("sim_speed_mps", 7.0))

        heading_bins = int(cfg.get("heading_bins", 72))
        max_expansions = int(cfg.get("max_expansions", 12000))
        max_time_ms = float(cfg.get("max_time_ms", 120.0))

        local_size_x_m = float(cfg.get("local_patch_size_x_m", grid_size_m))
        local_size_y_m = float(cfg.get("local_patch_size_y_m", grid_size_m))
        actor_filter_radius_m = float(
            cfg.get("actor_filter_radius_m", max(local_size_x_m, local_size_y_m) * 0.75)
        )

        # state-aware constraints / costs
        follow_corridor_half_width_m = float(cfg.get("follow_corridor_half_width_m", 1.6))
        lane_change_out_corridor_half_width_m = float(cfg.get("lane_change_out_corridor_half_width_m", 5.0))
        cruise_pass_lane_corridor_half_width_m = float(cfg.get("cruise_pass_lane_corridor_half_width_m", 2.0))
        lane_change_back_corridor_half_width_m = float(cfg.get("lane_change_back_corridor_half_width_m", 5.0))

        w_follow_center = float(cfg.get("w_follow_center", 2.0))
        w_lane_change_out_target = float(cfg.get("w_lane_change_out_target", 4.0))
        w_cruise_pass_lane_target = float(cfg.get("w_cruise_pass_lane_target", 1.5))
        w_lane_change_back_target = float(cfg.get("w_lane_change_back_target", 4.0))

        w_lane_change_out_progress = float(cfg.get("w_lane_change_out_progress", 0.2))
        w_cruise_pass_lane_progress = float(cfg.get("w_cruise_pass_lane_progress", 1.0))
        w_lane_change_back_progress = float(cfg.get("w_lane_change_back_progress", 0.2))

        lane_change_monotonic_tol_m = float(cfg.get("lane_change_monotonic_tol_m", 0.15))

        route_pts = self._route.points

        behavior = self._behavior_sm.update(
            ego=ego,
            world=world,
            route_pts=route_pts,
            t=t,
            dist_to_route_fn=self._dist_to_route,
        )

        lookahead_m = behavior.lookahead_m
        w_ref = behavior.w_ref
        target_speed = (
            float(behavior.target_speed)
            if behavior.target_speed is not None
            else default_target_speed
        )

        target_l = float(behavior.goal_lateral_offset_m)

        if behavior.state in ("LANE_CHANGE_OUT", "CRUISE_PASS_LANE"):
            goal = self._pick_pass_goal(
                ego=ego,
                route_pts=route_pts,
                lookahead_m=lookahead_m,
                lateral_offset_m=target_l,
            )
        elif behavior.state == "LANE_CHANGE_BACK":
            goal = self._pick_return_goal(
                ego=ego,
                route_pts=route_pts,
                lookahead_m=lookahead_m,
            )
        else:
            goal = self._pick_goal(
                ego=ego,
                route_pts=route_pts,
                lookahead_m=lookahead_m,
            )

        grid = self._occ_provider.get_local_patch(
            ego_x=ego.pose.x,
            ego_y=ego.pose.y,
            world=world,
            size_x_m=local_size_x_m,
            size_y_m=local_size_y_m,
            obstacle_inflation_m=inflation_m,
            actor_filter_radius_m=actor_filter_radius_m,
        )

        if bool(cfg.get("debug_print_behavior", True)) and t - self._last_print_t > 0.2:
            self._last_print_t = t
            print(
                f"[Behavior] {t:.2f}s | state={behavior.state} | reason={behavior.reason} | "
                f"w_ref={w_ref:.2f} | v={target_speed:.2f} | "
                f"target_l={target_l:.2f} | "
                f"lead_long={None if behavior.lead is None else round(behavior.lead.longitudinal, 2)} | "
                f"lead_lat={None if behavior.lead is None else round(behavior.lead.lateral, 2)} | "
                f"ego_l={getattr(self._behavior_sm, 'debug_ego_l', None)}"
            )

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
            behavior_state=behavior.state,
            target_l=target_l,
            follow_corridor_half_width_m=follow_corridor_half_width_m,
            lane_change_out_corridor_half_width_m=lane_change_out_corridor_half_width_m,
            cruise_pass_lane_corridor_half_width_m=cruise_pass_lane_corridor_half_width_m,
            lane_change_back_corridor_half_width_m=lane_change_back_corridor_half_width_m,
            w_follow_center=w_follow_center,
            w_lane_change_out_target=w_lane_change_out_target,
            w_cruise_pass_lane_target=w_cruise_pass_lane_target,
            w_lane_change_back_target=w_lane_change_back_target,
            w_lane_change_out_progress=w_lane_change_out_progress,
            w_cruise_pass_lane_progress=w_cruise_pass_lane_progress,
            w_lane_change_back_progress=w_lane_change_back_progress,
            lane_change_monotonic_tol_m=lane_change_monotonic_tol_m,
        )
        ms = (time.time() - t0) * 1000.0

        if not path:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={
                    "ms": ms,
                    "reason": "no_path",
                    "behavior_state": behavior.state,
                    "behavior_reason": behavior.reason,
                    "goal": (goal.x, goal.y, goal.yaw),
                },
            )

        traj = self._path_to_trajectory(
            path_xyz=path,
            dt=dt_out,
            horizon_s=horizon_s,
            v=target_speed,
        )

        return PlanResult(
            status=PlanStatus.OK,
            trajectory=traj,
            debug={
                "ms": ms,
                "path_len": len(path),
                "goal": (goal.x, goal.y, goal.yaw),
                "behavior_state": behavior.state,
                "behavior_reason": behavior.reason,
                "goal_lateral_offset_m": behavior.goal_lateral_offset_m,
                "lead_longitudinal": None if behavior.lead is None else behavior.lead.longitudinal,
                "lead_lateral": None if behavior.lead is None else behavior.lead.lateral,
                "ego_rear_s": getattr(self._behavior_sm, "debug_ego_rear_s", None),
                "lead_front_s": getattr(self._behavior_sm, "debug_lead_front_s", None),
                "ego_l": getattr(self._behavior_sm, "debug_ego_l", None),
                "target_l": getattr(self._behavior_sm, "debug_target_l", None),
            },
        )

    # ------------------------------------------------------------------
    # Goal selection
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

    def _pick_pass_goal(
        self,
        *,
        ego: EgoState,
        route_pts: List[Pose2D],
        lookahead_m: float,
        lateral_offset_m: float,
    ) -> Pose2D:
        base_goal = self._pick_goal(
            ego=ego,
            route_pts=route_pts,
            lookahead_m=lookahead_m,
        )
        return self._offset_goal_lateral(base_goal, lateral_offset_m)

    def _pick_return_goal(
        self,
        *,
        ego: EgoState,
        route_pts: List[Pose2D],
        lookahead_m: float,
    ) -> Pose2D:
        short_lookahead = min(lookahead_m, float(self.config.get("return_goal_short_lookahead_m", 6.0)))
        return self._pick_goal(ego=ego, route_pts=route_pts, lookahead_m=short_lookahead)

    def _offset_goal_lateral(self, base_goal: Pose2D, offset_m: float) -> Pose2D:
        nx = -math.sin(base_goal.yaw)
        ny = math.cos(base_goal.yaw)
        return Pose2D(
            x=base_goal.x + offset_m * nx,
            y=base_goal.y + offset_m * ny,
            yaw=base_goal.yaw,
        )

    # ------------------------------------------------------------------
    # Route geometry helpers
    # ------------------------------------------------------------------
    def _nearest_route_frame(
        self,
        *,
        x: float,
        y: float,
        route_pts: List[Pose2D],
    ) -> Tuple[float, float, float]:
        if len(route_pts) < 2:
            if route_pts:
                dx = x - route_pts[0].x
                dy = y - route_pts[0].y
                return 0.0, math.hypot(dx, dy), route_pts[0].yaw
            return 0.0, 0.0, 0.0

        best_s = 0.0
        best_l = 0.0
        best_yaw = route_pts[0].yaw
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
                best_yaw = yaw

            accum_s += seg_len

        return best_s, best_l, best_yaw

    def _dist_to_route(self, *, x: float, y: float, route_pts: List[Pose2D]) -> float:
        _, l, _ = self._nearest_route_frame(x=x, y=y, route_pts=route_pts)
        return abs(l)

    # ------------------------------------------------------------------
    # Search corridor by behavior
    # ------------------------------------------------------------------
    def _seg_allowed_by_state(
        self,
        *,
        seg: List[Tuple[float, float, float]],
        route_pts: List[Pose2D],
        behavior_state: str,
        target_l: float,
        follow_corridor_half_width_m: float,
        lane_change_out_corridor_half_width_m: float,
        cruise_pass_lane_corridor_half_width_m: float,
        lane_change_back_corridor_half_width_m: float,
    ) -> bool:
        for px, py, _ in seg:
            _, l, _ = self._nearest_route_frame(x=px, y=py, route_pts=route_pts)

            if behavior_state == "FOLLOW":
                if abs(l) > follow_corridor_half_width_m:
                    return False

            elif behavior_state == "LANE_CHANGE_OUT":
                lo = min(0.0, target_l) - 0.8
                hi = max(0.0, target_l) + 0.8
                if l < lo - lane_change_out_corridor_half_width_m or l > hi + lane_change_out_corridor_half_width_m:
                    return False

            elif behavior_state == "CRUISE_PASS_LANE":
                if abs(l - target_l) > cruise_pass_lane_corridor_half_width_m:
                    return False

            elif behavior_state == "LANE_CHANGE_BACK":
                lo = min(0.0, target_l) - 0.8
                hi = max(0.0, target_l) + 0.8
                if l < lo - lane_change_back_corridor_half_width_m or l > hi + lane_change_back_corridor_half_width_m:
                    return False

        return True

    # ------------------------------------------------------------------
    # Hybrid A*
    # ------------------------------------------------------------------
    def _search(
        self,
        *,
        ego: EgoState,
        goal: Pose2D,
        grid: LocalOccPatch,
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
        behavior_state: str,
        target_l: float,
        follow_corridor_half_width_m: float,
        lane_change_out_corridor_half_width_m: float,
        cruise_pass_lane_corridor_half_width_m: float,
        lane_change_back_corridor_half_width_m: float,
        w_follow_center: float,
        w_lane_change_out_target: float,
        w_cruise_pass_lane_target: float,
        w_lane_change_back_target: float,
        w_lane_change_out_progress: float,
        w_cruise_pass_lane_progress: float,
        w_lane_change_back_progress: float,
        lane_change_monotonic_tol_m: float,
    ) -> Optional[List[Tuple[float, float, float]]]:
        goal_tol_xy_m = float(self.config.get("goal_tol_xy_m", 2.5))
        goal_tol_yaw_rad = deg2rad(float(self.config.get("goal_tol_yaw_deg", 25.0)))

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

        def collision_free(seg):
            for sx, sy, syaw in seg:
                if self._ego_two_circles_in_collision(x=sx, y=sy, yaw=syaw, grid=grid):
                    return False
            return True

        if steer_samples <= 1:
            steers = [0.0]
        else:
            steers = [(-1.0 + 2.0 * i / (steer_samples - 1)) * steer_max for i in range(steer_samples)]

        sx, sy, syaw = ego.pose.x, ego.pose.y, ego.pose.yaw

        if self._ego_two_circles_in_collision(x=sx, y=sy, yaw=syaw, grid=grid):
            return None

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

            _f, _pid, k = heapq.heappop(open_heap)
            if k in closed:
                continue
            closed.add(k)

            cur = nodes[k]
            expansions += 1

            if cur.h < best_h:
                best_h = cur.h
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

                if not self._seg_allowed_by_state(
                    seg=seg,
                    route_pts=route_pts,
                    behavior_state=behavior_state,
                    target_l=target_l,
                    follow_corridor_half_width_m=follow_corridor_half_width_m,
                    lane_change_out_corridor_half_width_m=lane_change_out_corridor_half_width_m,
                    cruise_pass_lane_corridor_half_width_m=cruise_pass_lane_corridor_half_width_m,
                    lane_change_back_corridor_half_width_m=lane_change_back_corridor_half_width_m,
                ):
                    continue

                nx, ny, nyaw = seg[-1]
                nk = key_of(nx, ny, nyaw)

                _, cur_l, _ = self._nearest_route_frame(x=cur.x, y=cur.y, route_pts=route_pts)
                _, end_l, _ = self._nearest_route_frame(x=nx, y=ny, route_pts=route_pts)

                if behavior_state == "LANE_CHANGE_OUT":
                    if target_l > 0.0 and end_l < cur_l - lane_change_monotonic_tol_m:
                        continue
                    if target_l < 0.0 and end_l > cur_l + lane_change_monotonic_tol_m:
                        continue

                if behavior_state == "LANE_CHANGE_BACK":
                    if cur_l > 0.0 and end_l > cur_l + lane_change_monotonic_tol_m:
                        continue
                    if cur_l < 0.0 and end_l < cur_l - lane_change_monotonic_tol_m:
                        continue

                length = abs(sim_speed) * prim_dt * prim_steps
                ng = cur.g + length + 0.1 * abs(steer - cur.steer)

                if route_pts and w_ref > 0.0:
                    ng += w_ref * (end_l * end_l)

                goal_dir_x = math.cos(goal.yaw)
                goal_dir_y = math.sin(goal.yaw)
                progress = (nx - cur.x) * goal_dir_x + (ny - cur.y) * goal_dir_y

                if behavior_state == "FOLLOW":
                    ng += w_follow_center * (end_l * end_l)

                elif behavior_state == "LANE_CHANGE_OUT":
                    ng += w_lane_change_out_target * ((end_l - target_l) ** 2)
                    ng -= w_lane_change_out_progress * progress

                elif behavior_state == "CRUISE_PASS_LANE":
                    ng += w_cruise_pass_lane_target * ((end_l - target_l) ** 2)
                    ng -= w_cruise_pass_lane_progress * progress

                elif behavior_state == "LANE_CHANGE_BACK":
                    ng += w_lane_change_back_target * (end_l * end_l)
                    ng -= w_lane_change_back_progress * progress

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

        return self._reconstruct(nodes, best_key) if best_key in nodes else None

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
    # Ego collision: two-circle model
    # ------------------------------------------------------------------
    def _get_two_circle_params(self) -> Tuple[float, float, float]:
        wheelbase_m = float(self.config.get("ego_wheelbase_m", 2.875))
        width_m = float(self.config.get("ego_width_m", 1.85))
        front_overhang_m = float(self.config.get("ego_front_overhang_m", 0.868))
        rear_overhang_m = float(self.config.get("ego_rear_overhang_m", 0.977))
        safety_margin_m = float(self.config.get("ego_safety_margin_m", 0.02))

        radius_scale = float(self.config.get("ego_circle_radius_scale", 0.80))
        center_scale = float(self.config.get("ego_circle_center_scale", 1.0))

        x_min = -rear_overhang_m
        x_max = wheelbase_m + front_overhang_m
        length = x_max - x_min
        half_w = 0.5 * width_m

        rear_circle_x = x_min + 0.25 * length * center_scale
        front_circle_x = x_min + 0.75 * length * center_scale

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

    def _ego_two_circles_in_collision(
        self,
        *,
        x: float,
        y: float,
        yaw: float,
        grid: LocalOccPatch,
    ) -> bool:
        front_x, rear_x, radius = self._get_two_circle_params()

        c = math.cos(yaw)
        s = math.sin(yaw)

        fx = x + c * front_x
        fy = y + s * front_x

        rx = x + c * rear_x
        ry = y + s * rear_x

        if self._disc_in_collision(cx=fx, cy=fy, r=radius, grid=grid):
            return True
        if self._disc_in_collision(cx=rx, cy=ry, r=radius, grid=grid):
            return True

        return False