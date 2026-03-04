# framework/planning/local/hybrid_astar.py

#1. 从全局Route里选一个局部目标点（lookahead）
# 2.以自车为中心构建局部占据栅格
# 3.Hybrid A* 搜索， 从（x,y,raw）出发，用运动学自行车“离散转角原语”扩展，避开占据格
# 4. 把搜索到的路径 转换为 等dt的 Trajectory 
# 5.返回 PlanResult（ok, Trajectory,debug）; 失败返回FAIL

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from framework.planning.base_planning import BasePlanner
from framework.core.types import EgoState, WorldModel, Route, PlanResult, PlanStatus, Trajectory, TrajectoryPoint, Pose2D


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

#Hybrid A* 的节点不是纯栅格点，而是带航向yaw的连续状态
#

@dataclass
class _Node:
    x: float
    y: float
    yaw: float
    g: float
    h: float
    parent: Optional[Tuple[int, int, int]]  # key of parent
    steer: float
    direction: int  # +1 forward, -1 reverse
    # store a short segment for backtracking (optional)
    seg: Optional[List[Tuple[float, float, float]]] = None

#占据栅格
class _OccGrid:
    """
    Ego-centered occupancy grid in world frame:
    - We store grid origin (min_x, min_y) in world coordinates.
    - Convert world -> grid indices.
    """
    def __init__(self, *, ego_x: float, ego_y: float, size_m: float, res_m: float):
        self.size_m = size_m
        self.res_m = res_m
        self.half = size_m * 0.5
        self.w = int(math.ceil(size_m / res_m))
        self.h = int(math.ceil(size_m / res_m))
        self.min_x = ego_x - self.half
        self.min_y = ego_y - self.half
        self.occ = [[False] * self.w for _ in range(self.h)]

    def world_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        i = int((y - self.min_y) / self.res_m)
        j = int((x - self.min_x) / self.res_m)
        if 0 <= i < self.h and 0 <= j < self.w:
            return i, j
        return None

    #把圆形障碍画进格子
    def set_occupied_disc(self, x: float, y: float, r: float) -> None:
        """Mark a disc as occupied (in world coordinates)."""
        center = self.world_to_ij(x, y)
        if center is None:
            return
        ri = int(math.ceil(r / self.res_m))
        ci, cj = center
        for di in range(-ri, ri + 1):
            ii = ci + di
            if not (0 <= ii < self.h):
                continue
            for dj in range(-ri, ri + 1):
                jj = cj + dj
                if not (0 <= jj < self.w):
                    continue
                # circle check
                dx = dj * self.res_m
                dy = di * self.res_m
                if dx * dx + dy * dy <= r * r:
                    self.occ[ii][jj] = True

    #is_occupied(x,y):碰撞判断
    def is_occupied(self, x: float, y: float) -> bool:
        ij = self.world_to_ij(x, y)
        if ij is None:
            # outside local grid treated as collision (conservative)
            return True
        i, j = ij
        return self.occ[i][j]


class HybridAStarPlanner(BasePlanner):
    name: str = "hybrid_astar"

    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route
        self._map_info = map_info or {}
        self._last_goal_idx = 0
        self._last_solution: Optional[List[Pose2D]] = None

    #每tick做什么
    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        cfg = self.config

        dt_out = float(cfg.get("dt", 0.1))
        horizon_s = float(cfg.get("horizon_s", 5.0))
        target_speed = float(cfg.get("target_speed", 6.0))

        # 选目标 
        goal = self._pick_goal(ego, lookahead_m=float(cfg.get("lookahead_m", 25.0)))

        # 构建栅格 + 画障碍
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

        #搜寻 _search() 这里输出的是 path_xyz:List[(x,y,yaw)]
        t0 = time.time()
        path_xyz = self._search(
            ego=ego,
            goal=goal,
            grid=grid,
            max_expansions=int(cfg.get("max_expansions", 20000)),
            max_time_ms=float(cfg.get("max_time_ms", 50.0)),
        )
        ms = (time.time() - t0) * 1000.0

        if not path_xyz:
            return PlanResult(
                status=PlanStatus.FAIL,
                trajectory=None,
                debug={"planner": "hybrid_astar", "ms": ms, "reason": "no_path"},
            )

        # 转为轨迹 把按空间采样的路径 变成 按时间采样的轨迹
        traj = self._path_to_trajectory(
            path_xyz=path_xyz,
            dt=dt_out,
            horizon_s=horizon_s,
            v=target_speed,
        )

        return PlanResult(
            status=PlanStatus.OK,
            trajectory=traj,
            debug={
                "planner": "hybrid_astar",
                "ms": ms,
                "path_len": len(path_xyz),
                "goal": (goal.x, goal.y, goal.yaw),
            },
        )

    # Goal selection
    def _pick_goal(self, ego: EgoState, lookahead_m: float) -> Pose2D:
        # very simple: find nearest route point then advance by distance
        pts = self._route.points
        ex, ey = ego.pose.x, ego.pose.y

        # search around last_goal_idx for speed (optional)
        best_i = 0
        best_d2 = 1e18
        for i, p in enumerate(pts):
            dx = p.x - ex
            dy = p.y - ey
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        # accumulate forward distance to reach lookahead
        dist = 0.0
        j = best_i
        while j + 1 < len(pts) and dist < lookahead_m:
            dx = pts[j + 1].x - pts[j].x
            dy = pts[j + 1].y - pts[j].y
            dist += math.hypot(dx, dy)
            j += 1

        self._last_goal_idx = j
        return pts[j]

    # Hybrid A* core
    def _search(
        self,
        *,
        ego: EgoState,
        goal: Pose2D,
        grid: _OccGrid,
        max_expansions: int,
        max_time_ms: float,
    ) -> Optional[List[Tuple[float, float, float]]]:
        cfg = self.config

        L = float(cfg.get("wheelbase_m", 2.8))
        steer_max = math.radians(float(cfg.get("steer_max_deg", 30.0)))
        steer_samples = int(cfg.get("steer_samples", 7))
        sim_v = float(cfg.get("sim_speed_mps", 8.0))
        prim_dt = float(cfg.get("primitive_dt", 0.1))
        prim_steps = int(cfg.get("primitive_steps", 10))

        heading_bins = int(cfg.get("heading_bins", 72))
        allow_reverse = bool(cfg.get("allow_reverse", False))

        w_len = float(cfg.get("w_len", 1.0))
        w_steer = float(cfg.get("w_steer", 0.2))
        w_steer_change = float(cfg.get("w_steer_change", 1.0))
        w_reverse = float(cfg.get("w_reverse", 2.0))

        goal_tol = float(cfg.get("goal_tolerance_m", 2.0))
        goal_yaw_tol = math.radians(float(cfg.get("goal_tolerance_yaw_deg", 15.0)))

        def key_of(x: float, y: float, yaw: float) -> Tuple[int, int, int]:
            ij = grid.world_to_ij(x, y)
            if ij is None:
                # outside grid -> clamp to extreme; will be collision anyway
                i = -999999
                j = -999999
            else:
                i, j = ij
            yaw = _wrap_pi(yaw)
            k = int((yaw + math.pi) / (2.0 * math.pi) * heading_bins) % heading_bins
            return i, j, k

        def heuristic(x: float, y: float, yaw: float) -> float:
            # MVP heuristic: Euclidean + small yaw penalty
            dx = goal.x - x
            dy = goal.y - y
            d = math.hypot(dx, dy)
            yaw_err = abs(_wrap_pi(goal.yaw - yaw))
            return d + 0.1 * yaw_err

        def goal_reached(x: float, y: float, yaw: float) -> bool:
            dx = goal.x - x
            dy = goal.y - y
            if dx * dx + dy * dy > goal_tol * goal_tol:
                return False
            if abs(_wrap_pi(goal.yaw - yaw)) > goal_yaw_tol:
                return False
            return True

        def collision_free(seg: List[Tuple[float, float, float]]) -> bool:
            for (sx, sy, _syaw) in seg:
                if grid.is_occupied(sx, sy):
                    return False
            return True

        # steering set
        if steer_samples <= 1:
            steers = [0.0]
        else:
            steers = []
            for k in range(steer_samples):
                u = -1.0 + 2.0 * k / (steer_samples - 1)
                steers.append(u * steer_max)

        # A* open set
        start = (ego.pose.x, ego.pose.y, ego.pose.yaw)
        start_key = key_of(*start)
        start_node = _Node(
            x=start[0], y=start[1], yaw=start[2],
            g=0.0, h=heuristic(*start),
            parent=None, steer=0.0, direction=+1, seg=None
        )

        open_heap: List[Tuple[float, int, Tuple[int, int, int]]] = []
        heapq.heappush(open_heap, (start_node.g + start_node.h, 0, start_key))
        nodes: Dict[Tuple[int, int, int], _Node] = {start_key: start_node}
        closed: Dict[Tuple[int, int, int], float] = {}

        t_start = time.time()
        push_id = 1
        expansions = 0

        # simple “best goal” fallback
        best_key = start_key
        best_f = start_node.g + start_node.h

        while open_heap and expansions < max_expansions:
            if (time.time() - t_start) * 1000.0 > max_time_ms:
                break

            f, _pid, kkey = heapq.heappop(open_heap)
            if kkey in closed:
                continue

            cur = nodes[kkey]
            closed[kkey] = cur.g
            expansions += 1

            if f < best_f:
                best_f = f
                best_key = kkey

            if goal_reached(cur.x, cur.y, cur.yaw):
                return self._reconstruct(nodes, kkey)

            # expand with motion primitives
            for steer in steers:
                for direction in ([+1, -1] if allow_reverse else [+1]):
                    seg = self._simulate_primitive(
                        x=cur.x, y=cur.y, yaw=cur.yaw,
                        steer=steer, direction=direction,
                        v=sim_v, L=L,
                        dt=prim_dt, steps=prim_steps,
                    )
                    if not collision_free(seg):
                        continue
                    nx, ny, nyaw = seg[-1]

                    nkey = key_of(nx, ny, nyaw)
                    # cost increment
                    length = sim_v * prim_dt * prim_steps
                    dg = w_len * length + w_steer * abs(steer) + w_steer_change * abs(steer - cur.steer)
                    if direction < 0:
                        dg += w_reverse * length

                    ng = cur.g + dg
                    nh = heuristic(nx, ny, nyaw)
                    nf = ng + nh

                    prev = nodes.get(nkey)
                    if prev is None or ng < prev.g:
                        nodes[nkey] = _Node(
                            x=nx, y=ny, yaw=nyaw,
                            g=ng, h=nh,
                            parent=kkey, steer=steer, direction=direction,
                            seg=seg,
                        )
                        heapq.heappush(open_heap, (nf, push_id, nkey))
                        push_id += 1

        # failed to reach goal: return best-so-far partial
        return self._reconstruct(nodes, best_key) if best_key in nodes else None

    def _simulate_primitive(
        self,
        *,
        x: float, y: float, yaw: float,
        steer: float, direction: int,
        v: float, L: float,
        dt: float, steps: int,
    ) -> List[Tuple[float, float, float]]:
        seg: List[Tuple[float, float, float]] = []
        vv = v * float(direction)
        cx, cy, cyaw = x, y, yaw
        for _ in range(steps):
            cx += vv * math.cos(cyaw) * dt
            cy += vv * math.sin(cyaw) * dt
            cyaw = _wrap_pi(cyaw + (vv / L) * math.tan(steer) * dt)
            seg.append((cx, cy, cyaw))
        return seg

    def _reconstruct(self, nodes: Dict[Tuple[int, int, int], _Node], last_key: Tuple[int, int, int]) -> List[Tuple[float, float, float]]:
        out: List[Tuple[float, float, float]] = []
        k = last_key
        while True:
            n = nodes[k]
            if n.seg is not None:
                # prepend this segment (reverse order later)
                out.extend(reversed(n.seg))
            else:
                out.append((n.x, n.y, n.yaw))
            if n.parent is None:
                break
            k = n.parent
        out.reverse()
        # de-duplicate near-identical consecutive points
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

    # ----------------------------
    # Path -> Trajectory
    # ----------------------------
    def _path_to_trajectory(self, *, path_xyz: List[Tuple[float, float, float]], dt: float, horizon_s: float, v: float) -> Trajectory:
        # simple time-based sampling with constant v:
        # choose points along the path by arc-length s = v*t
        if len(path_xyz) < 2:
            pt = path_xyz[0] if path_xyz else (0.0, 0.0, 0.0)
            return Trajectory(points=[TrajectoryPoint(x=pt[0], y=pt[1], yaw=pt[2], v=v)], dt=dt)

        # build cumulative arc length
        xs = [p[0] for p in path_xyz]
        ys = [p[1] for p in path_xyz]
        yaws = [p[2] for p in path_xyz]

        s = [0.0]
        for i in range(1, len(path_xyz)):
            ds = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
            s.append(s[-1] + ds)
        total = s[-1]

        n_out = max(2, int(math.ceil(horizon_s / dt)) + 1)
        points: List[TrajectoryPoint] = []

        for k in range(n_out):
            sk = _clamp(v * (k * dt), 0.0, total)
            # find segment index
            idx = 0
            while idx + 1 < len(s) and s[idx + 1] < sk:
                idx += 1
            if idx + 1 >= len(s):
                idx = len(s) - 2
            s0, s1 = s[idx], s[idx + 1]
            ratio = 0.0 if s1 <= s0 else (sk - s0) / (s1 - s0)
            x = xs[idx] + ratio * (xs[idx + 1] - xs[idx])
            y = ys[idx] + ratio * (ys[idx + 1] - ys[idx])
            # yaw: interpolate safely by wrapping
            yaw0 = yaws[idx]
            yaw1 = yaws[idx + 1]
            dyaw = _wrap_pi(yaw1 - yaw0)
            yaw = _wrap_pi(yaw0 + ratio * dyaw)
            points.append(TrajectoryPoint(x=x, y=y, yaw=yaw, v=v))

        return Trajectory(points=points, dt=dt)