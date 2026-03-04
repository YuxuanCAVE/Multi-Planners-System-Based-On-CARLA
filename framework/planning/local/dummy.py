#测试用的planner
# framework/planning/dummy.py
from __future__ import annotations

from typing import Any, Dict, Optional

#框架里定义的数据结构和协议
from framework.core.types import (
    EgoState,
    WorldModel,
    Route,
    Trajectory,
    TrajectoryPoint,
    PlanResult,
    PlanStatus,
)

from framework.planning.base_planning import BasePlanner


def _dist2(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = ax - bx, ay - by
    return dx * dx + dy * dy


class DummyPlanner(BasePlanner):
    """
    最小规划器：沿着全局 route 截取前方 N 个点作为局部轨迹
    用来验证数据流：Scenario -> Planner -> Controller -> CARLA
    """
    #给runner用的标识符
    name = "dummy"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        #存全局路线
        self._route: Optional[Route] = None 
        #轨迹点直接的时间间隔
        self._dt: float = float(self.config.get("dt", 0.1))
        #往前取多少个点
        self._horizon_steps: int = int(self.config.get("horizon_steps", 40))
        #给每个轨迹点的目标速度
        self._target_speed: float = float(self.config.get("target_speed", 10.0))

    #保存全局route
    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        self._route = route

    #core part： 
    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        #没有route就返回empty
        if self._route is None or not self._route.points:
            return PlanResult(status=PlanStatus.EMPTY, trajectory=None, debug={"reason": "no_route"})

        pts = self._route.points

        # 找到 route 上离 ego 最近的点索引
        ex, ey = ego.pose.x, ego.pose.y
        best_i = 0
        best_d2 = float("inf")
        # 为简单起见线性搜索；后面可换 KD-tree
        for i, p in enumerate(pts):
            d2 = _dist2(ex, ey, p.x, p.y)
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        # 截取未来点（不够就到末尾）
        out: list[TrajectoryPoint] = []
        end_i = min(len(pts), best_i + self._horizon_steps)
        for i in range(best_i, end_i):
            p = pts[i]
            out.append(TrajectoryPoint(x=p.x, y=p.y, yaw=p.yaw, v=self._target_speed))

        #没点就EMPTY（保险）
        if not out:
            return PlanResult(status=PlanStatus.EMPTY, trajectory=None, debug={"reason": "empty_traj"})

        #包装成Trajectory，并返回OK
        traj = Trajectory(points=out, dt=self._dt)
        return PlanResult(
            status=PlanStatus.OK,
            trajectory=traj,
            debug={"nearest_index": best_i, "route_len": len(pts)},
        )
    

    """
    Scenario 先给一个全局Route（很多店构成）
    DummyPlanner每帧：
        找自己在哪个点附近
        截取前面40个点作为局部轨迹
    """


