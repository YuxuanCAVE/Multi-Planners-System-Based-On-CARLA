#定义统一数据结构

# framework/core/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Vec3:
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
#二维位置
class Pose2D:
    x: float
    y: float
    yaw: float  # radians


@dataclass(frozen=True)
#自车状态（位置， 朝向， 速度等）
class EgoState:
    pose: Pose2D
    speed: float  # m/s


@dataclass(frozen=True)
class Obstacle:
    """
    最小障碍物表示：位置 + 近似半径。
    先用的GRP
    """
    id: int
    position: Vec3
    velocity: Vec3
    radius: float  # meters (approx)


@dataclass(frozen=True)
#世界模型（周围车， 阻碍物，地图信息等）
class WorldModel:
    obstacles: List[Obstacle] = field(default_factory=list)




@dataclass(frozen=True)
class Route:
    """
    全局参考线：点序列（x,y,yaw）
    """
    points: List[Pose2D]


@dataclass(frozen=True)
#局部轨迹里的一个点
class TrajectoryPoint:
    x: float
    y: float
    yaw: float  # radians
    v: float    # m/s


@dataclass(frozen=True)
#一串 TrajectoryPoint组成的
class Trajectory:
    """
    局部轨迹：未来离散点序列（等 dt）
    """
    points: List[TrajectoryPoint]
    dt: float

#规划状态
class PlanStatus(str, Enum):
    OK = "ok"
    FAIL = "fail"
    EMPTY = "empty"


@dataclass(frozen=True)
#规划返回结果
class PlanResult:
    status: PlanStatus
    trajectory: Optional[Trajectory] = None
    debug: Dict[str, Any] = field(default_factory=dict)
