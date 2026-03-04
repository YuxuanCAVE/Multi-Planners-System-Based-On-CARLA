"""
规定算法接口（抽象基类）
做成统一插拔接口

抽象基类（ABC） + @abstract 用来“强制约束” 
——谁继承它，就必须实现reset和plan
"""
# framework/planning/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from framework.core.types import EgoState, WorldModel, Route, PlanResult


class BasePlanner(ABC):
    """
    所有规划算法统一接口
    """
    name: str = "base" #给框架一个 规划器名字 方便配置选择

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        #config 是可选字典
        # 如果传入None 就用空字典
    @abstractmethod
    def reset(self, *, route: Route, map_info: Dict[str, Any]) -> None:
        """
        场景开始 / 换路线 时调用
        - route: 全局参考线
        - map_info: 地图信息（可选，比如分辨率、车道宽等）
        """
        raise NotImplementedError

    @abstractmethod
    def plan(self, *, ego: EgoState, world: WorldModel, t: float) -> PlanResult:
        """
        每个 tick 调用，输出一段短时域可执行轨迹
        """
        raise NotImplementedError
