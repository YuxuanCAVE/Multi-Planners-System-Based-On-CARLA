from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import carla

from framework.core.types import Route

@abstractmethod
class BaseScenario(ABC):
    
    name: str = "base_scenario"

    def __init__(self, config: Optional[Dict[str,Any]] = None):
        self.config : Dict[str, Any] = config or {}

        self.ego_vehicle : Optional[carla.Vehicle] = None
        self.world : Optional[carla.World] = None
        self.actors : list[carla.Actor] = []
        self.sensors : list[carla.Sensor] = []

        self._done_info : Dict[str, Any] = {}

        super().__init__()

    #Lifecycle
    
    def setup(self, client: carla.Client) -> carla.World:

        raise NotImplementedError
    
    def get_route(self) -> Route:

        raise NotImplementedError
    def tick(self, t_sim: float) -> None:

        return
    
    def is_done(self) -> Tuple[bool, Dict[str,Any]]:

        raise NotImplementedError
    def destroy(self) -> None:
        """
        Cleanup all CARLA actors/sensors created by this scenario.
        Default implementation destroys sensors first, then other actors, then ego.

        Runner should call this in cleanup() regardless of success/failure.
        """
        # Destroy sensors first
        for s in list(self.sensors):
            try:
                s.stop()
            except Exception:
                pass
            try:
                s.destroy()
            except Exception:
                pass
        self.sensors.clear()

        # Destroy non-ego actors (if any)
        for a in list(self.actors):
            try:
                a.destroy()
            except Exception:
                pass
        self.actors.clear()

        # Destroy ego last
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
            self.ego_vehicle = None

        self.world = None

    # ---------------------------
    # Convenience hooks (optional)
    # ---------------------------
    def get_goal(self) -> Optional[Any]:
        """
        Optional: return the scenario goal representation (transform, waypoint, etc.).
        Runner doesn't need it, but metrics/visualization might.
        """
        return None

    def get_meta(self) -> Dict[str, Any]:
        """
        Optional: provide scenario metadata for recorder (map, spawn indices, etc.).
        """
        return {"scenario": self.name, "config": self.config}