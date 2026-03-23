from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import carla

from framework.core.types import Route


class BaseScenario(ABC):
    """
    Base class for all scenarios.

    Responsibilities:
    - store common runtime handles (world / ego / actors / sensors)
    - define standard scenario lifecycle
    - provide safe destroy logic
    """

    name = "base_scenario"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = config or {}

        self.world: Optional[carla.World] = None
        self.ego_vehicle: Optional[carla.Actor] = None
        self.actors: list[carla.Actor] = []
        self.sensor_suite: Optional[Any] = None

        self._done_info: Dict[str, Any] = {}

    # ---------------------------
    # Lifecycle API
    # ---------------------------
    @abstractmethod
    def setup(self, client: carla.Client) -> carla.World:
        """
        Create/load the world, spawn ego, spawn actors, initialize sensors, etc.
        Must return the CARLA world.
        """
        raise NotImplementedError

    @abstractmethod
    def get_route(self) -> Route:
        """
        Return the reference/global route for the planner/controller.
        """
        raise NotImplementedError

    def tick(self, t_sim: float) -> None:
        """
        Scenario update hook called every simulation step.
        Subclasses can override for scripted events.
        """
        return

    def is_done(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Return whether the scenario should terminate.
        Subclasses usually override this.
        """
        return False, {"reason": "running"}

    def get_goal(self) -> Optional[Any]:
        """
        Optional hook for evaluation/debug.
        """
        return None

    def get_meta(self) -> Dict[str, Any]:
        """
        Optional metadata for logging/recording.
        """
        return {
            "scenario": self.name,
            "config": self.config,
        }

    def get_sensor_snapshot(self) -> Dict[str, Any]:
        """
        Optional lightweight sensor summary for recorder.
        """
        return {}

    # ---------------------------
    # Cleanup
    # ---------------------------
    def destroy(self) -> None:
        """
        Safely destroy sensors, extra actors, and ego vehicle.
        """
        # sensor suite
        if self.sensor_suite is not None:
            try:
                self.sensor_suite.destroy()
            except Exception:
                pass
            finally:
                self.sensor_suite = None

        # extra actors
        for actor in reversed(self.actors):
            if actor is None:
                continue
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors.clear()

        # ego vehicle
        if self.ego_vehicle is not None:
            try:
                self.ego_vehicle.destroy()
            except Exception:
                pass
            finally:
                self.ego_vehicle = None

        self.world = None
        self._done_info = {}