# framework/map/occupancy_provider.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import carla


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class GridMeta:
    origin_x: float
    origin_y: float
    res_m: float
    width: int
    height: int


@dataclass
class LocalOccPatch:
    """
    Local occupancy patch in world coordinates.
    occ[i][j] = True means occupied.
    """
    origin_x: float
    origin_y: float
    res_m: float
    width: int
    height: int
    occ: List[List[bool]]

    def world_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        j = int((x - self.origin_x) / self.res_m)
        i = int((y - self.origin_y) / self.res_m)
        if 0 <= i < self.height and 0 <= j < self.width:
            return i, j
        return None

    def ij_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.origin_x + (j + 0.5) * self.res_m
        y = self.origin_y + (i + 0.5) * self.res_m
        return x, y

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
        rr = int(math.ceil(r / self.res_m))
        r2 = r * r
        for di in range(-rr, rr + 1):
            ii = ci + di
            if not (0 <= ii < self.height):
                continue
            row = self.occ[ii]
            dy = di * self.res_m
            for dj in range(-rr, rr + 1):
                jj = cj + dj
                if not (0 <= jj < self.width):
                    continue
                dx = dj * self.res_m
                if dx * dx + dy * dy <= r2:
                    row[jj] = True


class StaticOccupancyMap:
    """
    Global static occupancy map:
      True  = occupied / non-drivable
      False = free / drivable
    Built once at reset() from CARLA map.
    """

    def __init__(
        self,
        *,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        res_m: float,
        occ: List[List[bool]],
    ):
        self.meta = GridMeta(
            origin_x=float(origin_x),
            origin_y=float(origin_y),
            res_m=float(res_m),
            width=int(width),
            height=int(height),
        )
        self.occ = occ

    def world_to_ij(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        j = int((x - self.meta.origin_x) / self.meta.res_m)
        i = int((y - self.meta.origin_y) / self.meta.res_m)
        if 0 <= i < self.meta.height and 0 <= j < self.meta.width:
            return i, j
        return None

    def ij_to_world(self, i: int, j: int) -> Tuple[float, float]:
        x = self.meta.origin_x + (j + 0.5) * self.meta.res_m
        y = self.meta.origin_y + (i + 0.5) * self.meta.res_m
        return x, y

    def crop_patch(
        self,
        *,
        center_x: float,
        center_y: float,
        size_x_m: float,
        size_y_m: float,
    ) -> LocalOccPatch:
        """
        Crop a local patch from the global static map.
        Outside the global map -> occupied.
        """
        res = self.meta.res_m
        width = int(math.ceil(size_x_m / res))
        height = int(math.ceil(size_y_m / res))

        origin_x = center_x - 0.5 * size_x_m
        origin_y = center_y - 0.5 * size_y_m

        occ = [[True] * width for _ in range(height)]

        for i in range(height):
            for j in range(width):
                x = origin_x + (j + 0.5) * res
                y = origin_y + (i + 0.5) * res
                gij = self.world_to_ij(x, y)
                if gij is None:
                    occ[i][j] = True
                else:
                    gi, gj = gij
                    occ[i][j] = self.occ[gi][gj]

        return LocalOccPatch(
            origin_x=origin_x,
            origin_y=origin_y,
            res_m=res,
            width=width,
            height=height,
            occ=occ,
        )


class OccupancyMapProvider:
    """
    Shared occupancy provider for planners.

    Static map build strategy:
      pointwise Driving-lane classification using CARLA map.get_waypoint().
    """

    def __init__(self, *, static_res_m: float = 0.5):
        self.static_res_m = float(static_res_m)
        self.static_map: Optional[StaticOccupancyMap] = None

    # --------------------------------------------------
    # Static map build
    # --------------------------------------------------
    def build_static_from_carla_map(
        self,
        *,
        carla_map: carla.Map,
        route_points: List[Any],
        margin_m: float = 30.0,
        lane_type: carla.LaneType = carla.LaneType.Driving,
        free_space_relax_m: float = 0.0,
    ) -> None:
        """
        Build a static map only around the route bounding box + margin.
        Each cell center is queried against CARLA Driving lane semantics.
        """
        if not route_points:
            raise ValueError("route_points is empty, cannot build static occupancy map")

        xs = [float(p.x) for p in route_points]
        ys = [float(p.y) for p in route_points]

        min_x = min(xs) - margin_m
        max_x = max(xs) + margin_m
        min_y = min(ys) - margin_m
        max_y = max(ys) + margin_m

        res = self.static_res_m
        width = int(math.ceil((max_x - min_x) / res))
        height = int(math.ceil((max_y - min_y) / res))

        occ = [[True] * width for _ in range(height)]

        for i in range(height):
            row = occ[i]
            y = min_y + (i + 0.5) * res
            for j in range(width):
                x = min_x + (j + 0.5) * res
                loc = carla.Location(x=float(x), y=float(y), z=0.0)
                wp = carla_map.get_waypoint(
                    loc,
                    project_to_road=False,
                    lane_type=lane_type,
                )
                row[j] = (wp is None)

        if free_space_relax_m > 1e-6:
            occ = self._dilate_free_space(
                occ=occ,
                relax_margin_m=free_space_relax_m,
                res_m=res,
            )
            
        self.static_map = StaticOccupancyMap(
            origin_x=min_x,
            origin_y=min_y,
            width=width,
            height=height,
            res_m=res,
            occ=occ,
        )

        

    # --------------------------------------------------
    # Local patch build
    # --------------------------------------------------
    def get_local_patch(
        self,
        *,
        ego_x: float,
        ego_y: float,
        world: Any,
        size_x_m: float,
        size_y_m: float,
        obstacle_inflation_m: float = 1.0,
        actor_filter_radius_m: Optional[float] = None,
    ) -> LocalOccPatch:
        if self.static_map is None:
            raise RuntimeError("static_map is not built yet")

        patch = self.static_map.crop_patch(
            center_x=ego_x,
            center_y=ego_y,
            size_x_m=size_x_m,
            size_y_m=size_y_m,
        )

        self._overlay_dynamic_obstacles(
            patch=patch,
            ego_x=ego_x,
            ego_y=ego_y,
            world=world,
            inflation_m=obstacle_inflation_m,
            actor_filter_radius_m=actor_filter_radius_m,
        )
        return patch

    def _overlay_dynamic_obstacles(
        self,
        *,
        patch: LocalOccPatch,
        ego_x: float,
        ego_y: float,
        world: Any,
        inflation_m: float,
        actor_filter_radius_m: Optional[float],
    ) -> None:
        """
        Overlay current dynamic obstacles to local patch.
        Currently uses disc approximation, which is simple and robust.
        """
        rr2 = None
        if actor_filter_radius_m is not None:
            rr2 = actor_filter_radius_m * actor_filter_radius_m

        for obs in getattr(world, "obstacles", []):
            ox = float(obs.position.x)
            oy = float(obs.position.y)

            if rr2 is not None:
                dx = ox - ego_x
                dy = oy - ego_y
                if dx * dx + dy * dy > rr2:
                    continue

            r = max(0.0, float(getattr(obs, "radius", 0.8)) + inflation_m)
            patch.set_occupied_disc(ox, oy, r)

    def _dilate_free_space(
        self,
        occ: List[List[bool]],
        relax_margin_m: float,
        res_m: float,
    ) -> List[List[bool]]:
        """
        Expand free space outward by relax_margin_m.
        occ[i][j] = True means occupied.
        After this operation, some occupied cells near free cells become free.
        """
        if relax_margin_m <= 1e-6:
            return occ

        height = len(occ)
        width = len(occ[0]) if height > 0 else 0
        rr = int(math.ceil(relax_margin_m / res_m))
        r2 = relax_margin_m * relax_margin_m

        out = [row[:] for row in occ]

        for i in range(height):
            for j in range(width):
                if not occ[i][j]:
                    continue  # already free

                make_free = False
                for di in range(-rr, rr + 1):
                    ii = i + di
                    if not (0 <= ii < height):
                        continue
                    dy = di * res_m
                    for dj in range(-rr, rr + 1):
                        jj = j + dj
                        if not (0 <= jj < width):
                            continue
                        dx = dj * res_m
                        if dx * dx + dy * dy > r2:
                            continue
                        if occ[ii][jj] is False:
                            make_free = True
                            break
                    if make_free:
                        break

                if make_free:
                    out[i][j] = False

        return out