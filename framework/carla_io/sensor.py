from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import threading
import time

import carla

class _LatestBuffer:
    def __init__(self):
        self._lock = threading.Lock()
        self._data : Any = None
        self._ts : float = 0.0

    def set(self, data:Any, ts: float) -> None:
        with self._lock:
            self._data = data
            self._ts = ts

    def get(self) -> Tuple[Any, float]:
        with self._lock:
            return self._data, self._ts
        
@dataclass
class RGBCameraConfig:
    image_size_x: int = 1280
    image_size_y: int = 720
    fov: float = 90.0
    x: float = 1.5
    y: float = 0.0
    z: float = 1.5
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0
    sensor_tick: float = 0.0


class FrontRGBCamera:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.cfg = RGBCameraConfig(
            image_size_x=int(c.get("image_size_x", 1280)),
            image_size_y=int(c.get("image_size_y", 720)),
            fov=float(c.get("fov", 90)),
            x=float(c.get("x", 1.5)),
            y=float(c.get("y", 0.0)),
            z=float(c.get("z", 1.5)),
            pitch=float(c.get("pitch", 0.0)),
            yaw=float(c.get("yaw", 0.0)),
            roll=float(c.get("roll", 0.0)),
            sensor_tick=float(c.get("sensor_tick", 0.0)),
        )

        self.world = world
        self.vehicle = vehicle
        self.sensor: Optional[carla.Sensor] = None
        self.latest = _LatestBuffer()
        self._spawn()

    def _spawn(self) -> None:
        bp = self.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(self.cfg.image_size_x))
        bp.set_attribute("image_size_y", str(self.cfg.image_size_y))
        bp.set_attribute("fov", str(self.cfg.fov))
        bp.set_attribute("sensor_tick", str(self.cfg.sensor_tick))

        tf = carla.Transform(
            carla.Location(x=self.cfg.x, y=self.cfg.y, z=self.cfg.z),
            carla.Rotation(pitch=self.cfg.pitch, yaw=self.cfg.yaw, roll=self.cfg.roll),
        )
        self.sensor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        self.sensor.listen(self._on_image)
    
    def _on_image(self, image:carla.Image) -> None:
        self.latest.set(image, float(image.timestamp))
    
    def get_latest_image(self) -> Tuple[Optional[carla.Image],float]:
        img, ts = self.latest.get()
        return img,ts
    
    def stop_destroy(self) -> None:
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            try:
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None

@dataclass
class RadarConfig:

    x: float = 2.0
    y: float = 2.0
    z: float = 1.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0

    # radar parameters
    horizontal_fov: float = 30.0   # degrees
    vertical_fov: float = 10.0     # degrees
    range: float = 50.0            # meters
    points_per_second: int = 1500
    sensor_tick: float = 0.0       # 0 => every tick


class FrontRadar:
    """
    Front radar:
    - CARLA provides RadarMeasurement: list of RadarDetection
      each detection has:
        - depth (m): distance
        - azimuth (rad), altitude (rad)
        - velocity (m/s) radial (positive away from sensor)
    - We store latest measurement and compute useful summaries like min depth.
    """

    def __init__(self, world: carla.World, vehicle: carla.Vehicle, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.cfg = RadarConfig(
            x=float(c.get("x", 2.0)),
            y=float(c.get("y", 0.0)),
            z=float(c.get("z", 1.0)),
            pitch=float(c.get("pitch", 0.0)),
            yaw=float(c.get("yaw", 0.0)),
            roll=float(c.get("roll", 0.0)),
            horizontal_fov=float(c.get("horizontal_fov", 30.0)),
            vertical_fov=float(c.get("vertical_fov", 10.0)),
            range=float(c.get("range", 50.0)),
            points_per_second=int(c.get("points_per_second", 1500)),
            sensor_tick=float(c.get("sensor_tick", 0.0)),
        )

        self.world = world
        self.vehicle = vehicle
        self.sensor: Optional[carla.Sensor] = None

        self.latest = _LatestBuffer()
        self.latest_summary = _LatestBuffer()

        self._spawn()

    def _spawn(self) -> None:
        bp = self.world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", str(self.cfg.horizontal_fov))
        bp.set_attribute("vertical_fov", str(self.cfg.vertical_fov))
        bp.set_attribute("range", str(self.cfg.range))
        bp.set_attribute("points_per_second", str(self.cfg.points_per_second))
        bp.set_attribute("sensor_tick", str(self.cfg.sensor_tick))

        tf = carla.Transform(
            carla.Location(x=self.cfg.x, y=self.cfg.y, z=self.cfg.z),
            carla.Rotation(pitch=self.cfg.pitch, yaw=self.cfg.yaw, roll=self.cfg.roll),
        )
        self.sensor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
        self.sensor.listen(self._on_radar)

    def _on_radar(self, meas: carla.RadarMeasurement) -> None:
        ts = float(meas.timestamp)
        self.latest.set(meas, ts)

        # compute summary
        min_depth = None
        min_depth_vel = None
        count = 0

        # meas is iterable of RadarDetection
        for d in meas:
            count += 1
            if min_depth is None or d.depth < min_depth:
                min_depth = float(d.depth)
                min_depth_vel = float(d.velocity)

        summary = {
            "count": int(count),
            "min_depth_m": min_depth,
            "min_depth_radial_vel_mps": min_depth_vel,  # positive: moving away
        }
        self.latest_summary.set(summary, ts)

    def get_latest_radar(self) -> Tuple[Optional[carla.RadarMeasurement], float]:
        meas, ts = self.latest.get()
        return meas, ts

    def get_latest_summary(self) -> Tuple[Optional[Dict[str, Any]], float]:
        s, ts = self.latest_summary.get()
        return s, ts

    def stop_destroy(self) -> None:
        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass
            try:
                self.sensor.destroy()
            except Exception:
                pass
            self.sensor = None


# ----------------------------
# Sensor Suite: manage multiple sensors
# ----------------------------
class SensorSuite:
    """
    Convenience container:
    - attach camera + radar
    - expose getters
    - destroy cleanly
    """

    def __init__(
        self,
        world: carla.World,
        vehicle: carla.Vehicle,
        *,
        camera_cfg: Optional[Dict[str, Any]] = None,
        radar_cfg: Optional[Dict[str, Any]] = None,
        enable_camera: bool = True,
        enable_radar: bool = True,
    ):
        self.camera: Optional[FrontRGBCamera] = None
        self.radar: Optional[FrontRadar] = None

        if enable_camera:
            self.camera = FrontRGBCamera(world, vehicle, camera_cfg)
        if enable_radar:
            self.radar = FrontRadar(world, vehicle, radar_cfg)

    def get_front_image(self) -> Tuple[Optional[carla.Image], float]:
        if self.camera is None:
            return None, 0.0
        return self.camera.get_latest_image()

    def get_radar_summary(self) -> Tuple[Optional[Dict[str, Any]], float]:
        if self.radar is None:
            return None, 0.0
        return self.radar.get_latest_summary()

    def destroy(self) -> None:
        if self.camera is not None:
            self.camera.stop_destroy()
            self.camera = None
        if self.radar is not None:
            self.radar.stop_destroy()
            self.radar = None