# framework/runner.py
from __future__ import annotations

import math
import time
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import carla

from framework.core.types import EgoState, Obstacle, Pose2D, Vec3, WorldModel
from framework.core.types import PlanStatus,PlanResult
from framework.planning.base_planning import BasePlanner
from framework.control.pure_pursuit import PurePursuitController
from framework.evaluation.recorder import Recorder as RichRecorder

# metrics
from framework.evaluation.metrics import TrackingMetrics
def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def _hypot(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


@dataclass
class RunnerConfig:
    # carla
    host: str = "127.0.0.1"
    port: int = 2000
    timeout_s: float = 10.0

    # sim
    sync: bool = True
    fixed_delta_seconds: float = 0.05
    no_rendering_mode: bool = False

    # loop
    max_steps: int = 4000
    debug_draw: bool = True
    follow_spectator: bool = True

    # recorder
    enable_recorder: bool = True

    # world model filtering (VERY IMPORTANT for Frenet stability)
    obstacle_range_m: float = 60.0     # only collect vehicles within this radius
    max_obstacles: int = 60            # cap for performance
    ignore_static_parked: bool = False # keep False unless you know what you're doing


@runtime_checkable
class RecorderBase(Protocol):
    def start(self, *, meta: Dict[str, Any]) -> None: ...
    def step(
        self,
        *,
        t_sim: float,
        step_idx: int,
        ego_state: EgoState,
        world_model: WorldModel,
        plan: Any,
        control: carla.VehicleControl,
        sensors: Optional[Dict[str, Any]] = None,
    ) -> None: ...
    def finish(self, *, result: Dict[str, Any]) -> None: ...
    def close(self) -> None: ...


class NullRecorder:
    def start(self, *, meta: Dict[str, Any]) -> None:
        return

    def step(
        self,
        *,
        t_sim: float,
        step_idx: int,
        ego_state: EgoState,
        world_model: WorldModel,
        plan: Any,
        control: carla.VehicleControl,
        sensors: Optional[Dict[str, Any]] = None,
    ) -> None:
        return

    def finish(self, *, result: Dict[str, Any]) -> None:
        return

    def close(self) -> None:
        return


class CollisionSensor:
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self._world = world
        self._vehicle = vehicle
        self._sensor: Optional[carla.Sensor] = None
        self.collided: bool = False
        self._spawn()

    def _spawn(self) -> None:
        bp = self._world.get_blueprint_library().find("sensor.other.collision")
        self._sensor = self._world.spawn_actor(bp, carla.Transform(), attach_to=self._vehicle)
        self._sensor.listen(lambda event: self._on_collision(event))

    def _on_collision(self, event: carla.CollisionEvent) -> None:
        self.collided = True

    def destroy(self) -> None:
        if self._sensor is None:
            return
        try:
            self._sensor.stop()
        except Exception:
            pass
        try:
            self._sensor.destroy()
        except Exception:
            pass
        self._sensor = None


class Runner:
    """
    Clean simulation loop:
      connect -> scenario.setup -> enable sync -> planner.reset -> recorder.start
      for each tick:
        world.tick
        scenario.tick
        build ego/world
        planner.plan
        controller.compute_control (only if valid traj)
        apply_control
        recorder.step
        termination
      finish -> recorder.finish -> cleanup
    """

    def __init__(
        self,
        *,
        runner_cfg: Optional[Dict[str, Any]],
        scenario,
        planner: BasePlanner,
        controller: PurePursuitController,
        recorder: Optional[RecorderBase] = None,
        full_config: Optional[Dict[str, Any]] = None,
    ):
        cfg = runner_cfg or {}
        self.cfg = RunnerConfig(
            host=str(cfg.get("host", RunnerConfig.host)),
            port=int(cfg.get("port", RunnerConfig.port)),
            timeout_s=float(cfg.get("timeout_s", RunnerConfig.timeout_s)),
            sync=bool(cfg.get("sync", RunnerConfig.sync)),
            fixed_delta_seconds=float(cfg.get("fixed_delta_seconds", RunnerConfig.fixed_delta_seconds)),
            no_rendering_mode=bool(cfg.get("no_rendering_mode", RunnerConfig.no_rendering_mode)),
            max_steps=int(cfg.get("max_steps", RunnerConfig.max_steps)),
            debug_draw=bool(cfg.get("debug_draw", RunnerConfig.debug_draw)),
            follow_spectator=bool(cfg.get("follow_spectator", RunnerConfig.follow_spectator)),
            enable_recorder=bool(cfg.get("enable_recorder", RunnerConfig.enable_recorder)),
            obstacle_range_m=float(cfg.get("obstacle_range_m", RunnerConfig.obstacle_range_m)),
            max_obstacles=int(cfg.get("max_obstacles", RunnerConfig.max_obstacles)),
            ignore_static_parked=bool(cfg.get("ignore_static_parked", RunnerConfig.ignore_static_parked)),
        )

        self.scenario = scenario
        self.planner = planner
        self.controller = controller
        self.full_config: Dict[str, Any] = full_config or {}

        self.recorder: RecorderBase = recorder or self._make_default_recorder(cfg)
        self.metrics = self._make_metrics(cfg)
        self._metrics_started: bool = False

        self.client: Optional[carla.Client] = None
        self.world: Optional[carla.World] = None
        self.traffic_manager: Optional[carla.TrafficManager] = None
        self._collision: Optional[CollisionSensor] = None
        self._original_settings: Optional[carla.WorldSettings] = None

    # ---------------- main ----------------

    def run(self) -> Dict[str, Any]:
        self._connect()
        ego = self._setup_scenario_and_get_ego()
        self._enable_world_settings()

        # reset controller state (important when re-running)
        try:
            self.controller.reset()
        except Exception:
            pass

        # planner reset
        route = self.scenario.get_route()
        assert self.world is not None
        carla_map = self.world.get_map()
        self.planner.reset(route=route, 
                        map_info={
                            "map_name": carla_map.name,
                            "carla_map":carla_map,
                            },)
                            
        if self.cfg.debug_draw:
            self._draw_route(route)

        self._rec_start()

        start_wall = time.time()

        try:
            for step in range(self.cfg.max_steps):
                t_sim = self._tick(step)

                # allow scenario script/events
                if hasattr(self.scenario, "tick"):
                    try:
                        self.scenario.tick(float(t_sim))
                    except Exception:
                        pass

                ego_state = self._build_ego_state(ego)
                world_model = self._build_world_model(ego, ego_state)

                sensors = None
                if hasattr(self.scenario, "get_sensor_snapshot"):
                    try:
                        sensors = self.scenario.get_sensor_snapshot()
                    except Exception:
                        sensors = None

                plan_start = time.perf_counter()
                plan = self.planner.plan(ego=ego_state, world=world_model, t=float(t_sim))
                plan = self._attach_plan_timing(plan, start_t=plan_start)

                control = self._safe_stop_control()
                if self._is_plan_usable(plan):
                    if self.cfg.debug_draw:
                        self._draw_debug(plan.trajectory)
                    control = self.controller.compute_control(vehicle=ego, ego=ego_state, traj=plan.trajectory)
                # else: keep safe stop (throttle=0, brake=1)

                ego.apply_control(control)

                if self.cfg.follow_spectator:
                    self._spectator_follow(ego)

                # recorder: be tolerant to recorder implementations that don't accept sensors
                self._rec_step(
                    t_sim=float(t_sim),
                    step=int(step),
                    ego_state=ego_state,
                    world_model=world_model,
                    plan=plan,
                    control=control,
                    sensors=sensors,
                )
                self._metrics_step(
                    t_sim=float(t_sim),
                    step=int(step),
                    ego_state=ego_state,
                    plan=plan,
                    control=control,
                )

                done, out = self._check_termination(start_wall, step)
                if done:
                    return self._finalize_run(out)

            out = self._finish(start_wall, self.cfg.max_steps, reason="max_steps_reached")
            return self._finalize_run(out)

        finally:
            self._cleanup()

    # ---------------- recorder helpers ----------------

    def _make_default_recorder(self, runner_cfg: Dict[str, Any]) -> RecorderBase:
        if not self.cfg.enable_recorder:
            return NullRecorder()

        rec_cfg: Dict[str, Any] = {}
        if isinstance(runner_cfg.get("recorder"), dict):
            rec_cfg = runner_cfg["recorder"]
        elif isinstance(self.full_config.get("recorder"), dict):
            rec_cfg = self.full_config["recorder"]
        return RichRecorder(rec_cfg)

    def _make_metrics(self, runner_cfg: Dict[str, Any]) -> TrackingMetrics:
        met_cfg: Dict[str, Any] = {}
        if isinstance(runner_cfg.get("metrics"), dict):
            met_cfg = runner_cfg["metrics"]
        elif isinstance(self.full_config.get("metrics"), dict):
            met_cfg = self.full_config["metrics"]
        return TrackingMetrics(met_cfg)

    def _metrics_run_dir(self) -> Optional[Path]:
        # Use recorder's run dir so metrics outputs are saved together.
        run_dir = getattr(self.recorder, "run_dir", None)
        if isinstance(run_dir, Path):
            return run_dir
        return None

    def _rec_start(self) -> None:
        assert self.world is not None
        scenario_name = getattr(self.scenario, "name", "scenario")
        planner_name = getattr(self.planner, "name", self.planner.__class__.__name__)
        self.recorder.start(
            meta={
                "scenario": scenario_name,
                "planner": planner_name,
                "map": self.world.get_map().name,
                "sync": self.cfg.sync,
                "fixed_delta_seconds": self.cfg.fixed_delta_seconds,
                "no_rendering_mode": self.cfg.no_rendering_mode,
                "runner_cfg": vars(self.cfg),
                "config": self.full_config,
                "sensor_cfg": self.full_config.get("sensors", None),
            }
        
        )
        run_dir = self._metrics_run_dir()
        if run_dir is not None:
            self.metrics.start(run_dir=run_dir)
            self._metrics_started = True

    def _rec_step(
        self,
        *,
        t_sim: float,
        step: int,
        ego_state: EgoState,
        world_model: WorldModel,
        plan: Any,
        control: carla.VehicleControl,
        sensors: Optional[Dict[str, Any]],
    ) -> None:
        # recorder 兼容：有的 recorder.step 没有 sensors 参数
        try:
            self.recorder.step(
                t_sim=t_sim,
                step_idx=step,
                ego_state=ego_state,
                world_model=world_model,
                plan=plan,
                control=control,
                sensors=sensors,
            )
        except TypeError:
            # fallback: old signature
            self.recorder.step(
                t_sim=t_sim,
                step_idx=step,
                ego_state=ego_state,
                world_model=world_model,
                plan=plan,
                control=control,
            )

    def _metrics_step(
        self,
        *,
        t_sim: float,
        step: int,
        ego_state: EgoState,
        plan: Any,
        control: carla.VehicleControl,
    ) -> None:
        if not self._metrics_started:
            return
        traj = getattr(plan, "trajectory", None)
        target_speed = getattr(self.controller, "target_speed", None)
        self.metrics.step(
            t_sim=t_sim,
            step_idx=step,
            ego_state=ego_state,
            traj=traj,
            control=control,
            target_speed=float(target_speed) if isinstance(target_speed, (int, float)) else None,
        )

    def _finalize_run(self, out: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(out)
        if self._metrics_started:
            try:
                metrics_summary = self.metrics.finish()
                if metrics_summary:
                    result["metrics_summary"] = metrics_summary
            except Exception:
                pass
        self.recorder.finish(result=result)
        return result

    # ---------------- carla setup ----------------

    def _connect(self) -> None:
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout_s)

    def _setup_scenario_and_get_ego(self) -> carla.Vehicle:
        assert self.client is not None
        self.world = self.scenario.setup(self.client)
        assert self.world is not None

        ego = self.scenario.ego_vehicle
        if ego is None:
            raise RuntimeError("Scenario did not spawn ego_vehicle.")

        self._collision = CollisionSensor(self.world, ego)
        return ego

    def _enable_world_settings(self) -> None:
        assert self.world is not None
        self._original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = bool(self.cfg.sync)
        settings.fixed_delta_seconds = float(self.cfg.fixed_delta_seconds) if self.cfg.sync else None
        settings.no_rendering_mode = bool(self.cfg.no_rendering_mode)
        self.world.apply_settings(settings)

        if self.client is not None:
            self.traffic_manager = self.client.get_trafficmanager()
            self.traffic_manager.set_synchronous_mode(bool(self.cfg.sync))

    # ---------------- loop helpers ----------------

    def _tick(self, step: int) -> float:
        assert self.world is not None
        if self.cfg.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()
        return step * self.cfg.fixed_delta_seconds

    def _safe_stop_control(self) -> carla.VehicleControl:
        # always return a NEW object per tick
        return carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
    
    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)


    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)

    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)

    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)

    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)

    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)

    def _attach_plan_timing(self, plan: Any, *, start_t: float) -> Any:
        plan_ms = (time.perf_counter() - start_t) * 1000.0

        if not isinstance(plan, PlanResult):
            return plan

        debug = dict(plan.debug) if isinstance(plan.debug, dict) else {}
        timing = debug.get("timing")
        if not isinstance(timing, dict):
            timing = {}
        timing = dict(timing)
        timing["plan_total_ms"] = float(plan_ms)
        debug["timing"] = timing
        return replace(plan, debug=debug)

    def _is_plan_usable(self, plan: Any) -> bool:
        if getattr(plan, "status", None) != PlanStatus.OK:
            return False
        traj = getattr(plan, "trajectory", None)
        if traj is None:
            return False
        pts = getattr(traj, "points", None)
        return bool(pts)  # non-empty

    # ---------------- state builders ----------------

    def _build_ego_state(self, ego: carla.Vehicle) -> EgoState:
        tf = ego.get_transform()
        vel = ego.get_velocity()
        speed = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
        yaw = _wrap_pi(math.radians(tf.rotation.yaw))
        return EgoState(pose=Pose2D(x=tf.location.x, y=tf.location.y, yaw=yaw), speed=float(speed))

    def _build_world_model(self, ego: carla.Vehicle, ego_state: EgoState) -> WorldModel:
        """
        IMPORTANT:
        - Filter obstacles by distance to ego to avoid planner being killed by far-away vehicles.
        - Cap obstacle count.
        """
        assert self.world is not None

        ego_id = ego.id
        ex, ey = ego_state.pose.x, ego_state.pose.y
        R = float(self.cfg.obstacle_range_m)

        candidates = []
        for a in self.world.get_actors().filter("vehicle.*"):
            if a.id == ego_id:
                continue

            tf = a.get_transform()
            dx = float(tf.location.x) - ex
            dy = float(tf.location.y) - ey
            d = _hypot(dx, dy)
            if d > R:
                continue

            v = a.get_velocity()
            bb = a.bounding_box.extent
            radius = float(max(bb.x, bb.y)) + 0.5

            # optional: ignore almost-static parked cars (can be dangerous if used wrongly)
            if self.cfg.ignore_static_parked:
                speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                if speed < 0.3:
                    continue

            candidates.append((
                d,
                Obstacle(
                    id=int(a.id),
                    position=Vec3(float(tf.location.x), float(tf.location.y), float(tf.location.z)),
                    velocity=Vec3(float(v.x), float(v.y), float(v.z)),
                    radius=radius,
                )
            ))

        # sort by distance and cap
        candidates.sort(key=lambda x: x[0])
        obstacles = [ob for _, ob in candidates[: int(self.cfg.max_obstacles)]]

        return WorldModel(obstacles=obstacles)

    # ---------------- termination ----------------

    def _check_termination(self, start_wall: float, steps: int) -> tuple[bool, Dict[str, Any]]:
        if self._collision is not None and self._collision.collided:
            return True, self._finish(start_wall, steps, reason="collision")

        done, info = self.scenario.is_done()
        if done:
            reason = info.get("reason", "done") if isinstance(info, dict) else "done"
            extra = info if isinstance(info, dict) else None
            return True, self._finish(start_wall, steps, reason=reason, extra=extra)

        return False, {}

    def _finish(self, start_wall: float, steps: int, reason: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "reason": reason,
            "steps": int(steps),
            "sim_time_s": float(steps * self.cfg.fixed_delta_seconds),
            "wall_time_s": float(time.time() - start_wall),
        }
        if extra:
            out.update(extra)
        return out

    # ---------------- debug draw ----------------

    def _draw_debug(self, traj) -> None:
        assert self.world is not None
        dbg = self.world.debug
        for p in traj.points[::2]:
            dbg.draw_point(
                carla.Location(x=p.x, y=p.y, z=0.5),
                size=0.06,
                color=carla.Color(255, 50, 50),
                life_time=0.2,
            )

    def _draw_route(self, route) -> None:
        assert self.world is not None
        dbg = self.world.debug
        for p in route.points[::2]:
            dbg.draw_point(
                carla.Location(x=p.x, y=p.y, z=0.3),
                size=0.07,
                color=carla.Color(50, 255, 255),
                life_time=30.0,
            )

    def _spectator_follow(self, ego: carla.Vehicle) -> None:
        assert self.world is not None
        sp = self.world.get_spectator()
        tf = ego.get_transform()
        back = tf.get_forward_vector() * -8.0
        loc = tf.location + carla.Location(x=back.x, y=back.y, z=4.0)
        sp.set_transform(carla.Transform(loc, carla.Rotation(pitch=-15.0, yaw=tf.rotation.yaw)))

    # ---------------- cleanup ----------------

    def _cleanup(self) -> None:
        try:
            self.recorder.close()
        except Exception:
            pass

        try:
            if self.world is not None and self._original_settings is not None:
                self.world.apply_settings(self._original_settings)
        except Exception:
            pass

        try:
            if self._collision is not None:
                self._collision.destroy()
        except Exception:
            pass
        self._collision = None

        try:
            self.scenario.destroy()
        except Exception:
            pass
