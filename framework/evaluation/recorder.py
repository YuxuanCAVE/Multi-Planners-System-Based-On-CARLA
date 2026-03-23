from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import carla

from framework.core.types import EgoState, WorldModel


@dataclass
class RecorderConfig:
    save_dir: str = "runs"
    run_name: Optional[str] = None  # if None -> auto (scenario_planner_timestamp)

    # raw outputs
    save_csv: bool = True
    save_json: bool = True
    save_tick_record: bool = True
    save_trajectory_csv: bool = True
    save_control_csv: bool = True
    save_plan_path_csv: bool = True
    save_summary_json: bool = True
    save_summary_csv: bool = True

    # plots
    plot_ego_xy: bool = True
    plot_controls: bool = True
    plot_speed: bool = True
    plot_plan_summary: bool = True

    # logging detail
    record_world_obstacles: bool = False
    max_obstacles: int = 50

    # plan trajectory recording
    # "none" | "summary" | "full"
    record_trajectory: str = "summary"
    max_traj_points: int = 200

    # debug / perf
    flush_every_n: int = 0  # 0 means only flush at finish


class Recorder:
    """
    Recorder for simulation episodes.

    Outputs:
    - meta.json
    - result.json
    - final_summary.json / final_summary.csv
    - record.json / record.csv              (per-tick summary)
    - executed_trajectory.csv               (ego trajectory)
    - control_cmd.csv                       (control commands)
    - planner_path_last.csv                 (last valid planner trajectory)
    - png plots

    Design notes:
    - Keep per-tick rows relatively flat and light.
    - Keep case-level summary separate for downstream batch evaluation.
    - Best-effort extraction from `plan.debug`, `result`, and `sensors`.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.cfg = RecorderConfig(
            save_dir=str(c.get("save_dir", "runs")),
            run_name=c.get("run_name"),
            save_csv=bool(c.get("save_csv", True)),
            save_json=bool(c.get("save_json", True)),
            save_tick_record=bool(c.get("save_tick_record", True)),
            save_trajectory_csv=bool(c.get("save_trajectory_csv", True)),
            save_control_csv=bool(c.get("save_control_csv", True)),
            save_plan_path_csv=bool(c.get("save_plan_path_csv", True)),
            save_summary_json=bool(c.get("save_summary_json", True)),
            save_summary_csv=bool(c.get("save_summary_csv", True)),
            plot_ego_xy=bool(c.get("plot_ego_xy", True)),
            plot_controls=bool(c.get("plot_controls", True)),
            plot_speed=bool(c.get("plot_speed", True)),
            plot_plan_summary=bool(c.get("plot_plan_summary", True)),
            record_world_obstacles=bool(c.get("record_world_obstacles", False)),
            max_obstacles=int(c.get("max_obstacles", 50)),
            record_trajectory=str(c.get("record_trajectory", "summary")),
            max_traj_points=int(c.get("max_traj_points", 200)),
            flush_every_n=int(c.get("flush_every_n", 0)),
        )

        self._run_dir: Optional[Path] = None
        self._meta: Dict[str, Any] = {}
        self._rows: List[Dict[str, Any]] = []
        self._closed: bool = False

        # dedicated raw series for easier downstream evaluation
        self._trajectory_rows: List[Dict[str, Any]] = []
        self._control_rows: List[Dict[str, Any]] = []
        self._last_plan_points: List[Dict[str, Any]] = []

        # episode-level aggregates
        self._event_state: Dict[str, Any] = {
            "collision": False,
            "collision_count": 0,
            "collision_time_s": None,
            "collision_actor_type": None,
            "lane_invasion": False,
            "lane_invasion_count": 0,
            "timeout": False,
            "reach_goal": False,
            "goal_reached_time_s": None,
            "simulation_abort": False,
            "simulation_abort_reason": None,
        }

        self._stats: Dict[str, Any] = {
            "min_obstacle_distance_m": None,
            "max_speed_mps": None,
            "runtime_s": 0.0,
            "num_steps": 0,
            "planning_success_once": False,
            "planning_fail_count": 0,
            "last_plan_status": None,
            "last_planning_time_s": None,
        }

    # -----------------------------
    # Runner hooks
    # -----------------------------
    def start(self, *, meta: Dict[str, Any]) -> None:
        self._meta = dict(meta)

        scenario = str(meta.get("scenario", meta.get("scenario_type", "scenario")))
        planner = str(meta.get("planner", "planner"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_name = self.cfg.run_name or f"{scenario}_{planner}_{ts}"

        self._run_dir = Path(self.cfg.save_dir) / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._write_text("meta.json", json.dumps(self._meta, ensure_ascii=False, indent=2))

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
        if self._run_dir is None or self._closed:
            return

        row: Dict[str, Any] = {}

        # time
        row["step"] = int(step_idx)
        row["t"] = float(t_sim)

        # ego state
        ego_x = float(ego_state.pose.x)
        ego_y = float(ego_state.pose.y)
        ego_yaw = float(ego_state.pose.yaw)
        ego_speed = float(ego_state.speed)

        row["ego_x"] = ego_x
        row["ego_y"] = ego_y
        row["ego_yaw"] = ego_yaw
        row["ego_speed"] = ego_speed

        # control
        row["ctrl_steer"] = float(control.steer)
        row["ctrl_throttle"] = float(control.throttle)
        row["ctrl_brake"] = float(control.brake)

        # plan status
        plan_status = str(getattr(plan, "status", "unknown"))
        row["plan_status"] = plan_status
        self._stats["last_plan_status"] = plan_status

        # debug (best effort)
        debug = getattr(plan, "debug", None)
        if isinstance(debug, dict):
            self._merge_debug(row, debug)

            timing = debug.get("timing")
            if isinstance(timing, dict):
                planning_time = self._first_number(
                    timing.get("total_ms"),
                    timing.get("plan_ms"),
                    timing.get("total_s"),
                    timing.get("plan_s"),
                )
                if planning_time is not None:
                    if planning_time > 10.0:  # likely ms
                        planning_time /= 1000.0
                    self._stats["last_planning_time_s"] = planning_time
                    row["planning_time_s"] = planning_time

        # planning success heuristic
        if str(plan_status).lower() in ("ok", "success", "valid", "running"):
            self._stats["planning_success_once"] = True
        elif str(plan_status).lower() in ("failed", "error", "invalid"):
            self._stats["planning_fail_count"] += 1

        # planner trajectory
        traj = getattr(plan, "trajectory", None)
        self._append_trajectory_fields(row, traj)
        self._update_last_plan_points(traj)

        # obstacle distance
        min_obs_dist = self._compute_min_obstacle_distance(ego_x, ego_y, world_model)
        row["min_obstacle_distance_m"] = min_obs_dist
        self._update_min_stat("min_obstacle_distance_m", min_obs_dist)

        # world obstacle payload
        if self.cfg.record_world_obstacles:
            row["num_obstacles"] = int(len(world_model.obstacles))
            obs_pack = []
            for ob in world_model.obstacles[: self.cfg.max_obstacles]:
                obs_pack.append(
                    {
                        "id": int(ob.id),
                        "x": float(ob.position.x),
                        "y": float(ob.position.y),
                        "vx": float(ob.velocity.x),
                        "vy": float(ob.velocity.y),
                        "r": float(ob.radius),
                    }
                )
            row["obstacles"] = obs_pack
        else:
            row["num_obstacles"] = int(len(world_model.obstacles))

        # sensors / events
        self._update_events_from_sensors(t_sim=t_sim, sensors=sensors, row=row)

        # dedicated raw streams
        self._trajectory_rows.append(
            {
                "step": int(step_idx),
                "t": float(t_sim),
                "x": ego_x,
                "y": ego_y,
                "yaw": ego_yaw,
                "speed_mps": ego_speed,
            }
        )

        self._control_rows.append(
            {
                "step": int(step_idx),
                "t": float(t_sim),
                "steer": float(control.steer),
                "throttle": float(control.throttle),
                "brake": float(control.brake),
            }
        )

        self._stats["runtime_s"] = float(t_sim)
        self._stats["num_steps"] = int(step_idx) + 1
        self._stats["max_speed_mps"] = self._max_or_init(self._stats["max_speed_mps"], ego_speed)

        self._rows.append(row)

        if self.cfg.flush_every_n > 0 and (len(self._rows) % self.cfg.flush_every_n == 0):
            self._flush_partial()

    def finish(self, *, result: Dict[str, Any]) -> None:
        if self._run_dir is None or self._closed:
            return

        self._merge_events_from_result(result)
        summary = self._build_final_summary(result)

        # raw result
        self._write_text("result.json", json.dumps(result, ensure_ascii=False, indent=2))

        # per-tick record
        if self.cfg.save_tick_record:
            if self.cfg.save_json:
                self._write_text("record.json", json.dumps(self._rows, ensure_ascii=False, indent=2))
            if self.cfg.save_csv:
                self._write_csv("record.csv", self._rows)

        # dedicated exports
        if self.cfg.save_trajectory_csv:
            self._write_csv("executed_trajectory.csv", self._trajectory_rows)

        if self.cfg.save_control_csv:
            self._write_csv("control_cmd.csv", self._control_rows)

        if self.cfg.save_plan_path_csv and self._last_plan_points:
            self._write_csv("planner_path_last.csv", self._last_plan_points)

        # summary
        if self.cfg.save_summary_json:
            self._write_text("final_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
        if self.cfg.save_summary_csv:
            self._write_csv("final_summary.csv", [summary])

        # plots
        self._save_plots()

    def close(self) -> None:
        self._closed = True

    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _write_text(self, name: str, text: str) -> None:
        assert self._run_dir is not None
        (self._run_dir / name).write_text(text, encoding="utf-8")

    def _write_csv(self, name: str, rows: List[Dict[str, Any]]) -> None:
        assert self._run_dir is not None
        if not rows:
            return

        def normalize(v: Any) -> Any:
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return v

        fieldnames: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

        path = self._run_dir / name
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: normalize(r.get(k)) for k in fieldnames})

    def _flush_partial(self) -> None:
        assert self._run_dir is not None
        path = self._run_dir / "record.jsonl"
        with path.open("a", encoding="utf-8") as f:
            n = self.cfg.flush_every_n
            for r in self._rows[-n:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _merge_debug(self, row: Dict[str, Any], debug: Dict[str, Any]) -> None:
        for k in ("best_cost", "num_candidates", "ego_s0", "ego_l0"):
            if k in debug:
                row[k] = debug.get(k)

        best = debug.get("best")
        if isinstance(best, dict):
            for k in (
                "valid",
                "min_clearance_m",
                "max_curvature",
                "max_yaw_rate",
                "cost_total",
                "cost_offset",
                "cost_speed",
                "cost_curvature",
                "cost_collision",
                "l_target",
                "v_target",
                "collision",
            ):
                if k in best:
                    row[f"best_{k}"] = best.get(k)

        timing = debug.get("timing")
        if isinstance(timing, dict):
            for k, v in timing.items():
                if isinstance(v, (int, float)):
                    row[f"timing_{k}"] = float(v)

        for k, v in debug.items():
            if k in ("best", "timing"):
                continue
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[f"debug_{k}"] = v

    def _append_trajectory_fields(self, row: Dict[str, Any], traj: Any) -> None:
        if traj is None:
            row["traj_len"] = 0
            return

        pts = getattr(traj, "points", None)
        dt = getattr(traj, "dt", None)
        if dt is not None:
            row["traj_dt"] = float(dt)

        if not pts:
            row["traj_len"] = 0
            return

        row["traj_len"] = int(len(pts))

        p0 = pts[0]
        pN = pts[-1]
        for prefix, p in (("traj0", p0), ("trajN", pN)):
            row[f"{prefix}_x"] = float(getattr(p, "x", 0.0))
            row[f"{prefix}_y"] = float(getattr(p, "y", 0.0))
            if hasattr(p, "yaw"):
                row[f"{prefix}_yaw"] = float(getattr(p, "yaw"))
            if hasattr(p, "v"):
                row[f"{prefix}_v"] = float(getattr(p, "v"))

        if self.cfg.record_trajectory == "full":
            pack = []
            for p in pts[: self.cfg.max_traj_points]:
                pack.append(
                    {
                        "x": float(getattr(p, "x", 0.0)),
                        "y": float(getattr(p, "y", 0.0)),
                        "yaw": float(getattr(p, "yaw", 0.0)) if hasattr(p, "yaw") else 0.0,
                        "v": float(getattr(p, "v", 0.0)) if hasattr(p, "v") else 0.0,
                    }
                )
            row["traj_points"] = pack

    def _update_last_plan_points(self, traj: Any) -> None:
        pts = getattr(traj, "points", None) if traj is not None else None
        if not pts:
            return

        pack: List[Dict[str, Any]] = []
        for i, p in enumerate(pts[: self.cfg.max_traj_points]):
            pack.append(
                {
                    "idx": i,
                    "x": float(getattr(p, "x", 0.0)),
                    "y": float(getattr(p, "y", 0.0)),
                    "yaw": float(getattr(p, "yaw", 0.0)) if hasattr(p, "yaw") else 0.0,
                    "v": float(getattr(p, "v", 0.0)) if hasattr(p, "v") else 0.0,
                }
            )
        self._last_plan_points = pack

    def _compute_min_obstacle_distance(
        self,
        ego_x: float,
        ego_y: float,
        world_model: WorldModel,
    ) -> Optional[float]:
        if not getattr(world_model, "obstacles", None):
            return None

        best: Optional[float] = None
        for ob in world_model.obstacles:
            dx = float(ob.position.x) - ego_x
            dy = float(ob.position.y) - ego_y
            d = math.hypot(dx, dy) - float(getattr(ob, "radius", 0.0))
            if best is None or d < best:
                best = d
        return best

    def _update_events_from_sensors(
        self,
        *,
        t_sim: float,
        sensors: Optional[Dict[str, Any]],
        row: Dict[str, Any],
    ) -> None:
        if not isinstance(sensors, dict):
            return

        # collision
        collision = self._coerce_bool(
            sensors.get("collision"),
            sensors.get("has_collision"),
            sensors.get("collision_detected"),
        )
        if collision:
            self._event_state["collision"] = True
            self._event_state["collision_count"] += 1
            if self._event_state["collision_time_s"] is None:
                self._event_state["collision_time_s"] = float(t_sim)
            if self._event_state["collision_actor_type"] is None:
                actor_type = sensors.get("collision_actor_type")
                if actor_type is not None:
                    self._event_state["collision_actor_type"] = str(actor_type)

        # lane invasion
        lane_inv = self._coerce_bool(
            sensors.get("lane_invasion"),
            sensors.get("lane_invaded"),
        )
        if lane_inv:
            self._event_state["lane_invasion"] = True
            self._event_state["lane_invasion_count"] += 1

        # optional direct counters
        cnt = self._safe_int(sensors.get("collision_count"))
        if cnt is not None:
            self._event_state["collision_count"] = max(self._event_state["collision_count"], cnt)
            if cnt > 0:
                self._event_state["collision"] = True
                if self._event_state["collision_time_s"] is None:
                    self._event_state["collision_time_s"] = float(t_sim)

        lane_cnt = self._safe_int(sensors.get("lane_invasion_count"))
        if lane_cnt is not None:
            self._event_state["lane_invasion_count"] = max(self._event_state["lane_invasion_count"], lane_cnt)
            if lane_cnt > 0:
                self._event_state["lane_invasion"] = True

        # copy some event flags to row for convenience
        row["event_collision"] = self._event_state["collision"]
        row["event_collision_count"] = self._event_state["collision_count"]
        row["event_lane_invasion"] = self._event_state["lane_invasion"]
        row["event_lane_invasion_count"] = self._event_state["lane_invasion_count"]

    def _merge_events_from_result(self, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict):
            return

        for src_key, dst_key in (
            ("collision", "collision"),
            ("timeout", "timeout"),
            ("reach_goal", "reach_goal"),
            ("simulation_abort", "simulation_abort"),
        ):
            if src_key in result:
                self._event_state[dst_key] = bool(result.get(src_key))

        if "collision_count" in result and self._safe_int(result.get("collision_count")) is not None:
            self._event_state["collision_count"] = max(
                self._event_state["collision_count"],
                int(result["collision_count"]),
            )
            if self._event_state["collision_count"] > 0:
                self._event_state["collision"] = True

        if "goal_reached_time_s" in result and result["goal_reached_time_s"] is not None:
            self._event_state["goal_reached_time_s"] = float(result["goal_reached_time_s"])
            self._event_state["reach_goal"] = True

        if "fail_reason" in result and result["fail_reason"]:
            self._event_state["simulation_abort_reason"] = str(result["fail_reason"])

    def _build_final_summary(self, result: Dict[str, Any]) -> Dict[str, Any]:
        last_row = self._rows[-1] if self._rows else {}
        last_traj = self._trajectory_rows[-1] if self._trajectory_rows else {}

        goal_pose = self._extract_goal_pose(self._meta)
        final_x = self._safe_float(last_traj.get("x"))
        final_y = self._safe_float(last_traj.get("y"))
        final_yaw = self._safe_float(last_traj.get("yaw"))

        final_position_error_m = None
        final_heading_error_rad = None
        if goal_pose is not None and final_x is not None and final_y is not None:
            gx, gy, gyaw = goal_pose
            final_position_error_m = math.hypot(final_x - gx, final_y - gy)
            if final_yaw is not None and gyaw is not None:
                final_heading_error_rad = self._wrap_pi(final_yaw - gyaw)

        planning_success = bool(
            self._stats["planning_success_once"]
            or str(self._stats["last_plan_status"]).lower() in ("ok", "success", "valid", "running")
        )

        fail_reason = result.get("fail_reason") if isinstance(result, dict) else None
        if not fail_reason:
            fail_reason = self._infer_fail_reason(
                planning_success=planning_success,
                collision=self._event_state["collision"],
                timeout=self._event_state["timeout"],
                reach_goal=self._event_state["reach_goal"],
            )

        passed = (
            planning_success
            and not self._event_state["collision"]
            and not self._event_state["timeout"]
            and self._event_state["reach_goal"]
        )

        summary = {
            # meta
            "run_name": self._run_dir.name if self._run_dir is not None else None,
            "scenario": self._meta.get("scenario"),
            "scenario_type": self._meta.get("scenario_type", self._meta.get("scenario")),
            "planner": self._meta.get("planner"),
            "controller": self._meta.get("controller"),
            "map_name": self._meta.get("map_name", self._meta.get("map")),
            "case_id": self._meta.get("case_id"),
            "target_speed_mps": self._safe_float(
                self._meta.get("target_speed_mps", self._meta.get("target_speed"))
            ),
            # task level
            "pass": passed,
            "fail_reason": fail_reason,
            "runtime_s": self._safe_float(self._stats["runtime_s"]),
            "num_steps": self._safe_int(self._stats["num_steps"]),
            # planner
            "planning_success": planning_success,
            "planning_fail_count": self._safe_int(self._stats["planning_fail_count"]),
            "planning_time_s": self._safe_float(
                result.get("planning_time_s", self._stats["last_planning_time_s"])
                if isinstance(result, dict)
                else self._stats["last_planning_time_s"]
            ),
            "plan_status": self._stats["last_plan_status"],
            "traj_len": self._safe_int(last_row.get("traj_len")),
            # events
            "collision": self._event_state["collision"],
            "collision_count": self._event_state["collision_count"],
            "collision_time_s": self._event_state["collision_time_s"],
            "collision_actor_type": self._event_state["collision_actor_type"],
            "lane_invasion": self._event_state["lane_invasion"],
            "lane_invasion_count": self._event_state["lane_invasion_count"],
            "timeout": self._event_state["timeout"],
            "reach_goal": self._event_state["reach_goal"],
            "goal_reached_time_s": self._event_state["goal_reached_time_s"],
            # final state
            "final_x": final_x,
            "final_y": final_y,
            "final_yaw": final_yaw,
            "final_speed_mps": self._safe_float(last_traj.get("speed_mps")),
            "final_position_error_m": final_position_error_m,
            "final_heading_error_rad": final_heading_error_rad,
            # safety / misc
            "min_obstacle_distance_m": self._safe_float(self._stats["min_obstacle_distance_m"]),
            "max_speed_mps": self._safe_float(self._stats["max_speed_mps"]),
        }

        # merge selected scalar result fields
        if isinstance(result, dict):
            for k, v in result.items():
                if k in summary:
                    continue
                if isinstance(v, (int, float, str, bool)) or v is None:
                    summary[f"result_{k}"] = v

        return summary

    def _extract_goal_pose(self, meta: Dict[str, Any]) -> Optional[Tuple[float, float, Optional[float]]]:
        candidates = [
            meta.get("goal_pose"),
            meta.get("goal"),
            meta.get("target_pose"),
        ]
        for g in candidates:
            if isinstance(g, dict):
                x = self._safe_float(g.get("x"))
                y = self._safe_float(g.get("y"))
                yaw = self._safe_float(g.get("yaw"))
                if x is not None and y is not None:
                    return x, y, yaw
            elif isinstance(g, (list, tuple)) and len(g) >= 2:
                x = self._safe_float(g[0])
                y = self._safe_float(g[1])
                yaw = self._safe_float(g[2]) if len(g) >= 3 else None
                if x is not None and y is not None:
                    return x, y, yaw
        return None

    def _infer_fail_reason(
        self,
        *,
        planning_success: bool,
        collision: bool,
        timeout: bool,
        reach_goal: bool,
    ) -> Optional[str]:
        if collision:
            return "collision"
        if timeout:
            return "timeout"
        if not planning_success:
            return "planning_failed"
        if not reach_goal:
            return "goal_not_reached"
        return None

    # -----------------------------
    # Plotting
    # -----------------------------
    def _save_plots(self) -> None:
        if self._run_dir is None or not self._rows:
            return

        t = [r.get("t") for r in self._rows]
        xs = [r.get("ego_x") for r in self._rows]
        ys = [r.get("ego_y") for r in self._rows]
        speed = [r.get("ego_speed") for r in self._rows]
        steer = [r.get("ctrl_steer") for r in self._rows]
        throttle = [r.get("ctrl_throttle") for r in self._rows]
        brake = [r.get("ctrl_brake") for r in self._rows]

        if self.cfg.plot_ego_xy and self._all_numeric(xs) and self._all_numeric(ys):
            plt.figure()
            plt.plot(xs, ys, linewidth=1.5, label="ego")
            if self._last_plan_points:
                px = [p["x"] for p in self._last_plan_points]
                py = [p["y"] for p in self._last_plan_points]
                plt.plot(px, py, linewidth=1.2, linestyle="--", label="last_plan")
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("Ego trajectory")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.savefig(self._run_dir / "ego_xy.png", dpi=200, bbox_inches="tight")
            plt.close()

        if self.cfg.plot_speed and self._all_numeric(t) and self._all_numeric(speed):
            plt.figure()
            plt.plot(t, speed, linewidth=1.5)
            plt.xlabel("t (s)")
            plt.ylabel("speed (m/s)")
            plt.title("Ego speed")
            plt.grid(True)
            plt.savefig(self._run_dir / "ego_speed.png", dpi=200, bbox_inches="tight")
            plt.close()

        if (
            self.cfg.plot_controls
            and self._all_numeric(t)
            and self._all_numeric(steer)
            and self._all_numeric(throttle)
            and self._all_numeric(brake)
        ):
            plt.figure()
            plt.plot(t, steer, linewidth=1.2, label="steer")
            plt.plot(t, throttle, linewidth=1.2, label="throttle")
            plt.plot(t, brake, linewidth=1.2, label="brake")
            plt.xlabel("t (s)")
            plt.ylabel("control")
            plt.title("Controls")
            plt.grid(True)
            plt.legend()
            plt.savefig(self._run_dir / "controls.png", dpi=200, bbox_inches="tight")
            plt.close()

        if self.cfg.plot_plan_summary and self._all_numeric(t):
            best_cost = [r.get("best_cost") for r in self._rows]
            if self._all_numeric(best_cost) and any(v is not None for v in best_cost):
                plt.figure()
                plt.plot(t, best_cost, linewidth=1.5)
                plt.xlabel("t (s)")
                plt.ylabel("best_cost")
                plt.title("Planner best cost")
                plt.grid(True)
                plt.savefig(self._run_dir / "planner_best_cost.png", dpi=200, bbox_inches="tight")
                plt.close()

            clearance = [r.get("best_min_clearance_m") for r in self._rows]
            if self._all_numeric(clearance) and any(v is not None for v in clearance):
                plt.figure()
                plt.plot(t, clearance, linewidth=1.5)
                plt.xlabel("t (s)")
                plt.ylabel("min clearance (m)")
                plt.title("Planner min clearance (best)")
                plt.grid(True)
                plt.savefig(self._run_dir / "planner_min_clearance.png", dpi=200, bbox_inches="tight")
                plt.close()

            curv = [r.get("best_max_curvature") for r in self._rows]
            if self._all_numeric(curv) and any(v is not None for v in curv):
                plt.figure()
                plt.plot(t, curv, linewidth=1.5)
                plt.xlabel("t (s)")
                plt.ylabel("max curvature (1/m)")
                plt.title("Planner max curvature (best)")
                plt.grid(True)
                plt.savefig(self._run_dir / "planner_max_curvature.png", dpi=200, bbox_inches="tight")
                plt.close()

            min_obs = [r.get("min_obstacle_distance_m") for r in self._rows]
            if self._all_numeric(min_obs) and any(v is not None for v in min_obs):
                plt.figure()
                plt.plot(t, min_obs, linewidth=1.5)
                plt.xlabel("t (s)")
                plt.ylabel("distance (m)")
                plt.title("Minimum obstacle distance")
                plt.grid(True)
                plt.savefig(self._run_dir / "min_obstacle_distance.png", dpi=200, bbox_inches="tight")
                plt.close()

    # -----------------------------
    # Small utilities
    # -----------------------------
    def _update_min_stat(self, key: str, value: Optional[float]) -> None:
        if value is None:
            return
        cur = self._stats.get(key)
        if cur is None or value < cur:
            self._stats[key] = value

    @staticmethod
    def _max_or_init(cur: Optional[float], val: float) -> float:
        return val if cur is None else max(cur, val)

    @staticmethod
    def _safe_float(v: Any) -> Optional[float]:
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(v: Any) -> Optional[int]:
        try:
            return None if v is None else int(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_bool(*vals: Any) -> bool:
        for v in vals:
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                if v.lower() in ("1", "true", "yes", "y"):
                    return True
                if v.lower() in ("0", "false", "no", "n"):
                    return False
        return False

    @staticmethod
    def _first_number(*vals: Any) -> Optional[float]:
        for v in vals:
            if isinstance(v, (int, float)):
                return float(v)
        return None

    @staticmethod
    def _wrap_pi(a: float) -> float:
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a

    @staticmethod
    def _all_numeric(vals: List[Any]) -> bool:
        any_num = False
        for v in vals:
            if v is None:
                continue
            if isinstance(v, (int, float)) and not (
                isinstance(v, float) and (math.isnan(v) or math.isinf(v))
            ):
                any_num = True
            else:
                return False
        return any_num

    @property
    def run_dir(self) -> Optional[Path]:
        return self._run_dir