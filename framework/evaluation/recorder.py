# framework/evaluation/recorder.py
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt

import carla

from framework.core.types import EgoState, WorldModel


@dataclass
class RecorderConfig:
    save_dir: str = "runs"
    run_name: Optional[str] = None          # if None -> auto (scenario_planner_timestamp)
    save_csv: bool = True
    save_json: bool = True

    # Plot switches
    plot_ego_xy: bool = True
    plot_controls: bool = True
    plot_speed: bool = True
    plot_plan_summary: bool = True          # cost/clearance/etc if present

    # How much to log
    record_world_obstacles: bool = False    # can get large; keep False by default
    max_obstacles: int = 50

    record_trajectory: str = "summary"      # "none" | "summary" | "full"
    max_traj_points: int = 200              # when record_trajectory="full"

    # Debug / perf
    flush_every_n: int = 0                  # 0 means only flush at finish
    

class Recorder:
    """
    A rich episode recorder that:
    - creates a run directory
    - snapshots meta/config
    - records per-tick data to memory
    - writes record.csv + record.json + result.json
    - saves useful plots (png)

    Compatible with the Runner's RecorderBase Protocol:
      start(meta), step(...), finish(result), close()

    Notes:
    - plan is treated as Any; we attempt to extract common fields:
        plan.status, plan.trajectory.points, plan.debug (dict)
    - world obstacles optional (can be large)
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.cfg = RecorderConfig(
            save_dir=str(c.get("save_dir", "runs")),
            run_name=c.get("run_name"),
            save_csv=bool(c.get("save_csv", True)),
            save_json=bool(c.get("save_json", True)),
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

    # -----------------------------
    # Runner hooks
    # -----------------------------
    def start(self, *, meta: Dict[str, Any]) -> None:
        self._meta = dict(meta)

        scenario = str(meta.get("scenario", "scenario"))
        planner = str(meta.get("planner", "planner"))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_name = self.cfg.run_name
        if not run_name:
            run_name = f"{scenario}_{planner}_{ts}"

        self._run_dir = Path(self.cfg.save_dir) / run_name
        self._run_dir.mkdir(parents=True, exist_ok=True)

        # snapshot meta/config early
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
        row["ego_x"] = float(ego_state.pose.x)
        row["ego_y"] = float(ego_state.pose.y)
        row["ego_yaw"] = float(ego_state.pose.yaw)
        row["ego_speed"] = float(ego_state.speed)

        # control
        row["ctrl_steer"] = float(control.steer)
        row["ctrl_throttle"] = float(control.throttle)
        row["ctrl_brake"] = float(control.brake)

        # plan status
        row["plan_status"] = str(getattr(plan, "status", "unknown"))

        # plan debug (best effort, flattened)
        debug = getattr(plan, "debug", None)
        if isinstance(debug, dict):
            self._merge_debug(row, debug)

        # trajectory info
        traj = getattr(plan, "trajectory", None)
        self._append_trajectory_fields(row, traj)

        # world obstacles (optional)
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

        self._rows.append(row)

        # optional periodic flush
        if self.cfg.flush_every_n > 0 and (len(self._rows) % self.cfg.flush_every_n == 0):
            self._flush_partial()

    def finish(self, *, result: Dict[str, Any]) -> None:
        if self._run_dir is None or self._closed:
            return

        # write result
        self._write_text("result.json", json.dumps(result, ensure_ascii=False, indent=2))

        # write records
        if self.cfg.save_json:
            self._write_text("record.json", json.dumps(self._rows, ensure_ascii=False, indent=2))

        if self.cfg.save_csv:
            self._write_csv("record.csv", self._rows)

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

        # CSV cannot store nested dict/list cleanly; JSON-stringify those values.
        def normalize(v: Any) -> Any:
            if isinstance(v, (dict, list)):
                return json.dumps(v, ensure_ascii=False)
            return v

        # union of keys (robust across ticks)
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
        """Best-effort partial flush to avoid losing everything if crash."""
        # Keep it simple: write a rolling JSONL.
        assert self._run_dir is not None
        path = self._run_dir / "record.jsonl"
        with path.open("a", encoding="utf-8") as f:
            # write last N records
            n = self.cfg.flush_every_n
            for r in self._rows[-n:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def _merge_debug(self, row: Dict[str, Any], debug: Dict[str, Any]) -> None:
        """
        Flatten known useful fields.
        Keep everything else under 'debug_*' keys (shallow).
        """
        # Common top-level metrics
        for k in ("best_cost", "num_candidates", "ego_s0", "ego_l0"):
            if k in debug:
                row[k] = debug.get(k)

        # Some planners pack a dict under 'best'
        best = debug.get("best")
        if isinstance(best, dict):
            # useful summary if present
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

        # Shallow flatten of simple scalar debug values
        for k, v in debug.items():
            if k in ("best",):
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

        # Record full trajectory (optional)
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

        # 1) Ego x-y
        if self.cfg.plot_ego_xy and self._all_numeric(xs) and self._all_numeric(ys):
            plt.figure()
            plt.plot(xs, ys, linewidth=1.5)
            plt.xlabel("x (m)")
            plt.ylabel("y (m)")
            plt.title("Ego trajectory (x-y)")
            plt.axis("equal")
            plt.grid(True)
            plt.savefig(self._run_dir / "ego_xy.png", dpi=200, bbox_inches="tight")
            plt.close()

            # add trajectory endpoints if present
            tx = [r.get("trajN_x") for r in self._rows if r.get("trajN_x") is not None and r.get("trajN_y") is not None]
            ty = [r.get("trajN_y") for r in self._rows if r.get("trajN_x") is not None and r.get("trajN_y") is not None]
            if self._all_numeric(tx) and self._all_numeric(ty) and tx and ty:
                plt.figure()
                plt.plot(xs, ys, linewidth=1.2, label="ego")
                plt.scatter(tx, ty, s=6, alpha=0.5, label="traj end (per tick)")
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.title("Ego trajectory + planner trajectory endpoints")
                plt.axis("equal")
                plt.grid(True)
                plt.legend()
                plt.savefig(self._run_dir / "ego_xy_with_traj_end.png", dpi=200, bbox_inches="tight")
                plt.close()

        # 2) Speed over time
        if self.cfg.plot_speed and self._all_numeric(t) and self._all_numeric(speed):
            plt.figure()
            plt.plot(t, speed, linewidth=1.5)
            plt.xlabel("t (s)")
            plt.ylabel("speed (m/s)")
            plt.title("Ego speed")
            plt.grid(True)
            plt.savefig(self._run_dir / "ego_speed.png", dpi=200, bbox_inches="tight")
            plt.close()

        # 3) Controls over time
        if self.cfg.plot_controls and self._all_numeric(t) and self._all_numeric(steer) and self._all_numeric(throttle) and self._all_numeric(brake):
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

        # 4) Plan summary plots (best_cost / clearance / curvature etc.)
        if self.cfg.plot_plan_summary and self._all_numeric(t):
            # best_cost
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

            # clearance
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

            # curvature
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

    @staticmethod
    def _all_numeric(vals: List[Any]) -> bool:
        """True if list contains only numbers or None; and at least one number."""
        any_num = False
        for v in vals:
            if v is None:
                continue
            if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                any_num = True
            else:
                return False
        return any_num
    
    @property
    def run_dir(self) -> Optional[Path]:
        return self._run_dir
