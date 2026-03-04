# framework/evaluation/metrics.py
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

import carla
from framework.core.types import EgoState, Trajectory


def _wrap_pi(a: float) -> float:
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


@dataclass
class MetricsConfig:
    enabled: bool = True

    # reference selection
    use_lookahead: bool = True
    lookahead_base: float = 2.0
    lookahead_gain: float = 0.2

    # speed reference
    use_traj_speed_ref: bool = False  # if True: v_ref = traj.points[ref_i].v else v_ref = target_speed (if provided)

    # outputs
    save_csv: bool = True
    save_json: bool = True
    plot_errors: bool = True
    plot_controls: bool = True


class TrackingMetrics:
    """
    Controller-agnostic metrics:
    - cross-track error (signed)
    - heading error
    - speed error (optional)
    - control effort / jitter / saturation stats
    """

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        c = cfg or {}
        self.cfg = MetricsConfig(
            enabled=bool(c.get("enabled", True)),
            use_lookahead=bool(c.get("use_lookahead", True)),
            lookahead_base=float(c.get("lookahead_base", 2.0)),
            lookahead_gain=float(c.get("lookahead_gain", 0.2)),
            use_traj_speed_ref=bool(c.get("use_traj_speed_ref", False)),
            save_csv=bool(c.get("save_csv", True)),
            save_json=bool(c.get("save_json", True)),
            plot_errors=bool(c.get("plot_errors", True)),
            plot_controls=bool(c.get("plot_controls", True)),
        )
        self._run_dir: Optional[Path] = None
        self._rows: List[Dict[str, Any]] = []

        # for jitter metrics
        self._prev_steer: Optional[float] = None
        self._prev_throttle: Optional[float] = None
        self._prev_brake: Optional[float] = None

    def start(self, *, run_dir: Path) -> None:
        self._run_dir = run_dir
        self._rows = []
        self._prev_steer = None
        self._prev_throttle = None
        self._prev_brake = None

    def step(
        self,
        *,
        t_sim: float,
        step_idx: int,
        ego_state: EgoState,
        traj: Optional[Trajectory],
        control: Optional[carla.VehicleControl] = None,
        target_speed: Optional[float] = None,
    ) -> None:
        if not self.cfg.enabled or self._run_dir is None:
            return

        row: Dict[str, Any] = {"step": int(step_idx), "t": float(t_sim)}

        # ego
        row["ego_x"] = float(ego_state.pose.x)
        row["ego_y"] = float(ego_state.pose.y)
        row["ego_yaw"] = float(ego_state.pose.yaw)
        row["ego_speed"] = float(ego_state.speed)

        # reference errors
        if traj is not None and traj.points:
            cte_m, heading_err_rad, ref_i = self._compute_tracking_errors(ego_state, traj)
            row["cte_m"] = float(cte_m)
            row["heading_err_rad"] = float(heading_err_rad)
            row["ref_idx"] = int(ref_i)
            ref_p = traj.points[ref_i]
            row["ref_x"] = float(ref_p.x)
            row["ref_y"] = float(ref_p.y)
            row["ref_yaw"] = float(ref_p.yaw)

            # speed reference (optional)
            v_ref: Optional[float] = None
            if self.cfg.use_traj_speed_ref and hasattr(ref_p, "v"):
                v_ref = float(ref_p.v)
            elif target_speed is not None:
                v_ref = float(target_speed)

            if v_ref is not None:
                row["v_ref"] = float(v_ref)
                row["speed_err_mps"] = float(v_ref - ego_state.speed)
        else:
            row["cte_m"] = None
            row["heading_err_rad"] = None
            row["ref_idx"] = None

        # control metrics (optional)
        if control is not None:
            row["steer"] = float(control.steer)
            row["throttle"] = float(control.throttle)
            row["brake"] = float(control.brake)

            # saturation flags
            row["steer_sat"] = int(abs(control.steer) >= 0.999)
            row["throttle_sat"] = int(control.throttle >= 0.999)
            row["brake_sat"] = int(control.brake >= 0.999)

            # jitter (first difference)
            if self._prev_steer is not None:
                row["d_steer"] = float(control.steer - self._prev_steer)
            if self._prev_throttle is not None:
                row["d_throttle"] = float(control.throttle - self._prev_throttle)
            if self._prev_brake is not None:
                row["d_brake"] = float(control.brake - self._prev_brake)

            self._prev_steer = float(control.steer)
            self._prev_throttle = float(control.throttle)
            self._prev_brake = float(control.brake)

        self._rows.append(row)

    def finish(self) -> Dict[str, Any]:
        if not self.cfg.enabled or self._run_dir is None:
            return {}

        summary = self._summarize()

        if self.cfg.save_json:
            (self._run_dir / "metrics.json").write_text(
                json.dumps(self._rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (self._run_dir / "metrics_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        if self.cfg.save_csv:
            self._write_csv(self._run_dir / "metrics.csv", self._rows)

        if self.cfg.plot_errors:
            self._plot_errors()

        if self.cfg.plot_controls:
            self._plot_controls()

        return summary

    # -------------------------
    # Internals
    # -------------------------
    def _compute_tracking_errors(self, ego_state: EgoState, traj: Trajectory) -> Tuple[float, float, int]:
        ex, ey = float(ego_state.pose.x), float(ego_state.pose.y)
        ego_yaw = float(ego_state.pose.yaw)
        v = float(ego_state.speed)

        # nearest point
        nearest_i = 0
        best_d2 = float("inf")
        for i, p in enumerate(traj.points):
            dx = p.x - ex
            dy = p.y - ey
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                nearest_i = i

        ref_i = nearest_i
        if self.cfg.use_lookahead:
            Ld = max(0.5, self.cfg.lookahead_base + self.cfg.lookahead_gain * v)
            for i in range(nearest_i, len(traj.points)):
                p = traj.points[i]
                if math.hypot(p.x - ex, p.y - ey) >= Ld:
                    ref_i = i
                    break

        ref_p = traj.points[ref_i]
        path_yaw = float(ref_p.yaw)  # radians (your type confirmed)

        heading_err = _wrap_pi(path_yaw - ego_yaw)

        # signed cross-track error: left normal of path tangent
        n_x = -math.sin(path_yaw)
        n_y = math.cos(path_yaw)
        dx = ex - ref_p.x
        dy = ey - ref_p.y
        cte = dx * n_x + dy * n_y

        return cte, heading_err, ref_i

    def _summarize(self) -> Dict[str, Any]:
        def only_nums(key: str) -> List[float]:
            out = []
            for r in self._rows:
                v = r.get(key)
                if isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                    out.append(float(v))
            return out

        cte = only_nums("cte_m")
        hd = only_nums("heading_err_rad")
        se = only_nums("speed_err_mps")
        dsteer = only_nums("d_steer")

        def rmse(xs: List[float]) -> Optional[float]:
            if not xs:
                return None
            return math.sqrt(sum(x * x for x in xs) / len(xs))

        def mean(xs: List[float]) -> Optional[float]:
            if not xs:
                return None
            return sum(xs) / len(xs)

        def max_abs(xs: List[float]) -> Optional[float]:
            if not xs:
                return None
            return max(abs(x) for x in xs)

        steer_sat = sum(int(r.get("steer_sat", 0)) for r in self._rows)
        throttle_sat = sum(int(r.get("throttle_sat", 0)) for r in self._rows)
        brake_sat = sum(int(r.get("brake_sat", 0)) for r in self._rows)

        return {
            "num_samples": int(len(self._rows)),
            "cte_rmse_m": rmse(cte),
            "cte_max_abs_m": max_abs(cte),
            "heading_rmse_rad": rmse(hd),
            "heading_max_abs_rad": max_abs(hd),
            "speed_err_rmse_mps": rmse(se),
            "d_steer_rmse": rmse(dsteer),  # rough jitter indicator
            "steer_saturation_count": int(steer_sat),
            "throttle_saturation_count": int(throttle_sat),
            "brake_saturation_count": int(brake_sat),
            "cte_mean_m": mean(cte),
        }

    def _write_csv(self, path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        # union of keys (robust)
        fieldnames: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    fieldnames.append(k)

        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fieldnames})

    def _plot_errors(self) -> None:
        assert self._run_dir is not None
        t = [r.get("t") for r in self._rows if r.get("t") is not None]
        cte = [r.get("cte_m") for r in self._rows if r.get("cte_m") is not None]
        hd = [r.get("heading_err_rad") for r in self._rows if r.get("heading_err_rad") is not None]
        if t and cte:
            plt.figure()
            plt.plot(t[: len(cte)], cte, linewidth=1.5)
            plt.xlabel("t (s)")
            plt.ylabel("CTE (m)")
            plt.title("Tracking: Cross-Track Error (signed)")
            plt.grid(True)
            plt.savefig(self._run_dir / "metrics_cte.png", dpi=200, bbox_inches="tight")
            plt.close()
        if t and hd:
            plt.figure()
            plt.plot(t[: len(hd)], hd, linewidth=1.5)
            plt.xlabel("t (s)")
            plt.ylabel("Heading error (rad)")
            plt.title("Tracking: Heading Error")
            plt.grid(True)
            plt.savefig(self._run_dir / "metrics_heading_error.png", dpi=200, bbox_inches="tight")
            plt.close()

    def _plot_controls(self) -> None:
        assert self._run_dir is not None
        t = [r.get("t") for r in self._rows if r.get("t") is not None]
        steer = [r.get("steer") for r in self._rows if r.get("steer") is not None]
        if t and steer:
            plt.figure()
            plt.plot(t[: len(steer)], steer, linewidth=1.2)
            plt.xlabel("t (s)")
            plt.ylabel("steer")
            plt.title("Control: Steer")
            plt.grid(True)
            plt.savefig(self._run_dir / "metrics_steer.png", dpi=200, bbox_inches="tight")
            plt.close()