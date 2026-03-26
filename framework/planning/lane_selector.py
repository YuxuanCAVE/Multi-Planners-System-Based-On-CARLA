from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

@dataclass
class LeadInfo:
    actor_id: Any
    x: float
    y: float
    yaw: float
    speed: float
    longitudinal: float
    lateral: float
    distance: float
    raw_length_m: float


@dataclass
class LaneSelectorOutput:
    target_lane_offset_m: float
    reason: str
    tracked_lead_id: Any
    keep_cost: float
    pass_cost: float
    switched: bool


class LaneSelector:
    """
    Lightweight lane target selector.

    It is NOT a heavy behavior state machine.
    It only selects which lane center local planner should follow:
      - keep current lane center
      - pass lane center

    Anti-oscillation tools:
      1) dual margins for switch / return
      2) multi-frame confirmation
      3) minimum hold time
      4) geometric completion gates
      5) tracked lead memory
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.config = cfg

        self.keep_lane_offset_m = float(cfg.get("keep_lane_offset_m", 0.0))
        self.pass_lane_offset_m = float(cfg.get("pass_lane_offset_m", -3.5))

        # lead detection
        self.lead_detect_forward_m = float(cfg.get("lead_detect_forward_m", 50.0))
        self.lead_lane_half_width_m = float(cfg.get("lead_lane_half_width_m", 3.0))

        # trigger preference
        self.pass_trigger_dist_m = float(cfg.get("pass_trigger_dist_m", 22.0))
        self.min_speed_gain_to_pass_mps = float(cfg.get("min_speed_gain_to_pass_mps", 0.2))

        # anti-oscillation
        self.lane_switch_margin = float(cfg.get("lane_switch_margin", 8.0))
        self.lane_return_margin = float(cfg.get("lane_return_margin", 6.0))
        self.confirm_frames = int(cfg.get("lane_selector_confirm_frames", 3))
        self.min_hold_time_s = float(cfg.get("lane_selector_min_hold_time_s", 1.5))

        # geometry gates
        self.lane_center_reached_tol_m = float(cfg.get("lane_center_reached_tol_m", 0.6))
        self.return_pass_gap_m = float(cfg.get("return_pass_gap_m", 6.0))
        self.parallel_lateral_guard_m = float(cfg.get("parallel_lateral_guard_m", 2.2))

        # tracked lead persistence
        self.tracked_lead_keep_forward_m = float(cfg.get("tracked_lead_keep_forward_m", 45.0))
        self.tracked_lead_keep_backward_m = float(cfg.get("tracked_lead_keep_backward_m", 15.0))
        self.tracked_lead_keep_lateral_m = float(cfg.get("tracked_lead_keep_lateral_m", 8.0))
        self.tracked_lead_max_lost_cycles = int(cfg.get("tracked_lead_max_lost_cycles", 8))
        self.tracked_lead_release_gap_m = float(cfg.get("tracked_lead_release_gap_m", 8.0))

        self.reset()

    def reset(self) -> None:
        self.current_target_lane_offset_m: float = self.keep_lane_offset_m
        self.last_switch_t: float = -1e9

        self.left_better_count: int = 0
        self.keep_better_count: int = 0

        self.tracked_lead_actor_id: Any = None
        self.tracked_lead_lost_count: int = 0

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        ego: Any,
        world: Any,
        route_pts: List[Any],
        t: float,
    ) -> LaneSelectorOutput:
        tracked_lead = self._update_tracked_lead(ego=ego, world=world, route_pts=route_pts)

        keep_cost, pass_cost = self._estimate_lane_costs(
            ego=ego,
            tracked_lead=tracked_lead,
            route_pts=route_pts,
        )

        switched = False
        reason = "keep_current_target"

        time_ok = (t - self.last_switch_t) >= self.min_hold_time_s

        if self.current_target_lane_offset_m == self.keep_lane_offset_m:
            should_consider_pass = tracked_lead is not None and pass_cost + self.lane_switch_margin < keep_cost

            if should_consider_pass and time_ok:
                self.left_better_count += 1
            else:
                self.left_better_count = 0

            if self.left_better_count >= self.confirm_frames:
                self.current_target_lane_offset_m = self.pass_lane_offset_m
                self.last_switch_t = t
                self.left_better_count = 0
                self.keep_better_count = 0
                switched = True
                reason = "switch_to_pass_lane"
            else:
                reason = "stay_keep_lane"

        else:
            can_return = self._can_return_to_keep_lane(
                ego=ego,
                tracked_lead=tracked_lead,
                route_pts=route_pts,
            )
            should_return = can_return and keep_cost + self.lane_return_margin < pass_cost

            if should_return and time_ok:
                self.keep_better_count += 1
            else:
                self.keep_better_count = 0

            if self.keep_better_count >= self.confirm_frames:
                self.current_target_lane_offset_m = self.keep_lane_offset_m
                self.last_switch_t = t
                self.keep_better_count = 0
                self.left_better_count = 0
                switched = True
                reason = "return_to_keep_lane"
            else:
                reason = "stay_pass_lane"

        return LaneSelectorOutput(
            target_lane_offset_m=self.current_target_lane_offset_m,
            reason=reason,
            tracked_lead_id=None if tracked_lead is None else tracked_lead.actor_id,
            keep_cost=keep_cost,
            pass_cost=pass_cost,
            switched=switched,
        )

    # ------------------------------------------------------------------
    # lane cost estimates
    # ------------------------------------------------------------------
    def _estimate_lane_costs(
        self,
        *,
        ego: Any,
        tracked_lead: Optional[LeadInfo],
        route_pts: List[Any],
    ) -> Tuple[float, float]:
        """
        Lightweight lane-level scoring.
        Smaller is better.
        We intentionally keep this cheap:
          - no search
          - no rollout
          - only coarse geometry + tracked lead relation
        """
        _, ego_l = self._project_to_route_sl(x=ego.pose.x, y=ego.pose.y, route_pts=route_pts)
        ego_speed = self._get_speed(ego)

        # base cost: distance from current target lane centers
        keep_cost = 1.5 * abs(ego_l - self.keep_lane_offset_m)
        pass_cost = 1.5 * abs(ego_l - self.pass_lane_offset_m)

        if tracked_lead is None:
            # no blocking lead => keep lane is naturally preferred
            pass_cost += 15.0
            return keep_cost, pass_cost

        # current-lane blocking penalty
        if tracked_lead.longitudinal < self.pass_trigger_dist_m:
            keep_cost += 20.0 * (1.0 - tracked_lead.longitudinal / max(1e-3, self.pass_trigger_dist_m))

        # if ego is faster than lead, passing becomes more attractive
        speed_gain = max(0.0, ego_speed - tracked_lead.speed)
        if speed_gain > self.min_speed_gain_to_pass_mps:
            pass_cost -= 5.0 * min(speed_gain, 5.0)

        # parallel / unsafe return discourages keep-lane target when already in pass lane
        if self.current_target_lane_offset_m == self.pass_lane_offset_m:
            passed_gap = self._compute_pass_gap(ego=ego, lead=tracked_lead, route_pts=route_pts)
            if passed_gap < self.return_pass_gap_m:
                keep_cost += 15.0 * (1.0 - passed_gap / max(1e-3, self.return_pass_gap_m))

        return keep_cost, pass_cost

    def _can_return_to_keep_lane(
        self,
        *,
        ego: Any,
        tracked_lead: Optional[LeadInfo],
        route_pts: List[Any],
    ) -> bool:
        _, ego_l = self._project_to_route_sl(x=ego.pose.x, y=ego.pose.y, route_pts=route_pts)
        in_pass_lane_center = abs(ego_l - self.pass_lane_offset_m) < self.lane_center_reached_tol_m
        if not in_pass_lane_center:
            return False

        if tracked_lead is None:
            return True

        passed_gap = self._compute_pass_gap(ego=ego, lead=tracked_lead, route_pts=route_pts)
        if passed_gap < self.return_pass_gap_m:
            return False

        # also avoid return while still too parallel laterally
        if abs(tracked_lead.lateral) < self.parallel_lateral_guard_m and tracked_lead.longitudinal > -5.0:
            return False

        return True

    # ------------------------------------------------------------------
    # tracked lead
    # ------------------------------------------------------------------
    def _update_tracked_lead(
        self,
        *,
        ego: Any,
        world: Any,
        route_pts: List[Any],
    ) -> Optional[LeadInfo]:
        tracked = self._find_lead_by_id(ego=ego, world=world, actor_id=self.tracked_lead_actor_id)

        if tracked is not None:
            passed_gap = self._compute_pass_gap(ego=ego, lead=tracked, route_pts=route_pts)
            keep_track = (
                tracked.longitudinal > -self.tracked_lead_keep_backward_m
                and tracked.longitudinal < self.tracked_lead_keep_forward_m
                and abs(tracked.lateral) < self.tracked_lead_keep_lateral_m
                and passed_gap < self.tracked_lead_release_gap_m
            )

            if keep_track:
                self.tracked_lead_lost_count = 0
                return tracked

            self.tracked_lead_lost_count += 1
            if self.tracked_lead_lost_count <= self.tracked_lead_max_lost_cycles:
                return tracked

            self.tracked_lead_actor_id = None
            self.tracked_lead_lost_count = 0

        new_lead = self._find_current_lane_lead(ego=ego, world=world)
        if new_lead is not None:
            self.tracked_lead_actor_id = new_lead.actor_id
            self.tracked_lead_lost_count = 0
            return new_lead

        self.tracked_lead_actor_id = None
        self.tracked_lead_lost_count = 0
        return None

    def _find_current_lane_lead(self, *, ego: Any, world: Any) -> Optional[LeadInfo]:
        ex, ey, eyaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        c = math.cos(eyaw)
        s = math.sin(eyaw)

        best = None
        best_long = 1e18

        for obs in getattr(world, "obstacles", []) or []:
            info = self._obs_to_relative_lead_info(ego=ego, obs=obs)
            if info is None:
                continue
            if info.longitudinal <= 0.0:
                continue
            if info.longitudinal > self.lead_detect_forward_m:
                continue
            if abs(info.lateral) > self.lead_lane_half_width_m:
                continue
            if info.longitudinal < best_long:
                best_long = info.longitudinal
                best = info
        return best

    def _find_lead_by_id(self, *, ego: Any, world: Any, actor_id: Any) -> Optional[LeadInfo]:
        if actor_id is None:
            return None
        for obs in getattr(world, "obstacles", []) or []:
            if getattr(obs, "id", None) == actor_id:
                return self._obs_to_relative_lead_info(ego=ego, obs=obs)
        return None

    def _obs_to_relative_lead_info(self, *, ego: Any, obs: Any) -> Optional[LeadInfo]:
        pos = getattr(obs, "position", None)
        if pos is None:
            return None

        ox = getattr(pos, "x", None)
        oy = getattr(pos, "y", None)
        if ox is None or oy is None:
            return None

        ox = float(ox)
        oy = float(oy)

        ex, ey, eyaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        c = math.cos(eyaw)
        s = math.sin(eyaw)

        dx = ox - ex
        dy = oy - ey

        longitudinal = dx * c + dy * s
        lateral = -dx * s + dy * c
        distance = math.hypot(dx, dy)

        speed = 0.0
        vel = getattr(obs, "velocity", None)
        if vel is not None:
            vx = float(getattr(vel, "x", 0.0))
            vy = float(getattr(vel, "y", 0.0))
            vz = float(getattr(vel, "z", 0.0))
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)

        yaw = 0.0
        for name in ("yaw", "heading", "theta"):
            v = getattr(obs, name, None)
            if v is not None:
                yaw = float(v)
                break
        if abs(yaw) < 1e-6 and vel is not None:
            vx = float(getattr(vel, "x", 0.0))
            vy = float(getattr(vel, "y", 0.0))
            if math.hypot(vx, vy) > 0.2:
                yaw = math.atan2(vy, vx)

        raw_length_m = float(getattr(obs, "length", 4.5)) if getattr(obs, "length", None) is not None else 4.5

        return LeadInfo(
            actor_id=getattr(obs, "id", None),
            x=ox,
            y=oy,
            yaw=yaw,
            speed=speed,
            longitudinal=longitudinal,
            lateral=lateral,
            distance=distance,
            raw_length_m=raw_length_m,
        )

    # ------------------------------------------------------------------
    # route geometry
    # ------------------------------------------------------------------
    def _project_to_route_sl(self, *, x: float, y: float, route_pts: List[Any]) -> Tuple[float, float]:
        if len(route_pts) < 2:
            return 0.0, 0.0

        best_s = 0.0
        best_l = 0.0
        best_d2 = 1e18
        accum_s = 0.0

        for i in range(len(route_pts) - 1):
            ax = route_pts[i].x
            ay = route_pts[i].y
            bx = route_pts[i + 1].x
            by = route_pts[i + 1].y

            abx = bx - ax
            aby = by - ay
            ab2 = abx * abx + aby * aby
            seg_len = math.hypot(abx, aby)
            if ab2 <= 1e-9 or seg_len <= 1e-9:
                accum_s += seg_len
                continue

            apx = x - ax
            apy = y - ay
            t = clamp((apx * abx + apy * aby) / ab2, 0.0, 1.0)

            px = ax + t * abx
            py = ay + t * aby

            dx = x - px
            dy = y - py
            d2 = dx * dx + dy * dy

            if d2 < best_d2:
                yaw = math.atan2(aby, abx)
                left_nx = -math.sin(yaw)
                left_ny = math.cos(yaw)
                l_signed = (x - px) * left_nx + (y - py) * left_ny

                best_d2 = d2
                best_s = accum_s + t * seg_len
                best_l = l_signed

            accum_s += seg_len

        return best_s, best_l

    def _compute_pass_gap(self, *, ego: Any, lead: LeadInfo, route_pts: List[Any]) -> float:
        ego_s, _ = self._project_to_route_sl(x=ego.pose.x, y=ego.pose.y, route_pts=route_pts)
        lead_front_x = lead.x + math.cos(lead.yaw) * (0.5 * lead.raw_length_m)
        lead_front_y = lead.y + math.sin(lead.yaw) * (0.5 * lead.raw_length_m)
        lead_front_s, _ = self._project_to_route_sl(x=lead_front_x, y=lead_front_y, route_pts=route_pts)
        return ego_s - lead_front_s

    @staticmethod
    def _get_speed(obj: Any) -> float:
        v = getattr(obj, "speed", None)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass

        vel = getattr(obj, "velocity", None)
        if vel is not None:
            vx = float(getattr(vel, "x", 0.0))
            vy = float(getattr(vel, "y", 0.0))
            vz = float(getattr(vel, "z", 0.0))
            return math.sqrt(vx * vx + vy * vy + vz * vz)

        return 0.0
