from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LeadInfo:
    obs: Any
    actor_id: Any
    longitudinal: float
    lateral: float
    distance: float
    speed: float


@dataclass
class BehaviorOutput:
    state: str
    goal_lateral_offset_m: float
    lookahead_m: float
    w_ref: float
    target_speed: Optional[float]
    reason: str
    lead: Optional[LeadInfo]


class OvertakeStateMachine:
    """
    FOLLOW:
        Stay on original global route / lane.

    LANE_CHANGE_OUT:
        Explicit lane change from original lane to pass lane.

    CRUISE_PASS_LANE:
        Already in pass lane, keep going there until ego is safely ahead.

    LANE_CHANGE_BACK:
        Explicit lane change from pass lane back to original lane.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}
        self.config = cfg

        # lead detection
        self.lead_detect_forward_m = float(cfg.get("lead_detect_forward_m", 50.0))
        self.lead_lane_half_width_m = float(cfg.get("lead_lane_half_width_m", 3.0))

        # trigger out-lane-change
        self.pass_trigger_dist_m = float(cfg.get("pass_trigger_dist_m", 22.0))
        self.min_speed_gain_to_pass_mps = float(cfg.get("min_speed_gain_to_pass_mps", 0.2))

        # lane geometry
        self.pass_side = str(cfg.get("pass_side", "left")).lower()
        self.pass_lane_offset_m = float(cfg.get("pass_lane_offset_m", 3.5))
        self.lane_reached_tol_m = float(cfg.get("lane_reached_tol_m", 1.0))

        # pass / return safety
        self.return_trigger_gap_m = float(cfg.get("return_trigger_gap_m", 6.0))
        self.lead_half_length_m = float(cfg.get("lead_half_length_m", 2.2))
        self.ego_rear_overhang_m = float(cfg.get("ego_rear_overhang_m", 0.977))

        # timing
        self.min_lane_change_out_time_s = float(cfg.get("min_lane_change_out_time_s", 1.0))
        self.min_cruise_pass_lane_time_s = float(cfg.get("min_cruise_pass_lane_time_s", 0.8))
        self.min_lane_change_back_time_s = float(cfg.get("min_lane_change_back_time_s", 1.0))
        self.follow_reentry_cooldown_s = float(cfg.get("follow_reentry_cooldown_s", 2.0))

        # outputs for each state
        self.follow_lookahead_m = float(cfg.get("follow_lookahead_m", 18.0))
        self.lane_change_out_lookahead_m = float(cfg.get("lane_change_out_lookahead_m", 25.0))
        self.cruise_pass_lane_lookahead_m = float(cfg.get("cruise_pass_lane_lookahead_m", 35.0))
        self.lane_change_back_lookahead_m = float(cfg.get("lane_change_back_lookahead_m", 18.0))

        self.follow_w_ref = float(cfg.get("follow_w_ref", 1.5))
        self.lane_change_out_w_ref = float(cfg.get("lane_change_out_w_ref", 0.5))
        self.cruise_pass_lane_w_ref = float(cfg.get("cruise_pass_lane_w_ref", 0.5))
        self.lane_change_back_w_ref = float(cfg.get("lane_change_back_w_ref", 2.5))

        self.follow_target_speed = cfg.get("follow_target_speed", None)
        self.lane_change_out_target_speed = cfg.get("lane_change_out_target_speed", None)
        self.cruise_pass_lane_target_speed = cfg.get("cruise_pass_lane_target_speed", None)
        self.lane_change_back_target_speed = cfg.get("lane_change_back_target_speed", None)

        # final follow completion
        self.return_to_route_tol_m = float(cfg.get("return_to_route_tol_m", 0.8))

        self.reset()

    def reset(self) -> None:
        self.state: str = "FOLLOW"
        self.lead_actor_id: Any = None
        self.state_enter_t: float = 0.0
        self.last_return_finish_t: float = -1e9
        self.last_reason: str = "reset"

        # debug
        self.debug_ego_rear_s: Optional[float] = None
        self.debug_lead_front_s: Optional[float] = None
        self.debug_ego_l: Optional[float] = None
        self.debug_target_l: Optional[float] = None

    def update(
        self,
        *,
        ego: Any,
        world: Any,
        route_pts: List[Any],
        t: float,
        dist_to_route_fn,
    ) -> BehaviorOutput:
        current_lane_lead = self._find_current_lane_lead(ego=ego, world=world)
        tracked_lead = self._find_tracked_lead(world=world)

        if self.state == "FOLLOW":
            self._update_follow(ego=ego, lead=current_lane_lead, t=t)

        elif self.state == "LANE_CHANGE_OUT":
            self._update_lane_change_out(
                ego=ego,
                route_pts=route_pts,
                t=t,
            )

        elif self.state == "CRUISE_PASS_LANE":
            self._update_cruise_pass_lane(
                ego=ego,
                tracked_lead=tracked_lead,
                route_pts=route_pts,
                t=t,
            )

        elif self.state == "LANE_CHANGE_BACK":
            self._update_lane_change_back(
                ego=ego,
                route_pts=route_pts,
                t=t,
            )

        if self.state == "FOLLOW":
            lead_out = current_lane_lead
        else:
            lead_out = self._relative_lead_to_ego(
                ego=ego,
                lead=self._find_tracked_lead(world=world),
            )

        return self._build_output(lead=lead_out)

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------
    def _update_follow(self, *, ego: Any, lead: Optional[LeadInfo], t: float) -> None:
        if t - self.last_return_finish_t < self.follow_reentry_cooldown_s:
            self.last_reason = "follow_cooldown_after_return"
            return

        if lead is None:
            self.last_reason = "no_lead"
            return

        ego_v = self._get_speed(ego)
        if (
            lead.longitudinal < self.pass_trigger_dist_m
            and ego_v > lead.speed + self.min_speed_gain_to_pass_mps
        ):
            self.state = "LANE_CHANGE_OUT"
            self.lead_actor_id = lead.actor_id
            self.state_enter_t = t
            self.last_reason = "follow_to_lane_change_out"
            return

        self.last_reason = "keep_follow"

    def _update_lane_change_out(
        self,
        *,
        ego: Any,
        route_pts: List[Any],
        t: float,
    ) -> None:
        if t - self.state_enter_t < self.min_lane_change_out_time_s:
            self.last_reason = "lane_change_out_hold"
            return

        _, ego_l = self._project_to_route_sl(
            x=ego.pose.x,
            y=ego.pose.y,
            route_pts=route_pts,
        )
        target_l = self._target_pass_l()

        self.debug_ego_l = ego_l
        self.debug_target_l = target_l

        if abs(ego_l - target_l) < self.lane_reached_tol_m:
            self.state = "CRUISE_PASS_LANE"
            self.state_enter_t = t
            self.last_reason = "lane_change_out_to_cruise_pass_lane"
            return

        self.last_reason = "keep_lane_change_out"

    def _update_cruise_pass_lane(
        self,
        *,
        ego: Any,
        tracked_lead: Optional[LeadInfo],
        route_pts: List[Any],
        t: float,
    ) -> None:
        if t - self.state_enter_t < self.min_cruise_pass_lane_time_s:
            self.last_reason = "cruise_pass_lane_hold"
            return

        if tracked_lead is None or tracked_lead.obs is None:
            self.state = "LANE_CHANGE_BACK"
            self.state_enter_t = t
            self.last_reason = "cruise_pass_lane_to_lane_change_back_no_lead"
            return

        if self._has_passed_lead(
            ego=ego,
            lead_obs=tracked_lead.obs,
            route_pts=route_pts,
            required_gap_m=self.return_trigger_gap_m,
        ):
            self.state = "LANE_CHANGE_BACK"
            self.state_enter_t = t
            self.last_reason = "cruise_pass_lane_to_lane_change_back"
            return

        self.last_reason = "keep_cruise_pass_lane"

    def _update_lane_change_back(
        self,
        *,
        ego: Any,
        route_pts: List[Any],
        t: float,
    ) -> None:
        if t - self.state_enter_t < self.min_lane_change_back_time_s:
            self.last_reason = "lane_change_back_hold"
            return

        _, ego_l = self._project_to_route_sl(
            x=ego.pose.x,
            y=ego.pose.y,
            route_pts=route_pts,
        )

        self.debug_ego_l = ego_l
        self.debug_target_l = 0.0

        if abs(ego_l) < self.return_to_route_tol_m:
            self.state = "FOLLOW"
            self.lead_actor_id = None
            self.state_enter_t = t
            self.last_return_finish_t = t
            self.last_reason = "lane_change_back_to_follow"
            return

        self.last_reason = "keep_lane_change_back"

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------
    def _build_output(self, *, lead: Optional[LeadInfo]) -> BehaviorOutput:
        if self.state == "FOLLOW":
            return BehaviorOutput(
                state="FOLLOW",
                goal_lateral_offset_m=0.0,
                lookahead_m=self.follow_lookahead_m,
                w_ref=self.follow_w_ref,
                target_speed=self.follow_target_speed,
                reason=self.last_reason,
                lead=lead,
            )

        if self.state == "LANE_CHANGE_OUT":
            return BehaviorOutput(
                state="LANE_CHANGE_OUT",
                goal_lateral_offset_m=self._target_pass_l(),
                lookahead_m=self.lane_change_out_lookahead_m,
                w_ref=self.lane_change_out_w_ref,
                target_speed=self.lane_change_out_target_speed,
                reason=self.last_reason,
                lead=lead,
            )

        if self.state == "CRUISE_PASS_LANE":
            return BehaviorOutput(
                state="CRUISE_PASS_LANE",
                goal_lateral_offset_m=self._target_pass_l(),
                lookahead_m=self.cruise_pass_lane_lookahead_m,
                w_ref=self.cruise_pass_lane_w_ref,
                target_speed=self.cruise_pass_lane_target_speed,
                reason=self.last_reason,
                lead=lead,
            )

        return BehaviorOutput(
            state="LANE_CHANGE_BACK",
            goal_lateral_offset_m=0.0,
            lookahead_m=self.lane_change_back_lookahead_m,
            w_ref=self.lane_change_back_w_ref,
            target_speed=self.lane_change_back_target_speed,
            reason=self.last_reason,
            lead=lead,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _target_pass_l(self) -> float:
        return self.pass_lane_offset_m if self.pass_side == "left" else -self.pass_lane_offset_m

    def _obs_xy(self, obs: Any):
        pos = getattr(obs, "position", None)
        if pos is None:
            return None, None
        return getattr(pos, "x", None), getattr(pos, "y", None)

    def _obs_speed(self, obs: Any) -> float:
        vel = getattr(obs, "velocity", None)
        if vel is None:
            return 0.0
        vx = float(getattr(vel, "x", 0.0))
        vy = float(getattr(vel, "y", 0.0))
        vz = float(getattr(vel, "z", 0.0))
        return math.sqrt(vx * vx + vy * vy + vz * vz)

    def _find_current_lane_lead(self, *, ego: Any, world: Any) -> Optional[LeadInfo]:
        ex, ey, eyaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        c = math.cos(eyaw)
        s = math.sin(eyaw)

        best = None
        best_long = 1e18

        for obs in getattr(world, "obstacles", []) or []:
            ox, oy = self._obs_xy(obs)
            if ox is None or oy is None:
                continue

            dx = ox - ex
            dy = oy - ey

            longitudinal = dx * c + dy * s
            lateral = -dx * s + dy * c

            if longitudinal <= 0.0:
                continue
            if longitudinal > self.lead_detect_forward_m:
                continue
            if abs(lateral) > self.lead_lane_half_width_m:
                continue

            if longitudinal < best_long:
                best_long = longitudinal
                best = LeadInfo(
                    obs=obs,
                    actor_id=getattr(obs, "id", None),
                    longitudinal=longitudinal,
                    lateral=lateral,
                    distance=math.hypot(dx, dy),
                    speed=self._obs_speed(obs),
                )

        return best

    def _find_tracked_lead(self, *, world: Any) -> Optional[LeadInfo]:
        if self.lead_actor_id is None:
            return None

        for obs in getattr(world, "obstacles", []) or []:
            if getattr(obs, "id", None) == self.lead_actor_id:
                return LeadInfo(
                    obs=obs,
                    actor_id=getattr(obs, "id", None),
                    longitudinal=0.0,
                    lateral=0.0,
                    distance=0.0,
                    speed=self._obs_speed(obs),
                )

        return None

    def _relative_lead_to_ego(self, *, ego: Any, lead: Optional[LeadInfo]) -> Optional[LeadInfo]:
        if lead is None or lead.obs is None:
            return None

        ox, oy = self._obs_xy(lead.obs)
        if ox is None or oy is None:
            return None

        ex, ey, eyaw = ego.pose.x, ego.pose.y, ego.pose.yaw
        c = math.cos(eyaw)
        s = math.sin(eyaw)

        dx = ox - ex
        dy = oy - ey

        return LeadInfo(
            obs=lead.obs,
            actor_id=lead.actor_id,
            longitudinal=dx * c + dy * s,
            lateral=-dx * s + dy * c,
            distance=math.hypot(dx, dy),
            speed=lead.speed,
        )

    def _project_to_route_sl(self, *, x: float, y: float, route_pts: List[Any]):
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
            t = max(0.0, min(1.0, (apx * abx + apy * aby) / ab2))

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

    def _has_passed_lead(
        self,
        *,
        ego: Any,
        lead_obs: Any,
        route_pts: List[Any],
        required_gap_m: float,
    ) -> bool:
        eyaw = ego.pose.yaw
        ec = math.cos(eyaw)
        es = math.sin(eyaw)

        ego_rear_x = ego.pose.x - ec * self.ego_rear_overhang_m
        ego_rear_y = ego.pose.y - es * self.ego_rear_overhang_m

        lx, ly = self._obs_xy(lead_obs)
        if lx is None or ly is None:
            return False

        vel = getattr(lead_obs, "velocity", None)
        if vel is not None:
            lvx = float(getattr(vel, "x", 0.0))
            lvy = float(getattr(vel, "y", 0.0))
            lspd = math.hypot(lvx, lvy)
        else:
            lvx, lvy, lspd = 0.0, 0.0, 0.0

        if lspd > 0.3:
            lc = lvx / lspd
            ls = lvy / lspd
        else:
            lc = math.cos(eyaw)
            ls = math.sin(eyaw)

        lead_front_x = lx + lc * self.lead_half_length_m
        lead_front_y = ly + ls * self.lead_half_length_m

        ego_rear_s, _ = self._project_to_route_sl(
            x=ego_rear_x,
            y=ego_rear_y,
            route_pts=route_pts,
        )
        lead_front_s, _ = self._project_to_route_sl(
            x=lead_front_x,
            y=lead_front_y,
            route_pts=route_pts,
        )

        self.debug_ego_rear_s = ego_rear_s
        self.debug_lead_front_s = lead_front_s

        return ego_rear_s > lead_front_s + required_gap_m

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