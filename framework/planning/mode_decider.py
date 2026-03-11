from dataclasses import dataclass
from typing import Optional, List,Dict, Any


from framework.core.types import (
    
    WorldModel,
)


@dataclass
class ModeDecision:
    mode : str
    target_lane_l : float
    blocking_obstacle_index : Optional[int]
    reason : str


def decide_mode(
        
        world:WorldModel, 
        
        *,
        ego,    
        ref,
        current_lane_l: float,
        left_lane_l: Optional[float],
        right_lane_l: Optional[float],
        hint_i: Optional[int] = None,
        block_lookahead_m: float = 25.0,
        lane_half_width_m: float = 1.75,
        lane_block_margin_m: float = 0.5,
        front_gap_m: float = 18.0,
        rear_gap_m: float = 10.0,
        lane_occupancy_width_m: float = 1.5,
        ) -> ModeDecision:
    
        s_ego, l_ego, _, _, _ = ref.project_xy_to_sl(
            ego.pose.x,
            ego.pose.y,
            yaw_hint=ego.pose.yaw,
            hint_i=hint_i,
        )
        obs_sl: List[Dict[str, Any]] = []
        for i, ob in enumerate(world.obstacles):
            s_ob, l_ob, _, _, _ = ref.project_xy_to_sl(
                ob.position.x,
                ob.position.y,
                yaw_hint=None,
                hint_i=hint_i,
            )
            obs_sl.append({
                "index": i,
                "s": float(s_ob),
                "l": float(l_ob),
                "radius": float(ob.radius),
            })
    
        def is_blocking_current_lane(obs: Dict[str,Any]) -> bool:
            ds = obs["s"] - s_ego
            if ds <= 0.0 or ds > block_lookahead_m:
                return False
            
            lateral_thresh = lane_half_width_m + lane_block_margin_m + obs["radius"]
            return abs(obs["l"]- current_lane_l) <= lateral_thresh
        
        blocking = [obs for obs in obs_sl if is_blocking_current_lane(obs)]
        blocking.sort(key=lambda x: x["s"])
        nearest_blocking = blocking[0] if blocking else None

        if nearest_blocking is None:
             return ModeDecision(
                  mode= "KEEP_LANE",
                  target_lane_l = float(current_lane_l),
                  blocking_obstacle_index=None,
                  reason="current_lane_clear",
             )
        
        def is_lane_available(target_lane_l : float) -> bool:
            for obs in obs_sl:
                if obs["s"] < s_ego - rear_gap_m:
                    continue
                if obs["s"] > s_ego + front_gap_m:
                    continue
                lateral_thresh = lane_occupancy_width_m + obs["radius"]
                if abs(obs["l"] - target_lane_l) <= lateral_thresh:
                    return False
            return True
        
        left_available = left_lane_l is not None and is_lane_available(left_lane_l)
        right_available = right_lane_l is not None and is_lane_available(right_lane_l)

        if left_available:
            return ModeDecision(
                mode="CHANGE_LEFT",
                target_lane_l=float(left_lane_l),
                blocking_obstacle_index=int(nearest_blocking["index"]),
                reason="current_lane_blocked_left_available",
            )

        if right_available:
            return ModeDecision(
                mode="CHANGE_RIGHT",
                target_lane_l=float(right_lane_l),
                blocking_obstacle_index=int(nearest_blocking["index"]),
                reason="current_lane_blocked_right_available",
            )

        return ModeDecision(
            mode= "KEEP_LANE",
            target_lane_l = float(current_lane_l),
            blocking_obstacle_index=None,
            reason="current_lane_clear",
        )