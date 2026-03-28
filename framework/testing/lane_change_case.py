from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class LaneChangeCase:
    case_id: str
    ego_initial_speed: float
    lead_vehicle_initial_distance: float
    lead_vehicle_speed: float
    allow_left_lane_change: bool = True
    allow_right_lane_change: bool = False
    adjacent_rear_vehicle_exists: bool = False
    adjacent_rear_distance: float = 18.0
    adjacent_rear_speed: float = 7.0
    require_overtake_completion: bool = True
    require_return_to_original_lane: bool = False
    simulation_timeout_s: float = 120.0
    random_seed: int = 0
    expected_success: bool = True
    tags: List[str] = field(default_factory=list)
    scenario_overrides: Dict[str, Any] = field(default_factory=dict)
    planner_overrides: Dict[str, Any] = field(default_factory=dict)
    controller_overrides: Dict[str, Any] = field(default_factory=dict)
    runner_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LaneChangeCaseSuite:
    scenario_name: str
    base_config: str
    cases: List[LaneChangeCase]


def _read_structured_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        if yaml is None:
            raise RuntimeError("pyyaml is required for YAML case files")
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"Case file root must be object/dict: {path}")
    return data


def _as_case(raw: Dict[str, Any]) -> LaneChangeCase:
    return LaneChangeCase(
        case_id=str(raw["case_id"]),
        ego_initial_speed=float(raw.get("ego_initial_speed", 8.0)),
        lead_vehicle_initial_distance=float(raw.get("lead_vehicle_initial_distance", 20.0)),
        lead_vehicle_speed=float(raw.get("lead_vehicle_speed", 2.0)),
        allow_left_lane_change=bool(raw.get("allow_left_lane_change", True)),
        allow_right_lane_change=bool(raw.get("allow_right_lane_change", False)),
        adjacent_rear_vehicle_exists=bool(raw.get("adjacent_rear_vehicle_exists", False)),
        adjacent_rear_distance=float(raw.get("adjacent_rear_distance", 18.0)),
        adjacent_rear_speed=float(raw.get("adjacent_rear_speed", 7.0)),
        require_overtake_completion=bool(raw.get("require_overtake_completion", True)),
        require_return_to_original_lane=bool(raw.get("require_return_to_original_lane", False)),
        simulation_timeout_s=float(raw.get("simulation_timeout_s", 120.0)),
        random_seed=int(raw.get("random_seed", 0)),
        expected_success=bool(raw.get("expected_success", True)),
        tags=list(raw.get("tags", [])),
        scenario_overrides=dict(raw.get("scenario_overrides", {})),
        planner_overrides=dict(raw.get("planner_overrides", {})),
        controller_overrides=dict(raw.get("controller_overrides", {})),
        runner_overrides=dict(raw.get("runner_overrides", {})),
    )


def load_lane_change_case_suite(path: Path) -> LaneChangeCaseSuite:
    data = _read_structured_file(path)
    raw_cases = data.get("cases", [])
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"No cases found in {path}")
    return LaneChangeCaseSuite(
        scenario_name=str(data.get("scenario_name", "lane_change_overtake")),
        base_config=str(data.get("base_config", "configs/lane_change_actor.yaml")),
        cases=[_as_case(dict(c)) for c in raw_cases],
    )


def select_cases(
    cases: List[LaneChangeCase],
    *,
    case_id: Optional[str] = None,
    max_cases: Optional[int] = None,
) -> List[LaneChangeCase]:
    picked = cases
    if case_id:
        picked = [c for c in picked if c.case_id == case_id]
    if max_cases is not None:
        picked = picked[: max_cases]
    return picked
