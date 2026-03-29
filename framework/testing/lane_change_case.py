from __future__ import annotations

from dataclasses import dataclass
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
    ego_spawn_point_index: int
    goal_spawn_point_index: int
    lead_distance_m: float
    lead_target_speed_mps: float
    expected_success: bool = True


@dataclass
class LaneChangeCaseSuite:
    scenario_name: str
    base_config: str
    cases: List[LaneChangeCase]


def _read_structured_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
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
        ego_spawn_point_index=int(raw["ego_spawn_point_index"]),
        goal_spawn_point_index=int(raw["goal_spawn_point_index"]),
        lead_distance_m=float(raw["lead_distance_m"]),
        lead_target_speed_mps=float(raw["lead_target_speed_mps"]),
        expected_success=bool(raw.get("expected_success", True)),
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
