from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import copy
import importlib
import random

from framework.testing.lane_change_case import LaneChangeCase

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class LaneChangeExecutionResult:
    case_id: str
    success: bool
    fail_reason: str | None
    run_dir: str | None
    run_result: Dict[str, Any]
    input_case: Dict[str, Any]
    planner_name: str
    scenario_name: str
    base_config_path: str


def _parse_target(target: str) -> tuple[str, str]:
    if ":" in target:
        mod, cls = target.split(":", 1)
        return mod.strip(), cls.strip()
    mod, cls = target.rsplit(".", 1)
    return mod.strip(), cls.strip()


def _import_class(target: str):
    mod_name, cls_name = _parse_target(target)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load config yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Base config root must be dict")
    return data


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def build_case_run_config(base_cfg: Dict[str, Any], case: LaneChangeCase, output_dir: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    scenario_cfg = cfg.setdefault("scenario", {}).setdefault("config", {})
    planner_cfg = cfg.setdefault("planner", {}).setdefault("config", {})
    controller_cfg = cfg.setdefault("controller", {}).setdefault("config", {})
    runner_cfg = cfg.setdefault("runner", {})
    scenario_sec = cfg.setdefault("scenario", {})
    planner_sec = cfg.setdefault("planner", {})

    scenario_sec["name"] = "framework.scenarios.lane_change:ConfigurableRouteScenario"
    planner_sec["name"] = "framework.planning.local.a_star:HybridAStarMapPlanner"
    scenario_cfg["id"] = case.case_id
    scenario_cfg["timeout_s"] = float(case.simulation_timeout_s)
    scenario_cfg["random_seed"] = int(case.random_seed)
    scenario_cfg["ego_initial_speed_mps"] = float(case.ego_initial_speed)
    scenario_cfg.setdefault("lead_vehicle", {})
    scenario_cfg["lead_vehicle"]["enable"] = True
    scenario_cfg["lead_vehicle"]["distance_m"] = float(case.lead_vehicle_initial_distance)
    scenario_cfg["lead_vehicle"]["target_speed_mps"] = float(case.lead_vehicle_speed)
    scenario_cfg["adjacent_vehicle"] = {
        "enable": bool(case.adjacent_rear_vehicle_exists),
        "rear_distance_m": float(case.adjacent_rear_distance),
        "target_speed_mps": float(case.adjacent_rear_speed),
        "side": "left" if case.allow_left_lane_change else "right",
    }

    controller_cfg["target_speed"] = float(case.ego_initial_speed)
    planner_cfg["target_speed"] = float(case.ego_initial_speed)
    planner_cfg["planner_target_speed"] = float(case.ego_initial_speed)

    if case.allow_left_lane_change and not case.allow_right_lane_change:
        planner_cfg["pass_side"] = "left"
    elif case.allow_right_lane_change and not case.allow_left_lane_change:
        planner_cfg["pass_side"] = "right"
    elif not case.allow_left_lane_change and not case.allow_right_lane_change:
        raise ValueError(f"case={case.case_id}: both left/right lane change are disabled.")

    max_steps = int(case.simulation_timeout_s / float(runner_cfg.get("fixed_delta_seconds", 0.05)))
    runner_cfg["max_steps"] = max(1, max_steps)

    recorder_cfg = runner_cfg.setdefault("recorder", {})
    recorder_cfg["save_dir"] = str(output_dir)
    recorder_cfg["run_name"] = f"{case.case_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"

    _deep_update(scenario_cfg, case.scenario_overrides)
    _deep_update(planner_cfg, case.planner_overrides)
    _deep_update(controller_cfg, case.controller_overrides)
    _deep_update(runner_cfg, case.runner_overrides)

    return cfg


def run_lane_change_case(case: LaneChangeCase, *, base_config_path: Path, output_dir: Path) -> LaneChangeExecutionResult:
    random.seed(case.random_seed)

    cfg = build_case_run_config(_load_yaml(base_config_path), case, output_dir)
    run_result: Dict[str, Any]
    run_dir = None
    fail_reason = None

    try:
        planner_name = str(cfg.get("planner", {}).get("name", ""))
        scenario_name = str(cfg.get("scenario", {}).get("name", ""))
        if planner_name != "framework.planning.local.a_star:HybridAStarMapPlanner":
            raise RuntimeError(f"Unexpected planner in config: {planner_name}")
        if scenario_name != "framework.scenarios.lane_change:ConfigurableRouteScenario":
            raise RuntimeError(f"Unexpected scenario in config: {scenario_name}")

        ScenarioCls = _import_class(cfg["scenario"]["name"])
        PlannerCls = _import_class(cfg["planner"]["name"])
        ControllerCls = _import_class(cfg["controller"]["name"])

        scenario = ScenarioCls(cfg["scenario"].get("config", {}))
        planner = PlannerCls(cfg["planner"].get("config", {}))
        controller = ControllerCls(cfg["controller"].get("config", {}))

        from framework.runner import Runner

        runner = Runner(
            runner_cfg=cfg["runner"],
            scenario=scenario,
            planner=planner,
            controller=controller,
            full_config=cfg,
        )
        run_result = runner.run()
        run_dir_obj = getattr(runner.recorder, "run_dir", None)
        run_dir = str(run_dir_obj) if run_dir_obj is not None else None
    except Exception as exc:  # isolate each case
        fail_reason = "scenario_error"
        run_result = {"reason": "exception", "error": str(exc)}
        planner_name = str(cfg.get("planner", {}).get("name", ""))
        scenario_name = str(cfg.get("scenario", {}).get("name", ""))

    success = fail_reason is None
    return LaneChangeExecutionResult(
        case_id=case.case_id,
        success=success,
        fail_reason=fail_reason,
        run_dir=run_dir,
        run_result=run_result,
        input_case=asdict(case),
        planner_name=planner_name,
        scenario_name=scenario_name,
        base_config_path=str(base_config_path),
    )
