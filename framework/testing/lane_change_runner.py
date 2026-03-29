from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import ast
import copy
import subprocess
import sys
import traceback

from framework.testing.lane_change_case import LaneChangeCase

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


SCENARIO_TARGET = "framework.scenarios.lane_change:ConfigurableRouteScenario"
PLANNER_TARGET = "framework.planning.local.a_star:HybridAStarMapPlanner"


@dataclass
class LaneChangeExecutionResult:
    case_id: str
    success: bool
    fail_reason: Optional[str]
    run_result: Dict[str, Any]
    run_dir: Optional[str]
    config_path: str
    stdout: str
    stderr: str
    return_code: int
    input_case: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required to load YAML configs")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be dict: {path}")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    if yaml is None:
        raise RuntimeError("pyyaml is required to dump YAML configs")
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def build_case_run_config(base_cfg: Dict[str, Any], case: LaneChangeCase, run_output_dir: Path) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    scenario_sec = cfg.setdefault("scenario", {})
    planner_sec = cfg.setdefault("planner", {})

    # enforce real project chain targets
    scenario_sec["name"] = SCENARIO_TARGET
    planner_sec["name"] = PLANNER_TARGET

    scenario_cfg = scenario_sec.setdefault("config", {})
    scenario_cfg.setdefault("ego_spawn", {})["spawn_point_index"] = int(case.ego_spawn_point_index)
    scenario_cfg.setdefault("goal", {})["spawn_point_index"] = int(case.goal_spawn_point_index)
    scenario_cfg.setdefault("lead_vehicle", {})["distance_m"] = float(case.lead_distance_m)
    scenario_cfg.setdefault("lead_vehicle", {})["target_speed_mps"] = float(case.lead_target_speed_mps)

    # keep all existing runner/planner/controller logic; only isolate outputs per case
    runner_cfg = cfg.setdefault("runner", {})
    rec_cfg = runner_cfg.setdefault("recorder", {})
    rec_cfg["save_dir"] = str(run_output_dir)
    rec_cfg["run_name"] = case.case_id

    return cfg


def _extract_run_result(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        if line.startswith("Run finished:"):
            payload = line.split("Run finished:", 1)[1].strip()
            try:
                obj = ast.literal_eval(payload)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return {"raw": payload}
    return {}


def _latest_run_dir(save_dir: Path, run_name: str) -> Optional[Path]:
    direct = save_dir / run_name
    if direct.exists() and direct.is_dir():
        return direct
    candidates = [p for p in save_dir.glob(f"{run_name}*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def run_lane_change_case(case: LaneChangeCase, *, base_config_path: Path, output_dir: Path) -> LaneChangeExecutionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    gen_cfg_dir = output_dir / "generated_configs"
    raw_run_dir = output_dir / "raw_runs"
    gen_cfg_dir.mkdir(parents=True, exist_ok=True)
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_cfg = _load_yaml(base_config_path)
        cfg = build_case_run_config(base_cfg, case, raw_run_dir)
        cfg_path = gen_cfg_dir / f"{case.case_id}.yaml"
        _dump_yaml(cfg_path, cfg)

        cmd = [sys.executable, "scripts/run.py", "--config", str(cfg_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        run_result = _extract_run_result(proc.stdout)
        run_dir_obj = _latest_run_dir(raw_run_dir, case.case_id)
        run_dir = str(run_dir_obj) if run_dir_obj is not None else None

        success = proc.returncode == 0
        fail_reason = None if success else "scenario_error"

        return LaneChangeExecutionResult(
            case_id=case.case_id,
            success=success,
            fail_reason=fail_reason,
            run_result=run_result,
            run_dir=run_dir,
            config_path=str(cfg_path),
            stdout=proc.stdout,
            stderr=proc.stderr,
            return_code=int(proc.returncode),
            input_case=asdict(case),
        )

    except Exception as exc:
        return LaneChangeExecutionResult(
            case_id=case.case_id,
            success=False,
            fail_reason="scenario_error",
            run_result={
                "reason": "exception",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
            run_dir=None,
            config_path="",
            stdout="",
            stderr="",
            return_code=-1,
            input_case=asdict(case),
        )
