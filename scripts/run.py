# scripts/run.py
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

# --- Adjust this to your CARLA install path ---
CARLA_ROOT = Path(r"D:\CARLA0916\CARLA_0.9.16")
PYAPI = CARLA_ROOT / "PythonAPI"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure Python can import both `carla` and `agents`
sys.path.append(str(PYAPI / "carla"))
sys.path.append(str(PYAPI))


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config YAML not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a dict.")
    return data


def _parse_target(target: str) -> Tuple[str, str]:
    """
    Accept:
      - "pkg.mod:ClassName"
      - "pkg.mod.ClassName"
    Return (module, class_name)
    """
    target = target.strip()
    if ":" in target:
        mod, cls = target.split(":", 1)
        return mod.strip(), cls.strip()
    # last dot is class separator
    if "." not in target:
        raise ValueError(
            f"Invalid target '{target}'. Use 'package.module:ClassName' or 'package.module.ClassName'."
        )
    mod, cls = target.rsplit(".", 1)
    return mod.strip(), cls.strip()


def _import_class(target: str):
    mod_name, cls_name = _parse_target(target)
    mod = importlib.import_module(mod_name)
    try:
        cls = getattr(mod, cls_name)
    except AttributeError as e:
        raise ImportError(f"Module '{mod_name}' has no attribute '{cls_name}'") from e
    return cls


def _require_section(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    if key not in cfg or not isinstance(cfg[key], dict):
        raise ValueError(f"Missing required section '{key}' in YAML.")
    return cfg[key]


def _require_name_and_config(section: Dict[str, Any], section_name: str) -> Tuple[str, Dict[str, Any]]:
    name = section.get("name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Missing '{section_name}.name' (string) in YAML.")
    cfg = section.get("config", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"'{section_name}.config' must be a dict.")
    return name.strip(), cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    # Import carla after sys.path is set
    import carla  # noqa: F401

    cfg = _load_yaml(Path(args.config))

    # minimal section checks
    carla_sec = _require_section(cfg, "carla")
    scenario_sec = _require_section(cfg, "scenario")
    planner_sec = _require_section(cfg, "planner")
    controller_sec = _require_section(cfg, "controller")
    runner_sec = _require_section(cfg, "runner")

    # allow small CLI overrides for connection
    if args.host is not None:
        carla_sec["host"] = args.host
    if args.port is not None:
        carla_sec["port"] = args.port

    # instantiate scenario/planner/controller dynamically
    scenario_target, scenario_cfg = _require_name_and_config(scenario_sec, "scenario")
    planner_target, planner_cfg = _require_name_and_config(planner_sec, "planner")
    controller_target, controller_cfg = _require_name_and_config(controller_sec, "controller")

    ScenarioCls = _import_class(scenario_target)
    PlannerCls = _import_class(planner_target)
    ControllerCls = _import_class(controller_target)

    scenario = ScenarioCls(scenario_cfg)
    planner = PlannerCls(planner_cfg)
    controller = ControllerCls(controller_cfg)

    # runner cfg: inject carla host/port/timeout
    runner_cfg = runner_sec
    runner_cfg["host"] = str(carla_sec.get("host", "127.0.0.1"))
    runner_cfg["port"] = int(carla_sec.get("port", 2000))
    runner_cfg["timeout_s"] = float(carla_sec.get("timeout_s", 10.0))

    from framework.runner import Runner

    runner = Runner(
        runner_cfg=runner_cfg,
        scenario=scenario,
        planner=planner,
        controller=controller,
        full_config=cfg,   # recorder meta snapshot
    )

    result = runner.run()
    print("Run finished:", result)


if __name__ == "__main__":
    main()
