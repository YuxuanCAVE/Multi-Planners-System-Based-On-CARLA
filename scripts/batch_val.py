from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import yaml

from scripts.benchmark_metrics import compute_run_metrics, write_comparison_table


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be dict")
    return data


def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any) -> None:
    cur = cfg
    parts = dotted_key.split(".")
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair benchmark runner for multiple planners")
    parser.add_argument("--base-config", required=True, help="Base YAML config path")
    parser.add_argument("--planner", action="append", required=True, help="Planner class path, repeatable")
    parser.add_argument("--set", action="append", default=[], help="Override key=value, e.g. scenario.config.id=sc01")
    parser.add_argument("--run-script", default="scripts/run.py")
    parser.add_argument("--output", default="runs/benchmark_comparison.md")
    args = parser.parse_args()

    base_cfg = _load_yaml(Path(args.base_config))

    for item in args.set:
        if "=" not in item:
            raise ValueError(f"Invalid --set item: {item}")
        key, val = item.split("=", 1)
        _set_nested(base_cfg, key.strip(), yaml.safe_load(val.strip()))

    all_metrics = []
    for planner_name in args.planner:
        cfg = yaml.safe_load(yaml.safe_dump(base_cfg, sort_keys=False, allow_unicode=True))
        cfg.setdefault("planner", {})["name"] = planner_name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            tmp_path = Path(f.name)
        _dump_yaml(tmp_path, cfg)

        cmd = ["python", args.run_script, "--config", str(tmp_path)]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

        latest_run = max(Path("runs").glob("*"), key=lambda p: p.stat().st_mtime)
        all_metrics.append(compute_run_metrics(latest_run))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_table(all_metrics, output)
    print(f"Comparison table saved to: {output}")


if __name__ == "__main__":
    main()