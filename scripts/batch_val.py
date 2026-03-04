from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


from benchmark_metrics import compute_run_metrics, write_comparison_table


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


def _apply_sets(cfg: Dict[str, Any], set_items: List[str]) -> None:
    for item in set_items:
        if "=" not in item:
            raise ValueError(f"Invalid set item (expected key=value): {item}")
        key, val = item.split("=", 1)
        _set_nested(cfg, key.strip(), yaml.safe_load(val.strip()))


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _deepcopy_via_yaml(d: Dict[str, Any]) -> Dict[str, Any]:
    return yaml.safe_load(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))


def _latest_run_dir(runs_dir: Path) -> Path:
    candidates = [p for p in runs_dir.glob("*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {runs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_one(cfg: Dict[str, Any], runs_dir: Path, run_module: str, config_path: Path) -> None:
    # 用 -m 运行，避免 scripts 包导入问题
    cmd = ["python", "-m", run_module, "--config", str(config_path)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(Path.cwd()))
    # run.py 应该把结果写到 runs_dir 下；这里不强依赖返回值，后面用 latest_run_dir 取


def main() -> None:
    parser = argparse.ArgumentParser(description="Fair benchmark runner for multiple planners")

    # 新增：suite 一行用法
    parser.add_argument("--suite", help="Suite YAML path (contains base_config/global_set/planners)")


    parser.add_argument("--base-config", help="Base YAML config path")
    parser.add_argument("--planner", action="append", help="Planner class path, repeatable")
    parser.add_argument("--set", action="append", default=[], help="Override key=value")

    # 这些参数既可来自 suite，也可 CLI 覆盖
    parser.add_argument("--run-script", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    # -------- suite 模式 --------
    if args.suite:
        suite_path = Path(args.suite)
        suite = _load_yaml(suite_path)

        base_config_path = Path(suite["base_config"])
        run_script = suite.get("run_script", "scripts/run.py")
        runs_dir = Path(suite.get("runs_dir", "runs"))
        output_path = Path(suite.get("output", "runs/benchmark_comparison.md"))
        global_set = suite.get("global_set", [])
        planners = suite.get("planners", [])

        if not isinstance(global_set, list) or not all(isinstance(x, str) for x in global_set):
            raise ValueError("suite.global_set must be a list of strings like key=value")
        if not isinstance(planners, list) or not planners:
            raise ValueError("suite.planners must be a non-empty list")

        # 允许 CLI 覆盖
        if args.run_script:
            run_script = args.run_script
        if args.runs_dir:
            runs_dir = Path(args.runs_dir)
        if args.output:
            output_path = Path(args.output)

        # 将 scripts/run.py -> scripts.run（模块名）
        run_module = run_script.replace("/", ".").replace("\\", ".")
        if run_module.endswith(".py"):
            run_module = run_module[:-3]

        base_cfg = _load_yaml(base_config_path)
        all_metrics = []

        for p in planners:
            if not isinstance(p, dict):
                raise ValueError("Each suite.planners item must be a dict")
            label = p.get("label", p.get("name", "planner"))
            name = p.get("name")
            if not name:
                raise ValueError("Each planner must have a 'name' field")

            cfg = _deepcopy_via_yaml(base_cfg)

            # 1) global_set
            _apply_sets(cfg, global_set)
            # 2) planner-specific set
            _apply_sets(cfg, p.get("set", []))

            # 3) set planner name（按你原逻辑）
            cfg.setdefault("planner", {})["name"] = name

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                tmp_path = Path(f.name)
            _dump_yaml(tmp_path, cfg)

            print(f"\n=== Planner: {label} ({name}) ===")
            _run_one(cfg, runs_dir=runs_dir, run_module=run_module, config_path=tmp_path)

            latest_run = _latest_run_dir(runs_dir)
            metrics = compute_run_metrics(latest_run)
            # 可选：把 label 填进 metrics 便于表格显示（如果你的 write_comparison_table 支持）
            if isinstance(metrics, dict) and "label" not in metrics:
                metrics["label"] = str(label)
                metrics["planner_name"] = str(name)
            all_metrics.append(metrics)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_comparison_table(all_metrics, output_path)
        print(f"\nComparison table saved to: {output_path}")
        return

    # -------- 原 CLI 模式（兼容）--------
    if not args.base_config or not args.planner:
        raise SystemExit("Either use --suite, or provide --base-config and at least one --planner.")

    base_cfg = _load_yaml(Path(args.base_config))
    _apply_sets(base_cfg, args.set)

    run_script = args.run_script or "scripts/run.py"
    runs_dir = Path(args.runs_dir or "runs")
    output_path = Path(args.output or "runs/benchmark_comparison.md")

    run_module = run_script.replace("/", ".").replace("\\", ".")
    if run_module.endswith(".py"):
        run_module = run_module[:-3]

    all_metrics = []
    for planner_name in args.planner:
        cfg = _deepcopy_via_yaml(base_cfg)
        cfg.setdefault("planner", {})["name"] = planner_name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            tmp_path = Path(f.name)
        _dump_yaml(tmp_path, cfg)

        _run_one(cfg, runs_dir=runs_dir, run_module=run_module, config_path=tmp_path)

        latest_run = _latest_run_dir(runs_dir)
        all_metrics.append(compute_run_metrics(latest_run))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_comparison_table(all_metrics, output_path)
    print(f"Comparison table saved to: {output_path}")


if __name__ == "__main__":
    main()