#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from framework.evaluation.lane_change_evaluator import evaluate_lane_change_result
from framework.evaluation.result_writer import write_case_result, write_summary
from framework.testing.lane_change_case import load_lane_change_case_suite, select_cases
from framework.testing.lane_change_runner import run_lane_change_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch lane-change scenario tests")
    parser.add_argument("--cases", type=str, default="tests/scenario_cases/lane_change/lane_change_cases.json")
    parser.add_argument("--output-dir", type=str, default="runs/lane_change_batch")
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--base-config", type=str, default="configs/lane_change_actor.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite = load_lane_change_case_suite(Path(args.cases))
    cases = select_cases(suite.cases, case_id=args.case_id, max_cases=args.max_cases)
    if not cases:
        raise SystemExit("No case selected.")

    out_dir = Path(args.output_dir)
    case_out_dir = out_dir / "case_results"
    base_cfg = Path(args.base_config or suite.base_config or "configs/lane_change_actor.yaml")

    eval_results = []
    for case in cases:
        print(f"[lane-change] running case={case.case_id}")
        execution = run_lane_change_case(case, base_config_path=base_cfg, output_dir=out_dir)
        evaluation = evaluate_lane_change_result(case, execution, scenario_name=suite.scenario_name)
        write_case_result(case_out_dir, evaluation)
        eval_results.append(evaluation)
        print(f"[lane-change] done case={case.case_id} success={evaluation.success} fail_reason={evaluation.fail_reason}")

    summary_json, summary_csv = write_summary(out_dir, eval_results)
    print(f"summary json: {summary_json}")
    print(f"summary csv : {summary_csv}")


if __name__ == "__main__":
    main()
