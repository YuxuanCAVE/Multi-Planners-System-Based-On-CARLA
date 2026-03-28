from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import csv
import json

from framework.evaluation.lane_change_evaluator import LaneChangeEvalResult


def write_case_result(output_dir: Path, result: LaneChangeEvalResult) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{result.case_id}.json"
    path.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _avg(values: List[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _p95(values: List[float]) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = max(0, int(len(s) * 0.95) - 1)
    return s[idx]


def build_summary(results: Iterable[LaneChangeEvalResult]) -> Dict[str, Any]:
    rows = list(results)
    total = len(rows)
    success = sum(1 for r in rows if r.success)
    collision = sum(1 for r in rows if r.collision_count > 0)
    timeout = sum(1 for r in rows if r.timeout)

    planning_means = [r.planning_time_mean for r in rows if isinstance(r.planning_time_mean, (int, float))]
    planning_all = [r.planning_time_p95 for r in rows if isinstance(r.planning_time_p95, (int, float))]
    finish_times = [r.time_to_finish for r in rows if isinstance(r.time_to_finish, (int, float))]

    fail_counter = Counter(r.fail_reason for r in rows if r.fail_reason)

    return {
        "total_cases": total,
        "success_cases": success,
        "success_rate": (success / total) if total else 0.0,
        "collision_rate": (collision / total) if total else 0.0,
        "timeout_rate": (timeout / total) if total else 0.0,
        "average_planning_time_ms": _avg(planning_means),
        "p95_planning_time_ms": _p95(planning_all),
        "average_time_to_finish_s": _avg(finish_times),
        "fail_reason_counts": dict(fail_counter),
    }


def write_summary(output_dir: Path, results: Iterable[LaneChangeEvalResult]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_list = list(results)
    summary = build_summary(result_list)

    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [asdict(r) for r in result_list],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_csv_path = output_dir / "summary.csv"
    fieldnames = [
        "case_id",
        "success",
        "fail_reason",
        "timeout",
        "collision_count",
        "time_to_finish",
        "planning_time_mean",
        "planning_time_p95",
        "lane_change_started",
        "lane_change_completed",
        "overtake_completed",
        "returned_to_original_lane",
    ]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in result_list:
            row = asdict(item)
            writer.writerow({k: row.get(k) for k in fieldnames})

    return summary_json_path, summary_csv_path
