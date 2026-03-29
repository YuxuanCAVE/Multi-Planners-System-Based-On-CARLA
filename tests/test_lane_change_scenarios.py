from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from framework.evaluation.lane_change_evaluator import evaluate_lane_change_result
from framework.evaluation.result_writer import write_case_result
from framework.testing.lane_change_case import LaneChangeCase, load_lane_change_case_suite, select_cases
from framework.testing.lane_change_runner import run_lane_change_case

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def _requires_carla_or_skip(pytestconfig: pytest.Config) -> None:
    if not pytestconfig.getoption("run_carla"):
        pytest.skip("CARLA tests are disabled. Use --run-carla to execute integration tests.")
    if importlib.util.find_spec("carla") is None:
        pytest.skip("carla python package is not available in this environment.")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "lane_change_case" not in metafunc.fixturenames:
        return

    cfg = metafunc.config
    suite = load_lane_change_case_suite(Path(cfg.getoption("lane_change_cases")))
    max_cases_opt = cfg.getoption("lane_change_max_cases")
    max_cases = int(max_cases_opt) if max_cases_opt else None
    cases = select_cases(
        suite.cases,
        case_id=cfg.getoption("lane_change_case_id"),
        max_cases=max_cases,
    )
    metafunc.parametrize("lane_change_case", cases, ids=[c.case_id for c in cases])


@pytest.mark.lane_change
def test_lane_change_case(
    lane_change_case: LaneChangeCase,
    lane_change_base_config: Path,
    lane_change_output_dir: Path,
    lane_change_suite,
    pytestconfig: pytest.Config,
) -> None:
    _requires_carla_or_skip(pytestconfig)
    assert str(lane_change_base_config).endswith("configs/lane_change_actor.yaml")

    execution = run_lane_change_case(
        lane_change_case,
        base_config_path=lane_change_base_config,
        output_dir=lane_change_output_dir,
    )

    if yaml is not None and execution.config_path:
        run_cfg = yaml.safe_load(Path(execution.config_path).read_text(encoding="utf-8"))
        assert run_cfg["scenario"]["name"] == "framework.scenarios.lane_change:ConfigurableRouteScenario"
        assert run_cfg["planner"]["name"] == "framework.planning.local.a_star:HybridAStarMapPlanner"

    evaluation = evaluate_lane_change_result(
        lane_change_case,
        execution,
        scenario_name=lane_change_suite.scenario_name,
    )
    write_case_result(lane_change_output_dir / "case_results", evaluation)

    debug_msg = (
        f"case_id={evaluation.case_id}; "
        f"fail_reason={evaluation.fail_reason}; "
        f"collision_count={evaluation.collision_count}; "
        f"time_to_finish={evaluation.time_to_finish}; "
        f"planning_time_p95={evaluation.planning_time_p95}; "
        f"config_path={evaluation.config_path}; "
        f"run_dir={evaluation.run_dir}; "
        f"return_code={execution.return_code}"
    )
    assert evaluation.success is lane_change_case.expected_success, debug_msg
