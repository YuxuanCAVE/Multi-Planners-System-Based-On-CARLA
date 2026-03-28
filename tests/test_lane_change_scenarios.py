from __future__ import annotations

from pathlib import Path
import importlib.util

import pytest

from framework.evaluation.lane_change_evaluator import evaluate_lane_change_result
from framework.evaluation.result_writer import write_case_result
from framework.testing.lane_change_case import LaneChangeCase, load_lane_change_case_suite, select_cases
from framework.testing.lane_change_runner import run_lane_change_case


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
@pytest.mark.parametrize("_assert_expected_success", [True], ids=["assert_success"])
def test_lane_change_case(
    lane_change_case: LaneChangeCase,
    lane_change_base_config: Path,
    lane_change_output_dir: Path,
    lane_change_suite,
    pytestconfig: pytest.Config,
    _assert_expected_success: bool,
) -> None:
    _requires_carla_or_skip(pytestconfig)
    assert str(lane_change_base_config).endswith("configs/lane_change_actor.yaml")

    execution = run_lane_change_case(
        lane_change_case,
        base_config_path=lane_change_base_config,
        output_dir=lane_change_output_dir / "raw_runs",
    )
    evaluation = evaluate_lane_change_result(
        lane_change_case,
        execution,
        scenario_name=lane_change_suite.scenario_name,
    )

    write_case_result(lane_change_output_dir / "case_results", evaluation)

    debug_msg = (
        f"case_id={evaluation.case_id}; "
        f"planner={evaluation.planner_name}; "
        f"scenario={evaluation.scenario_impl}; "
        f"fail_reason={evaluation.fail_reason}; "
        f"collision_count={evaluation.collision_count}; "
        f"time_to_finish={evaluation.time_to_finish}; "
        f"planning_time_p95={evaluation.planning_time_p95}; "
        f"run_dir={evaluation.run_dir}"
    )
    assert evaluation.planner_name == "framework.planning.local.a_star:HybridAStarMapPlanner", debug_msg
    assert evaluation.scenario_impl == "framework.scenarios.lane_change:ConfigurableRouteScenario", debug_msg
    assert evaluation.success is lane_change_case.expected_success, debug_msg
