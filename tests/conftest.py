from __future__ import annotations

from pathlib import Path

import pytest

from framework.testing.lane_change_case import load_lane_change_case_suite, select_cases


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--lane-change-cases", action="store", default="tests/scenario_cases/lane_change/lane_change_cases.json")
    parser.addoption("--lane-change-base-config", action="store", default="configs/lane_change_actor.yaml")
    parser.addoption("--lane-change-output-dir", action="store", default="runs/lane_change_pytest")
    parser.addoption("--lane-change-case-id", action="store", default=None)
    parser.addoption("--lane-change-max-cases", action="store", default=None)
    parser.addoption("--run-carla", action="store_true", default=False, help="Actually run CARLA integration tests")


@pytest.fixture(scope="session")
def lane_change_suite(pytestconfig: pytest.Config):
    return load_lane_change_case_suite(Path(pytestconfig.getoption("lane_change_cases")))


@pytest.fixture(scope="session")
def lane_change_base_config(pytestconfig: pytest.Config, lane_change_suite):
    value = pytestconfig.getoption("lane_change_base_config")
    return Path(value or lane_change_suite.base_config or "configs/lane_change_actor.yaml")


@pytest.fixture(scope="session")
def lane_change_output_dir(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.getoption("lane_change_output_dir"))


@pytest.fixture(scope="session")
def selected_lane_change_cases(pytestconfig: pytest.Config, lane_change_suite):
    max_cases_opt = pytestconfig.getoption("lane_change_max_cases")
    max_cases = int(max_cases_opt) if max_cases_opt else None
    return select_cases(
        lane_change_suite.cases,
        case_id=pytestconfig.getoption("lane_change_case_id"),
        max_cases=max_cases,
    )
