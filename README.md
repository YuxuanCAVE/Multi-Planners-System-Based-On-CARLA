# CARLA Autonomous Driving Planning and Control Framework

## Project Overview

This repository implements a CARLA-based autonomous driving stack for route following and lane-change/overtake experiments. The codebase is built around a configurable runtime that composes scenarios, planners, controllers, logging, and evaluation from YAML.

Key points:

- CARLA version: **0.9.16**
- Simulation backend: CARLA synchronous stepping
- Local planners implemented in this repo:
  - `HybridAStarMapPlanner` in [`framework/planning/local/a_star.py`](framework/planning/local/a_star.py)
  - `FrenetPlanner` in [`framework/planning/local/frenet.py`](framework/planning/local/frenet.py)
- Controllers implemented in this repo:
  - Stanley controller
  - Pure Pursuit controller
- Built-in scenarios:
  - Route following
  - Lane change / overtake with scripted lead and adjacent vehicles
- Evaluation outputs:
  - per-tick logs
  - trajectory/control CSVs
  - tracking metrics
  - batch lane-change summaries

ROS2 status:

- No ROS2 nodes, `rclpy` imports, publishers, or subscribers are present in the current codebase.

## System Architecture

### High-Level Flow

```text
YAML config
  -> scripts/run.py dynamically imports scenario / planner / controller
  -> scenario.setup() loads CARLA world, spawns ego/actors, attaches optional sensors
  -> scenario.get_route() builds a global route from CARLA GlobalRoutePlanner
  -> runner.reset() passes route + CARLA map/world to the planner
  -> per simulation tick:
       world tick
       scenario.tick()
       ego/world state extraction
       planner.plan()
       controller.compute_control()
       ego.apply_control()
       recorder.step() + metrics.step()
       termination checks
  -> run artifacts written under runs/
```

### Module Breakdown

| Area | Main files | Responsibility |
| --- | --- | --- |
| Runtime / orchestration | `scripts/run.py`, `framework/runner.py` | Load config, instantiate modules, run the CARLA control loop, collect outputs |
| Shared types | `framework/core/types.py` | Common route, trajectory, ego state, obstacle, and planning result dataclasses |
| Scenarios | `framework/scenarios/base_scenario.py`, `framework/scenarios/lane_following.py`, `framework/scenarios/lane_change.py` | World loading, ego/actor spawning, route generation, scenario completion logic |
| CARLA I/O | `framework/carla_io/sensor.py` | Optional front RGB camera and radar attachment to the ego vehicle |
| Planning | `framework/planning/base_planning.py`, `framework/planning/local/a_star.py`, `framework/planning/local/frenet.py` | Planner interface plus concrete local planners |
| Planning support | `framework/planning/mapping.py`, `framework/planning/actor_model.py`, `framework/planning/lane_selector.py` | Static occupancy extraction, dynamic actor geometry/safety modeling, target lane selection |
| Control | `framework/control/stanley.py`, `framework/control/pure_pursuit.py`, `framework/control/vehicle/kinematics.py` | Lateral and longitudinal control plus bicycle rollout helpers |
| Evaluation | `framework/evaluation/recorder.py`, `framework/evaluation/metrics.py`, `framework/evaluation/lane_change_evaluator.py`, `framework/evaluation/result_writer.py` | Run logging, tracking metrics, lane-change case scoring, summary export |
| Testing / batch execution | `framework/testing/*.py`, `tools/run_lane_change_batch_tests.py`, `tests/*.py` | Scenario case definitions, case execution, pytest-based integration checks |

### Planning Stack

#### Hybrid A*

[`framework/planning/local/a_star.py`](framework/planning/local/a_star.py) implements the main Hybrid A* local planner used by the provided lane-following and lane-change configs.

Notable design choices:

- Builds a **static drivable occupancy map** from CARLA lane semantics once at reset.
- Crops a **local occupancy patch** around the ego each tick.
- Models the ego vehicle with a **two-circle collision approximation**.
- Separates dynamic obstacles into:
  - **hard collision** via oriented actor body geometry
  - **soft safety cost** via rectangular/elliptic safety regions
- Uses a **lane selector** to choose the target lane center for keep-lane vs pass-lane behavior without running multiple full searches.
- Optionally smooths the recovered path before resampling it into a time-parameterized trajectory.

#### Frenet Planner

[`framework/planning/local/frenet.py`](framework/planning/local/frenet.py) provides a second local planner based on route-relative `s-l` coordinates.

It uses:

- stable projection to the route reference line
- lateral offset sampling
- constant-speed trajectory rollout
- collision and curvature/yaw-rate validation
- cost-based candidate selection

### Control Stack

- [`framework/control/stanley.py`](framework/control/stanley.py): Stanley lateral control with optional lookahead target selection, steer-rate limiting, and simple proportional longitudinal control.
- [`framework/control/pure_pursuit.py`](framework/control/pure_pursuit.py): Pure Pursuit lateral control with PID-based longitudinal speed tracking.

### Simulation Integration

`framework/runner.py` is the central integration point with CARLA. It:

- connects to the simulator
- enables synchronous stepping
- spawns a collision sensor for termination detection
- converts CARLA actor state into lightweight planner-facing dataclasses
- filters nearby dynamic obstacles before passing them to the planner
- applies computed control commands back to the ego vehicle
- records logs and controller metrics

### Scenario Layer

- [`framework/scenarios/lane_following.py`](framework/scenarios/lane_following.py) loads a map, spawns ego/extra actors, and generates a route with CARLA's `GlobalRoutePlanner`.
- [`framework/scenarios/lane_change.py`](framework/scenarios/lane_change.py) extends that flow with a scripted lead vehicle and optional adjacent-lane rear vehicle for overtaking tests.

### Evaluation / Benchmarking

- [`framework/evaluation/recorder.py`](framework/evaluation/recorder.py) writes `meta.json`, `result.json`, `record.csv/json`, executed trajectory CSVs, control CSVs, planner path CSVs, and plots.
- [`framework/evaluation/metrics.py`](framework/evaluation/metrics.py) computes controller-facing metrics such as cross-track error, heading error, speed error, steer jitter, and saturation counts.
- [`scripts/benchmark_metrics.py`](scripts/benchmark_metrics.py) computes lightweight run summaries from saved artifacts.
- [`tools/run_lane_change_batch_tests.py`](tools/run_lane_change_batch_tests.py) runs case suites and writes batch summaries.

## Installation Guide

### 1. Prerequisites

- CARLA **0.9.16**
- Python environment compatible with your CARLA 0.9.16 PythonAPI build
- `pip`

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

- Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install / expose CARLA PythonAPI

This project imports both `carla` and CARLA's `agents.navigation.global_route_planner`, so your environment must expose:

- `CARLA_ROOT/PythonAPI`
- `CARLA_ROOT/PythonAPI/carla`

Example:

- Windows PowerShell:

```powershell
$env:CARLA_ROOT="D:\CARLA_0.9.16"
$env:PYTHONPATH="$env:PYTHONPATH;$env:CARLA_ROOT\PythonAPI;$env:CARLA_ROOT\PythonAPI\carla"
```

- Linux/macOS:

```bash
export CARLA_ROOT=/path/to/CARLA_0.9.16
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla
```

Note:

- [`scripts/run.py`](scripts/run.py) currently contains a hardcoded Linux `sys.path.append(...)` for one CARLA installation path. In a clean setup, prefer setting `PYTHONPATH` correctly or updating that file to match your machine.

### 5. Launch CARLA 0.9.16

Start the simulator before running the project. Example:

- Windows:

```powershell
CarlaUE4.exe -carla-port=2000
```

- Linux:

```bash
./CarlaUE4.sh -carla-port=2000
```

## Usage

### Run lane following

```bash
python -m scripts.run --config configs/lane_following.yaml
```

### Run lane change / overtake

```bash
python -m scripts.run --config configs/lane_change_actor.yaml
```

### Run batch lane-change evaluation

```bash
python tools/run_lane_change_batch_tests.py --cases tests/scenario_cases/lane_change/lane_change_cases.json
```

Optional filters:

```bash
python tools/run_lane_change_batch_tests.py --case-id overtake_easy_return --max-cases 1
```

### Run pytest integration tests

```bash
pytest tests/test_lane_change_scenarios.py --run-carla
```

### Visualize CARLA spawn points

```bash
python carla_spawn_point.py
```

## Project Structure

```text
configs/                     YAML experiment configs
framework/
  carla_io/                  Optional RGB camera and radar wrappers
  control/                   Stanley, Pure Pursuit, kinematic helpers
  core/                      Shared datatypes
  evaluation/                Recorder, metrics, evaluators, writers
  planning/                  Planner interfaces and planning utilities
  planning/local/            Hybrid A* and Frenet planners
  scenarios/                 Route-following and lane-change scenarios
  testing/                   Lane-change case models and runners
scripts/
  run.py                     Main single-run entry point
  batch_val.py               Multi-planner benchmark launcher
  benchmark_metrics.py       Post-run metric extraction
tests/                       Pytest integration tests and scenario case files
tools/
  run_lane_change_batch_tests.py
runs/                        Example committed run artifacts
carla_spawn_point.py         Spawn-point visualization helper
```

## Results / Benchmark

The repository already contains two example run folders under `runs/`. These are useful as reference artifacts, but they are not a large statistical benchmark.

Across the two committed example runs in `runs/`:

- success rate: **100%** (`result.json.reason == "reached_goal"` for both runs)
- collision count: **0 / 2**

Example artifact summary:

| Run artifact | Goal reached | Collision | Sim time (s) | Mean planning time (ms) | P95 planning time (ms) | Replan count | CTE RMSE (m) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `runs/lane_change_hybrid_astar_stanley` | Yes | No | 17.15 | 154.74 | 291.99 | 336 | 0.233 |
| `runs/town10_hybrid_astar_a_stanley_metrics` | Yes | No | 65.95 | 60.17 | 70.02 | 1291 | 0.239 |

Additional metrics available in the stored artifacts include:

- heading RMSE
- minimum obstacle distance
- steer saturation counts
- final trajectory/control CSV exports

## Future Work

- Remove the hardcoded CARLA PythonAPI path from `scripts/run.py` and replace it with a documented environment-variable based setup.
- Unify success/failure labels across `result.json`, `final_summary.json`, and `scripts/benchmark_metrics.py`.
- Add larger benchmark suites covering more towns, traffic densities, and controller/planner combinations.
- Add ROS2 interfaces if this project is intended to be integrated into a larger autonomy stack.
