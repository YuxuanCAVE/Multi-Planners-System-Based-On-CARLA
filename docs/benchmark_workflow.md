# Benchmark 组织建议（统一场景 + 指标 + Profiling + 公平对比）

建议**拆成多个 py 文件**，不要全部塞进一个脚本。

## 推荐拆分

- `scripts/run.py`
  - 单次实验入口（已有）。
- `scripts/batch_val.py`
  - 批量跑不同 planner，但保持同场景、同初始条件、同指标。
- `scripts/benchmark_metrics.py`
  - 统一指标定义与聚合：成功率、耗时、碰撞、越界、重规划次数、规划耗时统计（均值/P95）。
- `framework/runner.py`
  - 每步写入规划总耗时（`timing.plan_total_ms`）。
- `framework/evaluation/recorder.py`
  - 把 `debug.timing.*` 展平到 `record.csv`（列名 `timing_*`）。

## 为什么不要一个文件

- 统一指标是“评测逻辑”；跑 CARLA 是“执行逻辑”；两者耦合会很快失控。
- 后续扩展新指标/新 planner 时，分层后改动最小。
- 对公平性要求高：批量脚本只负责覆盖变量（planner），不改其他条件。

## 公平对比最低规范

- 固定：
  - 地图、天气、交通流、spawn 点、目标点、route source。
  - runner 配置（`fixed_delta_seconds`、`max_steps`、`obstacle_range_m` 等）。
- 只允许变化：
  - planner 类与 planner 超参。
- 输出：
  - 每次 run 的 `result.json + record.csv`。
  - 汇总表 `runs/benchmark_comparison.md`。

## 已加的性能项

- 规划 profiling：Runner 每次调用 planner 后记录 `plan_total_ms`。
- 碰撞检查加速（Frenet）：
  - 先做轨迹包围盒粗筛 obstacle；
  - 使用平方距离做碰撞判定，减少不必要开方；
  - 仅对候选障碍物做精算。
<<<<<<< ours
<<<<<<< ours
=======
=======
>>>>>>> theirs

## 一行命令跑批量对比（推荐）

先在 YAML 里写待测 planner 列表（示例见 `configs/benchmark_suite_example.yaml`），然后直接：

```bash
python scripts/batch_val.py --suite configs/benchmark_suite_example.yaml
```

如果你临时想热插拔一个 planner，不改 suite，也可以追加：

```bash
python scripts/batch_val.py --suite configs/benchmark_suite_example.yaml \
  --planner framework.planning.local.frenet:FrenetPlanner
```
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
