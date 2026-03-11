# Modular configs: single planner example

For a **single planner run**, you should still provide one full YAML to `scripts/run.py`.

- You keep small modular files in:
  - `configs/carla/`
  - `configs/scenarios/`
  - `configs/planners/`
  - `configs/controllers/`
  - `configs/runners/`
- Then combine them into one runnable config.

## Lattice example

Use:

```bash
python scripts/run.py --config configs/examples/lattice_single_run.yaml
```

This example corresponds to:

- `configs/carla/local.yaml`
- `configs/scenarios/scenario01_town10hd_clear.yaml`
- `configs/planners/lattice.yaml`
- `configs/controllers/stanley.yaml`
- `configs/runners/single_run.yaml`

## Optional helper script

A helper script is provided:

```bash
python scripts/compose_config.py \
  --carla configs/carla/local.yaml \
  --scenario configs/scenarios/scenario01_town10hd_clear.yaml \
  --planner configs/planners/lattice.yaml \
  --controller configs/controllers/stanley.yaml \
  --runner configs/runners/single_run.yaml \
  --out configs/examples/lattice_single_run.yaml
```

> Note: this helper uses `PyYAML` (`yaml` package), same as `scripts/run.py`.