RL Auto Router (Phase 0)
========================

This repository scaffolds an RL-based PCB auto-router with KiCad integration.

Modules
-------

- `env/`: routing environments (Gym-like)
- `algos/`: training/inference algorithms and trainers
- `heuristics/`: classical routing baselines (A*, Lee, etc.)
- `kicad_plugin/`: KiCad ActionPlugin for online inference
- `common/`: shared utilities, rules, geometry
- `scripts/`: convenience scripts for demos and CI
- `tests/`: unit and smoke tests

Setup
-----

1) Create and activate a virtual environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install dev dependencies:

```bash
pip install -e .[dev,infer,log]
```

3) Optional training (choose one):

```bash
pip install -e .[train]
# or use conda for CUDA-enabled torch
```

UV Setup（可选）
----------------

```bash
uv sync --extra dev
```

注意：不要只运行 `uv sync`；否则开发依赖（如 `pytest`）不会被安装。

KiCad Environment
-----------------

- Target: KiCad 7/8
- Units: internal nm, UI mm; layer mapping via `pcbnew` API
- Plugin packaging: see `scripts/package_plugin.sh`

Experiment Logging
------------------

- Choose `wandb` or `mlflow`. Configure via `configs/default.yaml`
- Disable by setting `logging.backend: none`

CI
--

- Runs pytest smoke tests
- Style/type checks: ruff, black, mypy
- Builds plugin zip and a dummy ONNX export to validate packaging

License
-------

MIT
