
# GNN Load Flow

This workspace contains a GNN-based load-flow training pipeline.

## Project layout

- `train/`: training CLI + loop
- `models/`: model architectures
- `data_loading/`: dataset class, collation, decoding, samplers
- `datasets/`: parquet datasets
- `configs/`: YAML configs (see `configs/default.yaml`)
- `results/`: outputs (logs, checkpoints, plots)
- `scripts/`: helper scripts (e.g., `scripts/bootstrap_uv.sh`, `scripts/train.py`)

Compatibility shims for older imports/paths are kept at the repo root (e.g. `train_valid_test2.py`).

## Run

Preferred:

- `python -m train --config configs/default.yaml --EPOCHS=1 ...`

MLflow (recommended: sqlite tracking backend):

- `python -m train --config configs/default.yaml --mlflow --mlflow_tracking_uri sqlite:///mlflow.db --mlflow_experiment SimpleGNN --EPOCHS=1 ...`
- UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`

Note: MLflow's filesystem tracking backend (e.g. `file:./mlruns`) is deprecated in recent MLflow releases (Feb 2026 timeframe). Use sqlite for the tracking backend to avoid the warning.

Also supported:

- `python run_train.py --config configs/default.yaml ...`
- `python scripts/train.py --config configs/default.yaml ...`
- `python train_valid_test2.py --config config.yaml ...`

