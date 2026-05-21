
## Parameter-Efficient Domain Adaptation of Physics-Informed Self-Attention based GNNs for AC Power Flow Prediction

This workspace provides code and tools for parameter-efficient domain adaptation of physics-informed self-attention based Graph Neural Networks (GNNs) for AC power flow prediction. It includes training pipelines, model architectures, data handling, and experiment management for research and development in power systems using advanced GNN techniques.

## Project layout

- `train/`: training CLI + loop
- `models/`: model architectures
- `data_loading/`: dataset class, collation, decoding, samplers
- `datasets/`: parquet datasets
- `configs/`: YAML configs (see `configs/default.yaml`)
- `results/`: outputs (logs, checkpoints, plots)
- `scripts/`: helper scripts (e.g., `scripts/bootstrap_uv.sh`, `scripts/run_sweep.py`)

## Run


## How to Run Training

**Standard training:**

```sh
python -m train --config configs/default.yaml --EPOCHS=1
```

**With MLflow tracking (recommended):**

```sh
python -m train --config configs/default.yaml --mlflow --mlflow_tracking_uri sqlite:///mlflow.db --mlflow_experiment SimpleGNN --EPOCHS=1
```

**Start the MLflow UI:**

```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

> **Note:** MLflow's filesystem tracking backend (e.g. `file:./mlruns`) is deprecated as of Feb 2026. Use a SQLite backend as shown above to avoid deprecation warnings.

