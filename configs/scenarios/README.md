Scenario overlays

These YAML files are *overlays* applied on top of a base config via `scripts/run_sweep.py`.

- Base config: full training setup (`configs/default.yaml`)
- Scenario overlay: only the changes for a specific method (baseline / LoRA / QLoRA / PTQ / QAT etc.)

Examples

- Baseline sweep:
  - `python scripts/run_sweep.py --base configs/default.yaml --scenario configs/scenarios/baseline.yaml --seeds 42-44`

- LoRA sweep (placeholder until LoRA is implemented):
  - `python scripts/run_sweep.py --base configs/default.yaml --scenario configs/scenarios/lora.yaml --seeds 42 43 44`

Forward extra training args to `python -m train` by putting them after `--`:

- `python scripts/run_sweep.py --base configs/default.yaml --scenario configs/scenarios/baseline.yaml --seeds 42-44 -- --EPOCHS 50 --BATCH 256`
