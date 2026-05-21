# Critical Code Review — SimpleGNN

Date: 2026-05-21
Scope: Whole repo (≈8.6 KLOC Python, excl. `.venv`, `backup/`, `mlruns/`, `outputs/`).

## 1. Redundant / unnecessary files

| Path | LOC | Verdict | Reason |
|---|---|---|---|
| `backup/legacy_code/gnsmsg_edge_selfattn_armijo.py` | 428 | **Delete** | `backup/` is in `.gitignore` but the file still sits on disk and matches grep/IDE searches. Git history is the source of truth. |
| `scripts/quantize_export.py` | 500 | **Decide** | Untracked. Last commit `a93ef3d` removed the legacy quantization export. This is either a forgotten rewrite or stale — commit it intentionally or delete. |
| `scripts/train.py` | 7 | **Delete** | Third entrypoint that does the same `from train.entrypoint import main` as `run_train.py` and `train/__main__.py`. No script imports it; not referenced in any config or README run-command. |
| `models/gnsmsg_edge_selfattn.py` | 13 | **Inline & delete** | Self-described "compatibility shim". Re-exports `GNSMsg_EdgeSelfAttn` from `models.edge_selfattn`. Adds an extra hop in every import. Migrate callers to `from models.edge_selfattn import GNSMsg_EdgeSelfAttn` and delete the shim. |
| `run_train.py` vs `train/__main__.py` | 12+7 | **Keep one** | Both invoke `train.entrypoint.main`. Pick `python -m train` (which uses `__main__.py`) as the single canonical CLI, drop `run_train.py`, OR vice versa. Documenting "preferred: `python -m train`" then shipping the alternative is mixed signals. |

Net win: ~960 LOC + 4 files removed, no behavior change.

## 2. God modules / SRP violations

### 2.1 `train/entrypoint.py:main()` — **506 lines, one function**
This is the worst offender. `main()` linearly performs:
1. arg parsing dispatch
2. repo-root resolution + run slug/id naming
3. MLflow run setup + tag/param logging + code snapshotting
4. dataloader construction
5. model construction + weight init
6. LoRA application + freeze/unfreeze policy
7. optimizer/scheduler construction
8. training loop launch
9. test evaluation
10. artifact writing (CSV, JSON) + plotting

Decompose into:
- `RunContext` (paths, slug, MLflow handle) — built in `setup_run()`
- `build_components(cfg) -> (model, loaders, optim)` — pure assembly
- `apply_peft_policy(model, cfg)` — encapsulates LoRA+freeze+unfreeze
- `Trainer.run(ctx, components)` — orchestrates `train_validate` + `evaluate_test` + artifact emission

Aim for `main()` ≤ 50 lines, each step ≤ 80.

### 2.2 `train/cli.py` — **752 LOC, ~70 `add_argument` calls + 150-line `DEFAULTS`**
Conflates: argparse spec, defaults table, dataclass schema, YAML loader, env-var path expansion, validation. Violations:
- **SRP**: split into `cli/defaults.py`, `cli/schema.py` (dataclass), `cli/parser.py` (argparse), `cli/loader.py` (YAML+env merge).
- **OCP**: every new flag forces edits in 3 places (DEFAULTS dict, dataclass, `add_argument`). Drive argparse from the dataclass (e.g., `dataclasses.fields` + a small `Field(... cli=...)` helper) so a new flag is added in *one* place.

### 2.3 `models/edge_selfattn/model.py:GNSMsg_EdgeSelfAttn` — 279 LOC, 117-line `forward`
Better than the others (private helpers are factored), but the class still owns:
- architecture (attention blocks, heads)
- iterative solver (`_armijo_alpha`, K-step unroll)
- physical constraints (`_apply_constraints`, `_wrap_theta`)
- mismatch metric

These are three responsibilities. Extract:
- `SolverStrategy` (Armijo / fixed-γ — Strategy pattern; `use_armijo` flag selects implementation)
- `ConstraintProjector` (theta wrap + voltage clip)
- Keep `GNSMsg_EdgeSelfAttn` as the *learned* component; the unroll lives in a `PowerFlowUnroller` that composes (model, solver, projector).

### 2.4 `train/loop.py` (331 LOC)
Mixes train / validate / evaluate / history. Acceptable but watch for growth — split into `loop/train.py`, `loop/eval.py`, `loop/history.py` when this crosses 500.

## 3. SOLID review

- **S (SRP)** — violated in §2.1–§2.3 above.
- **O (OCP)** — `train/modeling.py:create_model` uses string dispatch on `cfg.model_name`. Adding a model requires editing this function *and* `DEFAULTS["model"]` *and* possibly `peft_utils` target lists. Use a small **registry** (`MODEL_REGISTRY: dict[str, Callable]`, with `@register("name")` decorator).
- **L (LSP)** — N/A; single concrete model class.
- **I (ISP)** — `TrainConfig` is a kitchen-sink dataclass passed wholesale to most functions (`cfg.PINN`, `cfg.K`, `cfg.parquet_paths`, …). Group into sub-configs: `DataCfg`, `ModelCfg`, `OptimCfg`, `PeftCfg`, `RunCfg`. Functions then take only the slice they need.
- **D (DIP)** — `entrypoint.main` directly imports concrete `mlflow_utils`, concrete `MultiBucketBatchSampler`, concrete plotting. Acceptable for a small research codebase, but the MLflow coupling means there is no way to swap a tracker (e.g. WandB or noop) without editing the entrypoint. A 5-line `Tracker` protocol would cost nothing.

## 4. Design patterns missing

| Need | Current | Suggestion |
|---|---|---|
| Model selection | `if model_name == "X": …` | **Registry / Factory** keyed by name. |
| Solver step (Armijo vs γ) | `if self.use_armijo:` inside `forward` | **Strategy** object injected via `__init__`. |
| LoRA + freeze sequence | Imperative calls in `main()` | **Adapter / Builder** (`PeftPolicy(cfg).apply(model)`). |
| Tracker (MLflow) | Hard import in entrypoint | **Protocol/Port** with concrete `MlflowTracker` and `NullTracker`. |
| Plot suite | `train/plotting.py` + 5 `scripts/plot_*.py` | Extract shared `viz/` module; scripts become thin CLIs over it. |

## 5. Modular design

✓ **Good** package layout: `data_loading/`, `models/`, `train/`, `scripts/`, `configs/`.
✗ `scripts/` is a junk drawer (15 ad-hoc scripts, no shared utility module). The five `plot_*` scripts and `train/plotting.py` duplicate matplotlib boilerplate — extract a `viz/` package.
✗ Two near-duplicate naming pairs: `train/logger.py` (logger config) vs `train/logging_utils.py` (path helpers — *unrelated to logging*). Rename `logging_utils.py` → `run_paths.py`.
✗ Three LoRA scenario YAMLs (`lora_ft_hv.yaml`, `lora_ft_hv_heads.yaml`, `lora_ft_hv_r2_a8.yaml`) differ by 2–3 fields each. Collapse to one template plus CLI overrides, or generate via `scripts/run_sweep.py`.
✓ `models/edge_selfattn/` is correctly decomposed into `admittance.py`, `attention.py`, `mismatch.py`, `model.py`.

## 6. Recommended action list (priority order)

1. Delete `backup/legacy_code/`, `scripts/train.py`, decide on `scripts/quantize_export.py`.
2. Pick one CLI entrypoint; remove the other.
3. Decompose `train/entrypoint.py:main` (§2.1).
4. Split `train/cli.py` into 4 files (§2.2).
5. Introduce a model registry (replace string dispatch in `train/modeling.py`).
6. Group `TrainConfig` into sub-configs.
7. Rename `logging_utils.py` → `run_paths.py`.
8. Inline the `models/gnsmsg_edge_selfattn.py` shim; update imports.
9. Extract shared `viz/` module from `train/plotting.py` + `scripts/plot_*.py`.

Items 1–2 are pure deletions, zero risk; do them first.
