# Experimental Plan: Parameter-Efficient Domain Adaptation of Quantized Physics-Informed GATs for AC Power Flow

This document is the execution plan for the paper:

**“Parameter-Efficient Domain Adaptation of Quantized Physics-Informed Graph Attention Networks for AC Power Flow Prediction”**

Scope priorities (in order):
1. **Domain adaptation** (MV→HV and topology shift) with **LoRA** (parameter-efficient fine-tuning).
2. **Quantization for CPU deployment**: **PTQ (int8)** first; **QAT** second.
3. Optional: **QLoRA** (4-bit base + LoRA) only if feasible without derailing the above.

Design constraints:
- Do **not** break the current training setup.
- New capabilities must be **opt-in via config flags** and/or **new scripts**.
- All experiments must log **params/metrics/artifacts** via MLflow.

---

## 0) Reproducibility and logging (must-have)

### MLflow conventions
- Backend store (metadata): `sqlite:///results/mlflow.db`
- Artifact store (files): `file:./results/mlruns`
- Experiment naming: one experiment per major study (suggested below).

Required MLflow tags for every run:
- `domain_source`: e.g. `MVN_30000` or `MVN_mix`
- `domain_target`: e.g. `HVN_15000`
- `shift_type`: `voltage` | `topology` | `combined`
- `method`: `source_only` | `full_ft` | `lora_ft` | `ptq` | `qat`
- `seed`
- `target_budget`

Artifacts to log for every run:
- `config.yaml` (the exact config used)
- training log
- checkpoint(s)
- prediction metrics report (CSV/JSON)
- (for quantization) exported quantized model + benchmark results

### Fixed seeds
- Use **3 seeds** for primary results: `{42, 123, 999}`.
- Use **1 seed** for quick iteration.

---

## 1) Domains and splits

Assumptions:
- MV and HV datasets exist as Parquet files under `datasets/`.
- Each network ID indicates topology/scenario (e.g., `30000`, `30010`, `30020`, …).

### Domain definitions
- **Voltage regime domain**:
  - Source: MV (`MVN_*`)
  - Target: HV (`HVN_*`)
- **Topology domain**:
  - Each network ID is a topology domain (e.g., `MVN_30000` vs `MVN_30010`).

### Evaluation cases
Minimum publishable matrix:
- **Case A (Voltage shift)**: train on MV mix → test on HV mix
- **Case B (Topology shift within MV)**: train on MVN_a → test on MVN_b
- **Case C (Combined shift)**: train on MVN_a → test on HVN_b

### Target labeled-data budgets (for adaptation)
For fine-tuning on the target domain, use:
- `{0.1%, 0.5%, 1%, 5%, 10%, 50%}` of the target training split.

### Splitting protocol
- Keep the split method fixed across all methods.
- Recommended for paper: `split.mode: equal3` (already used) for consistency.

---

## 2) Methods to compare (per domain case)

For each domain case (A/B/C) and each target budget:

1. **Source-only transfer (no adaptation)**
   - Train on source domain.
   - Evaluate on target domain without updating weights.

2. **Full fine-tuning (upper bound)**
   - Load source checkpoint.
   - Fine-tune *all* parameters on target budget.

3. **LoRA fine-tuning (PEFT)**
   - Load source checkpoint.
   - Freeze base weights.
   - Inject LoRA into attention projection layers only.
   - Fine-tune only LoRA params (and optionally LayerNorm/bias in an ablation).

4. **Quantization for CPU**
   - Apply quantization to the best-performing FP32 model in each scenario:
     - **PTQ int8** (first)
     - **QAT int8** (second)

Optional (only if feasible):
- **QLoRA**: quantized base weights during fine-tuning + LoRA adapters.

---

## 3) Metrics (accuracy + physics + efficiency)

### Primary accuracy metrics (already in training)
- `rmse`
- `rmse_mag`
- `rmse_ang_deg`

### Physics consistency metrics (add for paper)
Compute on test sets:
- `mismatch/p_inf`: $\|\Delta P\|_\infty$
- `mismatch/q_inf`: $\|\Delta Q\|_\infty$
- `mismatch/combined_inf`: $\max(\|\Delta P\|_\infty, \|\Delta Q\|_\infty)$
- `violations/v_limit_frac`: fraction of buses with $V \notin [V_{min}, V_{max}]$

### Robustness under perturbations
Create perturbed test sets with multiplicative load perturbation on $(P,Q)$:
- perturbation levels: `{±5%, ±10%, ±20%, ±30%}`
- report accuracy and mismatch metrics as a function of perturbation

### Efficiency metrics (CPU-focused)
- Model file size (MB)
- Inference latency on CPU:
  - fixed batch size(s) (e.g., 1 and 64)
  - warmup iterations (e.g., 20)
  - timed iterations (e.g., 200)
  - report p50/p90 latency

---

## 4) Experimental outputs for the paper

### Tables
- **Table 1 (Domain adaptation)**: target-test accuracy for `source_only`, `full_ft`, `lora_ft` (mean ± std over 3 seeds)
- **Table 2 (PEFT efficiency)**: % trainable params + trainable count for LoRA vs full FT
- **Table 3 (Quantization)**: FP32 vs PTQ-int8 vs QAT-int8: accuracy, mismatch, size, latency

### Figures
- **Fig 1**: accuracy vs target budget (full FT vs LoRA)
- **Fig 2**: robustness curves under perturbations (RMSE + mismatch)
- **Fig 3**: latency vs accuracy tradeoff (FP32, PTQ, QAT)

---

## 5) Implementation plan (won’t break existing training)

### Phase 1 — Baseline lock-in
1. Add a baseline “paper” config (copy of default with explicit fields).
2. Run a short MLflow-tracked smoke run.
3. Freeze expected outputs/metrics format.

Acceptance criteria:
- Existing command still works.
- MLflow logs metrics + artifacts.

### Phase 2 — Domain adaptation fine-tuning
Add a *new* fine-tune script (recommended) or extend `python -m train` with a `finetune` mode.

Config keys to introduce (default no-op):
- `finetune.enabled: false`
- `finetune.resume_ckpt: <path>`
- `finetune.reset_optimizer: true/false`
- `finetune.load_strict: true/false`
- `finetune.target_budget: <fraction>`

### Phase 3 — LoRA (attention projections)
Implement LoRA injection for `nn.Linear` modules in:
- attention projections: q/k/v/out

Ablations:
- add FFN linears
- add heads

Config keys:
- `peft.enabled: false`
- `peft.method: lora`
- `peft.r: 4`
- `peft.alpha: 16`
- `peft.dropout: 0.05`
- `peft.target_modules: ["attn.q", "attn.k", "attn.v", "attn.out"]` (pattern-based)
- `peft.freeze_base: true`

Required logging:
- `%trainable_params`
- `trainable_param_count`

### Phase 4 — PTQ (CPU) as a separate script
Add `scripts/quantize_export.py`:
- loads a trained checkpoint
- applies PTQ for `nn.Linear` (dynamic int8)
- exports quantized model
- benchmarks CPU latency
- logs artifacts to MLflow

### Phase 5 — QAT (CPU)
Add QAT behind a config flag:
- `quant.qat.enabled: false`

QAT workflow:
- prepare model for QAT
- fine-tune for a small number of epochs
- convert and export quantized model

### Phase 6 — Robustness evaluation script
Add `scripts/eval_robustness.py`:
- loads a checkpoint/model
- evaluates on multiple perturbation levels
- writes a single CSV + plots
- logs to MLflow

---

## 6) Concrete run schedule (what to run to fill the paper)

### Experiments: Domain adaptation
For each shift case (A/B/C), for each target budget, for each seed:
- Train source model once per seed.
- Evaluate source-only transfer on target.
- Fine-tune full model on target.
- Fine-tune LoRA model on target.

Suggested MLflow experiments:
- `Paper_DomainAdapt_FP32`

### Experiments: Quantization
For each best FP32 model (per case) pick:
- best full-FT model
- best LoRA model

Then run:
- PTQ (int8) export + CPU benchmark
- QAT (int8) + export + CPU benchmark

Suggested MLflow experiment:
- `Paper_Quant_CPU`

---

## 7) Practical notes (to keep progress fast)

- Start with **Case A** (MV mix → HV mix) because it supports the headline claim.
- Keep QAT optional until PTQ results are solid.
- QLoRA is optional; do not block LoRA + PTQ/QAT on it.

---

## 8) Acceptance checklist (per feature)

- **Baseline**: unchanged behavior; logs cleanly to MLflow.
- **LoRA**: trainable params drop substantially; target accuracy close to full FT.
- **PTQ**: meaningful CPU speedup and size reduction with small accuracy loss.
- **QAT**: accuracy improves over PTQ at similar speed/size.
- **Robustness**: stable mismatch scaling under perturbations.
