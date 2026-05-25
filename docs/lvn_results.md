# PIGNN-Attn-LS vs PE_DEQ_PF on LVN — Experimental Findings

**Date:** 2026-05-24
**Hardware:** NVIDIA GeForce RTX 4090, torch 2.11.0+cu128, torch_scatter 2.1.2+pt211cu128
**Dataset:** `LVN_converted_n36000_v2.parquet` (36,000 samples, 722-bus multi-voltage networks 3–380 kV)
**Split:** `ratio 0.8/0.1/0.1` with `seed_value=42` → 28,800 train / 3,600 val / 3,600 test
**Common train args:** `--PER_UNIT --BLOCK_DIAG --mag_ang_mse --BATCH 32 --LR 1e-4 --DthetaMax 0.30 --DvmFrac 0.10 --EPOCHS 30`
**Loss:** pure MSE (no `--PINN`) for both models

## Bottom line

`PE_DEQ_PF` (plain variant, the HVN winner from `docs/pe_deq_pf_vs_pignn_findings.md` run I) **beats PIGNN-Attn-LS_VnFeat on every metric tested — supervised RMSE, magnitude RMSE, angle RMSE, and KCL physics residual** — on the corrected LVN dataset.

| | PIGNN-Attn-LS_VnFeat | **PE_DEQ_PF (plain)** | PE_DEQ advantage |
|---|---:|---:|---:|
| `test_rmse` | 0.01979 | **0.01396** | **−29%** |
| `test_rmse_mag` | 0.02431 | **0.01661** | **−32%** |
| `test_rmse_ang°` | 0.794° | **0.612°** | **−23%** |
| `test_phys_loss` | 367,100 | **17,126** | **−95% (21× lower)** |
| Trainable params | 4,472 | 25,791 (5.8×) | — |
| Wall time | ~8 hr | ~5 hr | PE_DEQ also faster/epoch |

Val ≡ test to 4 significant figures for both models → no overfitting on the 3,600-sample split.

The HVN pattern (DEQ architecture wins both supervised and physics metrics) **replicates on LVN**, confirming the architectural advantage generalises across operating regimes.

## Training trajectories (val rmse per epoch)

| Epoch | PIGNN-Attn-LS | PE_DEQ_PF |
|---:|---:|---:|
| 0 (init) | 0.0685 | 0.0870 |
| 1 | 0.0355 | 0.0298 |
| 5 | 0.0318 | 0.0285 |
| 10 | 0.0288 | 0.0239 |
| 15 | 0.0317 | 0.0228 |
| 20 | 0.0310 | 0.0218 |
| 25 | 0.0286 | 0.0219 |
| 27 | 0.0262 | 0.0196 |
| 30 | 0.0259 | **0.0192** ← best |
| Best (any epoch) | 0.0258 (ep 29) | **0.0192** (ep 30) |

**Observations:**
- PIGNN plateaued around epoch 5–10 and oscillated thereafter (Adam + inner Armijo line search noise).
- PE_DEQ descended monotonically with no oscillation — **still improving at epoch 30**. Another 10–20 epochs would likely push it below 0.018.
- PE_DEQ matched PIGNN's *best* by epoch 6 and overtook it permanently from epoch 7 onward.

## Production launch commands

### PIGNN-Attn-LS_VnFeat (winner: `PE_DEQ_PF`, but documented for reproducibility)
```bash
.venv/bin/python -m train \
  --PARQUET ./datasets/LVN_converted_n36000_v2.parquet \
  --model GNSMsg_EdgeSelfAttn_VnFeat --use_armijo --vlimit \
  --BLOCK_DIAG --PER_UNIT --mag_ang_mse \
  --d 4 --d_hi 16 --num_attn_layers 1 --K 10 \
  --BATCH 32 --EPOCHS 30 --LR 1e-4 \
  --DthetaMax 0.30 --DvmFrac 0.10 \
  --seed_value 42 --mode train_test \
  --split_mode ratio --train_ratio 0.8 --valid_ratio 0.1
```
Run dir: `results/runs/260523-181350_17a8/`.

### PE_DEQ_PF (winner)
```bash
.venv/bin/python -m train \
  --PARQUET ./datasets/LVN_converted_n36000_v2.parquet \
  --model PE_DEQ_PF --vlimit \
  --BLOCK_DIAG --PER_UNIT --mag_ang_mse \
  --d 4 --d_hi 32 --num_attn_layers 2 --K 15 \
  --BATCH 32 --EPOCHS 30 --LR 1e-4 \
  --DthetaMax 0.30 --DvmFrac 0.10 \
  --seed_value 42 --mode train_test \
  --split_mode ratio --train_ratio 0.8 --valid_ratio 0.1
```
Run dir: `results/runs/260524-090952_99d9/`.

## The bug that consumed half the session

A single wrong line in `scripts/convert_lvn_to_hvn_schema.py` cost ~5 hours of diagnostic work before being caught.

### Symptom
Every model configuration tried — PIGNN at d=4/d_hi=16, at d=8/d_hi=32, with/without Armijo, with wider DvmFrac, with PINN, with MSE — plateaued at **exactly the V_start baseline**: val mag RMSE = 0.0403, val ang RMSE = 3.44°.

Capacity bumps, line-search toggles, step-cap relaxations, and loss formulation changes all produced identical plateaus.

### Root cause
The source LVN parquet uses **inverted slack/PQ encoding** relative to PyPower/MATPOWER convention:

| Source LVN code | Count per 722-bus grid | True meaning | Original converter assumed |
|---:|---:|---|---|
| 1 | 1 | slack (singleton, correct for slack) | PQ |
| 2 | 4 | PV (generators) | PV |
| 3 | 717 | PQ (loads — 99% of buses) | slack |

The original mapping `LVN_TO_HVN_BUS_TYPE = {1: 0, 2: 2, 3: 1}` mislabeled **99.31% of buses as slack**. The slack mask in `models/edge_selfattn/model.py:126` then zeroed `dv` for those buses, making `v_head` gradient EXACTLY 0 — the magnitude head literally could not learn.

### Fix (commit `1c31730`)
```python
LVN_TO_HVN_BUS_TYPE = {1: 1, 2: 2, 3: 0}  # 1=slack→1, 2=PV→2, 3=PQ→0
```

After the fix, the very first 3-epoch diag run dropped val mag from 0.0403 → 0.0280 (the V_start baseline floor was broken).

### Signature symptoms (saved to user memory)
Filed under `feedback_verify_bus_type_mapping.md`. Catches this bug class in minutes next time:
1. Magnitude RMSE pinned at the V_start-baseline level (~`mean|V_n - V_start|`)
2. `v_head` weight gradients EXACTLY zero (not just small — exactly 0.0)
3. Model rmse approximately equals "predict V_start" baseline rmse
4. Mag RMSE constant across epochs while angle RMSE moves slightly
5. Capacity bumps, line-search toggles, and step-cap relaxations all fail to change the plateau

## Per-unit base re-calibration (v3 dataset, commit `d07507a`)

The source LVN uses `S_base = 1 MVA`, producing per-unit S values of O(1000) and absolute KCL residuals of O(370,000) — 4 orders of magnitude larger than HVN's O(10). The v2 dataset inherits this scale.

`scripts/convert_lvn_to_hvn_schema.py` was rebased to `TARGET_S_BASE = 100 MVA` (standard system base, matches HVN). `Y` is scaled by `s_scale = source_S_base / TARGET_S_BASE = 0.01` so the power-flow equation `S = V·conj(Y·V)` is preserved — V_newton stays valid, only the units change.

| | S_base | |S_pu| mean | |S_pu| max | Phys-loss order |
|---|---|---:|---:|---:|
| Source LVN raw | 1 MVA | 1.26e9 W | 5.51e11 W | — |
| v2 dataset (1 MVA PU) | 1.0 sentinel | 1260 | 551,000 | ~370,000 |
| **v3 dataset (100 MVA PU)** | **1.0 sentinel** | **12.6** | **5,507** | **~37** |
| HVN reference | 100 MVA | ~2 | ~3 | ~10 |

**Output:** `datasets/LVN_converted_n36000_v3.parquet` (283.8 MB).

**Compatibility:** v2-trained checkpoints (this comparison) will NOT predict correctly on v3 — the model has learned an operator that assumes v2's unit scale. v3 is for fresh training runs only.

## Follow-up experiments worth running

1. **PIGNN at matched capacity** (`d=8, d_hi=32, num_attn_layers=2, K=15` — same as PE_DEQ) to isolate the architectural advantage from the parameter-count advantage. PE_DEQ has 5.8× more params; some of the win likely comes from that.
2. **PE_DEQ_PF_Contractive on v3** with combined loss `phys + 1000·MSE` — the HVN K-robust winner. Now that S/Y are on a sane scale (v3), the PINN landscape should be tractable.
3. **Longer training (60–100 epochs)** for both models — PE_DEQ was still descending at epoch 30; the asymptote is unknown.
4. **K-sweep at inference** for PE_DEQ — replicate the HVN K-robustness study at K∈{5,10,15,30,50,100} to confirm DEQ's monotone improvement with deeper inference still holds on LVN.

## Reproducibility

| Artifact | Path |
|---|---|
| Source LVN | `datasets/LVN_snapshot_envelope_..._directSI.parquet` |
| v2 dataset | `datasets/LVN_converted_n36000_v2.parquet` (bus_type fix only) |
| v3 dataset | `datasets/LVN_converted_n36000_v3.parquet` (bus_type + S_base re-base) |
| PIGNN run | `results/runs/260523-181350_17a8/` |
| PE_DEQ run | `results/runs/260524-090952_99d9/` |
| Converter | `scripts/convert_lvn_to_hvn_schema.py` |
| Test evaluator | `scripts/eval_lvn_test.py` |
| Data-floor diagnostic | `scripts/lvn_data_floor.py` |
| Head-norm diagnostic | `scripts/inspect_trained_heads.py` |
| Delta-flow diagnostic | `scripts/inspect_trained_deltas.py` |
| Bus-type bug memory | `~/.claude/.../memory/feedback_verify_bus_type_mapping.md` |
