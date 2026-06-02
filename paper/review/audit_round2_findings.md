# Audit round 2 — models, loaders, training, reconverge (2026-05-30)

Second brutal-scrutiny pass, complementary to round 1 (core math: `core_verification_findings.md`).
Round 1 verified J_z/J_A/S/transfer numerically; it did NOT test the *data* (a wrong
adjacency or scrambled labels would pass an FD check on whatever operator it was handed).
Round 2 closes that gap. Two parallel adversarial agents, each running its own count-tests
with `.venv/bin/python` on the real datasets.

## VERDICT: clean. No bug corrupts any measurement.

### Data loaders — all 6 OK (count-tested, not read-tested)
| Dataset | N | E (measured=canonical) | C | feat | λ_max(Â) | sym | self-loops | splits |
|---|---|---|---|---|---|---|---|---|
| Cora | 2708 | 5278 ✓ | 7 | 1433 | 1.000000 | 0 | 100% | disjoint |
| Citeseer | 3327 | 4552 ✓ | 6 | 3703 | 1.0 | 0 | 100% | disjoint |
| Pubmed | 19717 | 44324 ✓ | 3 | 500 | 1.0 | 0 | 100% | disjoint |
| Amazon Photo | 7650 | 119081 ✓ | 8 | 745 | 1.0 | 0 | 100% | disjoint |
| WikiCS | 11701 | 215603 | 10 | 300 | 1.0 | 0 | 100% | disjoint |
| Amazon Fraud | 11944 | — | 2 (11123/821) | 25 | 1.0 | 0 | 100% | disjoint + explicit alignment assert |

- **Normalization** = `D^{-1/2}(A+Aᵀ+I)D^{-1/2}` confirmed (λ_max=1.0 is the defining property; rebuild error 0.0). Not row-stochastic, self-loops present.
- **Alignment** X↔y↔A: neighbour-label homophily ≫ chance (.728–.937) for all six — would collapse to chance if misaligned.
- **Labels** in [0,C-1], all classes present, no sentinel. **Splits** pairwise-disjoint (no train/test leakage).
- `train_ignn` trains on `train_mask` only; consumer scripts use no held-out accuracy split → unused val/test can't leak.

### Model / training / reconverge — all OK
- **Spectral norm of W**: Miyato `parametrizations.spectral_norm` on the *same* W used in `operator`. Post-train **eval ‖W‖₂ = 1.00000** (cap c=1) → A3 contractivity holds at audit time. (Mid-training transient ‖W‖₂→1.395 as power vectors lag is harmless: all measurements use `model.eval()`.)
- **operator** `ReLU(Â Z Wᵀ + X_proj)`: order/ReLU correct; `X_proj=U(X)` computed once, sliced for subgraphs, never recomputed. (z-structure already FD-verified in round 1.)
- **reconverge** (THE critical helper, used by every damage/transfer number): clean residual 5.96e-8, perturbed 0.0; at the contractivity boundary (product=1.0) the 200-iter result **exactly equals** a 5000-iter reference (gap 0.0). `tol=1e-7` sits below the float32 floor so it always runs the full 200 iters — safe (more, not fewer). Damage values are true fixed-point-to-fixed-point distances, not under-converged noise.
- **forward** fixed point: residual rel 3.2e-6 at 50 iters (negligible vs damage magnitudes 4–130).
- **contractive_pf** (cut power-flow model): confirmed NOT imported by any kept experiment.

## Two benign convention notes (verified harmless; optional reviewer-proofing only)
1. **Cora Planetoid `features[test_idx]=features[test_idx_sorted]`** — the pattern the loader docstring warns is wrong for Citeseer. Proven a **no-op for Cora** (test block contiguous 1708–2707, `tx_rows==test_idx_sorted`, edges 5278 exact, homophily .848). Citeseer/Pubmed use the correct `tx_rows` form.
2. **Edge-multiplicity weighting** — `adj+adjᵀ` on an already-symmetric Fraud/Photo `homo` doubles edge weights to 2; WikiCS keeps raw multi-edge weights (≤19). λ_max stays 1.0 and the model is trained AND audited on the same matrix, so every AEGIS claim is self-consistent on it; deviation from strict-binary GCN is small (≤0.11 Fraud, ≤0.32 WikiCS, per entry). Optional: `adj.data[:]=1` before renorm, or a one-line comment.

## Standing (codebase end-to-end)
- Round 1 (core math): correct; bounded benign bugs B1 (√2 magnitude), B2 (σ₁ report), B3/B4 (high-ρ Neumann, Amazon Photo only) — PARKED, none touch rankings/transfer/theory/four-quadrant.
- Round 2 (models/loaders/training/reconverge): **clean**.
- Remaining code work = apply the parked B1–B4 fixes, re-run `verify_core_implementation.py` to 10/10, re-derive σ₁ magnitudes + re-verify Amazon Photo +0.996.
