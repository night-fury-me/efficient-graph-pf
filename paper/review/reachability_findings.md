# Reachability go/no-go — is adversarial criticality reachable?

**10-PREFERRED-SEED VALIDATED (2026-06-02):** κ₀=0.5 → γ=1.06±0.09, ε_reach/ε\*=1.41±0.12,
ε_reach/ε_crit=2.17±0.18; κ₀=0.9 → γ=1.02±0.02, ε_reach/ε\*=1.51±0.18, ε_reach/ε_crit=8.72±1.03.
(γ=1 exponent + the ~1.4–1.5× active-fraction factor + the 2.2–8.7× norm-cert conservatism all
hold at 10 seeds.) 2 seeds initially failed on a transient GPU `cusolverDnXgeev` eigvals error →
fixed with a CPU fallback in `rho_eval`; all 10 clean. Per-seed detail below is seed 42.

**Script:** `scripts/exp_reachability.py` · **Data:** `results/exp_reachability.csv`
**Question:** can a budget-bounded WORST-CASE (critical-driving) attack drive a trained
implicit GNN to criticality (ρ(J_z)→1)? This gates the whole pseudospectral-criticality
breakthrough thesis.

## Method (debug-first + verified attack)
- **Debug-first:** the prior "unreachable" verdict (`exp_phase_transition.py`,
  amplification≈1.0008) was an **ARTIFACT** — that experiment varies κ by *retraining* and
  measures the **v₁ shift at a fixed tiny ε=0.01**; it never runs a critical-driving
  attack. Threat model confirmed: perturbation is added to the normalized Â directly, no
  renormalization, so ρ(Â+δÂ) CAN exceed 1.
- **Critical-driving attack:** projected gradient ascent (tangent-space, Frobenius sphere)
  maximizing ρ(J_z(Â+δ)) over symmetric edge-supported δ, ‖δ‖_F=ε, seeded with the analytic
  rank-1 warm start (top eigvec of Â). Toy self-check hits the analytic optimum exactly;
  J_z built via `diag(vec mask)·kron(Â+δ,W)`, verified bit-identical to `compute_jacobian`.
- Baselines: v₁ (the old S_c-optimal direction) and random. Margins κ₀∈{0.5, 0.9}.

## Result — REACHABLE in principle, with a γ=1 critical divergence
| κ₀ | ρ₀ (clean) | ε_crit | ε* | **ε_reach** | ε_reach/ε* | ε_reach/ε_crit | **γ (resolvent)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.219 | 1.000 | 1.518 | **2.273** | 1.50 | 2.27 | **1.023** |
| 0.9 | 0.390 | 0.111 | 0.629 | **1.060** | 1.69 | 9.54 | **1.021** |

- **The phase transition is real and clean.** The resolvent norm diverges as
  ‖(I−J_z)⁻¹‖ ∼ (1−ρ)^(−γ) with **γ = 1.02 on BOTH margins** (the predicted simple-pole
  exponent). The full nonlinear equilibrium genuinely destabilizes at the crossing
  (50/50 prediction flips, reconverged ρ→∞) — not a linearization artifact.
- **ε\* (spectral) governs, ε_crit (norm) does not.** ε_reach = 1.5–1.7×ε* on both
  margins, while ε_crit under-predicts by 2.3× (κ=0.5) → 9.5× (κ=0.9). The norm certificate
  is the wrong, increasingly-conservative quantity; the spectral budget is the right order.
- **The critical-driving attack is the right object.** At the near-critical κ₀=0.9 the old
  **v₁ attack caps at ρ=0.966 and CANNOT reach criticality**; only critical-driving does.
  (At the wide κ₀=0.5 margin v₁ does eventually reach ρ=1, but less efficiently.)

## The decisive caveat — criticality is a DISTANT limit, not a realistic threat
‖Â‖_F = 1.77 (50-node ego subgraph, ρ(Â)=0.52). So:
- κ₀=0.5: ε_reach = 2.27 = **128% of ‖Â‖_F**; κ₀=0.9: ε_reach = 1.06 = **60% of ‖Â‖_F**.
- The paper's own realistic budgets are ε ≤ 0.2. **Criticality is reached at 5–11× the
  largest realistic budget** — a perturbation comparable to *replacing the whole adjacency*.

Trained models sit at ρ₀ = 0.22–0.39 (a large robustness margin), and even aggressive
near-critical training (κ₀=0.9) lands at ρ₀=0.39 with ε_reach=60% of ‖Â‖. **Realistic-
budget robustness is governed by the first-order sensitivity σ₁(S_c) at the operating
point (the original AEGIS object), NOT by proximity to criticality.** The original paper's
breach rates (predictions changing at ε≤0.2) occur with ρ staying far below 1.

## Verdict for the breakthrough
- **GO (literal):** adversarial criticality IS reachable, exhibits a clean γ=1 pseudo-
  spectral phase transition, and ε* (not ε_crit) is the governing budget — confirmed at
  10/10... (seed 42 here; 10-seed pending).
- **NO-GO (the grand claim):** "criticality governs adversarial robustness" is **NOT
  supported** — criticality is a distant large-perturbation limit (≈‖Â‖_F scale), so it
  does not govern robustness in the realistic regime. Selling it as the governing law would
  be refuted by any reviewer computing ε_reach/‖Â‖_F.

## What genuinely survived (real, rigorous, reusable)
1. **γ=1 critical exponent** — proved (simple pole) + measured 1.02 on both margins.
2. **Spectral margin ≫ norm certificate** — the true breaking budget ε* (and ε_reach) is
   2–10× the conservative ε_crit; implicit GNNs are robust at a *measurable spectral margin*.
3. **Critical-driving attack** — a new, more efficient structural attack (the matching
   lower bound / two-sided companion to the certificate).
4. **C4 unification** (proved earlier) — one operator S_c → attack, certificate, defense.

## Implication (pivot)
The criticality-as-governing-law breakthrough is dead. The survivors point to a different,
honest, strong target: **the first TIGHT structural robustness certificate for implicit
GNNs** — ε* + the 1.5–1.7× nonlinear (active-fraction) correction predicts ε_reach within
a small constant, the critical-driving attack supplies the matching lower bound, and the
spectral margin is shown to be 2–10× the conservative norm certificate. Tight + two-sided +
a clean γ=1 structure. See `breakthrough_plan.md` (pivot logged).
