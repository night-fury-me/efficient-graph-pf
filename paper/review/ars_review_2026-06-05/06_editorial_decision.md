# Editorial Decision & Revision Roadmap

**Paper:** *AEGIS: A Matrix-Free Operator — Audit, Certify, and Defend Graph Neural Networks*
**Venue:** AAAI-2026 · **Panel:** EIC + R1 (methodology/theory) + R2 (domain) + R3 (impact) + Devil's Advocate
**Decision date:** 2026-06-05

## DECISION: **MAJOR REVISION**

The contribution is real and the scholarship is, in places, exemplary — but the submission's *front matter systematically overstates its own appendix*, and one headline certificate is not proven as written. Both are fixable without overturning the work; neither can be waved through. The Devil's Advocate raised one CRITICAL item (the `ε_crit` proof gap, independently identified by R1), which per panel rules precludes Accept on this version.

---
## Consensus across ≥4 reviewers (highest-priority, act on all)

1. **[5/5] Headline ≠ delivery.** EIC, R1, R2, R3, and DA all independently flag that the abstract/intro quote the most favorable projection of each result while the appendix quotes the honest one. This is the paper's defining problem and the through-line of the revision.
2. **[R1+DA CRITICAL · EIC endorses] The `ε_crit` subcritical certificate is not established by its proof for partially-active ReLU.** `app:proof_phase` eq. `Jzp-bound` proves safety only to `1/‖W‖−‖Â‖`, not to the advertised `(1−κ)/‖W‖` with κ=‖J_z‖; equality needs all-active (A4). **Blocking.**
3. **[R1+R2+DA] "τ=0.99" credits the edge weight to the operator.** The defensible contribution is the *marginal* `Δτ≈+0.25` of `v_ij` over a weight-only ranking; unweighted `v_ij` is +0.32 median with one negative cell.
4. **[R1+R3+DA] The "optimal attack" maximizes equilibrium shift, not misclassification** (0–1.8% flips at the headline ε=0.10). The "74–156×" advantage is on `‖ΔZ*‖`, not on decisions.
5. **[R1+DA · EIC endorses] Flagship `τ=+0.996` is measured at κ≈1.00**, outside the κ<1 regime that `prop:transfer` and the certificate assume.
6. **[R1+DA] Smoothing "vacuity" is a ball-matching artifact;** the real, defensible win over smoothing is *cost* (10^3–10^4×, zero-sample).

## Majority issues (≥2 reviewers)
- **[R1+R2+R3] Uneven "at scale":** audit scales to N=7,650, but certify is N=200-dense, defense is Cora-only, `ε_crit` is IGNN-only. The unified-at-scale story holds only for the audit.
- **[R2+R3] Defense novelty is thin** (a spectral/Lipschitz-style penalty) and **delocalization** (`app:ablations`) undercuts per-edge actionability; no matched spectral-norm baseline.
- **[R2+R3] Contractive-IGNN restriction** confines the unique theory to a model class practitioners rarely deploy; `S_K` for explicit GNNs is built by `O(|E|)` finite differences, breaking the "one query" claim there.

## Disagreements / arbitration
- **Severity of the unification critique.** DA argues it may be *definitional* (the same Jacobian necessarily appears in all three formulas); R2 sees genuine value in the *coupling* and its anticorrelation evidence. **Arbitration:** the coupling is the paper's best novelty, but DA-M1 is right that *no experiment isolates "value of one operator" from "value of having all three tools."* Required: an ablation/argument that the shared operator buys something the union of off-the-shelf tools does not.
- **Power-grid framing.** R3 wants it demoted to one sentence; EIC agrees it is a self-inflicted wound. No dissent. **Arbitration:** demote, and move the "audit the model, not the physics" scoping into the intro.
- **Accept-ability ceiling.** R2/R3/EIC believe the *honest, narrower* paper clears the bar; DA believes only the appendix's paper does. **These agree in practice:** revise the front matter down to the appendix's claims and the paper is competitive.

---
## Revision Roadmap (prioritized; ready for `academic-paper` revision mode)

### P0 — Blocking (must fix for the decision to flip)
1. **Repair the `ε_crit` proof.** *(Independently verified by `ml-theory-reviewer`, 2026-06-05 — see `07_theory_verification_ecrit.md`. Verdict: real gap, MAJOR, blocks in current wording; reproduced from primary source with rng-seeded numerics.)* The written Step 2 (`eq:Jzp-bound`) proves only the radius `ε_suff = 1/‖W‖−‖Â‖`; the stated `ε_crit=(1−κ)/‖W‖` (κ=‖J_z‖) is **strictly larger** for partial-ReLU (e.g. proven 0.111 vs stated 0.513 in one instance), so the certificate claims safety on `(ε_suff, ε_crit]` where the proof gives none, and the "(A1)–(A3) alone" header silently consumes (A4). **Correct fix (one line):** certify the **mask-agnostic** radius `ε_crit^suff := max(0, 1/‖W‖−‖Â‖)` via "M'=diag(φ') at z*' is a 0/1 projection, so `‖J_z'‖=‖M'(Â'⊗W)‖ ≤ ‖Â'‖‖W‖` unconditionally for every mask"; demote `(1−κ)/‖W‖` to a reported *operating margin*. Consumes A1+A2+threat-bound only — not A3/A4. **Do NOT** use the triangle bound `‖J_z'‖ ≤ κ+ε‖W‖`: verification showed it is *also unsound* over finite ε because the equilibrium crosses ReLU regions inside the ball (mask flips at ε≈0.22 ≪ ε_crit=0.54; adversarial violation +0.29), dropping a `(M'−M)(Â⊗W)` term. Alternative (weaker): keep `(1−κ)/‖W‖` but add (A4) all-active scope and fix the header. **The same one-line fix also closes `thm:cf2s`'s lower side (i)**, which re-uses `eq:Jzp-bound` verbatim and inherits the identical gap (verified in `08`) — one repair, two sites, no additive risk.
2. **Re-headline the transfer result.** Lead with the marginal `Δτ≈+0.25` of `v_ij` over the edge-weight baseline; report the weight-only ranking as the primary baseline in the main text; keep weighted/unweighted distinct and labeled everywhere (abstract included).
3. **Reconcile κ≈1.00.** Either run the headline transfer inside the certified regime (κ<1), or relabel `+0.996` as an empirical regularity that *persists past* contraction — and stop offering it as evidence for the contractive theory.
4. **Relabel the "attack."** State the attacked quantity (`‖ΔZ*‖`) and the actual flip rates at the same budget wherever the "optimal attack"/"74–156×" claims appear; or re-derive the advantage on a flip/margin metric.

### P1 — Major (expected by reviewers, strongly shapes scores)
5. **Align abstract/intro with the appendix's honesty** on the conformal guarantee: state coverage as "sound under (C1) exchangeability, empirically gated," not as an unconditional nominal-level guarantee.
6. **Fix the smoothing framing:** lead with cost; present both balls even-handedly; note conformal-set vs certified-radius are different guarantees.
7. **Isolate the value of the unification** (answer DA-M1): an experiment or argument that the single coupled operator outperforms / simplifies the union of an IFT attack + conformal + spectral-norm defense.
8. **Add a matched spectral-norm defense baseline** (answer R2/DA), and state the operational answer to delocalization (monitoring-prioritization vs remediation).
9. **State the scoping in the contributions list:** audit scales; certify is N=200 dense; defense is Cora/IGNN; `ε_crit` is IGNN-only; explicit-GNN `S_K` is finite-difference.
10. **Expand the SOTA head-to-head** beyond two rows: GR-BCD/PR-BCD across datasets and a budget sweep, including where AEGIS loses; an explicit threat-model positioning table vs AGNNCert / convex-relaxation / collective-smoothing certifiers.

### P2 — Minor (polish; cheap, raises clarity)
11. Make the APS worst-case in `lem:score-shift` rigorous, or scope the closed-form `Δ_r` to TPS.
12. Give the n / unit-of-analysis behind every p-value; avoid pseudo-replication.
13. Demote or honestly re-label the Fig. 1 radar semantics; demote power-grid/drug motivation to one "potential applications" sentence.
14. Engage influence-function fragility (Basu et al.) and connect your finite-difference checks as the rebuttal; add one sentence distinguishing `S_c` from per-edge over-squashing.
15. Reconcile the diagnostic-only release policy with the demonstrated offensive utility of `A_ij·v_ij`.

### Theory-verification ledger (independent `ml-theory-reviewer` passes — `07`, `08`)
Three sites were probed for the local-κ (`‖J_z‖`) vs global-spectral (`‖Â‖‖W‖`) conflation. Outcomes:
- **`ε_crit` subcritical certificate (`07`) — CONFIRMED gap · MAJOR · blocking.** Fix = mask-agnostic radius (P0 #1).
- **`thm:cf2s` lower side (i) (`08`) — CONFIRMED, but the SAME defect** (re-uses `eq:Jzp-bound`). The P0 #1 fix closes it too; not additive risk.
- **`thm:cf2s` upper side / "2–9× true break" (`08`) — framing only · MINOR–MODERATE [→ P1].** Part (ii) is sound and correctly all-active-scoped (it controls `ε_br^all`). But the abstract/intro "2–9×" is the *empirical* `ε_reach/ε_crit` from `rem:obs_o1` (a conjecture with two named open gaps), **not** the bracket's proven constant. *Fix:* label the 2–9× "empirical/measured" and do not let "deterministic guarantee" adjoin it.
- **Bracket constant `C/β` (`08`) — MINOR · tightness/honesty [→ P2].** The reported `≲16×` plugs the *partial* κ∈[0.14,0.59] into a `C` *derived* under all-active κ; the self-consistent value is `≈49–155×`. The bound was never violated (0/802 real models), so this is a reporting inconsistency, not a broken bound — but the bracket is looser than advertised. *Fix:* compute `C/β` consistently, or state which κ is used.
- **`‖z*‖ ≤ ‖X_proj‖/(1−κ)` curvature denominator (`08`) — REFUTED · NO gap.** At a ReLU fixed point the active coordinates solve the linear system exactly, so `vec(z*)=(I−J_z)⁻¹ D_M vec(X_proj)` with `‖D_M‖≤1`, making the **local-κ** bound rigorous (held 9909/9909; strictly tighter than the global `1/(1−‖Â‖‖W‖)` alternative, which was *vacuous* in 787 cases). `C_v` is therefore **not** under-estimated and **AEGIS-Conformal robust coverage is not threatened** on this account. *Meta-lesson: the conflation bites only for κ-bounds over a perturbation ball (mask changes), not for single-fixed-point quantities.*

---
## What is already strong (preserve in revision)
- The appendix's candor (`rem:obs_o1` naming two open gaps; `rem:exchange-honesty`; `tab:constants`) — this is the paper's credibility anchor; surface more of it in the main text rather than hiding it.
- `prop:transfer`'s complete-remainder (bilinearity) argument — correct and elegant.
- **Independently verified sound:** the `‖z*‖ → L_J → C_v` curvature chain feeding AEGIS-Conformal (local-κ bound is correct and tighter than the global alternative), and the bracket's all-active upper side (`08`). The conformal coverage guarantee is *not* threatened by the κ-conflation — keep these as-is.
- The matrix-free scaling of the audit path (N→7,650) — a real, verified engineering contribution.
- The sampling-free robust-conformal certificate at 10^3–10^4× lower cost than smoothing — per R3, potentially the broadest-impact result; consider re-centering the narrative on it.
- 10-seed protocol, reported std, finite-difference faithfulness checks — above-median reproducibility.

## One-line guidance to the authors
You wrote two papers: an over-claiming one on the first page and an honest, well-scoped one in the appendix. Publish the second one — fix the `ε_crit` proof, re-headline τ and the "attack," and let the coupling + cost story carry the contribution. The honest paper is competitive; the current framing is not.
