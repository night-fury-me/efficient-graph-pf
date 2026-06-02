# AEGIS Consolidation — Integration Plan (for approval, do NOT edit .tex yet)

Goal: fold the validated arsenal into one excellent, bulletproof top-tier paper, re-aimed
from "diagnostics" to a **certificate/robustness** contribution. This plan is for your
review; nothing in the `.tex` changes until you approve.

## 0. Decisions — LOCKED (2026-06-02)
1. **Venue/format:** ✅ **AAAI-27 Main Technical Track.** **7 pages two-column technical
   content** + unlimited references + supplementary appendix (reviewers NOT required to read
   it ⇒ anything load-bearing MUST be in the 7 main pages). **Abstract Jul 21, paper Jul 28,
   supp Jul 31, 2026; decision Nov 30, 2026.** Chosen for PhD-application timing (Nov 30
   decision lands before fall deadlines). Tighter main text than ICDM 10pp ⇒ ultra-focused
   flagship, heavier appendix. Rebuttal/author-feedback window Oct 19–25.
   *(History: ICDM-26 missed — abstract was May 30, not registered. ICLR-27 too late
   (decision Jan 2027).)*
2. **Flagship narrative:** ✅ **Balanced dual, anchored on AEGIS-Conformal** (rigorous +
   deployable) + the constant-factor two-sided spectral bracket + C4 unification (supporting).
3. **Over-squashing duality (H):** ✅ **Appendix ONLY** (no main mention — 7pp is too tight;
   the unification claim rides on C4, not H).

**Execution status (2026-06-02):**
- Theorem: drafted → adversarial review (NEEDS-REFRAME: "tight" overclaimed, upper bound
  all-active only, nonlinear empirical) → **revised to defensible v2** (`tightness_theorem_
  v2.md`): "constant-factor two-sided" (slack 10–18×), A4-scoped in-statement, nonlinear
  demoted to a fenced Observation, β/N1/N2 fixed, C a-posteriori, re-aimed onto C4. Reviewer:
  **low-risk as a SUPPORTING theorem.**
- **Flagship re-anchored on AEGIS-Conformal** (rigorous + deployable); spectral bracket +
  C4 unification = supporting theory. (Still the balanced dual — anchored on what survives
  hostile review.) §4/§5 of the structure swap emphasis accordingly.
- 10-seed reachability run: in progress (background, validates ε_reach/ε* across seeds).
- `.tex` still UNTOUCHED — awaiting 10-seed completion + drafting greenlight.

---

## 1. Re-aimed thesis, title, contributions
**Thesis (one sentence):** *One matrix-free sensitivity operator S_c characterizes the
structural robustness of implicit GNNs end-to-end — it identifies the optimal attack,
tightly brackets the true robustness boundary (closing the 2–10× conservatism of norm
certificates), and yields a deployable distribution-free coverage guarantee — unifying
attack, certification, and defense.*

**Title options** (certificate-forward, replacing "Matrix-Free Diagnostics for the
Adversarial Fault Lines"):
- A. *AEGIS: Spectral Sensitivity and Tight Robustness Certification for Implicit Graph Neural Networks*
- B. *The Spectral Margin: Tight, Distribution-Free Robustness Certificates for Implicit GNNs*
- C. *One Operator for Attack, Certification, and Defense in Implicit GNNs*

**Contributions (new list):**
1. The constrained sensitivity operator S_c (matrix-free, scales to N≈7,650) — attack v₁, per-edge ranking, per-node radius.
2. **Tight two-sided characterization of the structural robustness boundary** via the *spectral* budget ε* (γ=1 divergence; critical-driving attack as the matching upper bound), showing the norm certificate ε_crit is 2–10× conservative.
3. **AEGIS-Conformal:** the first distribution-free structural coverage certificate for implicit GNNs (non-vacuous where deterministic radii are thin; 10-seed, gate holds at nominal 1−α).
4. **Defense + coupling:** σ₁(S_c) regularization improves robustness; attack and defense are coupled through the one operator.
5. (Optional) **Spectral duality:** the same resolvent governs an expressivity↔robustness (over-squashing) trade-off.

---

## 2. Status ledger — every result, honestly
Legend: ✅ validated @10 preferred seeds · 🔬 seed-42/pilot, needs 10-seed · ✍️ theory done, needs writing · 🆕 needs new proof/experiment · ⏬ drop.

| Result | Status | 10-seed? | Destination |
|---|---|---|---|
| S_c operator, matrix-free, v₁ attack (Prop 1) | in paper ✅ | ✅ | Main §2–3 (keep) |
| r_v radius (Prop 2) | in paper ✅ | ✅ | Main §4 (reframe w/ Certify) |
| 39/39 τ=0.99 transfer (Prop 3) | in paper ✅ | ✅ | Main §6 (keep, strong) |
| Attack eff. (74–156×), breach rates, scalability | in paper ✅ | ✅ | Main §6 condensed / appendix |
| Thm 1 (3-regime) | in paper ✅ | — | **Re-aim** → tight spectral boundary (§5 flagship) |
| **Tight spectral boundary (ε*, γ=1, critical-driving)** | 🔬 + 🆕 | ❌ (seed 42) | **Main §5 flagship** — needs theorem + 10-seed |
| **AEGIS-Certify (sound radius)** | ✅ (campaign) | ✅ | Main §5 (sound baseline) — replaces rem:certificates |
| **AEGIS-Conformal** | ✅ Cora+Citeseer | ✅ | **Main §5/§6** (deployable guarantee) |
| **σ₁-regularizer defense (#1)** | ✅ | ✅ | Main §5 brief + appendix table |
| **Attack–defense coupling (#5)** | ✅ | ✅ | Main §5 brief / appendix |
| **MonDEQ breadth (#3)** | ✅ | ✅ | Appendix (breadth) |
| **Over-squashing duality (H)** | ✍️ + 🆕 | ❌ | Optional §7 or appendix — needs 1 experiment |
| Universal (P4) | ⏬ near-tautological | — | **Drop** (1-line at most) |
| rem:certificates concession | — | — | **Replace** with §5 certificates |
| Physics / power-grid | rescoped ✅ | — | Keep as model-auditing (1 dataset) |

---

## 3. Section structure (AAAI-27: 7 pp two-column + appendix)
> **AAAI re-aim (2026-06-02):** 7 pp two-column ≈ ~30% less main-text than the current
> ICDM 10 pp draft, so existing content compresses hard AND the flagship is added. Changes
> vs the table below (which was sized for ICLR single-column): **§7 over-squashing → appendix
> ONLY**; MonDEQ → appendix; ALL proofs → appendix; diagnostics (§3) compressed further;
> case study trimmed. Two-column packs denser, so the flagship (§4 certificates incl.
> Conformal + §5 spectral bracket) still fits. Detailed page-by-page re-triage during drafting.

### (ICLR-sized estimates, retained for reference)
| § | Content | Source | ~pp |
|---|---|---|---|
| 1 Intro | re-aimed thesis, contributions | new prose | 1.25 |
| 2 Setup | IGNN equilibrium, IFT, the resolvent, S_c, matrix-free | existing, condensed | 1.25 |
| 3 Diagnostics | v₁ attack, per-edge, r_v | existing, condensed | 1.0 |
| 4 **Spectral robustness boundary** (flagship theory) | re-aimed Thm 1 → tight two-sided ε* characterization; γ=1; critical-driving = matching bound; ε_crit 2–10× conservative | new theorem + existing | 1.5 |
| 5 **Certificates** | Certify (sound radius) + **Conformal** (deployable, non-vacuous, 10-seed table) | new | 1.25 |
| 6 Defense + empirics | σ₁-reg defense + coupling (brief); 39/39 transfer; attack eff.; breach; scalability | mixed, condensed | 1.25 |
| 7 (opt) Spectral duality | over-squashing margin↔reach Pareto | new | 0.5 |
| 8 Related + Conclusion | re-aimed | new prose | 0.75 |
| | **Main total** | | **~8.75** (9 w/o §7, or trim) |
| Appendix | all proofs; MonDEQ; full 10-seed tables; critical-driving details; over-squashing full; scalability | mixed | unlimited |

---

## 4. NEW work required before submission (sequenced)
1. **[YOU] decisions** (venue, flagship, §7 include?).
2. **Tightness theorem** (biggest): state + prove the two-sided bounded-gap spectral characterization (ε_crit ≤ true boundary ≤ C·ε*, C from η/active-fraction). Use ml-theory-writer + ml-theory-reviewer (adversarial). Re-aims Thm 1.
3. **10-seed completions** (hard rule): re-run `exp_reachability.py` at the 10 preferred seeds (ε*, γ=1, ε_reach, critical-driving) — ~hours on the 4090. Confirm Certify / regularizer / coupling / MonDEQ at 10 (mostly done — spot-verify).
4. **[if §7] over-squashing experiment** — measure the margin↔reach Pareto on trained models, 10-seed + lit positioning sentence (vs Arroyo'25, EIGNN).
5. **Reformat** ICDM→ICLR template (double→single column) [if ICLR].
6. **Draft prose:** abstract, intro/contributions, §4 flagship, §5 certificates, §6 defense, related, conclusion.
7. **Integrate + page-triage + figures** (serif 11 pt, glyphs in legend per house style).
8. **Adversarial internal review:** ml-theory-reviewer on the new theorem; academic-paper-reviewer on the full draft; fix; re-review.

---

## 5. Honest caveats to PRESERVE (so it stays bulletproof)
- **Do not conflate** the tight *stability/criticality boundary* (ε*, large-ε ≈‖Â‖ scale) with a tight *per-node classification* certificate. The boundary result says implicit GNNs are MORE robust than the norm certificate implies (a large, tightly-bracketed true margin). Deployable *small-ε* classification guarantees are Certify (sound) + Conformal (distribution-free).
- Criticality is reached only at ≈‖Â‖-scale perturbation (5–11× realistic budget) — frame as "the true safety margin is large and tight," never "easy to break."
- Conformal: exchangeability stated as a condition; n=200 dense-path (matrix-free = future work).
- Certify: small-ε / contractive-regime scope, reported honestly.
- Physics stays rescoped to model-auditing (the surrogate can't model voltage collapse).
- **Drop Universal** (the "non-GNN" probe was circular).

## 6. What this paper is (the honest pitch)
A unified S_c framework for implicit GNNs: a tight spectral characterization of the
robustness boundary (the right quantity, closing 2–10× conservatism), a deployable
distribution-free certificate (Conformal), a sound radius (Certify), a coupled defense, and
the original one-query diagnostics + 39/39 transfer — optionally an expressivity↔robustness
duality. Flagship tight theory + deployable guarantee + broad empirics = a strong ICLR/
NeurIPS submission. Not a paradigm-breakthrough; a genuinely excellent paper.
