# Baseline Faithfulness Audit — Randomized Smoothing Certificates

**Baseline:** RandSmoothing (variants `frob` = matched Gaussian Frobenius ball, σ=ε/√(2|E|);
`per_edge` = per-coordinate ball, σ=ε).
**Feeds:** `tab:smoothing` (paper/sections/experiments.tex ~111–126) and `app:smoothing` (appendix F).
**Reference methods:** Cohen, Rosenfeld & Kolter 2019 (Gaussian randomized smoothing, ICML; arXiv:1902.02918);
Bojchevski et al. 2020 (sparse smoothing); Lee et al. 2019.

**Verdict: MINOR-GAPS (faithful core, two provenance/soundness issues to fix before camera-ready).**

---

## 1. Implementation location — official vs hand-rolled

This is a **hand-rolled** Cohen certificate (no `locuslab/smoothing` import, no `statsmodels.proportion_confint`),
implemented twice:

| File | Function(s) | Role | Status |
|------|-------------|------|--------|
| `scripts/exp_conformal_vs_smoothing.py` | `cp_lower_bound` (L107), `smoothed_certificate` (L119), `smoothing_classify_node` (L150), main loop (L320–372) | **THE table source** — frob/per_edge head-to-head vs AEGIS-Conformal; writes `results/conformal_vs_smoothing.csv` | FAITHFUL core |
| `iem/examples/exp_smoothing_sweep.py` | `smoothing_certificate_with_accuracy` (L49) | Secondary "P0" σ-sweep over fixed σ∈{0.01,0.05,0.10,0.15}; **NOT** the frob/per_edge table source | 3 soundness deviations |

The CP lower bound and the Φ⁻¹ radius are computed from `scipy.stats.beta.ppf` and `scipy.stats.norm.ppf`
respectively — the same primitives `statsmodels`/`locuslab` use, so the hand-roll is acceptable **if** the
arguments are right (they are in script 1; partly wrong in script 2).

---

## 2. Official Cohen (2019) certification — the steps

For a base classifier f and smoothing noise N(0, σ²I):
1. **Select** the top class ĉ_A using a *first* batch of n₀ noise draws (argmax of counts).
2. **Estimate** p_A using a *second, independent* batch of n draws: n_A = #{draws classified ĉ_A}.
3. **Lower-confidence bound** p_A̲ = one-sided Clopper-Pearson lower bound at confidence 1−α
   = `Beta⁻¹(α; n_A, n−n_A+1)` (`proportion_confint(n_A, n, alpha=2α, method="beta")` lower endpoint).
4. **Abstain** if p_A̲ ≤ ½; otherwise certify radius **R = σ · Φ⁻¹(p_A̲)**.
5. Guarantee: the smoothed g(x) = ĉ_A for all ‖δ‖₂ ≤ R.

Sparse/discrete graphs (Bojchevski 2020 / Lee 2019) replace step 4–5 with a region-based / discrete bound,
not the Gaussian Φ⁻¹ form. **Not used here** — the impl applies the *continuous* Gaussian certificate to the
real-valued Â weights, which is internally consistent with AEGIS's Frobenius-ball threat model (Â ∈ ℝ^{N×N}),
so the Gaussian (not sparse) certificate is the correct reference for this paper. OK.

---

## 3. Our steps, per variant

### Script 1 — `scripts/exp_conformal_vs_smoothing.py` (the table source)

**Noise model** (`smoothing_classify_node`, L150–175): for each of M draws, ξ ~ N(0, σ²I_{|E|}) on the |E|
upper-triangular edge coords, mirrored to the lower triangle (`dA[i,j]=dA[j,i]=ξ`), A_pert = A+dA, reconverge
the equilibrium (warm-started from Z_sub, up to `max_iter`), classify `model.head(Z)[v].argmax()`, increment
count. Symmetric Gaussian on edges ⇒ E‖dA‖_F² = 2|E|σ². **Correct & untruncated.**

**CP bound** (`cp_lower_bound`, L107–116): `beta_dist.ppf(alpha, n_A, M−n_A+1)`; `n_A≥M` → `alpha**(1/M)`;
`n_A≤0` → 0. This is the exact **one-sided** CP lower bound at confidence 1−α. **Correct (matches Cohen).**

**Radius / abstention** (`smoothed_certificate`, L119–147):
`pA_lo = cp_lower_bound(...)`; if `pA_lo ≤ 0.5` → abstain (`radius=-inf/0`, `certified=False`);
else `radius = sigma * norm.ppf(pA_lo)`, `certified = radius >= eps`.
`pred_set = {c_A}` if certified else **all C classes** (sound coverage fallback). **Correct Cohen formula.**

**ε-matching** (L325–331): `frob` → `sigma = eps/sqrt(2*E)`; `per_edge` → `sigma = eps`. **Matches spec.**

**frob abstention is REAL, not an artifact.** The certified radius is bounded by
R = σ·Φ⁻¹(pA_lo) ≤ σ·Φ⁻¹(1−α^{1/M}); with σ_frob = ε/√(2|E|) and |E| in the thousands, σ_frob ≈ 3.6e-4·ε…
(CSV note: `sigma_fro=3.59e-04` at ε=0.01). For `certified` we need R ≥ ε, i.e. Φ⁻¹(pA_lo) ≥ √(2|E|) ≈ 60+.
Φ⁻¹ caps at ≈ Φ⁻¹(1−α^{1/M}) ≈ 4–5 even when all M samples agree. So R ≪ ε *structurally* on the matched
ball ⇒ Cert = 0.00 is a **genuine geometric consequence** of the Cohen radius vs the Frobenius ε-ball, exactly
the paper's intended point. **This claim is sound.**

**Confidence / M** (L187, L191, L195): α=0.1 default; M (`--M`) default 200 (smoke), extrapolated to
`--extrap-M` 10000 (`wall_sm_extrap = wall_smoke * extrap_M/M`, L351). α handled correctly. **Wall-clock**: both
arms timed with `time.time()` on the **same device** (comment L206–207), AEGIS measured directly, smoothing
*linearly extrapolated* from a smaller M (see Gap G2).

### Script 2 — `iem/examples/exp_smoothing_sweep.py` (secondary, NOT the table)

`smoothing_certificate_with_accuracy` (L49–125): noise only on **existing edges** (`edge_mask`, L72), σ scaled
`raw_sigma=σ√2` then symmetrized, **`A_pert.clamp(min=0)`** (L86). CP at L111 uses `beta_dist.ppf(alpha/2, …)`
(two-tailed convention) with default α=0.001; gate at L106 uses **raw empirical** p_A>0.5; **L112 clamps
`p_lower = max(p_lower, 0.5+1e-6)`**. Fixed σ∈{0.01,0.05,0.10,0.15}, N_SAMPLES=1000, 10 seeds. No ε-matching.

---

## 4. GAPS

| # | Gap | Severity | File:line | Fix |
|---|-----|----------|-----------|-----|
| G1 | **CSV provenance does not match the table.** Committed `results/conformal_vs_smoothing.csv` has only ONE `RandSmoothing` row per ε (note: `sigma_cert=eps/2.326`, `M_smoke=600`, cov=1.0, set=7.0, wall_1e4 = 22,467s / 34,667s). The table shows THREE rows/ε with **distinct** frob vs per_edge cov (0.95/0.99), Cert (0.96/0.77), and walls (7,600/15,000/10,900/36,300s) that are **absent** from this CSV. The numbers in `tab:smoothing` are therefore either from an un-committed newer run or hand-entered. | **HIGH** (reproducibility) | `results/conformal_vs_smoothing.csv` vs `experiments.tex` L116–124 | Re-run script 1 with `--sigma-match frob,per_edge` over the 10 seeds, commit the CSV it emits (with `matching` + `cert_frac` columns, which the current CSV lacks), and regenerate the table from it. |
| G2 | **Wall-clock is extrapolated, not measured, at M=10⁴.** `wall_sm_extrap = wall_smoke·(10000/M)` (L351) with M_smoke=200–600. The CSV walls (22,467s/34,667s) and the table walls (7,600–36,300s) are linear projections, and they **disagree with each other** and with the headline multipliers. Ratios from the CSV (AEGIS 3.9s vs 22,467s/34,667s) give ≈5,700×/8,800×, **not** the appendix's "11,700–57,000×" nor cleanly the main text's 10³–10⁴×. | **MEDIUM** (the multiplier is the headline) | L350–351; experiments.tex L111; appendix `app:smoothing` | Either run the true M=10⁴ smoothing (preferred — memory says "bulletproof over hand-waving") or reconcile the extrapolation so the CSV walls, table walls, and the 11,700–57,000× range derive from one consistent computation. State extrapolation explicitly in the caption. |
| G3 | **Single-batch p_A (selection bias).** Script 1 picks ĉ_A AND computes the CP bound on the **same** M draws (`smoothed_certificate` argmax + CP on one `counts`), vs Cohen's two-batch n₀/n split. This is slightly **anti-conservative** (over-certifies marginally). | **LOW** | `exp_conformal_vs_smoothing.py` L134–141 | Note in appendix as a single-pass variant, or split n₀/n. Direction *favors* smoothing, so it does not weaken AEGIS's case. |
| G4 | **Script 2 clamps the CP bound up to 0.5+1e-6** (`p_lower = max(p_lower, 0.5+1e-6)`), so a node that should abstain (pA_lo≤½) instead gets a tiny positive radius → can **over-count** `frac_certified`. | **MEDIUM** (but script 2 is NOT the table) | `exp_smoothing_sweep.py` L112 | Remove the floor; abstain (radius=0) when pA_lo≤½. Only matters if any paper number comes from this file — confirm it does not. |
| G5 | **Script 2 noise is truncated Gaussian** (`A_pert.clamp(min=0)`, L86) and edge-masked; Cohen's radius assumes *untruncated* N(0,σ²). Realized smoothing dist ≠ Gaussian ⇒ the σ·Φ⁻¹ radius is not strictly valid for it. | **MEDIUM** (script 2 only) | `exp_smoothing_sweep.py` L86, L72 | Drop the clamp/mask for the certified arm, or use a discrete/sparse (Bojchevski) certificate that matches the actual noise. Script 1 (the table) does **not** clamp — OK there. |
| G6 | **Two-tailed α convention** in script 2 (`beta_dist.ppf(alpha/2,…)`, L111) vs one-sided in script 1 (`ppf(alpha,…)`). Inconsistent across files; script 2's is more conservative (smaller radius), so not unsound, just non-uniform. | **LOW** | `exp_smoothing_sweep.py` L111 | Standardize on the one-sided 1−α convention (script 1's). |

---

## 5. VERDICT & justification

**MINOR-GAPS.** The Monte-Carlo certification that actually feeds `tab:smoothing`
(`scripts/exp_conformal_vs_smoothing.py`) is a **faithful** Cohen (2019) implementation: sound symmetric
untruncated Gaussian noise model on the edge coords, **correct one-sided Clopper-Pearson lower bound**
(`Beta⁻¹(α; n_A, M−n_A+1)`), **correct radius R = σ·Φ⁻¹(pA_lo)** with **abstention at pA_lo ≤ ½**, and the correct
σ-matching (frob σ=ε/√(2|E|), per_edge σ=ε). The **"frob abstains (Cert 0.00)" claim is a real geometric
consequence** of the Cohen radius being capped at ≈σ·Φ⁻¹(1−α^{1/M}) ≪ ε on the matched Frobenius ball — not a
bug or an artifact. The baseline does **not** unfairly understate smoothing; if anything the single-batch p_A
(G3) and the abstain-to-all-classes fallback are mildly *favorable* to smoothing's coverage.

It is **not FAITHFUL-clean** because of two issues a certification reviewer will probe: (G1) the committed CSV
does not contain the per-variant rows/columns shown in the table, and (G2) the M=10⁴ wall-clock is a linear
extrapolation whose numbers don't reconcile across the CSV, the table, and the headline "11,700–57,000×". These
are provenance/reproducibility defects, not certificate-soundness defects. Script 2's deviations (G4–G6) are
real but it does **not** back the paper table; confirm no number traces to it.

---

## 6. Paper numbers at risk

- **`tab:smoothing` (experiments.tex L116–124):** all RandSmoothing rows. Cov (frob 1.00 / per_edge 0.95,0.99),
  Set (7.00 / 1.26, 2.39), **Cert (frob 0.00 / per_edge 0.96, 0.77)**, Wall@10⁴ (7,600 / 15,000 / 10,900 /
  36,300 s) — none of these per-variant values are present in the committed source CSV (G1). The **Cert 0.00 on
  frob is methodologically sound** and will survive re-run; the per_edge Cert/Set and all wall-clocks need a
  committed run to back them.
- **"runs 10³–10⁴× cheaper" (experiments.tex L111) and appendix "11,700–57,000×" (`app:smoothing`):** rest on
  extrapolated walls (G2). CSV-implied ratios are ≈5,700×–8,800×; table-implied ≈7,600×–36,300×. The
  "11,700–57,000×" figure is **not reproducible from the committed artifacts** and must be regenerated or
  re-derived consistently.
- **Not at risk:** AEGIS-Conformal's own coverage/set-size and the *qualitative* conclusion (smoothing must
  retreat to a larger ball to certify, at 10³–10⁴× the cost) — both hold under the faithful script-1 logic.

**Recommended action (per the team's bulletproof rule):** re-run `exp_conformal_vs_smoothing.py` at the true
M=10⁴ (or a defensible M with a stated, reconciled extrapolation) over the 10 preferred seeds with
`--sigma-match frob,per_edge`, commit the emitted CSV including `matching`/`cert_frac`, and regenerate both the
table and the multiplier range from that one artifact.
