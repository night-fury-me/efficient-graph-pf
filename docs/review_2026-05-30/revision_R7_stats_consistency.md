# Revision R7 (stats hygiene) + P3 (consistency) — DONE

Most of R7/P3 was **already satisfied** in the current draft (more rigorous than R1/R2's review snapshot). Verified each, fixed the one real gap.

## R7 — statistics hygiene
- **± labeling (the one FIX):** added "reported $\pm$ are standard deviations over seeds" to Setup. Verified against code — `agg()`/`np.std` over the 10 seeds → inline/table `±` = seed SD. (Figure uncertainty bands are separately labeled 95% CI in their captions, so no conflict.)
- **Sign test — named + DEFENDED (no correction needed):** the transfer significance is a **single one-sided binomial sign test** over the 39 cells (`experiments.tex` L172), i.e. "is the fraction of positive cells > 0.5?" — *one* aggregate test, not 39 separate per-cell tests. So R1's "33-cell multiplicity correction" does not apply (verified the concern before acting, à la the L_J/W2 case). For reference, 39/39 positive gives an exact one-sided binomial p = 0.5³⁹ ≈ 2e-12; the paper states the conservative p<10⁻⁵.
- **rSVD error — already cited + bounded:** L128 reads "the rSVD error~\cite{halko2011finding} is bounded by the spectral gap", with σ₁ matching dense to **0.03%** at N=200 and analytical Neumann residual κ²⁰⁰ ∈ [1e-105, 1e-48]. Done.
- **"one query" — already defined:** intro L8 defines it as one matrix-free `S_c` construction (a randomized SVD over the equilibrium resolvent), not three analyses; framework describes the JVP/Neumann/rSVD internals. Not a single matvec — adequately transparent.

## P3 — consistency (VERIFIED already consistent; no edit)
- **AGNNCert cell:** `0.163` everywhere (`tab:baselines` L107). No `0.187` in the current tex. ✓
- **`r_cert/r_v` range:** `[4.4, 15.0]` in *both* the footnote (L110) and the prose (L114, "$4.4$--$15.0\times$ tighter"). No `[4.9, 10.2]` anywhere. ✓
- R2's flagged mismatches were from an earlier review snapshot and have since been reconciled.

## Build
10 pages, 0 overfull, 0 undefined references.
