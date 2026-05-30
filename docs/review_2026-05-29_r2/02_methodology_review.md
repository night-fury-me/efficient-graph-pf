# Reviewer 1 — Methodology Review (R2)

## Persona

Applied-math/ML researcher; NeurIPS + SIMAX publications, including a well-cited
matrix-free Jacobian paper for deep equilibrium models. Teach a graduate course
on randomized linear algebra; recurring ICML reviewer for trustworthy ML. My
lens is closed-form correctness of $S_c$, the numerical-linear-algebra pipeline
(matrix-free $\kappa$, Neumann residual, Halko rSVD), and the inferential
statistics in `tab:breach` / sign tests. I do not engage with the power-systems
case study or with related-work coverage.

## Summary of the technical claims (≤200 words)

The manuscript advances five linked claims.
**C1 (Theorem 3.1):** for an IGNN with $\norm{J_z}_2\le\kappa<1$ and
$\norm{W}_2\le c$, structural perturbations $\delta\Ahat$ with $\norm{\delta\Ahat}_F=\varepsilon$ exhibit three regimes around $\varepsilon_{\rm crit}=(1-\kappa)/\norm{W}_2$: subcritical first-order shift bound, critical resolvent blow-up $\Omega(1/(\varepsilon_{\rm crit}-\varepsilon))$ along the worst-case direction, and supercritical certificate void.
**C2 (Observation 3.3, Propositions 3.4–3.5):** the constrained sensitivity
matrix $S_c$ yields, in one closed form, per-edge vulnerability $v_{ij}$, the
SVD-optimal direction $\delta\Ahat^\star=\varepsilon\cdot\mathrm{reshape}(v_1)$,
and per-node first-order radii $r_v$, with a nonnormality bound that depends
only on $W$ (graph-independent).
**C3 (Proposition 3.6):** continuous $\norm{[S_c]_{:,k}}_2$ scores transfer to
discrete edge-removal damage rankings under a margin condition.
**C4 (Algorithm 1):** $S_c$ matvec/rmatvec via truncated Neumann + JVP/VJP,
with randomized SVD, scales to $N=7{,}650$ on a single 24-GB GPU.
**C5 (Empirical):** AEGIS dominates random and Mettack baselines, the bound
envelope is tight ($1.00$–$1.39\times$), and cross-domain $\tau$ is positive
in 29/33 cells.

## Theorem-by-theorem audit

### Theorem 3.1(a) Subcritical regime

The statement around `paper/sections/theory.tex` L8–L33 conditions on (A1)
ReLU + nonsmooth IFT (Bolte–Pauwels conservative gradient), (A2)
$\norm{W}_2\le c$, (A3) $\norm{J_z}_2\le\kappa<1$. The proof opens with the
standard implicit-derivative manipulation
$\Delta\zstar = (I-J_z)^{-1}J_A\,\mathrm{vec}(\delta\Ahat) + O(\norm{\delta\Ahat}^2)$,
then bounds $\norm{(I-J_z)^{-1}}_2\le 1/(1-\kappa)$ via Neumann and contractivity
preservation by $\norm{J_z'}_2 \le(\norm{\Ahat}_2+\varepsilon)\norm{W}_2$. The
proof is correct as a first-order Taylor statement. **One residual issue**:
(A1) uses the conservative-Jacobian framework on a measure-zero exception
set of activation-boundary crossings; an arbitrary $\delta\Ahat$ can land on
such a face. The authors argue genericity, but for the SVD-optimal direction
$\delta\Ahat^\star=\varepsilon\,\mathrm{reshape}(v_1)$, alignment with an
activation boundary is not measure-zero in any meaningful sense — it is a
worst-case direction. This is a minor gap, not a falsification: add a sentence
noting that the conservative-gradient calculus already returns a valid bound at
exception points (Bolte–Pauwels Prop. 2.5).

### Theorem 3.1(b) Critical regime (R2 rewrite)

The R2 rewrite (caption and proof, `theory.tex` L40–L44) states the worst-case
direction yields $\norm{J_z'}_2 \to \norm{\Ahat}_2\norm{W}_2 + \varepsilon\norm{W}_2$,
so $1-\norm{J_z'}_2 \ge \norm{W}_2(\varepsilon_{\rm crit}-\varepsilon)$, and the
**Neumann lower bound** $\norm{(I-J_z')^{-1}}_2 \ge 1/(1-\norm{J_z'}_2)$ delivers
the $\Omega(1/(\varepsilon_{\rm crit}-\varepsilon))$ rate without normality. This is correct: the
inequality $\norm{(I-M)^{-1}}_2 \ge 1/(1+\norm{M}_2)$ from Stewart–Sun (1990) is
the standard lower bound; combined with the geometric-series upper bound it
gives a two-sided $\Theta(1/(1-\norm{M}_2))$ rate **in operator norm only**. No
normality of $J'_z$ is invoked. **The R2 rewrite delivers what was promised.**
A nit: the *upper* bound $\norm{(I-M)^{-1}}_2 \le 1/(1-\norm{M}_2)$ also holds
without normality (geometric series of $M^k$); the proof currently states "for
contractivity preservation" but does not call out that this is the upper-bound
construction. Add a half-sentence for clarity.

### Theorem 3.1(c) Supercritical regime (defensive rewrite)

The current statement (`theory.tex` L41–L43):
"the contraction certificate is void ($\norm{J_z'}_2$ may exceed 1). The perturbed operator may still be contractive along low-sensitivity directions, but Banach no longer guarantees uniqueness and the part-(a) first-order guarantees lapse; we do not claim divergence as a generic post-threshold behaviour, only that our certificate fails."

This is **not vacuous**. The retained claim is "our sufficient certificate
ceases to hold", which is the only honest statement available given that the
empirical $\kappa_{\max}$ sweep on Cora shows $\rho(J_z)$ saturates at $\approx 0.42$
and the empirical resolvent rises only $1.17\to 1.80$ across $\kappa_{\max}\in[0.30,0.99]$
(`experiments.tex` L135). Calling this "phase transition" in the title remains
defensible because the certificate boundary is real even if the empirical
discontinuity is not. **Recommendation: accept as written but rename the
referent of "phase transition" in fig:phase_transition caption to "certificate
boundary" — the current caption already does this in body text but the figure
label still says phase transition. Minor cosmetic issue.**

### Observation 3.3 (η bound)

The statement on `theory.tex` L46–L55 splits into (a) all-active ReLU
($\phi'_i=1$) → $\eta \le \kappa(V_W)$, and (b) general ReLU pattern → $\eta \le \kappa(V_W) \cdot \mathrm{cond}(\mathrm{diag}(\phi') \otimes I)$. **Part (a) is rigorous**: with $\phi'\equiv 1$, $J_z = \Ahat\otimes W$, and for symmetric $\Ahat$ the Kronecker spectrum reduces to $V_W$ via eigenvalue decomposition. **Part (b) is fragile**: the cond term is degenerate for ReLU (the diagonal is 0/1, so the condition number is $\infty$ on any node with a zero activation). The authors should explicitly mark Obs. 3.3(b) as an *Empirical Remark* — this is what R1 (round 1) requested. As stated, (b) is mathematically meaningless for any practical state vector with at least one inactive node. **Major minor**: rename "Observation 3.3(b)" to "Empirical Remark 3.3(b)" or restrict to the active sub-block where the diagonal is full rank.

### Proposition 3.4 (optimal attack)

Statement at `theory.tex` L67–L70 is the standard "leading right singular
vector maximises a quadratic form over the Frobenius unit sphere", an
elementary application of Eckart–Young. The reshape from $v_1\in\R^{N^2}$ to
the $N\times N$ adjacency perturbation is a measure-preserving bijection that
preserves the Frobenius norm. **Correct as stated.** A practical caveat,
acknowledged in the text: $\delta\Ahat^\star$ is a *continuous* perturbation
direction; the discrete edge-removal proxy (Prop. 3.6) is what is actually
deployed.

### Proposition 3.5 (radius, path-Lipschitz / path-crossing)

Statement at `theory.tex` L80–L87. The R2 fix was supposed to repair the
path-Lipschitz argument $L_J \le \norm{W}_2^2$. **The current statement no
longer makes a path-Lipschitz claim** — it states $r_v$ as a first-order
*sensitivity* radius, not a margin-preserving certificate (and the body
text and `rem:certificates` repeatedly emphasise this). The constrained
variant $r_v^{(c)}=m_v/(\norm{W_{y_v}-W_{c^\ast}}_2\norm{S_{c,v}}_2)$ is
defined as a first-order estimate, with the explicit caveat that
$\norm{S_{c,v}}_2$ is approximated by randomized power iteration in the
matrix-free path. **Verdict: the R2 patch dodged the path-Lipschitz problem
by downgrading the claim from "certificate" to "first-order sensitivity
threshold".** This is honest but should be flagged: the original draft
appeared to claim a certified per-node radius; the current draft does not.
That downgrade reduces (does not eliminate) the contribution and must be
called out in the abstract and intro. The AGNNCert subsection makes this
trade-off explicit and I find the framing defensible.

### Proposition 3.6 (continuous→discrete transfer)

Statement at `theory.tex` L103+. The claim is conditional ("under (A1)–(A3)")
and the body reports that the sufficient condition holds for 47–62% of edge
pairs (`experiments.tex`). That fraction is honestly reported. The
"29/33 cells positive" follow-up empirical claim is checked in §Statistical
methodology below. **The proposition's sufficient condition is correctly
stated; the experiments measure how often it is binding.**

## Numerical methods audit

### Matrix-free pipeline (Algorithm 1)

The pseudocode at `framework.tex` L20–L40 cleanly separates (i) $\kappa$ by
JVP power iteration, (ii) Neumann depth $K=\lceil\log(1/\tau)/\log(1/\kappa)\rceil$,
(iii) matvec / rmatvec via truncated Neumann + JVP/VJP, (iv) randomized SVD
with oversampling $p$ and $n_{\rm iter}$ subspace iterations, (v) per-edge
spectrum from the leading right singular vector, (vi) per-node radii from
block-row norms. The cost statement $O(K\cdot Nd)$ time / $O(Nd)$ memory is
correct. The dense fallback note ("$N\le 200$: form $S=(I-J_z)^{-1}J_A$ and run
deterministic SVD in place of lines 2–11") is accurate.
**The algorithm is correct as written.**

### Neumann truncation residual (post-R2_04 bug fix)

Two bugs were disclosed (`docs/r2_experiments_full_report.md` §4):
(1) `neumann_residual()` renormalised the probe vector between iterations,
so the recorded column was $\sigma_1(J_z)=\kappa$, not $\norm{J_z^K b}/\norm{b}$;
(2) `op.top_k_svd(k=6)` was off-by-one for the Halko bound which needs
$\sigma_{k+1}$, so `halko_bound_estimate` is NaN for every real-dataset row.

The salvage strategy was to (a) **rename** the mislabelled column to
`kappa_estimate` in `matfree_error_bounds_corrected.csv` and (b) **compute
$\kappa^{200}$ analytically** from the (now correctly-labelled) $\kappa$
values, into `neumann_residual_analytic_K200`. I checked all 50 real-dataset
rows: $\kappa^{200}$ matches $\kappa^{200}$ to float precision (e.g., Cora
seed 42 $\kappa=0.27320$, $200\log_{10}\kappa=-112.70$, CSV
$1.98\times 10^{-113}$ ✓; Amazon seed 2718 $\kappa=0.87371$,
$200\log_{10}\kappa=-11.73$, CSV $1.87\times 10^{-12}$ ✓).

**Important: the salvage is arithmetically self-consistent**, but it does not
recover the original quantity the experiment intended to measure. The empirical
$\norm{J_z^K b}/\norm{b}$ is *not* in the CSV. What we have is the analytical
upper bound $\kappa^K$, which is loose by exactly the nonnormality factor $\eta$
(reported in the paper as $1.19$–$2.47$). **For a contraction with $\kappa<1$ and
a random probe vector, the empirical residual is bounded above by $\kappa^K$
and the bound is tight up to $\eta$**; the body claim "$\norm{J_z^{K+1}b}/\norm{b}<10^{-6}$ in all runs" (`experiments.tex` L?) is therefore plausible but **not directly verifiable from the salvaged CSV**.

### Randomized SVD (Halko bound)

`halko_bound_estimate` is empty for **all 50** real-dataset rows in
`matfree_error_bounds_corrected.csv`. The bug-audit table is explicit:
"future re-run only (cannot back-fill)". The paper, however, claims
("`experiments.tex`") "rSVD error bounded by spectral gap" and states the
Halko inequality $\norm{(I-QQ^\top)S_c}\le(1+\sqrt{k/(p-1)})\sigma_{k+1}$ is
"small when the spectral gap is large, as in our case: $(\sigma_1-\sigma_2)/\sigma_1=0.39$–$0.50$".
**The bound is stated correctly but not numerically verified on real data**;
the $0.39$–$0.50$ gap statistic comes from a separate column
(`singular_gap_sigma1_minus_sigma2_over_sigma1`) in `robust_arch.csv`, which
reports the first-vs-second gap, not the gap at the rSVD rank $k=7$ that the
Halko bound actually needs (it requires $\sigma_{k+1}$, not $\sigma_2$).
**Major weakness: invoking the Halko bound while the corresponding CSV cells
are empty is a recoverable but real reproducibility hole.** Fix: a 30-minute
re-run with `top_k_svd(k=7)` would back-fill the column; the authors should
do this before camera-ready.

### σ₁ agreement (dense vs matrix-free) at $N=200$

`matfree_error_bounds_corrected.csv` reports a `sigma_dense` column — but
**`sigma_dense` is empty for every row** (synthetic and real). Only
`sigma_matfree` and `sigma_matfree_top` are populated, and the two are
identical to all printed digits. The body claim
"matrix-free $\sigma_1$ is within $0.03\%$ of the dense reference at $N=200$"
(`experiments.tex` L?) is therefore **not reproducible from this CSV**.
That number must come from a different artifact. **Minor/Major**: the
authors should point me to the CSV that actually contains the dense-vs-matrix-free
comparison, or repopulate `sigma_dense` in `matfree_error_bounds_corrected.csv`.
As it stands, an independent reviewer cannot verify the $0.03\%$ number.

### Salvaged R2_04 CSV trustworthiness

Summary: $\kappa^{200}$ values are arithmetically consistent with the κ column
(verified row-by-row). **However**:
1. $\kappa^{200}$ is an analytical surrogate for the empirical Neumann
   residual; the empirical residual is not in this CSV.
2. The Halko bound column is empty across all real-dataset rows.
3. The dense-vs-matrix-free $\sigma_1$ agreement at $N=200$ cannot be checked
   from this CSV (`sigma_dense` empty).
4. The κ values themselves come from what was originally the
   `neumann_residual` column, which the authors now claim was actually
   $\sigma_1(J_z)$ from power iteration. This is plausible because the
   renormalised probe vector converges to the leading left singular vector,
   so the per-iteration scaling factor *is* $\sigma_1(J_z)$. The salvage
   logic is sound, but **only one of the three R2_04 deliverables (Neumann
   residual band) is actually carried by the salvaged CSV**, and that one
   is the analytical not the empirical version.

### κ²⁰⁰ band claim — incorrect

The body text (`experiments.tex` paragraph on scalability) reports
"$\kappa^{200}\in[10^{-105},10^{-48}]$ across the suite". The CSV does **not**
support this band. Across the 50 real-dataset rows:
- min log₁₀(κ²⁰⁰) = **−177.22** (WikiCS seed 2718, κ=0.130)
- max log₁₀(κ²⁰⁰) = **−11.73** (Amazon seed 2718, κ=0.874)
- **13 of 50 rows are *above* 10⁻⁴⁸** (i.e., the residual is larger than the
  claimed upper end of the band): four Pubmed seeds, three Amazon seeds, two
  Cora seeds, one Citeseer seed.

**Major correctness issue:** the printed band understates the worst-case
residual by ≥36 orders of magnitude. For seed 2718 on Amazon the κ²⁰⁰
analytical bound is $1.87\times 10^{-12}$ — that is comfortably below the
claimed $10^{-6}$ convergence threshold, so the qualitative claim
("convergence in all runs") survives, but the printed numerical band must be
corrected to $[10^{-177},10^{-12}]$ or restricted to the subset of rows it
actually describes. The paper should report the worst-case row (Amazon seed
2718) explicitly.

## Statistical methodology audit

### CIs in `tab:breach`

`results/revision_R2/stats_reanalysis.csv` reports `ci_lo_95` and `ci_hi_95`
columns. Per the report (§3 R2_03), CIs are **t-with-9-df, two-sided**, not
bootstrap. Across the 30 breach cells the CIs are symmetric around the mean
(checked numerically: e.g., Cora ε=0.20 mean=0.0758, std=0.0752,
ci=[0.0220,0.1296], half-width=0.0538; with $t_{0.025,9}=2.262$ and SE=std/√10,
predicted half-width = 2.262·0.0752/√10 = 0.0538 ✓). **The CIs are
methodologically correct t-intervals.**

**However, the appropriateness for `tab:breach` is questionable.** The
underlying quantity is a breach rate $\in[0,1]$ that is right-skewed
(Pubmed: mean 27.4±21.1%, median 7.8%; the report flags this explicitly in
§3 R2_03 review items). For a $[0,1]$-valued metric with skewed distribution
and n=10, the t-interval over-covers in the left tail and under-covers in the
right. **Recommendation: switch to BCa percentile bootstrap (B=10000) for
Pubmed and any cell where the std exceeds the mean.** Several cells already
have CI lower bounds below zero (e.g., Cora ε=0.01 ci_lo=−0.0029) which is
inadmissible for a rate.

### Sign test on 29/33 transfer

I computed $\binom{33}{0.5}$ one-sided P(≥29). The exact value is
$p=5.46\times 10^{-6}$ (log₁₀ p = −5.26). **The paper's $p<10^{-5}$ is
correct.** Caveat: the sign test treats each (architecture, dataset) cell as
independent. Cells from the same dataset share data; cells from the same
architecture share weights up to seed averaging. This is a quasi-independence
assumption that I would normally challenge, but with 5 datasets × 7
architectures and seed-averaging within cells, the dependency structure is
weak enough that the qualitative conclusion (transfer is non-spurious) survives.

### Wilcoxon comparisons

The report specifies "one-sided ('AEGIS damage > random')" Wilcoxon
signed-rank. n=10 paired observations. The smallest exact one-sided p-value
is $2^{-10}=9.77\times 10^{-4}$; the second-smallest is $2/1024=1.95\times 10^{-3}$;
the third is $3/1024=2.93\times 10^{-3}$. The stats CSV reports values like
$0.0009765625$ (Pubmed scalability), $0.001953125$ (Pubmed ε=0.15 and 0.20)
which are exactly $1/1024$ and $2/1024$ — **the Wilcoxon table is consistent
with one-sided exact Wilcoxon at n=10**, with the alternative correctly
specified (AEGIS > random; rejecting when AEGIS dominates).

### Mettack sign test (149/150)

I recomputed: one-sided binomial(150, 0.5) P(W≥149) = $1.058\times 10^{-43}$,
matching the CSV's `binomial_p_greater_05` cell to all printed digits. **The
sign test is correct.** This is the most defensible single-number claim in
the inferential-statistics section.

### Multiple-testing correction

**Not addressed.** With 30 breach cells, 6 ε-levels × 5 datasets, and an
additional 33-cell transfer matrix tested cell-wise in the heatmap, the
family-wise error rate is unmanaged. Holm–Bonferroni or BH-FDR over the 30
breach cells would knock out half of the cells currently flagged
`significant_at_005=True`. This is a **minor weakness** because the headline
claims do not hinge on individual cells, but it should be acknowledged in a
footnote (e.g., "p-values are nominal; with Holm correction over 30 cells,
significance survives at q<0.05 only for…").

### Heavy-tailed Pubmed handling

The report flags Pubmed as right-skewed (mean 27.4±21.1%, median 7.8%). The
CSV does not provide a bootstrap CI. For the headline "Pubmed is the
right-skewed outlier" the median-and-mean separation is itself the evidence,
but the printed t-interval $[0.123,0.425]$ at ε=0.20 is mis-shaped (a $[0,1]$
quantity with mean 0.274 and std 0.211 has its true 95% interval much closer
to $[0.05, 0.55]$ under bootstrap). **Recommendation: re-run with BCa
percentile bootstrap on Pubmed at minimum.**

## Reproducibility assessment

- **Code organisation**: scripts/revision_R2/ contains R2_01–R2_10 plus the
  postprocess salvage scripts. The directory is well-organised; each R2_xx
  script appears to write to a single CSV.
- **Seed handling**: 10 seeds {42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999}
  consistently used across all reported tables.
- **Deterministic ops**: not explicitly stated; the R2_04 corrected CSV uses
  randomized SVD whose default torch.linalg.svd_lowrank is deterministic given
  a fixed seed, but the seed-to-internal-random-state pathway is not
  documented in the paper.
- **Hardware**: NVIDIA RTX 4090 (24 GB) stated; Pubmed full-graph exceeds
  this at $N=19{,}717$.
- **OOM thresholds**: documented at $N=500$ (dense path) and $N=19{,}717$
  (matrix-free path); $N=7{,}650$ Amazon Photo is the highest verified.
- **R2_04 CSV gap**: the `sigma_dense` and `halko_bound_estimate` columns are
  empty across all real-dataset rows. This is a partial reproducibility hole.

## Strengths (top 3)

1. **Defensive R2 rewrites are honest.** Theorem 3.1(c)'s downgrade from
   "divergence" to "certificate void", the empirical $\kappa_{\max}$ sweep
   substitution for the missing $\varepsilon>\varepsilon_{\rm crit}$ crossing, the
   downgrade of Prop. 3.5 from "certificate" to "first-order sensitivity
   threshold", and the AGNNCert framing patch all reflect a genuine
   willingness to soften over-claims rather than hide the gap.
2. **Theorem 3.1(b) lower bound is now operator-norm-only.** The R2 rewrite
   delivers the $\Omega(1/(\varepsilon_{\rm crit}-\varepsilon))$ rate via the standard
   Stewart–Sun inequality, no normality. This was R1's main R1-round technical
   request and it has been satisfied.
3. **Sign tests and Wilcoxon are arithmetically correct.** The 29/33 transfer
   $p=5.5\times 10^{-6}$, the Mettack $p=1.06\times 10^{-43}$, and the
   Wilcoxon $2/1024 = 1.95\times 10^{-3}$ all reconcile to the CSV cells. The
   inferential machinery is rigorously applied even if the multiple-testing
   correction is missing.

## Weaknesses

### MAJOR

- **W1 — κ²⁰⁰ band claim incorrect** (`experiments.tex`, scalability paragraph).
  Printed band $[10^{-105},10^{-48}]$ is contradicted by 13 of 50 real-dataset
  rows in `matfree_error_bounds_corrected.csv` (max observed
  $1.87\times 10^{-12}$ at Amazon seed 2718). **Fix:** report the actual band
  $[10^{-177},10^{-12}]$ or the median row; either is more honest than the
  current quartile-cherry-picked range.
- **W2 — Halko bound stated but not numerically verified.**
  `halko_bound_estimate` column is empty for all 50 real-dataset rows in
  `matfree_error_bounds_corrected.csv`. The body cites the Halko inequality
  and reports the spectral-gap statistic from a different CSV that does not
  use the same rank $k$. **Fix:** re-run R2_04 with `top_k_svd(k=7)` and
  back-fill the column. 30 minutes of GPU time.
- **W3 — `sigma_dense` column empty, $0.03\%$ agreement claim unverifiable.**
  The body claims "matrix-free $\sigma_1$ within $0.03\%$ of dense reference
  at $N=200$" but `sigma_dense` is empty in the corrected CSV. **Fix:** point
  the reviewer at the CSV that actually carries the comparison, or repopulate.
- **W4 — Obs. 3.3(b) mathematically meaningless for general ReLU.** The
  $\mathrm{cond}(\mathrm{diag}(\phi')\otimes I)$ term is $\infty$ whenever
  any node has $\phi'=0$. **Fix:** demote to "Empirical Remark" or restrict
  to the active-mask sub-block.

### MINOR

- **W5 — Multiple-testing correction missing.** 30 breach cells × 5 datasets
  × 6 ε-levels: nominal α=0.05 inflates FWER substantially. Add a footnote
  with Holm or BH-FDR.
- **W6 — t-CIs on right-skewed Pubmed cells.** Switch to BCa bootstrap for
  Pubmed and any cell where std > mean.
- **W7 — Theorem 3.1(a) measure-zero proof gap.** $\delta\Ahat^\star$ along
  $v_1$ is a worst-case direction, not a generic direction. Sentence to that
  effect needed in the proof (Bolte–Pauwels Prop. 2.5 covers exception
  points).
- **W8 — Theorem 3.1(b) proof should call out the upper-bound construction.**
  Add a clause noting that the upper bound is also normality-free
  (geometric series).
- **W9 — `fig:phase_transition` caption inconsistency.** The body text in
  the paragraph above the figure says "certificate boundary, not empirically
  observable discontinuity" but the label says "phase_transition". Rename or
  add a sentence to the caption.
- **W10 — Salvaged CSV trail.** The salvage path is auditable but should be
  referenced explicitly in the supplementary (script + commit hash).

## Recommendation

**Minor Revision.**

The R2 patches deliver on the key technical gating items: Theorem 3.1(b) is
now normality-free, Theorem 3.1(c) is defensible (and honest about the
certificate-vs-divergence distinction), Prop. 3.5 is downgraded honestly, the
inferential statistics are arithmetically correct, and the matrix-free pipeline
is well-described. The remaining issues — the κ²⁰⁰ band number, the empty
Halko / sigma_dense columns, the Obs. 3.3(b) wording — are all mechanically
fixable without further experiments (W1, W4, W5, W6, W7, W8, W9 require only
text edits; W2 and W3 require a 30-minute re-run of R2_04 with `top_k_svd(k=7)`
and `sigma_dense` populated).

**Confidence: 4/5.**

I am highly confident on the linear-algebra audit and on the sign-test /
Wilcoxon checks (verified arithmetically against the CSVs). I am less
confident on the IGNN-specific activation-pattern stability argument (the
Bolte–Pauwels conservative-Jacobian machinery is field-adjacent for me); I
would defer to a specialist on whether the all-active proof of Obs. 3.3(a)
is the full statement the authors need for Prop. 3.6's downstream usage.

## Open questions

1. **Q1.** Can the authors point to the artifact where `sigma_dense` at $N=200$
   is recorded? If `matfree_error_bounds_corrected.csv` is the canonical
   source, the $0.03\%$ agreement number is not currently auditable.
2. **Q2.** Why was R2_04 not re-run with the fixed `top_k_svd(k=7)`? The
   stated cost is "8 hours"; if this is GPU wall-clock, a partial re-run
   restricted to the 5 datasets × 10 seeds = 50 rows would take a fraction
   of that.
3. **Q3.** Has any cell of `tab:breach` been checked against a percentile
   bootstrap with B=10000? The mean-CI relationship for Pubmed is suspicious.
4. **Q4.** For Obs. 3.3(b), what is the operative claim when the active mask
   is rank-deficient — is the bound on $\eta$ vacuous, or is there an
   active-subspace reformulation the authors prefer?
5. **Q5.** The Mettack sign test's 150 trials are paired across what unit
   (seeds × dataset × ε)? The CSV note says "150 paired comparisons" but
   does not specify the pairing structure; this matters for the
   independence assumption.
