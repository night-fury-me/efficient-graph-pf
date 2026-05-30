# Reviewer 2 — Methodology (Adversarial ML + Numerical Linear Algebra)

## Summary

The methodology blends (i) IFT applied to the fixed-point operator of a contractive implicit GNN, (ii) a Neumann-series truncation that avoids materialising the resolvent, (iii) a duplication-matrix–like edge-supported projection $P_c$, and (iv) a randomised SVD (Halko–Martinsson–Tropp) for the leading singular triple. Statistically, results are reported as mean ± std over 10 seeds across 9 datasets and 7 architectures. The dense $\sigma_1$ vs matrix-free $\sigma_1$ agreement at $N=200$ (0.03 %) is a credible internal validation; the Neumann residual $<10^{-6}$ confirms truncation control.

## Theorem 3.1 (Phase Transition)

### Part (a) — Subcritical regime

Application of the implicit function theorem to $G(z, \hat A) = z - F(z, \hat A) = 0$ at $z^*$, combined with the Neumann bound $\|(I-J_z)^{-1}\|_2 \le 1/(1-\kappa)$, is standard and correct. The piecewise-affine handling of ReLU via Bolte–Pauwels conservative IFT is the appropriate device.

**Concern A.1 (must address).** The proof writes "$F_\theta$ is piecewise-affine on each ReLU linear region," but inside the fixed-point iteration the activation pattern at $z^*$ can change with arbitrarily small $\delta\hat A$ if $z^*$ lies on a region boundary. The conservative IFT applies on the relatively-open interior of a region and on a measure-zero exceptional set across boundaries; the proof should state explicitly that the activation pattern at $z^*$ is generically stable under sufficiently small $\delta \hat A$, and that the measure-zero exceptional set is the set of $\hat A$ where $z^*$ lies on a region boundary. This is a one-paragraph clarification, not a structural fix.

### Part (b) — Critical regime

The claim is that along directions aligned with the top singular vector of $\hat A$, $\|(I - J'_z)^{-1}\|_2$ diverges as $\Omega(1/(\varepsilon_{\rm crit} - \varepsilon))$.

**Concern A.2 (must address).** The proof writes "$\|J'_z\|_2 \to 1$ forces $\|(I-J'_z)^{-1}\|_2 \to \infty$." This is true. But:

- The lower bound $\|(I-J'_z)^{-1}\|_2 \ge 1/(1-\|J'_z\|_2)$ is tight only when $J'_z$ is *normal*. For non-normal $J'_z$, the inverse can diverge *faster* (more violent), not slower; the qualitative direction is correct but the rate is not "$\Omega(1/(\varepsilon_{\rm crit}-\varepsilon))$" without invoking the pseudospectral index $\eta$. State the bound as $\|(I-J'_z)^{-1}\|_2 \ge \eta / (1-\|J'_z\|_2)$ (or equivalently use the operator-norm lower bound only) and cite Observation 3.3 to control $\eta$.
- The "worst-case direction" claim chooses $\delta\hat A$ along the top singular vector of $\hat A$, but the perturbation that maximises $\|J'_z\|_2$ is along the top singular vector of $J'_z$ itself (which, for a Kronecker product $\hat A \otimes W$ at fixed $W$, *coincides* with the leading singular vector of $\hat A$ — but only when $\phi' \equiv 1$). For active ReLU masks the alignment is not exact. State this caveat.

### Part (c) — Supercritical regime

"The contraction certificate is void" is correct; "the part-(a) first-order guarantees lapse" is correct. The current wording slips into characterising the post-threshold behavior, which the proof does not establish.

**Concern A.3 (must address).** Restate (c) defensively: *"For $\varepsilon \ge \varepsilon_{\rm crit}$ our certificate fails. We do not claim divergence or oscillation as a general regime; empirically, in the supercritical band X% of seeds exhibit non-convergent iteration."* The current paper does not run any experiment with $\varepsilon > \varepsilon_{\rm crit}$, which is itself a problem (see Devil's Advocate). At minimum, the theorem statement should not over-claim what (c) certifies.

## Proposition 3.5 (Continuous → discrete transfer)

The first-order bridge $d_k = w_k v_k + R_k$ with $|R_k| \le L_J w_k^2 / (2(1-\kappa)^2)$ is correct in form.

**Concern A.4 (must address).** The proof states $L_J \le \|W\|_2^2$ for IGNN, justified by "ReLU activations make $J_z$ locally constant; jumps by at most $\|W\|_2^2$ across activation boundaries." A Lipschitz constant on a *path* requires bounding the cumulative jump, not the per-boundary jump. Provide a one-line argument: along a path of length $\|\delta\hat A\|_F$ the iterate crosses at most finitely many ReLU boundaries; the cumulative variation of $J_z$ is bounded by the operator norm of the activation-boundary jump, which is $\le \|W\|_2^2$ per boundary crossing. State the worst-case under "at most $K$ boundary crossings on the path" or invoke an averaged Lipschitz constant.

## Observation 3.3 ($\eta$ bound)

The fully-positive-activation case ($\phi' \equiv 1$) is correct and elegant.

**Concern A.5 (should address).** The general-ReLU bound $\eta \le \kappa(V_{J_z})$ and the assertion "$\kappa(V_{J_z})$ empirically stays near $\kappa(V_W)$ ($\eta = 1.02$–$1.28$)" is an empirical claim, not an observation. Re-label as "Empirical Remark" or scope Observation 3.3 to the fully-active case.

## Algorithm 1 (matrix-free pipeline)

- **Line 2** writes the matvec as $\sum_{j=0}^K J_z^j \mathrm{JVP}_F^{\hat A}(P_c v)|_{z^*}$. State explicitly that the JVP is at fixed $z^*$ with $\hat A$ as the varying argument.
- **Lines 3–7** are a power-iteration prelude to Halko Algorithm 4.4 / 5.1; cite the specific algorithm so readers can audit. The current "rSVD~\cite{halko2011finding}" is too loose for reproducibility.
- **Line 9** $v_{ij} \leftarrow \|\mathrm{matvec}(e_{ij})\|_2$ is $O(|E| \cdot K \cdot Nd)$ for per-edge column evaluation, which the paper acknowledges is impractical at $|E| = 119$K (Amazon Photo). Make this conditional / dense-only in the algorithm, or specify a randomized sketch for large $|E|$.

## Statistical evaluation

- 10 seeds per cell is acceptable; the cross-architecture coverage is generous.
- **Concern A.6 (must address).** No explicit hypothesis test for "transfer is positive" across cells. With 29/33 positive cells, a one-sided sign test gives $p \approx 2 \times 10^{-6}$ — trivially significant — but reporting it locks the claim. Add a one-line test in §V or a footnote in Table 2.
- **Concern A.7 (must address).** Subgraph-to-full-graph Kendall $\tau = 0.16$ on Cora (current §Limitations) is the dominant evaluation regime. Provide at least one full-graph $\tau$ (Amazon Photo at $N = 7{,}650$ is computable; report it). Without this, the subgraph-rankings claim cannot be extended to citation-scale graphs.

## Reproducibility

- Code release with tiered access is stated; the description is clear.
- Hyperparameters: $k = 10$, $p = 10$, $n_{\rm iter} = 2$, Neumann tol $\tau = 10^{-6}$ — all stated. Power-iteration count and seed protocol are listed in §V.

## Decision lean

**Major Revision.** Concerns A.1–A.7 are recoverable. Theorem 3.1 needs (a) a one-paragraph activation-pattern stability statement, (b) the non-normal lower bound restated via $\eta$ or as an operator-norm-only bound, (c) a defensive restatement of regime (c). Proposition 3.5 needs the path-Lipschitz argument. Statistical tests and full-graph $\tau$ are quick additions.
