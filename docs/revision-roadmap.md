# AEGIS Revision Roadmap

**Source:** Simulated peer-review panel (5 reviewers, 2026-05-27)
**Verdict:** Minor Revision — all items addressable without fundamental re-design

---

## Priority Classification

All review items are parsed, deduplicated, and classified into three tiers. Cross-references show which reviewer(s) raised each item. Items are ordered by dependency: upstream fixes (theory) before downstream (experiments, framing).

---

## Tier 1 — Required (4 items, block acceptance)

### T1.1 · Fix norm conflation in Theorem 1(a) proof
- **Reviewers:** R1-C1, R0-C1
- **Paper location:** `sections/theory.tex`, Theorem 1 proof (Part a, subcritical regime)
- **Issue:** The proof bounds $\|(\hat{A} + \delta A) Z W^\top\|$ mixing $\|\delta A\|_F$ and $\|\delta A\|_2$. Since $\|\cdot\|_2 \leq \|\cdot\|_F$, the bound is *correct but loose*. The statement doesn't clarify which norm $\varepsilon$ refers to.
- **Action:** (1) Add one line to the proof: "Since $\|\delta A\|_2 \leq \|\delta A\|_F = \varepsilon$, the operator norm of the perturbation is bounded by $\varepsilon$." (2) In the theorem statement, specify: "$\|\delta A\|_F = \varepsilon$" (already there) and add a remark that $\varepsilon_{\text{crit}}$ is a Frobenius-norm sufficient condition (conservative relative to the operator norm).
- **Effort:** Low (editorial, ~30 min)
- **Files to edit:** `sections/theory.tex`

### T1.2 · Report $\kappa = \|J_z\|_2$ or justify its omission
- **Reviewers:** R2-C1, R4-A5, R0-C2, R1-Q1
- **Paper location:** `sections/experiments.tex`, Notation paragraph + convergence diagnostics table (Sec. V-E or Appendix)
- **Issue:** Formal bounds use $\kappa$ but tables only report $\rho$. The $\eta = 1.02$–$1.28$ values imply $\kappa \leq 1.28\rho$, but $\kappa$ is never tabulated. Hostile reviewer: "why not just compute it? It's one SVD."
- **Action (preferred):** Compute $\kappa$ via `torch.linalg.svdvals(J_z)[0]` in the analysis pipeline and add a $\kappa$ column to the convergence table. Revise the Notation paragraph to state both are reported.
- **Action (fallback):** If re-running is impractical, add a sentence: "Computing $\kappa$ directly requires materializing $J_z \in \mathbb{R}^{Nd \times Nd}$; for the 50-node subgraph with $d=64$, this is a $3200 \times 3200$ SVD per seed. We report $\rho$ as a computationally cheaper diagnostic and bound the gap via $\eta$."
- **Effort:** Low–Med (code change: add 1 SVD call; table update)
- **Files to edit:** `iem/adversarial.py` (add $\kappa$ to output dict), experiment scripts, `sections/experiments.tex`

### T1.3 · Mettack: 10 seeds or remove
- **Reviewers:** R2-C2, R0-C3
- **Paper location:** `sections/experiments.tex`, Mettack paragraph + any associated table
- **Issue:** If the Mettack comparison uses fewer than 10 seeds, it breaks the paper's own statistical protocol.
- **Action (preferred):** Verify seed count. If already 10, add a note. If <10, re-run with 10 seeds — the Mettack comparison is a sanity check, not a main result, so even weak results are fine.
- **Action (fallback):** Remove the Mettack table entirely and keep only the inline text ("149/150 wins"). The adaptive PGD comparison (Sec. V-C) is the fairer and more important baseline.
- **Effort:** Low (if removing) to Med (if re-running)
- **Files to edit:** `sections/experiments.tex`, possibly `iem/examples/adversarial_baselines.py`

### T1.4 · Clarify "Converged" column in Appendix Table IV
- **Reviewers:** R3-C1, R0-C4, R1-Q3 (related)
- **Paper location:** Appendix, phase transition table
- **Issue:** "Converged = No" is ambiguous — does it mean (a) IGNN fixed-point iteration failed during AEGIS analysis, or (b) IGNN training at high $\rho$ failed? These have very different implications.
- **Action:** Add a footnote or remark: "Converged refers to the IGNN fixed-point iteration during AEGIS analysis at the given $\rho$ setting; models were trained to convergence at all $\rho$ values."
- **Effort:** Low (editorial, ~15 min)
- **Files to edit:** Appendix `.tex` file

---

## Tier 2 — Strongly Recommended (8 items, strengthen paper significantly)

### T2.1 · Reframe headline: lead with $\varepsilon = 0.10$ tightness
- **Reviewers:** R4-A1, R0-R1
- **Paper location:** `sections/abstract.tex`, `sections/introduction.tex` (contributions), `sections/conclusion.tex`
- **Issue:** Tightness = 1.00 at $\varepsilon = 0.01$ is tautological ($O(\varepsilon^2) = O(10^{-4})$). The $\varepsilon = 0.10$ result (within 15%) is the genuinely impressive claim.
- **Action:** In abstract/intro, lead with: "Predictions remain within 15% of ground truth at $\varepsilon = 0.10$, well into the nonlinear regime." Reposition $\varepsilon = 0.01$ as implementation validation.
- **Effort:** Low (editorial, ~45 min)
- **Files to edit:** `sections/abstract.tex`, `sections/introduction.tex`, `sections/conclusion.tex`

### T2.2 · Add significance tests to defense ablation
- **Reviewers:** R2-W3, R0-R2
- **Paper location:** `sections/experiments.tex`, Sec. V-F (defense ablation)
- **Issue:** No statistical tests accompany the accuracy-drop comparisons.
- **Action:** Run paired Wilcoxon signed-rank test (10 paired seeds) on accuracy drop: protected vs. unprotected. Report p-value. If $p > 0.05$, soften the claims.
- **Effort:** Low (add ~5 lines of scipy code, update 1 paragraph)
- **Files to edit:** defense ablation script, `sections/experiments.tex`

### T2.3 · Discuss admittance-weighted adjacency as future work
- **Reviewers:** R3-W1, R0-R3
- **Paper location:** `sections/case_study.tex` (setup paragraph), `sections/conclusion.tex` (limitations)
- **Issue:** Binary adjacency treats all lines equally; real grids use admittance weights.
- **Action:** Add 2–3 sentences: "We use binary adjacency for compatibility with the classification-benchmark pipeline. Admittance-weighted edges ($A_{ij} = 1/X_{ij}$) would better capture electrical coupling and are a natural extension; the $S_c$ framework requires only differentiability of message passing w.r.t. edge weights, which admittance-weighted GNNs satisfy."
- **Effort:** Low (editorial, ~20 min)
- **Files to edit:** `sections/case_study.tex`, `sections/conclusion.tex`

### T2.4 · Validate subgraph representativeness via inter-root $\tau$
- **Reviewers:** R4-A3, R0-R4, R2-W2
- **Paper location:** `sections/experiments.tex`, Sec. V-D (subgraph ablation)
- **Issue:** Different BFS roots may yield different vulnerability rankings. No inter-root variance is reported.
- **Action:** For 2–3 datasets, run AEGIS from 5 different BFS roots on the same graph. Report pairwise Kendall $\tau$ between resulting vulnerability rankings. If $\tau_{\text{inter-root}} > 0.5$, rankings are stable. If low, discuss scope limitations.
- **Effort:** Medium (new experiment, ~2–4 hours)
- **Files to edit:** New script or extend `adversarial_scalability.py`, `sections/experiments.tex`

### T2.5 · Reconcile Citeseer appendix vs. main-text accuracy
- **Reviewers:** R4-A6, R0-R5
- **Paper location:** Appendix B (per-seed breakdown)
- **Issue:** Appendix reports 42–47%, Table I reports 66.0%. Likely different experimental conditions.
- **Action:** Check if appendix uses a different hidden dim, no early stopping, or an earlier code version. Either (a) re-run appendix with final config and update numbers, or (b) add a note: "This table uses $d=32$ without early stopping to isolate the effect of early stopping on certification stability; main-text results use $d=64$ with early stopping."
- **Effort:** Low (if editorial clarification) to Med (if re-running)
- **Files to edit:** Appendix `.tex`

### T2.6 · Rename "per-node robust radius" → "per-node sensitivity radius"
- **Reviewers:** R4-A7, R0 (implicit)
- **Paper location:** `sections/theory.tex` (Proposition 3), `sections/experiments.tex`, `sections/conclusion.tex`
- **Issue:** "Robust radius" invites comparison with formal robustness certificates (Zügner, Bojchevski). AEGIS provides first-order sensitivity radii, not global certificates.
- **Action:** Global find-replace: "robust radius" → "sensitivity radius" or "first-order tolerance radius." Add a sentence in Related Work distinguishing from certificate-based methods.
- **Effort:** Low (global rename, ~30 min)
- **Files to edit:** `sections/theory.tex`, `sections/experiments.tex`, `sections/related_work.tex`, `sections/conclusion.tex`

### T2.7 · Explain IGNN accuracy gap from published results
- **Reviewers:** R2-W1
- **Paper location:** `sections/experiments.tex`, setup or first results paragraph
- **Issue:** IGNN 77.5% on Cora vs. 83.5% published. Readers may suspect implementation bugs.
- **Action:** Add: "Our IGNN uses spectral-norm constrained weights (A2) with $\|W\|_2 \leq 0.9$, which reduces accuracy by ~6% relative to unconstrained IGNN (Gu et al., 2020). This is the accuracy–guarantee tradeoff: contractivity enables formal vulnerability analysis but limits model capacity."
- **Effort:** Low (editorial, ~15 min)
- **Files to edit:** `sections/experiments.tex`

### T2.8 · Mention constrained variant of Proposition 3
- **Reviewers:** R1-W2
- **Paper location:** `sections/theory.tex`, after Proposition 3
- **Issue:** Using constrained $S_c$ rows instead of full $S_v$ gives strictly larger (less conservative) per-node radii.
- **Action:** Add a remark: "Under the constrained threat model (Sec. II-B), the per-node radius can be tightened to $r_v^c = \gamma_v / \sigma_1([S_c]_v)$, where $[S_c]_v$ denotes the block-rows of $S_c$ corresponding to node $v$. Since $\sigma_1([S_c]_v) \leq \sigma_1(S_v)$, we have $r_v^c \geq r_v$; the constrained radius is always at least as large."
- **Effort:** Low (editorial, ~20 min)
- **Files to edit:** `sections/theory.tex`

---

## Tier 3 — Nice-to-Have (5 items, polish for camera-ready)

### T3.1 · Compare AEGIS timing vs. DC power flow
- **Reviewers:** R3-W3, R3-Q3
- **Action:** Time AEGIS's single-pass analysis vs. DC approximation on case14/case30. Add timing column to IEEE table.
- **Effort:** Medium (~2 hours)

### T3.2 · Add case300 results
- **Reviewers:** R3-W4
- **Action:** Run AEGIS on IEEE case300 to demonstrate scalability. If model quality is poor, report honestly.
- **Effort:** Medium–High (training + analysis, ~1 day)

### T3.3 · Heatmap visualization for cross-architecture $\tau$ table
- **Reviewers:** R2-M1
- **Action:** Replace or supplement Table IV with a color-coded heatmap (dataset × architecture).
- **Effort:** Low (~1 hour)

### T3.4 · Cite Trefethen & Embree for pseudospectral threshold
- **Reviewers:** R1-M2
- **Action:** Add citation to Observation 1 (nonnormality index).
- **Effort:** Low (1 citation, ~10 min)

### T3.5 · Bound $\eta$ in terms of graph structure
- **Reviewers:** R1-Q2
- **Action:** Derive a bound $\eta \leq f(\text{degree distribution}, \|W\|_2)$. This is new theory and may not be feasible for a minor revision.
- **Effort:** High (new theoretical result)

---

## Dependency Graph

```
T1.1 (norm fix) ──→ T2.8 (constrained Prop 3 remark)
T1.2 (report κ) ──→ T2.1 (reframe headline)
T1.4 (converged column) ── standalone
T1.3 (Mettack seeds) ── standalone
T2.4 (inter-root τ) ──→ T2.1 (reframe intro claims about subgraph)
T2.5 (Citeseer reconcile) ── standalone
T2.6 (rename radius) ──→ T2.8 (constrained variant uses new name)
T2.7 (IGNN accuracy gap) ── standalone
```

**Suggested execution order:**
1. T1.1, T1.4, T2.5, T2.7 (independent editorial fixes, can be parallelized)
2. T1.2 (requires code change, then table update)
3. T1.3 (verify seed count, decide keep/remove)
4. T2.6 → T2.8 (rename first, then add constrained remark)
5. T2.1 (reframe headline — do last among editorial, after all numbers are final)
6. T2.2, T2.3 (independent, can run in parallel)
7. T2.4 (new experiment — run overnight, update text next day)

---

## Estimated Total Effort

| Tier | Items | Effort |
|------|-------|--------|
| Required | 4 | ~1 day |
| Strongly Recommended | 8 | ~2–3 days |
| Nice-to-Have | 5 | ~2–3 days |
| **Total** | **17** | **~5–7 days** |
