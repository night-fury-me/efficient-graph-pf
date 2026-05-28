# Reviewer 1 — Methodology Review

**Role.** Adversarial-robustness methodologist with hands-on PRBCD / GR-BCD / Mettack / randomized smoothing experience.
**Mode.** Independent.
**Page-budget calibration.** 10-page IEEEtran cap binding.

---

## 1. Summary as I read it

The authors compute a sensitivity matrix $S = (I-J_z)^{-1} J_A$ (IFT-resolvent for IGNN; finite-difference / unrolled Jacobian for explicit GNNs), project to a $|E|$-dimensional symmetric perturbation subspace ($S_c = S P_c$), and read off three outputs: the leading right singular vector $v_1$ as a perturbation direction, $\|[S_c]_{:,k}\|_2$ as per-edge scores, and $r_v = m_v / (\|W_{y_v} - W_{c^*}\|_2 \cdot \|S_{c,v}\|_2)$ as per-node first-order thresholds. The matrix-free computation uses truncated Neumann series + JVP/VJP + randomized SVD (Halko 2011). For contractive IGNN, $\varepsilon_\text{crit} = (1-\kappa)/\|W\|_2$ is the budget at which the contraction certificate fails.

The empirical study is large (330 runs + IEEE cases) and uses 10 seeds throughout, which is positively noted up front.

## 2. Strengths

**S1. 10-seed discipline is excellent.** Tables 1, 2, 4, 5, 6, scalability, IEEE — all 10 seeds. Standard deviations reported with $\pm$ notation. Wilcoxon signed-rank where appropriate (Mettack comparison, defense ablation, LODF). This is above average for an adversarial-ML paper.

**S2. Finite-difference sanity check (τ=0.999) for $S_c$ column norms is the right control.** It rules out implementation bugs as the source of the rankings; the IFT machinery is doing what the closed form says.

**S3. Honest scalability boundary.** The authors state $N \approx 7{,}650$ on a 24GB GPU and explicitly note Pubmed ($N=19{,}717$) is OOM. They do not claim scalability they have not demonstrated.

**S4. Comprehensive attack-method table (Table on attack quadrants).** Cls-PGD, Shift-PGD, Random, SVD across 4 datasets is the right framing. 1000-random-directions reaching only 0.45–0.49 × $\sigma_1(S_c)$ is a clean confirmation that the SVD direction is well separated, not a generic optimum.

**S5. The phrase shift from "optimal attack" to "maximally sensitive perturbation direction"** (after Cls-PGD beats Shift-PGD on classification flips) is methodologically honest. Many papers would have left the original framing.

## 3. Weaknesses (numbered for R&R cross-reference)

### W1. **Tightness ratio semantics — the bound is loose, not the linearisation underestimating.** [Major]

The paper defines tightness ratio = (actual shift) / (predicted shift) and notes "tightness ≥ 1 means the linearisation underestimates damage, the safe direction for a diagnostic" (§Cross-domain). At ε=0.20, tightness reaches 1.36 (Cora), 1.39 (Citeseer). The framing presents this as a virtue ("safe direction") but it is the conventional definition of a **loose bound**.

A *tight* bound is one whose ratio is 1.00 ± noise; 1.36 means the bound under-predicts the realised shift by 36%. A diagnostic that under-predicts by 36% is not a safety threshold a practitioner can rely on at that ε.

**Concretely, ask the authors to:**
- Rename "tightness ratio" to "first-order envelope ratio" or simply "actual/predicted shift", removing the implication that ≥ 1 is a diagnostic virtue.
- Restrict the headline tightness claim ("1.00 ± 0.01") to ε=0.01 explicitly in the abstract, not as a global claim.
- Acknowledge that the bound is genuinely tight only at small ε.

**Fix fits the budget:** ≤ 3 sentences in §Cross-domain + abstract rewording.

### W2. **Subgraph protocol τ=0.16 with full-graph rankings is a structural problem, not a limitation footnote.** [Critical]

The paper reports that on Cora, "50-node BFS rankings correlate weakly with full-graph rankings (τ=0.16)." This is acknowledged in limitations but the entire §Cross-domain analysis (Tables tightness_eps, breach, attack_full) is run on 50-node subgraphs. If those rankings don't transfer to the full graph, then:

- The per-edge $v_{ij}$ output is only valid at the subgraph scale.
- The "$S_c$ unifies per-edge rankings, attack direction, per-node radii" headline holds at subgraph scale but breaks at graph scale.

The authors offer the matrix-free pipeline (validated to $N=7{,}650$) as a remedy, but the matrix-free pipeline is not run to verify that **its** rankings agree with full-graph dense rankings for the same architecture. The only stability statement at full-graph scale is the random-SVD spectral-gap argument (Halko bound) — that bounds the rSVD error, not the ranking stability.

**Concretely, ask the authors to:**
- Run the matrix-free pipeline on full Cora and report per-edge $\tau$ vs the dense $N=200$ subgraph rankings as a self-consistency check.
- If the τ is high (which I expect — the matrix-free path is computing the same $S_c$), state it explicitly. This resolves the issue.
- If τ is low, restrict the per-edge ranking claim to dense-path regimes only.

**Fix fits the budget:** ≤ 1 table row + 2 sentences. The experiment is essentially free since the pipeline already exists.

### W3. **Missing PRBCD as a baseline.** [Major]

The current SOTA structural attack at scale is PRBCD (Geisler et al. 2021, NeurIPS — "Robustness of Graph Neural Networks at Scale"). The paper cites Geisler 2021 *as GR-BCD* (the smaller variant). GR-BCD is benchmarked on Pubmed k=10 in Table baselines. PRBCD scales to OGB and is the natural comparison for a paper claiming $N=7{,}650$ scalability.

If PRBCD outperforms AEGIS at the operational scale, the paper's contribution is at risk. If PRBCD is comparable or worse, the paper is stronger. Either way, this comparison must appear.

**Concretely, ask the authors to:**
- Add one row to the baselines table comparing AEGIS vs PRBCD on the largest tractable graph (e.g., full Cora).
- If space-constrained, drop the AGNNCert row (it's already noted as a loose-radius comparison at 4.9–10.2×) and use the freed row for PRBCD.

**Fix fits the budget:** Row swap in existing table; ≤ 4 lines of text.

### W4. **Tightness and breach are on different quantities.** [Moderate]

The tightness claim (≥ 1 across ε) is on the **equilibrium shift** $\|\Delta z^*\|_F$. The breach claim (< 2% at ε ≤ 0.10) is on the **prediction flip rate** at the linear-head readout. These are different quantities: $\Delta z^*$ can be large without crossing a classification margin.

Currently the abstract reads as if tightness ≈ 1 and breach < 2% are co-evidence. They are not: they are evidence for two different things. A reader could conclude "the linearisation is tight at small ε" — true for $\Delta z^*$, not for prediction outcomes.

**Concretely, ask the authors to:**
- Add one sentence clarifying that tightness is measured on equilibrium shift, breach on prediction flip; the two metrics measure complementary regimes.
- Move the breach-rate-by-ε discussion adjacent to the tightness discussion so readers see the linkage.

**Fix fits the budget:** ≤ 2 sentences.

### W5. **Mettack-only structural comparison is thin.** [Moderate]

The paper compares against Mettack on 3 datasets (149/150 wins). Mettack is a 2019 method. Beyond GR-BCD (W3), the literature has FGAttack, PGD-based variants, Heuristic-AttackSelection (Mujkanovic 2022). The "3–10× more damage than Mettack" framing is true but the bar is low.

The "$+6$–$148\%$ over degree-proportional / edge-betweenness / spectral baselines" with Wilcoxon $p < 0.001$ is more defensible because it's against heuristics, not against an attack baseline. State the latter more prominently.

**Concretely, ask the authors to:**
- Reframe the Mettack number as "calibration vs 2019 SOTA" rather than "headline structural attack comparison".
- Make the heuristic comparison the headline structural number; the GR-BCD row in Table baselines is the headline attack-vs-attack number; PRBCD (W3) would complete the picture.

**Fix fits the budget:** Rewording.

### W6. **AGNNCert IBP comparison is unfair.** [Moderate]

The paper compares AEGIS's $r_v$ (Cora median 0.187) against AGNNCert's IBP radius (1.414, "10.2× looser"). But these are different mathematical objects: IBP gives **sound certificates** (no perturbation within the radius can flip the prediction); $r_v$ gives a **first-order sensitivity threshold** that the authors themselves acknowledge can be violated. The "10.2× looser" framing reads as if AEGIS dominates AGNNCert, when actually AGNNCert is providing a stronger guarantee at a less-tight radius.

The Remark on Certificate Semantics is good and correct, but it is in the theory section; the Table baselines header should at least flag this asymmetry, otherwise a casual reader will mis-attribute the comparison.

**Concretely, ask the authors to:**
- In Table baselines caption: add "AGNNCert: sound IBP radius; AEGIS $r_v$: first-order sensitivity threshold. Direct numerical comparison reflects different semantics."
- Move "Cora med. $r_v$" column header to "Cora med. radius (semantics differ)".

**Fix fits the budget:** Caption rewording, ≤ 30 words.

### W7. **Breach rate reporting uses median because of high variance — needs more transparency.** [Minor]

§Cross-domain: "medians are preferable for high-variance settings… $3/10$ seeds at $0\%$" at ε=0.10. This is honest but raises a question: if 30% of seeds breach 0 nodes and 70% breach many, the median is a soft headline. The mean and the seed-breach distribution should also be reported (in a small footnote or compact table).

**Fix fits the budget:** Add (mean ± SD) alongside median for the affected ε rows.

### W8. **Pseudospectral index η reporting is partial.** [Minor]

Theorem 1 commentary states "$\eta = 1.02$–$1.28$" but doesn't separate by dataset or architecture. Observation 1 shows η is governed by W not by graph topology — empirical η per-dataset (already in Table cross_domain via $\kappa$?) would close the loop.

**Fix fits the budget:** One extra column in Table cross_domain.

## 4. Minor / typographic

- "GAT$^\dagger$" footnote dagger meaning is explained, but the symbol appears in the abstract before any explanation. Consider inlining the definition (e.g., "edge-weighted GAT variant") on first abstract mention.
- "Tightness $\geq 1$ means the linearisation underestimates damage, the safe direction for a diagnostic" reads as advocacy not description; reword (W1).
- Table breach: standard deviation shown; consider showing IQR for medians.
- Algorithm 1 line 1: "$\kappa \leftarrow \|J_z\|_2$ via power iteration on the JVP of $F_z$" — specify number of power iterations.

## 5. Reproducibility

- 10 seeds, exact seed list to be released ✓
- PyTorch + spectral-norm + Adam config stated ✓
- Dataset splits to be released ✓
- PandaPower contingency script to be released ✓
- Tiered code-release plan stated ✓

The reproducibility plan is above-average for this venue band. One open question: will the **trained model checkpoints** be released, or only training scripts? With 330 runs, retraining is a real cost barrier for a reproducer.

**Concretely:** state explicitly whether checkpoints are released (tiered or open).

## 6. Recommendation

**Major Revision.** W1 (tightness framing) and W2 (subgraph τ=0.16) are critical to address. W3 (PRBCD) is critical for the paper's claim to scale. W4–W8 are tractable and fit the 10-page budget. None of these require new sections — the issues are framing, baseline selection, and one self-consistency experiment.

## 7. Scores

| Dimension | Score (0–100) |
|---|---|
| Research design | 68 |
| Statistical validity | 72 |
| Baseline coverage | 58 |
| Reproducibility | 78 |
| Numerical claim faithfulness | 64 |
| **Methodology overall** | **66** |
