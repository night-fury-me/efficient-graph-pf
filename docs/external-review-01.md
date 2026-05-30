## Overall verdict

**The idea is strong and ICDM-relevant, but the current draft is not yet safe for submission in its present form.** I would currently rate it as **borderline / weak-reject for ICDM Research Track**, mainly because the paper makes several mathematically and experimentally ambitious claims that reviewers may find overstated or insufficiently justified. With targeted revisions, especially to the theory, threat-model semantics, power-grid case study, and baseline framing, it could become a **credible ICDM submission**.

ICDM 2026 is a good venue fit: the Research Track explicitly covers foundations, algorithms, theory, deep learning/statistical methods, graph data mining, data mining systems, security/privacy, cyber-physical systems, complex networks, and engineering applications. Submissions are judged on technical quality, relevance, originality, significance, and clarity, and are limited to **10 IEEE two-column pages including references and appendices**. ([icdm2026.neu.edu.cn][1]) Your PDF is also 10 pages, so any added proof/detail must come from compression rather than expansion.

---

## My high-level score

| Dimension                       |                  Current score | Comment                                                                                                                                                                        |
| ------------------------------- | -----------------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ICDM relevance                  |                       **8/10** | Graph mining + adversarial robustness + scalable diagnostics + power-grid case study fit ICDM well.                                                                            |
| Novelty                         |                       **7/10** | The unified constrained sensitivity object is promising, but reviewers may see parts as influence-functions/Jacobian-SVD adapted to GNN edges unless positioning is sharpened. |
| Technical correctness           |                       **5/10** | Theorem 1 and Prop. 7 have statements that look too strong or mathematically fragile. This is the biggest risk.                                                                |
| Experimental strength           |                     **6.5/10** | Broad coverage, but many headline results rely on small 50-node BFS subgraphs; baseline comparison is uneven.                                                                  |
| Power-grid credibility          |                     **5.5/10** | Interesting, but currently closer to an illustrative case study than an operational contingency-screening result.                                                              |
| Clarity/presentation            |                       **6/10** | Ambitious and dense, but overloaded with claims, numbers, and terminology. Needs simplification.                                                                               |
| Acceptance standing now         |   **Borderline / weak reject** | Strong idea, but too many reviewer attack surfaces.                                                                                                                            |
| Acceptance standing after fixes | **Borderline accept possible** | Especially if theory is corrected and claims are narrowed.                                                                                                                     |

---

## What the paper is trying to contribute

The paper proposes **AEGIS**, a one-pipeline adversarial diagnostic method for GNNs. The central object is the constrained sensitivity matrix (S_c), intended to provide three outputs: an SVD-optimal first-order perturbation direction, per-edge vulnerability scores, and per-node first-order sensitivity radii. The method is motivated for structural perturbations in GNNs and extended from contractive implicit GNNs to explicit architectures such as GCN, SAGE, GIN, APPNP, and an edge-weighted GAT variant. The paper also reports scalability up to Amazon Photo with (N=7{,}650), broad architecture/dataset transfer, comparisons against attacks/certifiers, and a power-flow contingency case study. 

That is a good ICDM story: **graph data mining + robustness + scalable algorithm + applied cyber-physical validation**. The core problem is not relevance; the core problem is whether the current claims survive skeptical review.

---

# Major strengths

## 1. The unified diagnostic object is genuinely appealing

The strongest part of the paper is the idea that one constrained structural sensitivity operator can produce:

1. a global first-order perturbation direction,
2. per-edge vulnerability rankings,
3. per-node sensitivity radii.

This is a clean conceptual contribution. ICDM reviewers often like methods that are not only attacks or not only certificates, but provide actionable diagnostics for graph mining systems. The paper does a good job distinguishing AEGIS from structural attacks, smoothing/certification methods, and robust-architecture defenses. 

## 2. The matrix-free implementation is practically valuable

The matrix-free Neumann/JVP/VJP formulation is an important practical ingredient. The draft claims dense/matrix-free agreement within 0.03% at (N=200), and scalability to (N=7{,}650) on a single GPU. That is relevant for ICDM because scalability and system practicality matter more there than in a purely theory venue. 

## 3. The paper has a stronger-than-average empirical breadth

The paper covers 9 datasets, 7 architectures, multiple baselines, 10 seeds, explicit GNN extension, defense ablation, and power-flow benchmarks. That breadth is a plus for ICDM. The continuous-to-discrete transfer table and the architecture heatmap are among the most compelling empirical parts. 

## 4. Power-grid case study can differentiate the paper

The power-grid angle is a good ICDM differentiator. ICDM has explicit scope for cyber-physical systems, complex networks, and engineering applications. ([icdm2026.neu.edu.cn][1]) If strengthened, this could make the paper stand out from a standard adversarial-GNN paper.

---

# Major weaknesses that may hurt acceptance

## 1. Theorem 1 has a serious mathematical fragility

The critical-regime claim is the most dangerous part of the paper.

The paper states that as (\epsilon \to \epsilon_{\mathrm{crit}}), the resolvent diverges using a “Neumann lower bound” of the form

[
|(I-J_z')^{-1}|_2 \gtrsim \frac{1}{1-|J_z'|_2}.
]

This is not generally valid. The standard Neumann argument gives an **upper bound**:

[
|(I-M)^{-1}|_2 \le \frac{1}{1-|M|_2}
\quad \text{when } |M|_2 < 1.
]

A lower-bound divergence requires additional structure: for example, an eigenvalue or singular/eigen-direction approaching (+1), not merely (|M|_2 \to 1). A matrix can have norm near 1 while (I-M) remains well-conditioned if the dangerous direction is not aligned with eigenvalue (1). This is a likely reviewer-killer because the paper heavily sells a “three-regime characterization” and “critical budget.” The paper’s current Theorem 1 statement and proof are therefore too strong. 

**Fix:** Recast (\epsilon_{\mathrm{crit}}) as a **sufficient contraction-preservation radius**, not as a guaranteed phase transition. Then state a separate conditional critical result:

> If along a perturbation path (J_z'(\epsilon)) has an eigenvalue approaching (1) and the corresponding eigenvectors are not pathologically ill-conditioned, then the resolvent norm diverges at rate proportional to the inverse spectral gap.

That would be mathematically safer.

---

## 2. The “phase transition” language is too strong

The draft calls the theory a “phase-transition” result. But the experiments themselves show that trained models stay well below the worst-case boundary: the empirical resolvent grows mildly, and the trained ReLU pattern keeps (\rho(J_z)\le 0.42) even when the spectral cap is high. That actually supports a **safety-envelope** interpretation, not a true observed phase transition. 

**Fix:** Replace “phase transition” with something like:

> “contraction-regime envelope”
> “sufficient stability boundary”
> “three-regime safety characterization”

Use “phase transition” only if you demonstrate an actual sharp transition in either convergence, resolvent norm, or classification damage.

---

## 3. Proposition 7 may not correctly model normalized adjacency deletion

The continuous-to-discrete bridge says edge deletion corresponds to setting ([\delta \hat A]*{ij}=[\delta \hat A]*{ji}=-w_k). But for standard normalized adjacency,

[
\hat A = D^{-1/2}(A+I)D^{-1/2},
]

removing one edge changes the degrees of the incident nodes, and therefore changes **all normalized incident edge weights**, not only the deleted edge entry. The current proof appears to treat deletion as zeroing one normalized edge while keeping the normalization fixed. That is a mismatch between the discrete graph operation and the continuous perturbation model. 

**Fix:** Either:

1. explicitly define the discrete deletion experiment as **fixed-normalization edge masking**, or
2. add the derivative/remainder terms caused by degree-renormalization, or
3. run experiments showing that fixed-normalization deletion and recomputed-normalization deletion yield similar rankings.

Without this, a reviewer can attack the key continuous-to-discrete claim.

---

## 4. “One-query” is rhetorically risky

The paper says “one-query,” but the algorithm uses randomized SVD, multiple MATVEC/RMATVEC calls, Neumann iterations, and then per-edge MATVEC calls for column norms. In adversarial ML, “one-query” often means one forward model query or black-box query. Here it means something closer to **one matrix-free diagnostic pass** or **one linearized analysis stage**. 

**Fix:** Change the title or clarify immediately:

> “One-Pass Matrix-Free Adversarial Diagnostics…”

or

> “One Linearized Solve for Adversarial Diagnostics…”

If you keep “one-query,” define it very precisely in the abstract and introduction to avoid reviewer backlash.

---

## 5. The per-node radius is not a certificate, but the wording sometimes sounds certificate-like

The paper is careful in Remark 6 that (r_v) is a first-order sensitivity threshold, not a sound probabilistic certificate. But elsewhere the language “preserves classification” and “radii” may be read as certification. 

The formula also uses the current closest competitor (c^*). For a rigorous classification-preservation bound, you usually need to account for **all competing classes**, e.g.

[
r_v = \min_{c\ne y_v}
\frac{f_{y_v}(z_v^*) - f_c(z_v^*)}
{|(W_{y_v}-W_c)S_v|_2}.
]

Using only (c^*) is locally intuitive but not always sufficient if another class has a larger sensitivity direction.

**Fix:** Replace the radius theorem with the min-over-classes version. Then present the (c^*)-only version as a cheaper approximation or empirical variant.

---

## 6. The power-grid case study is interesting but not yet operationally convincing

The current power-grid section is good for ICDM relevance, but weak for power-systems credibility. The model uses binary adjacency and 5 bus features, and the data are generated by uniform load scaling over 70–130% of nominal load. The paper itself admits that this does not cover seasonal peaks, dispatch shifts, or renewable ramps. 

The bigger issue: contingency screening in power systems is usually about thermal overloads, voltage violations, post-contingency flows, and operating limits. Your ground truth is (\ell_2) voltage-angle deviation. That is a valid proxy, but not the same as operational N-1 severity.

Also, the LODF comparison is not clean. LODF is naturally a DC line-flow outage measure; comparing it against AC voltage-angle deviation can make LODF look weak for the wrong reason. The draft later says LODF gets (\tau=0.44)–(0.58) on credible cases, while earlier Table IV gives a much weaker LODF number for case118. This needs careful explanation. 

**Fix:** Reframe the power-grid result as:

> “a proof-of-concept vulnerability-attribution case study on learned AC-surrogate sensitivity,”

not as “operator-grade contingency screening.”

Add at least one of the following if possible:

* ranking against thermal overload or voltage-limit violation,
* line-flow change as the target, not only voltage-angle deviation,
* line parameters/admittance/limits as edge features,
* non-uniform load perturbations,
* generator dispatch variation,
* comparison to LODF on its native DC-flow metric.

---

# Experimental concerns

## 1. Too much of the main evaluation depends on 50-node BFS subgraphs

The paper openly states that many IGNN experiments use 50-node BFS ego-subgraphs and that on Cora, these cover only about 1.8% of edges with low agreement to full-graph ranking.  That is honest, but dangerous. ICDM reviewers may ask: if the method scales, why are many main results still subgraph-local?

**Fix:** Move full-graph results to the main table for Cora, Citeseer, Amazon Photo, and ideally at least one larger benchmark. Keep 50-node subgraphs only as dense-vs-matrix-free validation.

## 2. Baseline framing is uneven

The table against GR-BCD shows AEGIS is slightly better on Pubmed but worse on Cora for the reported damage value. Yet the narrative still sounds broadly favorable.  A skeptical reviewer will notice.

**Fix:** Be more direct:

> “AEGIS is not designed to maximize classification loss under a discrete budget; it is a label-free equilibrium-sensitivity diagnostic. Against GR-BCD, it is competitive on Pubmed but weaker on Cora under direct attack damage. Its value is interpretability, one-pass ranking, and compatibility with radii/direction diagnostics.”

That honesty will increase trust.

## 3. Classification performance is secondary, but adversarial papers need classification impact

The paper emphasizes equilibrium shift. That is valid for the theory, but adversarial ML reviewers will still ask: does this actually change predictions, cause loss increase, or expose meaningful vulnerability? You report breach rates, but they are often low. That weakens the “attack” framing.

**Fix:** Frame AEGIS as **diagnostic/early-warning**, not primarily as a stronger attack. Put classification flips/loss in a supporting role.

---

# Clarity and writing problems

The paper is very dense. The abstract alone contains too many claims: one-query, three diagnostics, closed-form regimes, 2–4× margin, 29/33 cells, (\tau=0.998), PGD comparison, masking defense, power-flow P@10, disclosure. This reads impressive, but also like overclaiming.

There are also presentation issues that reviewers will notice:

* “Vulnarability” typo in Fig. 1.
* “perturnbation” typo in Fig. 1.
* “theorem 6” is referenced, but Remark 6 is not a theorem.
* Proposition/Theorem numbering is inconsistent: e.g., Proposition 4 is referred to as theorem 4.
* The title says “One-Query,” but the algorithm is not a single forward query.
* The GAT notation (GAT^\dagger) needs to be defined earlier and more cautiously.
* “Code released under tiered disclosure” may conflict with reproducibility expectations unless handled carefully.

ICDM also requires triple-blind review for the Research Track, and the guidelines emphasize anonymization, careful self-citation, anonymized code/data if shared, and reproducibility details. ([icdm2026.neu.edu.cn][1]) Your reference [51] includes names and appears very close to the authors’ own power-flow work; if it is your group’s paper, cite it in the third person only if necessary and ensure it does not compromise anonymity.

---

# ICDM track recommendation

Submit to the **Research Track**, not the Applied Track, unless you significantly strengthen the power-grid deployment story.

The Research Track is the better fit because the main contribution is an algorithmic/theoretical graph-mining method with scalability and robustness diagnostics. ICDM Research Track explicitly includes algorithms, theory, deep learning, graph data, systems, security/privacy, cyber-physical systems, and engineering applications. ([icdm2026.neu.edu.cn][1])

The Applied Track would be suitable only if the paper were primarily about a deployed or deeply validated power-grid diagnostic system. The Applied Track solicits deployed systems, real-world applied ML/data mining, new benchmarks, and comprehensive experimental analyses. ([ICDM 2026][2]) Your current power-grid section is promising but not strong enough to carry the paper as an applied submission.

---

# Most important revision plan before submission

## Priority 1: Fix the theory

This is non-negotiable. Revise Theorem 1 so that:

* (\epsilon_{\mathrm{crit}}) is a **sufficient contraction radius**, not a guaranteed critical phase transition.
* Remove or condition the claimed lower-bound divergence.
* Add assumptions needed for resolvent blow-up.
* Fix the radius theorem using min-over-classes.
* Fix continuous-to-discrete deletion under degree renormalization.

## Priority 2: Reduce overclaiming

Change the core claim from:

> “globally maximally sensitive perturbation direction / one-query attack”

to:

> “globally optimal first-order hidden-state sensitivity direction under the constrained continuous edge-weight model.”

That is still strong, but defensible.

## Priority 3: Strengthen full-graph experiments

Move full-graph evidence into the main experimental story. At minimum, add a compact table:

| Dataset | (N) | (|E|) | runtime | memory | AEGIS vs random | Kendall vs deletion |
|---|---:|---:|---:|---:|---:|---:|

Include Cora/Citeseer full graph if possible, not only Amazon Photo.

## Priority 4: Reframe the power-grid section

Make it honest and precise:

* “voltage-angle deviation ranking,” not general contingency severity,
* “learned surrogate diagnostic,” not operator-ready N-1 screening,
* LODF compared on its native metric or clearly qualified,
* case300 excluded from headline claims.

## Priority 5: Clean presentation

The paper needs a full pass for numbering, notation, typos, and claim hierarchy. Reviewers are more forgiving of a narrow paper with clean claims than a broad paper with fragile claims.

---

# Suggested positioning statement

A safer ICDM framing would be:

> “AEGIS is a scalable matrix-free sensitivity diagnostic for graph neural networks. It does not replace certified robustness or discrete attack optimization; instead, it provides a unified first-order vulnerability spectrum—edge rankings, dominant structural sensitivity direction, and node-level sensitivity thresholds—from one constrained structural Jacobian. For contractive implicit GNNs, we provide a sufficient stability envelope, and for explicit GNNs we empirically validate the same diagnostic operator across architectures and domains.”

This framing avoids claiming too much while preserving the main contribution.

---

## Final ICDM standing

**Current draft:** likely **borderline / weak reject** at ICDM Research Track because reviewers may challenge the theory, “one-query” terminology, subgraph-heavy evaluation, and power-grid operational claims.

**After targeted revision:** credible **borderline accept / weak accept** candidate, especially if you make the paper more honest: “matrix-free first-order adversarial diagnostics for graph mining” rather than “one-query adversarial attack with phase transition.”

The idea is good enough for ICDM. The main task is to make the claims mathematically defensible and reviewer-proof.

[1]: https://icdm2026.neu.edu.cn/CallforResearchTrack/list.htm?utm_source=chatgpt.com "Call for Research Track Papers"
[2]: https://icdm2026.neu.edu.cn/CallforAppliedTrackPapers/list.htm "Call for Applied Track Papers"
