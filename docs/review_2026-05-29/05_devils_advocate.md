# Devil's Advocate Review — AEGIS

**Role.** Hostile-but-fair top-tier reviewer; allergic to unification narratives that fray at operational scale.
**Mode.** Independent of other reviewers.
**Page-budget calibration.** Acknowledged. Hostility is targeted at *claim language* and *evidence-claim alignment*, not at "you should have written more."

---

## Strongest Counter-Argument (250 words)

The paper's headline contribution — that $S_c$ unifies three outputs (direction, edge ranking, per-node radius) from a single closed-form computation — is a presentation device, not a practitioner pain point.

Practitioners doing GNN robustness analysis do not "shop for a unified $S_c$ that gives all three at once." They pick the certificate they trust (randomized smoothing for sound radii, despite cost); they pick the attack they fear (PRBCD for scalable structural attacks); they pick the edge ranking they need (Mettack-style meta-gradients or, in power systems, LODF / brute-force N-1). These tools are already coupled in practice: a smoothing certificate plus PRBCD attack plus an LODF screen is a standard pipeline.

AEGIS replaces this pipeline with a single Neumann-resolvent + randomized SVD computation. **But the three outputs it produces are individually weaker than the dedicated tool in each lane:**
- The per-edge ranking transfers poorly at scale (Cora τ = 0.16 between 50-node subgraph and full graph), so it is *not* a substitute for full-graph attack analysis.
- The per-node $r_v$ is explicitly *not* a probabilistic certificate (the authors' own Remark), so it does not substitute for smoothing.
- The SVD direction is "maximally sensitive" for equilibrium shift, but Cls-PGD beats Shift-PGD on classification flips — so it's not the best attack for the metric a practitioner actually cares about.

The unification is intellectually pleasing but operationally a Swiss-army-knife problem: one tool, three jobs, none done as well as the dedicated tool.

This counter-argument can be answered. The authors could:
1. Re-state the contribution as "a unified analytical *framework* for diagnostic vulnerability mapping" rather than "a unified replacement for three tools";
2. Show empirically that the rankings transfer from matrix-free to dense path at the same scale (this should be cheap);
3. Acknowledge in the abstract that each output is a *complementary*, not a *replacement*, diagnostic.

---

## Issue list

### CRITICAL

**C1. Per-edge ranking transfer breaks at operationally relevant scale.**
- **Where.** §Cross-domain explicitly: "on Cora ($N=2{,}708$), 50-node rankings correlate weakly with full-graph rankings ($\tau=0.16$)"; §Conclusion limitation (3). All §Cross-domain through §Defense-ablation results are run on 50-node BFS subgraphs.
- **Why critical.** The unification claim's headline output (per-edge ranking) breaks where the framework would actually be deployed. The authors flag this, but the flagging does not undo the headline.
- **Effect on decision.** Per Checkpoint Rule #4, this is a CRITICAL issue and the editorial decision cannot be Accept on the current presentation. The fix is achievable: run the matrix-free path on full Cora, report agreement with dense $N=200$ rankings (R1 W2). If the matrix-free path gives stable rankings at full scale, the criticism dissolves. If it does not, the per-edge ranking claim must be restricted.

**C2. "Tightness ≥ 1" is reframed as virtue but is a bound failure.**
- **Where.** §Cross-domain commentary on Table tightness_eps: "Tightness $\geq 1$ means the linearisation underestimates damage, the safe direction for a diagnostic."
- **Why critical.** A first-order envelope that under-predicts by 36% (Cora, ε=0.20) is not a diagnostic *for* the perturbation regimes the paper claims to characterize (subcritical → critical). The "safe direction" framing converts a numerical limitation into a rhetorical strength. This is the same fallacy as calling p > 0.05 "evidence of no effect."
- **Effect on decision.** Reframe semantics + restrict the headline tightness claim to small ε. Without this, the abstract's "1.00 ± 0.01 at ε=0.01" reads as if the bound is globally tight, which is false.

**C3. "Three-regime characterisation" is worst-case along one direction, not a generic phase transition.**
- **Where.** Theorem 1(b): "along worst-case directions (aligned with the top singular vector of $\hat A$), $\|(I-J_z')^{-1}\|_2$ diverges as $\Omega(1/(\varepsilon_\text{crit} - \varepsilon))$; generic directions diverge more slowly."
- **Why critical.** The abstract advertises "a three-regime characterisation" as a theoretical contribution; the regime structure is conditional on direction. There is no empirical demonstration of the three regimes (subcritical, critical, supercritical) end-to-end.
- **Effect on decision.** Either qualify the abstract language ("worst-case three-regime characterisation along the leading sensitivity direction"), or add an empirical phase-transition figure (the repository has `exp_phase_transition.py`).

### MAJOR

**M1. PRBCD absent.** Current SOTA structural attack at scale; cited only as the smaller GR-BCD variant. The "scales to $N = 7{,}650$" claim has no head-to-head SOTA-attack comparison at that scale. (R1 W3, R2 W5)

**M2. Sober-look defense literature absent.** Mujkanovic 2022 + Gosch 2024 challenge prior defense-evaluation methodology. The §Defense-ablation reports +42% / +61% reduction in attack success via vulnerability-guided masking — this number must be checked against the sober-look protocol or the comparison risks the same fragility those papers exposed.

**M3. "Formal track" applies only to a deliberately weakened model.** IGNN under spectral-norm constraint loses ~6% accuracy on Cora (77.5% vs ~83%). Practitioners deploying high-accuracy explicit GNNs receive only the computational tool, *not* the regime guarantees. The abstract's "for contractive implicit GNNs we additionally derive a critical perturbation budget" understates this trade-off.

**M4. The SVD direction is not the strongest attack on the metric practitioners care about.** Table attack_full shows Cls-PGD beats Shift-PGD on Cora (2.51 vs 3.05 equilibrium shift) but on **classification flips** Cls-PGD often wins (the text concedes this and renames "optimal attack" → "maximally sensitive direction"). Honest, but the abstract still leads with "globally maximally sensitive perturbation direction" without specifying for which loss.

**M5. AGNNCert "10.2× looser" framing.** AGNNCert provides sound certificates; AEGIS $r_v$ provides first-order thresholds that the authors' own Remark says can be violated. The Table baselines comparison numerically dominates AGNNCert without flagging the semantic gap in the headline cell. (R1 W6)

**M6. Cross-domain case-study claim is conditional on a model that gets case118 θ wrong by ~4°.** Vulnerability rankings depend on the GNN's learned PF; the GNN's $\theta$ RMSE = 0.076 p.u. is not operationally accurate. The τ = +0.62 reflects the model's view, not the grid's. (R3 W4)

**M7. case300 failure indicates the framework does not reach operational scale.** Real grids are 1000s of buses; case300 is the first benchmark in that range and the GNN does not converge. The "scalability stress-test only" framing softens this but the framework's value proposition (efficiency over brute-force at scale) is unevidenced at scale. (R3 W3)

### MINOR

**m1. Tightness vs breach are on different quantities** (equilibrium shift vs prediction flip); the abstract bundles them implicitly. (R1 W4)

**m2. Mean vs median breach reporting.** Median is used because "3/10 seeds at 0%". State the full breach distribution. (R1 W7)

**m3. Abstract's τ range $[-0.28, +0.89]$ includes a number recovered to ~+0.03 at full-graph.** The lower bound is from a 50-node subgraph that doesn't transfer. The abstract should report the full-graph cell.

**m4. "Mining graph structure" in the title is never operationalized.** Title verb mismatch. (EIC §5(b))

---

## Ignored alternative explanations / paths

**A1. Spectral-graph-theoretic baseline.** Why is $S_c$ better than ranking edges by their contribution to spectral gap of $\hat A$ (i.e., $|\lambda_2(\hat A) - \lambda_2(\hat A - \delta A)|$)? The "spectral baseline" mentioned in §Cross-domain ("+6–148% AtkAdv over degree-proportional, edge-betweenness, and spectral baselines") is reported as a number but the spectral baseline definition is not given in detail. A reader cannot tell if the comparison engages with the strongest spectral attack.

**A2. The IFT-resolvent for explicit GNNs via fixed-point reparameterization.** The unrolled $S_K$ for explicit GNNs is computed via finite-difference forward passes. An alternative — treat the explicit GNN as the truncation of an implicit fixed-point and apply the IFT-resolvent argument with truncation error — would unify the explicit and implicit treatments under the same machinery. This is not pursued.

**A3. Per-node radius as a tail-risk distribution.** $r_v$ is a scalar threshold. An alternative — estimate the distribution of perturbations that flip node $v$'s prediction (a Bayesian or empirical posterior) — would give practitioners a richer object than a single threshold. The "first-order vs probabilistic" Remark draws the contrast but does not pursue the middle ground.

---

## Missing stakeholder perspectives

**P1. The defender deploying the framework.** What does a defender *do* with a per-edge vulnerability ranking? §Defense-ablation suggests "edge protection masking" — but in production, edges aren't protectable individually; what's the operational defense action? An end-to-end defender workflow would clarify the contribution.

**P2. The financial-graph practitioner.** The abstract names "financial graphs" as one of three safety-critical domains, but no financial-graph case study or even a financial-graph dataset appears. Cora / Citeseer / Pubmed / WikiCS are not financial graphs. The motivating domain in the lead sentence is unsupported.

**P3. The regulator.** "Tiered access" + NERC CIP references invoke a regulator-facing posture. But the framework's outputs (per-edge ranking, attack direction) are operational, not regulatory artifacts. A regulator does not consume $S_c$; they consume an audit report. The dual-use disclosure is well-handled but the regulator's stake is mentioned without being addressed.

---

## "So what?" test

If this paper were rejected today, what would the field lose?

**It would lose:** a clean operational name ($S_c$) for an object many GNN-robustness researchers compute informally; a credible matrix-free implementation that scales to $N \approx 7{,}650$; a useful illustration that the same Banach-style IFT machinery surfaces in adversarial GNN analysis and (loosely) in power-systems N-1.

**It would not lose:** a new attack (the SVD direction is dominated by Cls-PGD on classification metric); a new certificate (radii are not certificates); a new defense (defense-informed masking is a follow-up, not the contribution); operator-grade contingency screening (LODF still wins on case14–30 and on speed everywhere).

The contribution is **real but modest**: an analytical lens + a scalable pipeline, packaged with an honest empirical study. The presentation overclaims relative to the contribution. With the overclaiming softened (C1, C2, C3 addressed), this is a defensible paper.

---

## Observations (non-defects)

- The 10-page budget discipline is real. The authors visibly compressed.
- The limitations section is one of the better ones I have seen in this venue band; the issues I raise are mostly issues of *framing*, not of unflagged defects.
- The ethical-disclosure section (90-day notification, tiered access, candor about post-publication non-control) is admirable.
- The finite-difference sanity check (τ=0.999 for $S_c$ column norms) is the right kind of paranoia.

---

## Recommendation

**Major Revision.** Three CRITICAL issues (C1, C2, C3) prevent acceptance on current presentation. Each is addressable within the 10-page budget without restructuring. The Major issues (M1–M7) are individually tractable, though the editor will need to pick which subset to require; my own priorities are M1 (PRBCD), M3 (formal-track applies to weakened model), and M5 (AGNNCert framing).

Per Checkpoint Rule #4, the presence of any CRITICAL precludes Accept.
