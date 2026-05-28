# Reviewer 3 — Perspective Review (cross-disciplinary)

**Role.** Power-systems engineer with strong CS background; PSCC / IEEE TPWRS / TSG; PandaPower + Grid2Op user; familiar with LODF, PTDF, ACOPF, contingency screening, NERC CIP, EU NIS2.
**Mode.** Independent.
**Page-budget calibration.** 10-page IEEEtran cap binding. The case study is one of nine sections, so this review evaluates the cross-domain claim within that budget reality.

---

## 1. The cross-disciplinary positioning

The paper makes a *cross-domain* claim: a vulnerability framework developed for GNN adversarial analysis (Cora / Citeseer / Pubmed / Amazon Photo / WikiCS) also surfaces structurally meaningful information on AC power-flow benchmarks (IEEE 14 / 30 / 57 / 118). This is the "proof-of-concept cross-domain demonstration" of §Case Study. The §Conclusion limitation list correctly states this is not operator-grade screening.

A cross-domain claim from an ML paper into power systems is a strong move. It is also a load-bearing piece of the paper's universality narrative: if $S_c$ is just for citation networks, the paper is narrower than the abstract suggests.

## 2. Strengths

**S1. The case study is honestly scoped.** "Proof-of-concept", "not operator-grade", "operator-grade deployment requires admittance-weighted edges, broader operating coverage, and thermal/voltage-violation severity rather than the $\ell_2$ target used here" — this is the right disclosure language. I have seen many ML-into-power-systems papers that did not say this.

**S2. case300 is included as a stress-test only.** The authors do not hide that case300's GNN does not converge to physical PF physics ($\theta$ RMSE 0.394 p.u. ≈ 22.6°). Reporting this transparently rather than silently dropping case300 is creditable.

**S3. The LODF comparison is the right baseline.** LODF (Line Outage Distribution Factor) is the textbook DC-PF linear sensitivity used in industry for fast N-1 screening; it is precisely the contrast every ML-for-PF paper must engage with. Bringing LODF onto Table baselines and acknowledging it runs <0.13 s vs AEGIS 2–23 s is the right kind of honesty.

**S4. The 90-day notification to PandaPower / Grid2Op maintainers and NERC / ENTSO-E working groups is the right disclosure protocol.** I would not normally see this in an ML conference paper.

**S5. Wilcoxon $p < 0.01$ for AEGIS > LODF on case57/118 is statistical due-diligence.** Many cross-domain ML papers don't even compute it.

## 3. Weaknesses

### W1. **The "without line-impedance data" advantage is a false economy.** [Major]

The paper claims AEGIS "[produces] a vulnerability spectrum, attack direction, and per-node thresholds without line-impedance data." In operational settings this advantage does not apply: line impedance is part of the static system data every transmission operator maintains. The case study's framing ("does not require impedance data") is a feature of the GNN learning curve, not an operational advantage.

Where this *might* matter:
- Distribution networks with incomplete impedance data (e.g., low-voltage feeders) — but those are not the IEEE benchmarks used here.
- Synthetic / sandboxed grids for ML research — but operators don't deploy from those.

**Concretely:** soften "without line-impedance data" to "from learned representations, without explicit impedance parameters" and acknowledge that this is a research convenience, not an operational advantage.

**Fix fits the budget:** ≤ 1 sentence rewording.

### W2. **AEGIS's improvement over LODF (τ +0.62 vs +0.44–0.58 on case57/118) is incremental and the runtime trade-off is unfavorable.** [Major]

LODF runs in <0.13 s from static reactances. AEGIS requires GNN training (the dominant cost; "2–23 s" includes training) plus $S_c$ computation. For operators doing N-1 screening at 15-minute SCADA cycles, AEGIS's wall-clock budget is fine. But the *information yielded* is comparable to LODF for case57/118:

| Method | τ vs N-1 | Runtime | Engineering inputs |
|---|---|---|---|
| LODF (DC linear sensitivity) | 0.44–0.58 | <0.13 s | Line reactances (operator has these) |
| AEGIS (per-edge $v_{ij}$) | 0.62–0.67 | 2–23 s | Trained GNN, $S_c$ pipeline |
| Brute-force N-1 | 1.0 (ground truth) | 0.1–2 s | Topology + power-flow solver |

The brute-force baseline is 0.1–2 s — *faster than AEGIS for these grids*. The case for AEGIS over brute force at case14–118 scale is unclear from the operational metric.

The likely real-world case is *larger* grids where brute-force N-1 becomes expensive. But case300 fails — so the framework has not been demonstrated at the scale where its computational advantage would actually appear.

**Concretely:** position the case study as a *qualitative correspondence* demonstration ("structural isomorphism between $S_c$ vulnerability and N-1 severity, validated on small benchmarks") rather than an operational tool comparison. The current framing invites the question "why use this over brute-force N-1?" which the paper cannot answer with case14–118.

**Fix fits the budget:** Reposition the case-study Conclusion paragraph.

### W3. **case300 failure is a structural gap, not just a footnote.** [Major]

The framework's value proposition (computational efficiency, "without line-impedance data") only matters at the scale where brute-force N-1 is impractical. case300 is the first benchmark in the typical operational range; case1354 / case2848 / case6470 are the actual North American & European interconnection sizes. The GNN does not converge on case300, and the paper presents this as a scalability stress-test rather than as evidence that the framework does not yet reach the operational regime.

The matrix-free pipeline is validated to $N = 7{,}650$ (Amazon Photo) — but Amazon Photo is a product co-purchase graph, not a power grid. The GNN architecture (contractive IGNN with spectral-normalized W and 64-dim hidden layer + 2-head readout) is too small to learn case300 power-flow physics. This is not a $S_c$ limitation; it is a fitness-for-PF limitation that contaminates the case study's claims at scale.

**Concretely:** add one sentence clarifying that the matrix-free $S_c$ pipeline could in principle scale to larger grids, but the bottleneck is GNN learning capacity for PF physics at those scales, not $S_c$ computation. This separates the framework's scalability (genuine to $N \approx 7{,}650$) from the case study's scalability (limited to case14–118).

**Fix fits the budget:** ≤ 2 sentences.

### W4. **$|V|$ and $\theta$ RMSE on case118 are large.** [Moderate]

Case118: $|V|$ RMSE = 0.014 p.u., $\theta$ RMSE = 0.076 p.u. ≈ 4.4 degrees. For an AC PF surrogate, 4.4° angle error is large — it would not be deployable as a state-estimation aid. The paper does not claim it would be, but the contingency-ranking correlation (τ = 0.62) is computed against brute-force N-1 *on the GNN's learned representation* — so the rankings are conditioned on a model that gets θ wrong by 4° at case118.

This is consistent with the paper's framing ("vulnerability rankings are conditioned on the GNN model's learned representation, not on physical grid parameters"), but raises a concern: the τ = +0.62 may reflect what a moderately accurate GNN-PF surrogate *thinks* about contingencies, not what *actually* threatens the grid.

**Concretely:** add a sentence in §Case Study acknowledging that the τ value is conditional on the GNN's learned PF; an exactly-solved PF would yield a different (likely higher) τ since N-1 is computed via PandaPower's exact AC solver. This is implicit but should be explicit.

**Fix fits the budget:** ≤ 1 sentence.

### W5. **NERC CIP-005 / CIP-007 mention is decorative.** [Moderate]

The Conclusion's dual-use section references NERC CIP-005 and CIP-007 (cyber security perimeter, system security management). These are real regulatory references but they don't actually constrain the threat model in the paper: the paper's threat model is white-box $\ell_2$-bounded edge perturbations, which doesn't map to the SCADA-network attack surface that CIP-005/007 address.

Where the framing *would* be coherent: an attacker who has compromised the SCADA network and can inject false topology data into the state estimator. Then the GNN sees an adversarial graph and AEGIS's vulnerability rankings tell the defender which edges the false-data attack would target. This is the right framing, and the paper hints at it but doesn't pull it through.

**Concretely:** either (a) develop the SCADA / false-data-injection framing in 2–3 sentences (CIP-005 = network perimeter compromise + CIP-007 = endpoint hardening → adversary in graph data → AEGIS-style vulnerability map); or (b) drop the NERC references as they currently do not bind anything.

**Fix fits the budget:** Either swap.

### W6. **Per-unit RMSE shown with std < 0.003 — too tight to be plausible at case118.** [Minor / verify]

"Std < 0.003 for all entries" in Table ieee caption. For case118 with θ RMSE mean 0.076, a std < 0.003 across 10 seeds suggests very stable training. Verify: is this 10 seeds *of the same training run*, or 10 *retrainings* of the GNN? The latter is what the rest of the paper implies. If retrainings, a 0.003 std on a 0.076 mean is unusually tight; readers will want to know the random-seed control protocol.

**Concretely:** clarify in caption whether seeds vary across training initialization, data ordering, or both.

### W7. **Binary adjacency vs admittance-weighted advantage is interesting but counterintuitive for power systems readers.** [Minor]

"Binary adjacency outperforms admittance-weighted (P@10 = 0.81 vs 0.27 on case118)." This is striking — a power systems reader expects admittance-weighting (which encodes physical impedance) to dominate, since N-1 severity is governed by physical flows. The "all-or-nothing character of a trip" explanation is plausible but understated: this is a substantive finding worth emphasizing. A trip removes the line entirely, so a binary mask correctly models the topology change while admittance weighting injects irrelevant magnitude information.

**Concretely:** make this an explicit sub-finding ("topological discontinuity dominates magnitude in N-1 severity") rather than a parenthetical. This is actually one of the more *interesting* cross-domain insights and is currently buried.

**Fix fits the budget:** Reorder sentences.

## 4. The cross-disciplinary value

Setting aside the operational-tool framing: does this paper contribute something to the cross-disciplinary conversation between GNN robustness and power-systems contingency analysis?

**Yes, modestly.** The structural isomorphism — *edge adjacency perturbation in a GNN's IFT sensitivity ↔ line outage in N-1 severity ranking* — is a real conceptual link. Power systems researchers will recognize that the GNN's $J_z$ at equilibrium is doing something analogous to the linearized PF Jacobian, and the resolvent $(I-J_z)^{-1}$ to the post-contingency steady state. Saying this explicitly would help.

The paper does not say it explicitly. The Related Work cites Donon 2019 (GNN-PF) and Nakiganda 2023 (GNN contingency screening) but does not draw the analogy between $J_z^{-1}$ and the post-contingency Jacobian. Doing so would lift the case study from "we ran it on power grids" to "the same Banach-style inverse argument links GNN equilibrium sensitivity and DC-PF linear sensitivity."

**Concretely:** add 1–2 sentences in §Case Study identifying the conceptual analogy. This would make the cross-disciplinary contribution real.

## 5. Recommendation

**Major Revision** at the case-study level (W1–W3 are substantive; W4–W7 are tractable). At the main-paper level, the cross-domain demonstration is the weakest part of the paper as currently framed; the easiest path to a stronger paper is to **shrink** the case study (2 pages → 1 page) and use the freed space for PRBCD comparison and a phase-transition figure (R1's W3, R2's W2). The case study would lose nothing important: its current value is qualitative correspondence + structural-isomorphism insight, both of which fit in a compressed presentation.

## 6. Scores

| Dimension | Score (0–100) |
|---|---|
| Cross-disciplinary positioning | 64 |
| Operational realism | 52 |
| Honesty of scope | 80 |
| Insight transfer | 65 |
| **Perspective overall** | **65** |
