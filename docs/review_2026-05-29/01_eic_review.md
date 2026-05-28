# EIC Review — AEGIS

**Role.** Editor-in-Chief / area editor at an IEEE venue spanning ML security + graph mining.
**Mode.** Independent, paper-blind to other reviewers' reports.
**Page-budget calibration.** 10-page IEEEtran conference cap is binding; treats it as a constraint, not a defect.

---

## 1. One-sentence contribution test

The author's stated contribution: *"a single closed-form computation that produces a maximally sensitive perturbation direction, per-edge vulnerability rankings, and per-node first-order sensitivity radii from the constrained sensitivity matrix $S_c$, for any GNN with continuous edge-weight-modulated message passing."*

This is statable in one sentence, falsifiable, and concrete. It survives the test. It is **not** the same as "we prove a theorem about GNN robustness" — the contribution is a *unification of three outputs*, which is a different (and more application-oriented) framing.

## 2. Venue fit

The IEEEtran conference format + tiered-access disclosure language + "safety-critical settings" framing places this naturally in:
- **IEEE TIFS / S&P-affiliated**: the threat-model framing and tiered code release fit; the formal track (Theorem 1) is mathematically substantive.
- **ICDM (IEEE) / IEEE Big Data**: the graph mining angle and 9-dataset × 7-architecture matrix fit.
- **IEEE SaTML**: the adversarial-ML focus fits cleanly.

**Less natural for**: power-systems venues (IEEE PES Transactions / PSCC), because the case study is explicitly proof-of-concept and underperforms LODF on operational metrics by the authors' own admission.

If the target is IEEE TIFS / SaTML, the framing works. If the target is IEEE PES, the paper undersells.

## 3. Originality

There are three originality claims to evaluate:

| Claim | My read |
|---|---|
| Unifying three outputs from one matrix | **Plausible** — I am not aware of prior work that produces all three (direction, ranking, radius) from one IFT-resolvent computation. The related-work positioning (four threads, each producing at most one of the three) is fair. |
| The constrained projection $P_c$ ($N^2 \to |E|$) is novel | **Modest** — symmetrization of the edge-pair Jacobian is a standard construction in matrix calculus; the novelty is in pairing it with the IFT-resolvent and naming it $S_c$. The authors' description in §Theory is honest about this ("the ingredients of Theorem 1 are classical … the contribution is the constrained projection $S_c$ that turns a vacuous bound … into a tight one"). |
| The critical budget $\varepsilon_\text{crit} = (1-\kappa)/\|W\|_2$ is a new formal result | **Modest, framing-dependent** — this is a direct consequence of the contraction mapping theorem + operator norm. The framing as an "$\varepsilon_\text{crit}$" specific to structural GNN perturbations is the contribution; the math is classical. |

Net: there is real originality, but the abstract's "we additionally derive a critical perturbation budget" overstates the novelty of $\varepsilon_\text{crit}$ relative to its derivation. The unification claim is the strongest originality lever; lean on it more.

## 4. Significance & readership

Audience reach is broad — adversarial-ML, GNN-theory, and (proof-of-concept) power-systems readers all get something. But broad reach in 10 pages produces shallow treatment per audience:

- Adversarial-ML reader: where is PRBCD? Gosch 2024? Mujkanovic 2022?
- GNN-theory reader: how does $S_c$ relate to Lipschitz-GNN sensitivity (Gama 2020 cited but not contrasted)?
- Power-systems reader: case300 fails; case14–118 is below operationally relevant scale.

In a 10-page budget, the EIC's question is: *which audience pays the rent?* My recommendation is to commit to the adversarial-ML / GNN-theory dual reader and demote the power-grid case study from "cross-domain demonstration" to "structural plausibility check" (one paragraph + one figure) — this would free space for the missing PRBCD / Mujkanovic comparisons that R1 and DA will demand.

## 5. Editorial concerns

**(a) Scope-vs-depth tension.** 9 datasets × 7 architectures × 4 domains × IEEE 14–118 = many cells, each shallowly defended. The decision to keep them all is brave; whether reviewers tolerate this depends on whether the framing makes the breadth a *strength* (universality of $S_c$) or a *liability* (no single comparison fully defended).

**(b) Title vs body alignment.** Title says "Mining Graph Structure for Adversarial Vulnerability Analysis." Body delivers vulnerability analysis but the "mining" language is not used in the body — verb mismatch is mild but worth fixing.

**(c) Abstract overload.** The abstract packs 7 numerical claims into 200 words. This is dense but defensible if each is exactly defended in the body. **Spot-check fails:** the abstract says "$\tau \in [-0.28, +0.89]$" for the continuous-to-discrete transfer; the lower bound (−0.28) is on a *50-node subgraph* of Amazon Photo / IGNN. The body acknowledges this recovers to ~+0.03 on the full graph. The abstract should clarify, or report the full-graph value as the headline.

**(d) Dual-use disclosure language.** The 90-day stakeholder notification + tiered code release is professionally handled. The authors' candor that "tiered access is not a strong technical control after publication" is the right tone. EIC nods.

**(e) 10-page binding.** With every column used, any reviewer demand for "add a section on X" must be matched by a "remove section on Y" recommendation. I will hold R1–R3 + DA to this.

## 6. Recommendation (EIC, paper-blind)

**Major Revision.** The paper's contribution is real and clearly stated, the empirical breadth is impressive given the budget, and the limitations are visibly disclosed. But three issues need substantive response, not editorial caveats:

1. **Tightness semantics** (R1's lane): "tightness ≥ 1" is presented as a virtue but is a bound failure — clarify or rename.
2. **Subgraph τ=0.16 on Cora** (R1 + DA's lane): the framework's per-edge ranking output, the headline of the unification claim, breaks at the scale where graphs are interesting. Either restrict the per-edge ranking claim to graphs where it holds, or demonstrate that the full-graph matrix-free path preserves rankings (currently only τ=0.95 across BFS-centre choices is reported, which is a weaker statement).
3. **Missing SOTA baselines** (R1 + DA's lane): PRBCD and a sober-look defense reference (Mujkanovic 2022 / Gosch 2024) must be addressed even if only briefly, given the adversarial-ML positioning.

The paper is unlikely to be accepted as-is at a top IEEE venue. With the above tightened — within the 10-page budget — it is competitive.

## 7. Quality scores (preliminary, for synthesizer reference)

| Dimension | Score (0–100) | Note |
|---|---|---|
| Significance | 72 | Real practitioner gap; broad applicability |
| Originality | 64 | Unification is novel; pieces are classical |
| Technical rigor | 68 | Theorem statement careful; tightness framing wobbles |
| Clarity | 70 | Dense but readable; abstract overclaims by 1 cell |
| Empirical depth | 65 | Broad but each cell shallow; PRBCD absent |
| Reproducibility | 78 | 10 seeds throughout; tiered code release plan stated |
| Ethics / disclosure | 82 | 90-day notification + tiered access; honest about controls |
| **Overall (EIC dimension)** | **70** | Borderline accept → Major Revision |
