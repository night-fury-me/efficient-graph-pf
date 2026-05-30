# Phase 2 — Editorial Synthesis & Decision

## Aggregate verdict

All five reviewers converge on **Major Revision**. Devil's Advocate qualifies the verdict with *"leaning Reject if Critical Issue #1 is not addressed."* The recommended decision is therefore **Major Revision**, with the empirical demonstration of the phase-transition theorem treated as a *gating* requirement.

## Consensus issues (≥3 reviewers agree)

| ID | Issue | EIC | R1 | R2 | R3 | DA |
|----|-------|:--:|:--:|:--:|:--:|:--:|
| C1 | **Title / abstract overreach** vs scope of formal guarantees | ✓ | – | – | – | ✓ |
| C2 | **Phase-transition theorem never empirically crossed** (all experiments are subcritical) | – | implicit | – | – | ✓ |
| C3 | **Theorem 3.1(b) tightness for non-normal $J'_z$** + (c) regime overclaim | – | ✓ | – | – | ✓ |
| C4 | **Insertion attacks not in threat model** | – | – | ✓ | – | ✓ |
| C5 | **Defense ablation lacks adaptive attacker** | – | – | ✓ | – | ✓ |
| C6 | **Subgraph τ=0.16 dominates citation-scale evaluation; need full-graph τ** | – | ✓ | ✓ | – | – |
| C7 | **Ranking semantics: equilibrium-shift vs classification** | – | ✓ | ✓ | – | – |
| C8 | **Case study: correspondence vs derivation; binary-vs-admittance; AC/DC ground truth** | – | – | – | ✓ | – |
| C9 | **"Tightness 1.00" headline is at non-adversarial ε** | – | ✓ | – | – | ✓ |
| C10 | **Cross-architecture transfer needs sign test; deep-vs-shallow caveat lead** | – | ✓ | ✓ | – | ✓ |

## Disagreement / arbitration

**R1 vs Devil's Advocate on theorem severity.** R1 reads Theorem 3.1(b) as fixable with a one-paragraph clarification (normal-vs-non-normal lower bound, activation-pattern stability). Devil's Advocate reads it as load-bearing and demands an experimental cross of $\varepsilon_{\rm crit}$. **Arbitration:** both are right. The proof itself is recoverable as R1 describes; the empirical anchor Devil's Advocate demands is non-negotiable for a paper titled around a phase-transition characterisation. The author must do both.

**R3 on case-study framing vs EIC's "current framing is acceptable as proof-of-concept."** EIC accepts the disclaimed framing; R3 demands tighter treatment of (a) correspondence vs derivation, (b) operating envelope, (c) AC-vs-DC ground truth. **Arbitration:** EIC's "acceptable" presupposes R3's corrections are made. The case study can remain in the paper as proof-of-concept, but the four specific concerns (P.1–P.4) must be addressed in the body, and the abstract / conclusion qualifiers strengthened.

**No reviewer recommends Accept-as-is. No reviewer recommends Reject outright** (Devil's Advocate's *lean* Reject is conditional on Critical Issue #1).

## Decision

**Major Revision.**

### Gating revision items (must address — sufficient to clear gate)

| Gate | Description | Source | Estimated work |
|------|-------------|--------|----------------|
| G1 | **Run at least one $\varepsilon > \varepsilon_{\rm crit}$ experiment**; report breach / divergence behavior. Show the regime characterisation either empirically or as scope-limited. | DA C2 | 1–2 days compute + 1 paragraph + 1 figure |
| G2 | **Restate Theorem 3.1(b)** with the operator-norm-only lower bound (no claim of $\Omega(1/(\varepsilon_{\rm crit}-\varepsilon))$ rate beyond what the pseudospectral index $\eta$ permits). **Restate (c)** defensively. **Add the activation-pattern-stability paragraph** to (a). | R1 A.1–A.3 | 1 day writing |
| G3 | **Fix Prop. 3.5 path-Lipschitz proof** ($L_J \le \|W\|_2^2$ requires a path-crossing argument). | R1 A.4 | half day |
| G4 | **Title + abstract scope alignment.** Either rename to scope the formal guarantees, or restructure the abstract to make the implicit/explicit split unmistakable. Move the AGNNCert sound-vs-first-order distinction into the abstract or §I. | EIC, DA, R2 | 1 day writing |
| G5 | **Insertion-attack scoping** in §II Threat Model and §Limitations. Either extend $S_c$ to a candidate-insertion set or state the gap clearly. | R2 D.5, DA | half day |
| G6 | **Adaptive attacker column** in defense ablation. | R2 D.6 | 1–2 days compute |
| G7 | **PR-BCD head-to-head** on Pubmed + Amazon Photo. | R2 D.2 | 1–2 days compute |
| G8 | **Full-graph τ** for at least Amazon Photo. | R1 A.7, R2 D.7 | 1 day compute |
| G9 | **Case study corrections:** (i) restate isomorphism as empirical correlation; (ii) AC/DC ground-truth metric in table caption; (iii) add PTDF baseline; (iv) report admittance-normalisation details. | R3 P.1–P.4 | 1–2 days |
| G10 | **Statistical anchor** for 29/33 positive-transfer claim (sign test or one-sided Wilcoxon). | R1 A.6 | 1 hour |

### Recommended but non-gating

- Operational-utility paragraph for the case study (R3 P.5).
- Tighten "2–8× over random" framing in abstract to also report PGD ratio (DA).
- Re-label Observation 3.3 general-ReLU case as Empirical Remark (R1 A.5).
- Algorithm 1 step-9 conditional / dense-only annotation (R1).
- Ethics framing as "best-practice notification" rather than capability-driven (DA).

## Estimated total revision cost

- Compute: 5–8 days (G1, G6, G7, G8 dominate).
- Writing: 4–6 days across §I, §III, §V, §VI, §VII, §VIII.
- Total: ~2 weeks of focused effort for one author.

## Re-review trigger

Re-submit Major Revision when G1–G10 are addressed; one of the original reviewers (preferably R1 or Devil's Advocate) re-reviews against G1–G10 specifically; the other reviewers verify their concerns are addressed via point-by-point response.

## Decision letter (draft, for author)

> Thank you for submitting *AEGIS: Structural Sensitivity for Adversarial Vulnerability Analysis of GNNs*. The reviewing committee finds the unification claim — that one matrix-free computation yields per-edge rankings, the SVD-optimal attack direction, and per-node sensitivity radii — to be a substantive contribution. The 330-run empirical study is thorough, and the ethics treatment is responsible.
>
> The committee unanimously requests **Major Revision** with ten gating items (G1–G10 above). The principal concerns are: (i) the phase-transition theorem is the headline theoretical claim but every reported experiment is in the subcritical regime; (ii) the title and abstract scope the formal guarantees more broadly than the IGNN-class proof supports; (iii) the threat model excludes edge insertion, which the related-work positioning does not make clear; (iv) the defense ablation needs an adaptive attacker, and (v) the case study requires tighter treatment of the AC/DC ground truth and the binary-vs-admittance discussion.
>
> A revised manuscript that addresses G1–G10 will be re-reviewed. We estimate ~2 weeks of focused effort.
