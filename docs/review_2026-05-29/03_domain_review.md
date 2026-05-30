# Reviewer 3 — Domain (GNN adversarial robustness)

## Summary

AEGIS is positioned against four threads: structural attackers (Nettack, Mettack, PR-BCD), certified robustness (smoothing, IBP / AGNNCert), implicit networks (DEQ, IGNN), and influence-function sensitivity (Lorraine). The framework is the first to unify per-edge rankings + global SVD direction + per-node radii in a single closed-form computation. Empirical study spans 9 datasets including 3 heterophilic ones (Texas, Cornell, Wisconsin) — a deliberate stress test that the framework largely passes.

## Positioning against attackers

The four-quadrant framing (gradient-based × gradient-free, equilibrium-shift × classification-loss) is sensible but used asymmetrically: two of the four quadrants are AEGIS-internal (SVD-optimal closed-form vs. Shift-PGD IFT-gradient — same objective, different optimiser) and only Cls-PGD probes the classification quadrant.

**Concern D.1 (must address).** The head-to-head with GR-BCD shows $\tau = +0.69$ on Pubmed $k=10$ but $\tau = +0.16$ on Cora $k=5$. The Cora result is the dangerous number and is not explained. Two interpretations: (a) AEGIS's equilibrium-sensitivity ranking diverges from GR-BCD's classification-loss ranking on sparse graphs, consistent with §Discrete edge removal where $\tau$ with Cls-PGD is +0.19 to −0.06; (b) GR-BCD finds a different vulnerability mode that AEGIS misses. Add a per-dataset $k$-vs-$\tau$ curve and a one-paragraph diagnosis.

**Concern D.2 (must address).** PR-BCD~\cite{geisler2021robustness} is the strongest scalable baseline and is mentioned only in passing. Run PR-BCD on at least Pubmed and Amazon Photo at the same budget; report $\tau$ to AEGIS rankings.

## Ranking semantics

$v_{ij} = \|[S_c]_{:,k}\|_2$ ranks edges by their effect on the equilibrium $z^*$, not on the final classification. The current abstract reads as if $v_{ij}$ is a classification-vulnerability ranker.

**Concern D.3 (must address).** Rephrase the abstract claim "per-edge vulnerability rankings" to "per-edge equilibrium-sensitivity rankings (which transfer to classification damage in 29/33 architecture–dataset cells)." This is a small wording fix but it prevents downstream confusion in citing work.

## AGNNCert comparison

The footnote in Table 1 ("AGNNCert is a sound IBP certificate; AEGIS $r_v$ is a first-order sensitivity threshold") is correct but buried.

**Concern D.4 (must address).** Move the sound-vs-first-order distinction into the abstract or introduction. Reviewers who skim Table 1 will read $r_v = 0.187$ vs $r_v = 1.414$ as a 7.6× loss to AGNNCert without context. The current Remark 3.4 (in §3) is good; cross-reference it from the table caption.

## Threat model — the insertion gap

The threat model restricts perturbations to edge-supported entries: $[\delta \hat A]_{ij} = 0$ for $(i,j) \notin E$. Continuous-to-discrete transfer (Prop 3.5) bridges to *removal* of existing edges.

**Concern D.5 (must address).** The graph-adversarial literature treats edge *insertion* as the dominant threat (Nettack inserts edges into a target node's neighborhood). AEGIS as stated does not address insertion. Two options:

1. Extend $S_c$'s edge basis from $E$ to a candidate set $\bar E \supseteq E$ (the full $N(N-1)/2$ for small graphs, a $k$-hop candidate set for large graphs). The computation remains matrix-free; $|E|$ in the cost table becomes $|\bar E|$.
2. State explicitly in §II (Threat model) and §VIII (Limitations) that AEGIS targets edge-deletion / edge-weight-perturbation attacks and does not address insertion. Currently the limitation paragraph mentions "binary adjacency masks" but not "no insertion."

## Defense ablation

The defense-informed-masking ablation (Cora IGNN, $N=50$, paired Wilcoxon $p < 0.002$) is solid in isolation: 42 ± 8 % damage reduction at $k=5$ vs 11 ± 6 % random. But the test is against a *non-adaptive* attacker who does not re-optimise after masking.

**Concern D.6 (must address).** Add an *adaptive-attacker* column: recompute $S_c$ on the masked graph and re-run the SVD attack. Either (a) the defense still beats random, in which case the headline strengthens, or (b) the adaptive attacker eats most of the gain, in which case scope the claim to "non-adaptive AEGIS attackers." Without this column the result is consistent with the "sober look" critique cited in §Related Work — exactly the trap the paper means to avoid.

## Subgraph evaluation

50-node BFS subgraphs are the default (tightness 1.013 ± 0.003, 66× faster than $N=200$). The 50-node-vs-full-graph Kendall $\tau$ on Cora is 0.16 (§Limitations).

**Concern D.7 (must address).** Subgraph evaluation is the dominant regime for citation graphs. $\tau = 0.16$ between subgraph and full-graph rankings means the subgraph rankings approximate the full-graph rankings *weakly*. Report at least the Amazon Photo full-graph $\tau$ (computable with the matrix-free path) and recommend a default subgraph size on citation-scale graphs.

## Heterophilic datasets

Texas / Cornell / Wisconsin results — $\tau$ comparable to homophilic (Cornell $+0.44$, Wisconsin $+0.21$). Good faith effort. IGNN's Texas $\tau = -0.13$ with high variance is the warning sign; GCN-4 dominates on all three. Reportage is honest.

## Decision lean

**Major Revision.** The framework is publishable but needs (a) PR-BCD head-to-head, (b) adaptive-attacker column for the defense ablation, (c) explicit insertion-attack scoping in the threat model, and (d) ranking-semantics precision in the abstract.
