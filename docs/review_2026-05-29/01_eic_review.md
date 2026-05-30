# Reviewer 1 — Editor-in-Chief

## Summary

AEGIS proposes a single object, the constrained sensitivity matrix $S_c$, that delivers (i) the SVD-optimal first-order adversarial direction, (ii) per-edge vulnerability rankings, and (iii) per-node first-order sensitivity radii from one matrix-free computation. The formal track (Theorem 3.1, $\varepsilon_{\rm crit}$, three regimes) covers contractive implicit GNNs; the same pipeline extends as a computational tool to seven explicit architectures via the unrolled Jacobian $S_K$. Empirical evaluation spans 9 datasets, 7 architectures, 4 domains (330 runs), with a power-flow case study on IEEE case14–118 reported as a proof-of-concept.

## Originality

Moderate–high. The IFT for fixed points of implicit networks is established (DEQ, IGNN, El Ghaoui). The fresh moves are: (a) targeting structural sensitivity $\partial Z/\partial A$ rather than input sensitivity $\partial Z/\partial x$, (b) the constrained projection $P_c$ that prunes non-edge entries and restores tightness ($\sigma_1(S_c)$ envelope ≈ 1.00 vs. the unconstrained $\sqrt{r}$ slack of 7–14×), and (c) the unification claim — one matrix delivers attack direction + edge rankings + node radii. The unification is the load-bearing novelty. Each ingredient in isolation is standard.

## Significance

The result is significant **if** the cross-architecture transfer holds in deployment. The deeper-is-better empirical pattern (GCN-4, APPNP, GAT$^\dagger$ ≫ GCN-2; 29/33 positive transfer cells) is publishable. The IGNN-only formal guarantees cost ~6% Cora accuracy, which means the "audited safety" track applies to a model that is not the strongest classifier. For practitioners who care most about deeper explicit GNNs, AEGIS delivers a computational tool but no closed-form regime guarantee.

## Relevance to readership

ICDM / IEEE BigData / IEEE TKDE readers will find the technical contribution well-matched. SaTML readers will demand a sharper threat model. NeurIPS/ICLR readers will press the case-study softness.

## Framing concerns (the principal EIC issue)

1. **Title overpromises.** "Adversarial Vulnerability Analysis of GNNs" suggests architecture-agnostic guarantees. The formal apparatus binds to a contractive subclass. Suggested title: *"Structural Sensitivity for Implicit GNNs, with Empirical Transfer to Explicit Architectures."*
2. **Abstract lead.** The first headline — "first-order envelope tight ($1.00 \pm 0.01$) at $\varepsilon=0.01$" — is the *weakest* claim: tightness at a magnitude where 0% of predictions flip. Lead instead with the unification, the cross-architecture transfer pattern, and the matrix-free scalability.
3. **Power-grid framing.** The case study is consistently labelled "proof-of-concept" in the conclusion and the abstract softens with "cross-domain demonstration." Good. The title still implicitly invites the operator-grade reading. Either commit to the case study with admittance-weighted edges + Grid2Op, or relegate it to a closing subsection.

## Decision lean

**Major Revision.** The technical content is publishable; the framing and a small number of theoretical / empirical gaps need to be tightened.

## Specific asks (for editorial decision intake)

- E-EIC-1 Tighten title and abstract to match the architecture scope of the formal results.
- E-EIC-2 State the unification claim as the primary contribution; relegate "first-order tightness at small $\varepsilon$" to a supporting role.
- E-EIC-3 Decide explicitly whether the power-grid case study is proof-of-concept (current framing, acceptable) or a contribution claim (needs more evidence). Currently it sits ambiguously.
