# Reviewer 5 — Devil's Advocate

I read this paper as a hostile referee. The construction is competent, the empirical study is broad, and the framing has been polished through prior rounds. But there are four genuine problems beneath the surface and several smaller framing slips that I want to name before they propagate into the citation graph.

## Critical issue #1 — The phase-transition theorem is empirically untested

Theorem 3.1 characterises three regimes — subcritical ($\varepsilon < \varepsilon_{\rm crit}$), critical ($\varepsilon \to \varepsilon_{\rm crit}$), supercritical ($\varepsilon > \varepsilon_{\rm crit}$). The headline experimental budgets are $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$. Computed values of $\varepsilon_{\rm crit}$ from $\kappa = 0.14$–$0.59$ and $\|W\|_2 \le c \in [0.5, 0.95]$ give $\varepsilon_{\rm crit}$ in the rough range $0.4$–$1.8$. **Every experimental point is subcritical.** The "phase transition" is never crossed. Regime (b) is never empirically observed; regime (c) is never empirically observed. The theorem is a statement about a regime nobody visits.

This is a structural problem, not a wording problem. The phase-transition framing is the load-bearing theoretical claim — it is what distinguishes AEGIS from "yet another sensitivity-based attacker." If the regime characterisation is never observed, then the formal track collapses to part (a): a first-order shift bound. Part (a) is fine but is a one-line consequence of IFT + Neumann — not novel.

Required: run experiments at $\varepsilon = 1.5 \cdot \varepsilon_{\rm crit}$ and $\varepsilon = 0.9 \cdot \varepsilon_{\rm crit}$ for at least one dataset–architecture cell. Show what actually happens near the threshold. If divergence / oscillation are observed, the theorem becomes empirically grounded. If not, the regime characterisation is overclaimed and should be downgraded to "the contraction certificate fails for $\varepsilon \ge \varepsilon_{\rm crit}$."

## Critical issue #2 — "Tightness 1.00" at a regime where nothing flips

Abstract headline: "first-order envelope tight ($1.00 \pm 0.01$) at $\varepsilon = 0.01$." Table on breach rate: at $\varepsilon = 0.01$, predictions flip in 0.0 % – 0.6 % of nodes. The "tight" regime is the regime where the perturbation is too small to matter. At $\varepsilon = 0.20$, where breach rates reach 27 % (Pubmed), the envelope ratio is 1.38 — i.e., the linearisation under-predicts by 38 %.

This is honest in the body of the paper (the tightness table reports all values). The framing problem is that the headline is at the *least* interesting $\varepsilon$. Quoting "1.00 ± 0.01 at $\varepsilon = 0.01$" in the abstract is true but selective. The adversarial-ML reader should see "envelope is tight where attacks fail; loose where attacks succeed." Reframe.

## Critical issue #3 — Insertion attacks are not in the threat model

The threat model restricts $\delta \hat A$ to the edge-supported subspace ($[\delta \hat A]_{ij} = 0$ for $(i,j) \notin E$). The continuous-to-discrete transfer bridges to edge *removal*. Edge *insertion* — the dominant threat in Nettack/Mettack — is outside the model. The paper does not say this clearly. A reader scanning the related-work section and the abstract will assume AEGIS competes with Nettack on its home turf. It does not.

Required: name the gap in the threat model. State explicitly that AEGIS targets edge-deletion / edge-weight-perturbation and does not (yet) address insertion. Either extend the formalism to a candidate-insertion set $\bar E$ or scope the claim. The current "no new edges" parenthetical in §II is a flag, not a clear delimitation.

## Critical issue #4 — Title scope ≠ formal scope

The title says "Adversarial Vulnerability Analysis of GNNs." The formal track (Theorem 3.1, $\varepsilon_{\rm crit}$, three regimes) applies to a contractive implicit GNN subclass that costs ~6 % Cora accuracy relative to standard explicit GNNs. The explicit-GNN extension (§3.4 / Observation 3.6) is computational only — no $\varepsilon_{\rm crit}$, no regime characterisation, only an empirical tightness number.

This means: the paper has *two* contributions sitting under one title. (i) A formal apparatus for contractive implicit GNNs with degraded accuracy. (ii) A computational tool that empirically transfers to standard GNNs but without the formal guarantees. A practitioner reading the title and abstract gets the impression that the formal guarantees apply across the board. They do not.

Required: either rename to *"Structural Sensitivity for Contractive Implicit GNNs"* (with the explicit extension as an empirical bonus), or restructure so the formal/empirical split is explicit at the abstract level. The current abstract does mention this ("the formal track applies only to this subclass") but the title does not.

## Smaller framing slips

- **"2–8× more damage than random"** — random in $N(N-1)/2$ dimensions is a vanishingly weak baseline. The relevant comparison is iterative attackers; Cls-PGD reaches 72–92 % of AEGIS's equilibrium-shift damage at modest extra compute. Reporting "2–8× over random" without noting "0.7–1.0× vs PGD" is selective.
- **"29 of 33 architecture–dataset cells positive transfer"** — the 4 negative cells include GCN-2 (the canonical architecture) on multiple datasets. The paper acknowledges this ("deeper-than-2-layer models transfer most reliably") but the lead claim is the 29/33 fraction. Lead with the architectural caveat.
- **Novelty inflation.** The IFT-for-adjacency move is fresh; the Neumann + randomised SVD pipeline is standard; the constrained projection $P_c$ is a duplication-matrix construction with the right cite (Magnus). The unification claim is the real contribution. The framing should make this distinction.
- **Ethics overreach.** The 90-day notification to PandaPower / Grid2Op / NERC / ENTSO-E is overkill for a method whose case-study $\tau = +0.37$ is below operator-grade. The notification is responsible practice but the implication — that AEGIS presents a credible grid-attack capability — is not supported by the case-study numbers. State the notification as "consistent with adversarial-ML best practice for dual-use methods" rather than as a response to the case-study capability.

## Decision lean

**Major Revision, leaning Reject if Critical Issue #1 is not addressed.**

The technical content is real. The contributions are real. But the headline framings overreach systematically, and the phase-transition theorem is not anchored in any post-threshold experiment. Fix Critical Issue #1 and the paper is publishable; leave it unfixed and the formal track is decorative.
