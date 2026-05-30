# Phase 2.5 — Revision Coaching (Socratic)

The decision is **Major Revision** with ten gating items. Before you start editing files, work through the six questions below. These are not rhetorical — write down your answers (or scratch-pad them in this file) and let the answers shape your revision plan.

## Q1 — Phase transition (gate G1)

**The headline theorem characterises three regimes, but every reported experiment is subcritical.** Two paths:

- **Path A** — keep the theorem as-is and run experiments at $\varepsilon = 1.5 \cdot \varepsilon_{\rm crit}$ and $\varepsilon = 0.9 \cdot \varepsilon_{\rm crit}$ for IGNN on Cora (smallest, fastest). Report what actually happens: divergent iteration? bounded but loose? oscillation? A single dataset–architecture cell with one $\varepsilon$ sweep is sufficient to anchor the regime characterisation empirically.

- **Path B** — keep the experimental envelope as it is and downgrade the regime characterisation: parts (b) and (c) become *"for $\varepsilon \ge \varepsilon_{\rm crit}$ our certificate fails; we do not characterise post-threshold behavior."* The theorem becomes a clean subcritical bound with a named threshold, no more.

> **Question to you:** Which path costs less and which path makes the paper stronger? Is the post-threshold experiment expensive (~1–2 days) or fast (~hours)? If fast, do Path A unconditionally — the empirical anchor is worth far more than the compute.

## Q2 — Title scope (gate G4)

The formal track applies to a contractive implicit subclass with ~6% Cora-accuracy cost. The explicit-architecture extension is computational only.

> **Question to you:** What is the paper's primary contribution? Reading my five reviewer reports, three reviewers (EIC, R2, DA) treat the **unification** (one matrix → three outputs) as the headline; one (R1) treats the **theorem** as the headline. The empirical extension to 7 architectures is supporting evidence either way.
>
> If unification is the headline, the title can stay broad ("Structural Sensitivity for Adversarial Vulnerability Analysis of GNNs") provided the abstract makes the implicit/explicit split unmistakable in the first three sentences. If the theorem is the headline, the title should narrow to "Implicit GNNs."
>
> Which positioning do you actually want to defend?

## Q3 — Insertion attacks (gate G5)

Your threat model excludes edge insertion. Nettack and Mettack are insertion-dominant. The related-work section reads as if AEGIS competes head-to-head with them; in the threat model it does not.

> **Question to you:** Is extending $S_c$ to a candidate-insertion set $\bar E$ technically feasible inside the matrix-free pipeline? My read: yes, with a cost increase from $O(|E|)$ to $O(|\bar E|)$ per-edge column eval and no change to the matvec / rmatvec routines. The candidate set $\bar E$ can be the 2-hop neighborhood of each target node — finite, much smaller than $N^2/2$.
>
> Two sub-questions: (i) Does the empirical transfer hold for insertions (does $v_{ij}$ on candidate edges correlate with Nettack/Mettack insertion damage)? (ii) Do you have time to run this experiment, or will you scope insertion to *future work* in §Limitations and §Threat Model?

## Q4 — Case study (gate G9)

R3's four concerns: correspondence vs derivation, narrow envelope, binary-vs-admittance, AC/DC ground-truth.

> **Question to you:** The case study currently sits awkwardly. Three options:
>
> (a) **Keep as proof-of-concept** — adopt R3's corrections (re-label isomorphism as empirical correlation, strengthen Table 1 caption with envelope qualifier, name the AC/DC ground-truth metric, report admittance normalisation), and lead the section with "this is not an operator-grade screening tool." Lightest effort.
>
> (b) **Strengthen toward operator-grade** — add PTDF baseline, Grid2Op operating envelope, admittance-weighted edges. Heaviest effort; arguably belongs in a follow-up paper.
>
> (c) **Move to appendix** — keep the case study as a one-paragraph cross-domain validation in the main paper, full table in appendix. Costs the cross-domain framing but eliminates 80% of R3's concerns.
>
> Which option do you want? Be honest about your appetite for power-systems revision work; (a) is the realistic 2-week answer, (b) is a different paper, (c) is the safe move if you decide the case study is not load-bearing.

## Q5 — Adaptive attacker (gate G6)

Your defense ablation shows 42 ± 8% damage reduction at $k=5$ for top-AEGIS-edge masking vs 11 ± 6% for random masking, paired Wilcoxon $p < 0.002$. The attacker does not re-optimise after masking.

> **Question to you:** What happens when you recompute $S_c$ on the masked graph? Either (i) the new SVD direction recovers most of the lost damage (defense gains shrink), in which case scope to "non-adaptive AEGIS attackers" with a clean disclaimer; or (ii) the masking changes the spectral structure in a way the attacker cannot fully recover (defense gains shrink modestly), in which case you have a *stronger* result than the current paper claims.
>
> This is a one-day experiment you should run before you decide which framing to use. Don't guess.

## Q6 — Headline reframing (gates G4, G9, G10)

Your current abstract leads with: *"first-order envelope tight ($1.00 \pm 0.01$) at $\varepsilon=0.01$ across 7 architectures and 9 datasets."* That is your most technically sound but **least practically interesting** claim. At $\varepsilon = 0.01$, breach rates are 0.0%–0.6% — nothing is breaking.

> **Question to you:** What would you want a reviewer to remember if they read only your abstract?
>
> My suggestion (push back if you disagree): lead with the **unification** ("one matrix-free computation yields per-edge rankings, the SVD-optimal direction, and per-node radii"), then the **architectural transfer pattern** ("positive transfer in 29/33 cells, dominated by deeper-than-2-layer models"), then **scalability** ("$N = 7{,}650$ on a single GPU"), then *briefly* the formal subcritical bound for the IGNN subclass. Leave the "1.00 tightness" detail for §VI.

## Revision plan template

After answering Q1–Q6, fill in this template:

```
Q1 path chosen:                    [A / B]
Q2 positioning:                    [unification / theorem]
Q3 insertion scope:                [extend candidate-set / future work]
Q4 case study:                     [keep as PoC / strengthen / move to appendix]
Q5 adaptive-attacker outcome:      [TBD — run experiment first]
Q6 abstract reframing draft:       [paste 3-sentence draft here]

Gating items to address:           G1, G2, G3, G4, G5, G6, G7, G8, G9, G10
Compute budget needed:             ___ days
Writing budget needed:             ___ days
Submission re-target date:         ___
```

When the plan is filled in, work G2 + G3 (proof tightening, half-day) first to flush out any cascading wording changes, then G1 + G6 + G7 + G8 (compute, parallelisable), then G4 + G5 + G9 + G10 (writing). Total ~2 weeks of focused effort.

Good luck. Tag me when the revision is ready and I'll re-review against G1–G10.
