# Devil's Advocate Report

*Mandate: build the strongest case against the paper's central claim, independent of the other reviewers. Charitable to the math, hostile to the framing.*

## Strongest Counter-Argument (the case for rejection)
The paper's thesis is that `S_c` is a *unification* — one operator that audits, certifies, and defends — and that this unification is the contribution. The counter-argument is that the unification is **largely notational**, and that every headline number is the most flattering projection of a weaker underlying result.

Strip the framing and what remains is: equilibrium IFT sensitivity (known) restricted to edges by a duplication matrix (known), whose top singular value is read three ways. But "the same matrix appears in three formulas" is not by itself a scientific advance — `∂z*/∂A` *necessarily* governs sensitivity-based attacks, sensitivity-based radii, and sensitivity penalties, because they are all functions of the one Jacobian. The "coupling" (−0.65 anticorrelation between attack magnitude and certified radius) is not a discovery; it is the definitional fact that a large `σ_1` simultaneously means "easy to attack" and "small certified radius." The paper dresses an identity as an empirical finding.

Now the numbers. (1) "τ=0.99" is mostly the **edge weight** `A_ij`, a quantity available without `S_c` at all; the operator's own marginal contribution is +0.25, and unweighted it is +0.32 with a *negative* cell. (2) The "optimal attack" with "74–156×" advantage **flips <2% of labels** — it is an attack on the hidden state, not the classifier. (3) "Non-vacuous where smoothing degenerates" is true only on a Frobenius ball deliberately sized to make smoothing abstain; on smoothing's natural ball it certifies 77–96%. (4) The flagship "+0.996" is measured at **κ≈1.00**, outside the contractive regime the entire theory assumes. (5) The headline "distribution-free guarantee at the nominal level" is conditional on an exchangeability assumption the authors themselves admit does not hold transductively. Each individual reframe is disclosed *somewhere* in the appendix — but the pattern is that the front matter is uniformly the optimistic projection and the appendix is uniformly the honest one. A skeptical reviewer reads that pattern as engineered optimism.

If every headline is walked back in its own appendix, the defensible paper is the appendix's paper — and that paper is "a useful, cheap, honestly-scoped structural-sensitivity tool," not "a unifying operator that audits, certifies, and defends." The gap between those two papers is the case for rejection of *this* version.

## Issue List

### CRITICAL
- **[DA-C1] The certificate the paper "sells" is not proven as stated.** `thm:phase_transition`'s subcritical safe radius `ε_crit=(1−κ)/‖W‖` (κ=‖J_z‖) is established by a proof step (`app:proof_phase`, eq. `Jzp-bound`) that only yields the smaller radius `1/‖W‖−‖Â‖` unless the model is all-active. For a partially-active ReLU model the *advertised* safe radius exceeds the *proven* one — an optimistic (unsound-as-written) certificate. (Concurs with R1-A1.) *This is fixable (triangle-inequality rewrite), but on the submitted text the headline certificate is not justified, which under panel rules precludes Accept.*

### MAJOR
- **[DA-M1] The "unification" may be definitional, not substantive.** The paper must show the coupled operator buys something the *union of off-the-shelf tools* (an IFT attack + smoothing/conformal + spectral-norm reg) does not. As written, no experiment isolates "value of using one operator" from "value of having all three tools." Without that, "unification" is a narrative, not a result.
- **[DA-M2] Headline τ is carried by the edge weight, not the sensitivity.** (Concurs with R1-B2 from the hostile angle.) A reviewer can reproduce ~most of τ=0.99 by ranking edges on `A_ij` alone — a baseline requiring none of the paper's machinery. The contribution is the +0.25 increment; the paper hides it behind 0.99.
- **[DA-M3] The "attack" does not attack the prediction.** (Concurs with R1-B3.) An "Adversarial Evaluation of Graph Integrity" whose optimal perturbation flips 0–1.8% of labels at the quoted budget invites the question: integrity of *what*? Equilibrium movement is not, on its own, a security-relevant failure.
- **[DA-M4] Flagship experiment is outside the theory's validity envelope.** (Concurs with R1-B1.) κ≈1.00 on Amazon Photo voids `prop:transfer`'s subcritical hypothesis; offering it as the lead evidence *for* the theory is a category error.
- **[DA-M5] Selective ball-matching against smoothing.** (Concurs with R1-B4.) The vacuity claim is an artifact of the chosen `σ`. The legitimate, large, defensible win is *cost*; the vacuity framing is not.

### MINOR
- **[DA-m1] Radar plot rhetoric.** "Nonzero mass on all seven axes" is unfalsifiable-flavored: a method that does ε of everything and excels at nothing satisfies it. As the *first figure*, it primes the reader to accept breadth as merit.
- **[DA-m2] p-values without an n.** `p≈10^{-160}`, `p<10^{-43}` are quoted without the unit of analysis; such values usually signal pseudo-replication (pooling node-level observations as if independent).
- **[DA-m3] "One query" does not hold for the deployment-relevant explicit GNNs**, where `S_K` is built by `O(|E|)` finite differences (`app:explicit`). The marquee efficiency claim is an IGNN property.

## Ignored Alternative Explanations / Paths
- **Edge weight as the real predictor.** The most parsimonious explanation for the transfer result is "high-weight edges matter most when removed," which needs no sensitivity theory. The paper must rule this out (the +0.25 increment is the start, but a weight-only ablation should be the headline baseline, not a buried line).
- **Spectral-norm regularization as the real defense.** The defense's effect may be entirely attributable to generic Lipschitz control, not to anything `S_c`-specific. A matched spectral-norm-penalty baseline is missing.
- **Smoothing variants.** Sparse/localized smoothing and anisotropic smoothing are dismissed on cost but not run as accuracy/coverage competitors on the matched task.

## Missing Stakeholder Perspectives
- **The attacker who reads the diagnostic-only release** (R3-5): the "safe" release still ships an effective edge-attack ranking.
- **The practitioner with a non-contractive model**: receives a tool whose certificate and boundary theory do not apply, and whose "one query" becomes `|E|` probes.

## Observations (non-defects — credit where due)
- The appendix's honesty is genuinely exemplary: `rem:obs_o1` labels its own conjecture and *names two open proof gaps*; `rem:exchange-honesty` concedes the load-bearing assumption; `tab:constants` pre-empts the "which ×?" confusion. This is rarer than it should be and it is the main reason the recommendation is Revise, not Reject — the honest paper is *in here*, it just isn't the one on the first page.
- `prop:transfer`'s complete-remainder (bilinearity) argument is correct and elegant.
- The matrix-free scaling of the audit path is a real, verifiable engineering contribution.

**DA verdict:** one CRITICAL (the as-written certificate gap) → cannot Accept. The honest, narrower paper is publishable; the submitted framing is not.
