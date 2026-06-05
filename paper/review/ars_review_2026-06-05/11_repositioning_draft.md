# Repositioning Draft — "Coupling + Cost" framing (abstract + intro)

**Goal:** pivot the positioning from "wins a little of everything" (the radar's weak "nonzero on all 7 axes")
to the three contributions that survive every correction in `06`–`10` fully intact:
**(C) coupling** — one operator is attack + certificate + defense; **(S) scaling** — matrix-free audit to N=7,650;
**(K) cheap conformal** — sampling-free, 10³–10⁴× below smoothing. All numbers below are the *honest* ones
(τ reported as estimator + sensitivity-lift decomposition; "attack" = equilibrium damage; conformal sound
under exchangeability; deterministic radius dropped from the abstract as a secondary result).

---
## ABSTRACT

### BEFORE
> Deploying a graph neural network safely takes two things current tools provide only in isolation: a map
> of which edge perturbations would break its predictions, and an actionable robustness guarantee. AEGIS
> delivers both from a single matrix-free object … From this one object come three capabilities. AEGIS
> audits: its leading singular direction, column norms, and node margins give an **optimal attack** … It
> certifies: AEGIS-Conformal … turning split conformal prediction into a distribution-free, sampling-free
> **guarantee that holds at the nominal level** … Empirically, **this continuous sensitivity predicts
> discrete attack damage** … (**rank correlation τ=0.99**), and one analytic query inflicts 74–156× the
> per-query damage of a 50-step PGD attack.

### AFTER (coupling + cost; honest numbers)
> Auditing a graph neural network, certifying its robustness, and defending it are handled today by three
> disjoint toolkits: structural attacks that rank edges but issue no guarantee, certifiers that bound
> robustness but name no vulnerable edge, and robust architectures that harden a model but expose neither.
> We show these are three readings of one matrix-free object, the constrained sensitivity matrix $S_c$,
> which records how strongly each edge moves the model's equilibrium prediction and is applied to vectors
> without ever being formed: one randomized-SVD pass reads its leading singular direction, column norms,
> and node margins as the most damaging edge perturbation, a per-edge vulnerability ranking, and per-node
> safe radii at once. The coupling is the contribution. The same $\sigma_1(S_c)$ that locates the worst
> perturbation is the penalty that suppresses it---attack magnitude and certified radius are anticorrelated
> ($-0.65$ across 10 seeds)---and the same closed-form score-shift bound turns split conformal prediction
> into \AEGIS-Conformal, a sampling-free robustness certificate, sound under exchangeability, that costs
> $\mathbf{10^3}$--$\mathbf{10^4\times}$ less than randomized smoothing and stays non-vacuous on the
> Frobenius ball where matched smoothing abstains. Matrix-free evaluation scales the audit from $N\approx300$
> to $N=\mathbf{7{,}650}$ on one GPU. Across 6 datasets, 7 architectures, and 4 domains, the
> sensitivity-weighted score $A_{ij}v_{ij}$ reproduces brute-force single-edge-removal damage at Kendall
> $\tau=\mathbf{0.99}$---of which the sensitivity contributes $\mathbf{+0.25}$ (median) to $\mathbf{+0.65}$
> (uniform-weight graphs) beyond the edge weight alone---and one analytic query matches a 50-step PGD attack
> on equilibrium damage at $\mathbf{74}$--$\mathbf{156\times}$ lower per-query cost.

**What changed & why**
- Opens on **coupling** (the only-AEGIS-does-all-three claim) instead of the generic "two things in isolation."
- τ is reported as **estimator (0.99) + sensitivity lift (+0.25–0.65)** — preempts the edge-weight-baseline critique; the +0.65 is your fraud-graph strength.
- "optimal attack" → **"matches PGD on equilibrium damage"** (the metric you actually measure; flips are <2% at this budget).
- Conformal: **"sound under exchangeability"** added; the **cost win (10³–10⁴×)** is now the headline, not the vacuity-of-smoothing framing.
- Deterministic `ε_crit`/2–9× **dropped from the abstract** (secondary; lives in intro/theory) — declutters and avoids the `ε_glob` nuance up front.

---
## INTRODUCTION — opening

### BEFORE (first paragraph)
> Graph neural networks are now deployed where structural errors carry real consequences: fraud accounts
> evade detection …, drug-interaction models misfire …, and **power grids can miss contingencies** … A
> practitioner faces a question current tools leave unanswered: which edges, if perturbed, would cause
> predictions to fail, and by how much?

### AFTER
> Graph neural networks are deployed where structural errors carry real consequences---a flagged fraud
> account can evade detection by perturbing a few behavioural-similarity edges, the audit we demonstrate
> in \cref{sec:fraud_case}. Deployment raises three questions in turn: which edges, if perturbed, would
> break the model; how much structural perturbation it provably tolerates; and how to harden it. Today
> each is answered by a separate literature---structural attacks rank edges by gradient search but return
> no budget or certificate~\cite{...}; smoothing and certifiers return per-node or collective certificates
> but neither an edge ranking nor a direction~\cite{...}; robust architectures harden models without
> surfacing which edges are vulnerable~\cite{...}---and none answers the other two (\cref{fig:positioning}).
> \AEGIS supplies all three from one object: the constrained sensitivity matrix $S_c$ (\cref{sec:framework}).

**Radar reframe (`fig:positioning` caption):** the operative claim becomes *"each thread answers one of the
three questions and none the other two; \AEGIS is the only method covering all three,"* which the radar
genuinely shows — instead of the weak *"nonzero mass on all seven axes."* (Drugs/power-grid demoted: the
honest "we audit the model, not the physics" scoping moves up from the conclusion, per `04`/`06`.)

---
## INTRODUCTION — contributions

### AFTER (re-centered on coupling + cost; honest empirics)
> \textbf{Contributions.}
> \textbf{(1) One operator, three readings.} $S_c$ specialises equilibrium IFT sensitivity to \emph{structural}
> edge perturbations via $P_c$; one matrix-free query reads the SVD-optimal direction, per-edge ranking, and
> per-node radii together, never forming $S_c$ (matching dense $\sigma_1$ to $0.03\%$ at $N{=}200$, scaling
> to $N{=}7{,}650$).
> \textbf{(2) Coupling.} The same $\sigma_1(S_c)$ is the attack direction, the (inverse) certified budget, and
> the defense penalty; attack magnitude and certified radius are anticorrelated ($-0.65$, $10/10$ seeds,
> \cref{sec:defense}). No prior object couples the three.
> \textbf{(3) Sampling-free robust conformal.} The closed-form score-shift bound makes \AEGIS-Conformal a
> distribution-free (under exchangeability) certificate at $10^3$--$10^4\times$ below smoothing, non-vacuous
> where matched smoothing abstains (\cref{sec:conformal}); for contractive models a closed-form deterministic
> radius adds a global guarantee the measured break exceeds by $2$--$9\times$ ($10$ seeds, \cref{thm:phase_transition}).
> \textbf{(4) Empirical evaluation} (390 runs): $A_{ij}v_{ij}$ tracks brute-force removal at $\tau{=}0.99$
> (sensitivity lift $+0.25$ median, $+0.65$ on uniform-weight graphs, over the edge-weight baseline),
> equilibrium-damage parity with 50-step PGD at $74$--$156\times$ lower cost (\cref{tab:attack_full}), and a
> fraud audit (\cref{sec:fraud_case}).

---
## Net effect on positioning
- **Strengthened / clarified (untouched by any correction):** coupling, matrix-free scaling, cheap conformal — now the spine of the pitch.
- **Honestly trimmed:** τ headline carries its decomposition; "attack" = equilibrium damage; conformal qualifier explicit; radar claim earned.
- **Reviewer-defensibility:** the front matter now says exactly what the appendix proves — the "oversells-its-own-appendix" rejection trigger is gone.
