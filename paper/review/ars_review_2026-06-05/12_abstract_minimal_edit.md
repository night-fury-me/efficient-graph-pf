# Abstract — minimal-edit version (chosen)

Keeps the current abstract's opening, development, and style. **Sentences 1–4 verbatim**
(opening hook + `S_c` object + "three capabilities" + *audits*). Only *certifies / defends / empirical*
are edited, and only for the coupling reframe + the honesty corrections already verified in `06`–`10`.
Supersedes the full-rewrite abstract in `11` (which restructured the opening — not wanted).

---
## Updated abstract (drop-in for `sections/abstract.tex`)

```latex
Deploying a graph neural network safely takes two things current tools provide only in isolation: a map
of which edge perturbations would break its predictions, and an actionable robustness guarantee. \AEGIS
delivers both from a single matrix-free object, the \emph{constrained sensitivity matrix} $S_c$, which
records how strongly each edge can move the model's equilibrium prediction and scales to $N{=}7{,}650$
nodes on one GPU. From this one object come three capabilities. \AEGIS \emph{audits}: its leading
singular direction, column norms, and node margins give an optimal attack, an edge-vulnerability ranking,
and per-node safe radii in one pass. It \emph{certifies}: \AEGIS-Conformal bounds in closed form how far
a bounded perturbation shifts the conformity scores, turning split conformal prediction into a
sampling-free certificate---distribution-free, sound under an exchangeability condition---that holds at
the nominal level under the attack it certifies. It stays non-vacuous on the matched Frobenius ball where
randomized smoothing degenerates to the full label set at $10^4\times$ the cost; for contractive models a
closed-form safe radius adds a deterministic certificate the \emph{measured} break exceeds by
$\mathbf{2}$--$\mathbf{9\times}$ ($10$ seeds). It \emph{defends}: penalizing $\sigma_1(S_c)$ trades clean
accuracy for certified robustness, so the same operator that finds the worst attack tunes the defense
against it---attack magnitude and certified radius anticorrelate ($-0.65$, $10$ seeds). Empirically, this
continuous sensitivity predicts discrete single-edge-removal damage across 6 datasets, 7 architectures,
and 4 domains (rank correlation $\tau{=}0.99$ for the edge-weighted score, the sensitivity itself adding
$\mathbf{+0.25}$ median to $\mathbf{+0.65}$ beyond the edge weight alone), and one analytic query inflicts
$\mathbf{74}$--$\mathbf{156\times}$ the per-query equilibrium damage of a 50-step PGD attack.
```

---
## Exactly what changed (each one necessary; nothing else touched)

| # | Sentence | Change | Why it's necessary |
|---|---|---|---|
| 1 | *certifies* | "a distribution-free, sampling-free **guarantee** that holds at the nominal level" → "a sampling-free **certificate**—distribution-free, **sound under an exchangeability condition**—that holds at the nominal level" | `rem:exchange-honesty` concedes (C1) doesn't hold for free transductively; the unqualified "guarantee … at the nominal level" was the overclaim flagged in `02`/`06`. |
| 2 | *certifies* | "non-vacuous where smoothing degenerates" → "non-vacuous **on the matched Frobenius ball** where smoothing degenerates" | The vacuity is a ball-matching artifact (`tab:smoothing`); naming the ball makes it honest and keeps the real win (cost). |
| 3 | *certifies* | "a deterministic **guarantee** the **empirical** break exceeds by 2–9×" → "a deterministic **certificate** the **measured** break exceeds by 2–9× **(10 seeds)**" | PATCH 4 (reviewer-approved in `10`): the 2–9× is `ε_reach/ε_crit`, `ε_reach` being `rem:obs_o1`'s *measured* (conjecture-backed) break, not a guaranteed extension. |
| 4 | *defends* | "penalizing the leading sensitivity trades clean accuracy for certified robustness" → "penalizing **$\sigma_1(S_c)$** … **so the same operator that finds the worst attack tunes the defense against it—attack magnitude and certified radius anticorrelate ($-0.65$, 10 seeds)**" | **The positioning move.** Surfaces the *coupling* — your strongest, fully-defensible novelty — inside the existing *defends* sentence, no restructuring. |
| 5 | *empirical* | "this continuous sensitivity predicts discrete **attack** damage … (τ=0.99)" → "predicts discrete **single-edge-removal** damage … (**τ=0.99 for the edge-weighted score, the sensitivity itself adding +0.25 median to +0.65 beyond the edge weight alone**)" | The τ decomposition we discussed: keeps 0.99 but credits the sensitivity honestly (+0.25–0.65), preempting the edge-weight-baseline critique. "removal" not "attack" = the metric actually measured. |
| 6 | *empirical* | "the per-query damage of a 50-step PGD attack" → "the per-query **equilibrium** damage of a 50-step PGD attack" | One word: the attacked quantity is equilibrium shift, not misclassification (flips <2% at this budget). |

**Untouched:** opening hook, `S_c`-object sentence, "three capabilities," the entire *audits* sentence, and the audits/certifies/defends structure — the style you wanted preserved.

---
## Length-neutral variant (≈ current length: ~214 words vs ~209) — ✅ CHOSEN (2026-06-05)

Keeps **all six corrections** and the **verbatim opening (S1–S4)**. To absorb the coupling (+15 words)
and the τ decomposition (+7) without growing the abstract, **one secondary clause is dropped — the
deterministic-radius "2–9×" sentence** — which relocates to intro contribution (2) / `thm:phase_transition`,
where the `ε_glob`/`ε_crit` nuance belongs anyway. The smoothing-vs-cost comparison is merged into the
*certifies* sentence; nothing else is lost.

```latex
Deploying a graph neural network safely takes two things current tools provide only in isolation: a map
of which edge perturbations would break its predictions, and an actionable robustness guarantee. \AEGIS
delivers both from a single matrix-free object, the \emph{constrained sensitivity matrix} $S_c$, which
records how strongly each edge can move the model's equilibrium prediction and scales to $N{=}7{,}650$
nodes on one GPU. From this one object come three capabilities. \AEGIS \emph{audits}: its leading
singular direction, column norms, and node margins give an optimal attack, an edge-vulnerability ranking,
and per-node safe radii in one pass. It \emph{certifies}: \AEGIS-Conformal bounds in closed form the
worst-case conformity-score shift, turning split conformal into a distribution-free, sampling-free
certificate---sound under exchangeability---that holds at the nominal level under the attack it certifies,
and stays non-vacuous on the matched Frobenius ball where smoothing abstains, at $10^4\times$ lower cost.
It \emph{defends}: penalizing $\sigma_1(S_c)$ trades clean accuracy for certified robustness, so the
operator that finds the worst attack also tunes the defense (attack--radius anticorrelation $-0.65$, 10
seeds). Empirically, this continuous sensitivity predicts single-edge-removal damage across 6 datasets, 7
architectures, and 4 domains (edge-weighted $\tau{=}0.99$; the sensitivity adds $\mathbf{+0.25}$ to
$\mathbf{+0.65}$ over the weight), and one query inflicts $\mathbf{74}$--$\mathbf{156\times}$ the per-query
equilibrium damage of a 50-step PGD attack.
```

**What this variant trades vs the longer one above:**
- **Dropped from the abstract:** "for contractive models a closed-form safe radius adds a deterministic certificate the measured break exceeds by 2–9×" → moves to intro contribution (2) / `thm:phase_transition` (the deterministic radius is secondary to the conformal certificate, and the `ε_glob` caveat reads cleaner in the body).
- **Merged:** the smoothing/cost comparison now closes the *certifies* sentence.
- **Tightened, no meaning lost:** "bounds … how far a bounded perturbation shifts the conformity scores" → "bounds … the worst-case conformity-score shift"; "degenerates to the full label set" → "abstains"; τ aside compressed to "the sensitivity adds +0.25 to +0.65 over the weight."
- **Identical to current:** opening hook, `S_c` sentence, "three capabilities," the whole *audits* sentence.

**Alternative if you want the deterministic clause kept:** I reclaim the ~15 words by lightly tightening the existing prose instead (no content dropped, but it touches a bit more of the current wording). Say which you prefer.
