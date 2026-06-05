# Baseline audit — PGD attacks (Madry 2018; topology-attack Xu 2019)

**Scope.** Faithfulness audit of the three reported PGD variants: (i) 50-step
equilibrium-shift PGD ("Shift-PGD", labelled *IFT-gradient PGD*), (ii)
classification-loss PGD ("Cls-PGD"), (iii) the reachability critical-driving PGD.
Verified against the AEGIS threat model `‖δÂ‖_F ≤ ε`, symmetric, edge-only,
continuous (`background.tex:20-25`).

## Implementation locations

| Variant | Function | File:line |
|---|---|---|
| Cls-PGD / Shift-PGD (subgraph, tab:attack_full) | `pgd_attack(objective=...)` | `scripts/exp_full_attack_table.py:137` |
| Cls-PGD standalone | `pgd_classification_attack` | `scripts/exp_classification_attack.py:72` |
| Shift-PGD standalone | `pgd_shift_attack` | `scripts/exp_classification_attack.py:152` |
| Cls/Shift-PGD (full-graph) | `pgd_attack_fullgraph` | `scripts/exp_fullgraph_attack_table.py:172` |
| Critical-driving PGD (reachability) | `pgd_critical_attack` | `scripts/exp_reachability.py:275` |
| True IFT operator `(I−J_z)^{-1}` (AEGIS only, NOT the baseline) | `structural_sensitivity_matrix` | `iem/adversarial.py:156,170` |

All PGD variants are **hand-rolled**. No official topology-attack
(`deeprobust`/`Xu2019`) code is used. The IGNN forward solve is contractive:
`κ = ‖J_z‖₂ ≤ c < 1`, default `c=0.9`, forward `max_iter=100, tol=1e-6`
(`iem/examples/ignn_cora.py`).

## Official reference algorithm (defining steps)

**Madry 2018 PGD** (the cite at `experiments.tex:35`): for `T` steps, ascend the
*classification* loss; project onto the `ε`-ball each step; **random restart(s)**
inside the ball; report the worst restart. Madry-PGD is an Lp-ball attack — it
does not define a graph/topology variant.

**Xu 2019 topology attack** (the correct reference for *structure* PGD, IJCAI
2019, the cite the protocol names): optimise over a **relaxed edge variable
`s ∈ [0,1]^|E|`** (probability of flipping each edge). Per step: gradient of the
*attack/CW classification* loss w.r.t. `s`; gradient/PGD step; **project onto the
box `[0,1]` ∩ the budget simplex `1ᵀs ≤ Δ`** (bisection on the dual). After the
`T`-step relaxed solve, **random sampling**: draw `K` Bernoulli samples
`s_k ~ Bern(s*)`, keep the feasible sample maximising the attack loss → discrete
`δA`. Typical `T≈200`, multiple restarts. The two load-bearing ingredients are
**(a) the [0,1] relaxation + budget-simplex projection** and **(b) the random
discrete sampling step**.

## Our steps, per variant

**Cls-PGD** (`pgd_attack` objective=`classification`; `pgd_classification_attack`):
1. `delta` over edge weights, init random (subgraph) / zeros (standalone).
2. 50 outer steps. Each: build `A_pert = A + delta` on the edge support; inner
   solve `for _ in range(50): Z=operator(Z,...)` under `enable_grad`;
   `loss = -CE(head(Z), y)`; `grad = autograd.grad(loss, delta)`.
3. Update `delta -= (ε/10)·grad.sign()`; per-coord clamp `±ε/√|E|`; L2 renorm to
   `‖delta‖₂ ≤ ε`.

**Shift-PGD** (objective=`shift`; `pgd_shift_attack`): identical loop, but
`loss = -‖Z* − Z_clean‖²` (equilibrium shift). Same sign-step / clamp / L2 proj.

**Critical-PGD** (`pgd_critical_attack`): 120 iters, tangent-space projected
**ascent** on the Frobenius sphere `‖δA‖_F = ε`, objective `ρ(J_z(A+δ))`, two LRs
{0.3, 1.0}, analytic rank-1 warm start + 1 random seed; keeps the best.

## GAPS

| Gap | Severity | File:line | Fix |
|---|---|---|---|
| **"IFT-gradient PGD" uses NO IFT gradient.** Shift-PGD backprops through a *truncated 50-step unrolled solve*, not `(I−J_z)^{-1}`. The true IFT operator exists at `iem/adversarial.py:170` but feeds only AEGIS's SVD. Paper labels it "IFT-gradient PGD" / "IFT gradients" / "VJP through the resolvent" — all three are misnomers. | **HIGH (mislabel)** | `exp_full_attack_table.py:137-170`; `exp_classification_attack.py:152`; captions `experiments.tex:39`, `exp_fullgraph_attack_table.py:12-13,44` | Either (a) implement the real IFT grad (1 linear solve `(I−Jᵀ)x=∂L/∂z` via the existing resolvent) — at κ≤0.9 this changes little, so cheap to do; or (b) relabel everywhere to "unrolled-solver PGD" / "backprop-through-solver". Mislabel is independently rejectable. |
| **Inner solve capped at 50 iters vs model `max_iter=100`.** `for _ in range(50)` rarely hits the `1e-7` break at κ≈0.9 (residual ≈ 0.9⁵⁰ ≈ 5e-3), so the gradient is taken at a *not-fully-converged* `Z` → a weakened, slightly-wrong gradient for the PGD opponent. | MED | all four PGD fns (inner `range(50)`) | Raise inner solve to ≥100 iters or loop to `tol=1e-6`, matching the victim's forward. |
| **No random restarts.** Madry-PGD and Xu-PGD both report best-of-`R` restarts; ours runs a single trajectory (Cls/Shift seed once; standalone Cls/Shift start at `delta=0`, deterministic). Single-trajectory PGD systematically under-estimates the attack optimum → inflates AEGIS's advantage. | MED→HIGH | `exp_full_attack_table.py:137`, `exp_classification_attack.py:72,152`, `exp_fullgraph_attack_table.py:172` | Add 5–10 random restarts, report max damage. Critical-PGD already does warm-start + random seed; Cls/Shift do not. |
| **`grad.sign()` (L∞ step) under an L2/Frobenius ball.** The threat set is `‖δ‖_F ≤ ε`, but the step direction is `sign(grad)` (the L∞-PGD step). The correct ascent direction for an L2 ball is the **normalised gradient `grad/‖grad‖`**. `sign()` points to a box corner, not the steepest L2 ascent → a measurably weaker L2 attack. | MED | `exp_full_attack_table.py` (delta update), `exp_classification_attack.py:72,152` (`grad.sign()`) | Use `delta += step·grad/grad.norm()` (normalised steepest ascent) for the Frobenius ball; keep `sign()` only if the ball were L∞. |
| **Spurious per-coordinate clamp `±ε/√|E|`.** This box-clamps each edge to a corner of an L∞ box before the L2 renorm. It is not in either reference and artificially caps the per-edge magnitude, shrinking the feasible L2 set the attacker can reach. | MED | `exp_full_attack_table.py`, `exp_classification_attack.py:72,152`, `exp_fullgraph_attack_table.py:172` (`clamp_(-ε/√n, ε/√n)`) | Drop the per-coord clamp; the L2 renorm alone enforces `‖δ‖_F ≤ ε`. |
| **No [0,1] relaxation + random sampling (Xu 2019).** For the *structure* attack the protocol names Xu's relaxed-`s`-in-`[0,1]` + budget-simplex + Bernoulli-sampling algorithm. Ours optimises continuous real-valued edge *weights* with an L2 ball. This is defensible because the AEGIS threat model is explicitly *continuous weight perturbation* (`background.tex:25`), so the topology-attack discretisation is out of model — but the paper should say so, else a reviewer expecting Xu-PGD sees a different attack. | LOW (scope, not bug) | `background.tex:25`; captions | Add one sentence: PGD baselines run in the same continuous Frobenius threat model as AEGIS; Xu-2019 discrete topology attack is reported separately (Mettack, `experiments.tex:54`). |
| **Cls-PGD ascends classification loss — CONFIRMED correct.** `loss=-CE(head(Z),y)`, grad ascends CE. Name matches objective. | OK | `exp_classification_attack.py:72` | — |
| **Shift-PGD ascends `‖ΔZ*‖` — CONFIRMED correct.** `loss=-‖Z*−Z_clean‖²`. Name matches objective. | OK | `exp_classification_attack.py:152` | — |

## Verdict per variant

- **Shift-PGD / "IFT-gradient PGD" — UNFAITHFUL (as labelled).** It optimises the
  right objective (`‖ΔZ*‖`) but the gradient is **plain backprop through a
  truncated solver, not the IFT/implicit gradient** the name and three captions
  claim. Protocol step 4 fails outright. This is the PR-BCD-class defect (the
  method does not do what its label asserts). Fix = implement the resolvent VJP
  *or* rename. Note: because κ≤0.9 (strongly contractive), the numerical gap
  between unrolled-50 and true-IFT is small, so the *numbers* are likely close —
  but the **claim** is false and a knowledgeable reviewer will catch it.

- **Cls-PGD — MINOR-GAPS (under-powered, not mislabelled).** Objective correct.
  But it is a *weak* PGD opponent: `sign()` step on an L2 ball, a spurious
  per-coord clamp, no restarts, and a 50-iter inner solve. Each independently
  biases the comparison toward AEGIS. Stack of four → the "74–156×" / "1.5–3.1×"
  margins are inflated by an unknown but non-trivial amount.

- **Critical-PGD (reachability) — FAITHFUL.** Genuine projected gradient on the
  Frobenius sphere with tangent-space projection, multi-LR, warm-start + random
  seed, keeps best. Proper L2 PGD; no notable gap.

## Paper numbers at risk

- `experiments.tex:35` + `abstract.tex` — **"74–156× per-query advantage" over
  50-step PGD**, and `tab:attack_full` (Cls-PGD: Cora 2.51, Citeseer 2.97, WikiCS
  0.67 vs AEGIS 3.70/4.63/2.10): at risk from the weak-PGD stack (sign-step,
  clamp, no restarts, 50-iter solve). Margin likely shrinks if PGD is
  strengthened; *direction* (AEGIS ≥ PGD) probably survives, magnitude may not.
- **Appendix F** — Cls-PGD "15–70% less at 50× the cost" / "1.5–3.1×" (fig. 9
  caption, `aaai_aegis.aux:398`): same exposure as above.
- **Appendix F / captions** — **IFT-gradient PGD "72–92%"** (`shift_pgd_over_svd_dmg`,
  e.g. 0.88, 0.90 in `full_attack_table.csv`): the **number may stand** (κ small)
  but the **"IFT-gradient" framing is unsupported by the code** and must be
  renamed or re-implemented; "solver validation, not an independent baseline"
  partly hedges this but the explicit "IFT" label remains false.
