# Strategic Response Plan — AEGIS revision (honesty without self-sabotage)

**Thesis:** Addressing a concern ≠ hedging it. For this paper, every gating concern has a path that is *both* honest *and* positioning-preserving — usually a cheap experiment that strengthens (FIX) or a precise restatement that converts a caveat into a feature (REFRAME). Genuine concession-that-weakens (CONCEDE) is forced in exactly **zero** places. The only non-negotiable is the `ε_crit` *math*, and its honest form is stronger than the current one.

**Disposition legend:** FIX = run experiment / change math (closes + strengthens) · DEFEND = reviewer is wrong / already handled, push back · REFRAME = restate honestly, keep the hook · CONCEDE = genuinely scope down.

---

## Disposition table

| # | Concern | Disp. | Strategic move | Net effect on positioning |
|---|---|---|---|---|
| C-2 | `ε_crit` not worst-case sufficient (measured-κ optimistic) | **REFRAME + small FIX** | State the all-active worst-case `(1−‖W‖₂)/‖W‖₂` explicitly; move the "2–4×" margin onto ρ(J_z) where it's genuine and *more* impressive (holds even at cap 0.99). | **Stronger** — preempts the attack; the safety story becomes "training provably respects the boundary." |
| C-1 | PF topology confound; no centrality null | **FIX (+REFRAME)** | Run degree / betweenness / current-flow-betweenness on IEEE N-1. Beat them → objection dies; tie → reframe as label-free, surrogate-only, admittance-free attribution. Recast "binary beats admittance" as *physically correct*, not incriminating. | **Stronger or neutral** — never weaker; a win here is a headline. |
| C-3a | Prop 3(b) only pairwise; τ=0.99 empirical | **REFRAME** | Theorem supplies the *mechanism* (`d_k=w_k v_k+R_k`, bounded remainder); the global τ is its empirical confirmation. Normal theory↔experiment relationship. | Neutral — stop implying (b) proves the statistic; keep the result. |
| C-3b | Phase-transition regime never reached | **REFRAME (feature)** | "Trained models sit inside the regime with 2–4× margin" is a *positive safety result*, not a weakness. Don't apologize that models are safe — that's the point. | **Stronger** — reframes a "so what" into the deliverable. |
| C-4 | Headline results on 50-node subgraph (τ=0.16 faithful) | **FIX** | Re-run four-quadrant + 42% defense on full graph (matrix-free path exists). The paper *already* shows the full-graph edge advantage *amplifies* (9.82× Citeseer). | **Stronger** — the better numbers are full-graph. |
| C-5 | 512-query "black-box" is a random strawman | **FIX (low risk)** | Swap in NES/SimBA at matched budget. Even if it recovers more than 44%, the surrogate-transfer control (cos=0.99, zero shared gradients) remains the primary non-circularity evidence. | Neutral→stronger; removes an easy reviewer jab. |
| C-6a | "No method returns all three" overclaim | **REFRAME** | "no prior *single matrix-free object* yields all three in one pass." Keeps the hook, becomes unfalsifiable-by-counterexample. | Neutral — hook retained, made precise. |
| C-6b | AGNNCert "complementary" spins near-zero τ | **DEFEND + REFRAME** | Lead with the low correlation: "they flag *different* nodes (τ≈0) — which is precisely why they are complementary, not redundant." The low τ is the *argument*, not a liability. | **Stronger** — turns the weak number into evidence. |
| C-6c | "PGD direction" misattribution (abstract) | **FIX (trivial)** | Name the zero-gradient surrogate transfer; it's *more* impressive than a PGD match. | **Stronger.** |
| #3 | τ "+0.62" vs table "+0.67" | **FIX (trivial)** | Correct to +0.67 — the real number is higher. | **Stronger.** |
| R3 | Missing oversquashing/curvature, Topology-Attack, GCORN, node-injection cites | **FIX (+DEFEND)** | Cite *and distinguish*: Topping/Di Giovanni analyze sensitivity for expressivity/oversquashing; AEGIS for adversarial vulnerability via the equilibrium resolvent. Each cite sharpens the delta. | **Stronger** — shows command of literature, preempts "missed prior art." |
| P1-8 | PF surrogate trained on uniform load only | **REFRAME + bounded CONCEDE** | Push the limit onto the *surrogate*, not the method: "AEGIS adds a label-free vulnerability layer over *any* PF surrogate; operating-envelope breadth is a property of the surrogate." Scope claim to "in-distribution rank triage." | Neutral — limitation lands on the surrogate, not AEGIS. |
| P1-9 | No code artifact | **FIX** | Release the diagnostic path (consistent with the gating of the attack path). | **Stronger** (reproducibility). |
| Prop 1 | maximizer is σ₁(S_c), not σ₁(S) | **FIX (math)** | Restate over the symmetric/edge-supported subspace. Tighter and correct; no narrative cost. | Neutral. |

---

## The four highest-leverage moves (do these first)

### 1. Re-point the `ε_crit` margin (REFRAME the crux) — non-negotiable math, strong outcome
Do **not** write "ε_crit is only locally sufficient." Instead split the two quantities cleanly:
> *"Since ‖Â‖₂=1 for the renormalized adjacency, the closed-form **sufficient** budget in the worst (all-active) case is `ε_crit^suff = (1−‖W‖₂)/‖W‖₂`, independent of the trained activation pattern. The **operative** safety quantity, however, is the spectral radius: across the full κ_max∈[0.30,0.99] sweep the trained ReLU pattern holds ρ(J_z)≤0.42, a 2–4× margin to criticality that **persists even when the spectral-norm cap is pushed to 0.99** (resolvent grows only 1.17→1.80)."*

This keeps Theorem 1 intact, states the worst case rigorously, and moves the impressive "2–4×" onto ρ(J_z) where it is *genuinely* true and even more striking. A reviewer who tries the κ-substitution attack finds you already there.

### 2. Run the PF centrality baseline (FIX the CRITICAL confound) — highest single ROI
This is the one experiment that converts a CRITICAL into a strength. Outcomes:
- **S_c beats centrality** → headline: "label-free, admittance-free S_c outperforms graph centrality on N-1." Objection dead.
- **S_c ties centrality** → reframe (still strong): centrality needs the explicit topology; S_c recovers the same ranking from a *learned surrogate alone*, no graph, no labels, and generalizes to classification where centrality is meaningless.
Either way you win. The strategic error would be to *pre-concede* ("our PF result may be topology-driven") instead of running the 1-hour baseline.

### 3. Promote the full-graph numbers (FIX C-4) — the better results are already yours
The paper buries that the edge advantage *amplifies* to 9.82×/3.25× at full-graph scale while leaning on weaker 50-node subgraph numbers for the headline tables. Re-run the four-quadrant + defense full-graph and lead with those. This simultaneously closes the "subgraph isn't faithful" jab and raises the numbers.

### 4. Weaponize the complementarity (DEFEND C-6b) — the weak τ is the argument
The near-zero AGNNCert correlation is not something to hedge — it's the *proof* of complementarity. State it first, then the decision rule. A certifier and a sensitivity diagnostic that agreed would be redundant; disagreement is the value proposition.

---

## Hedge discipline (the anti-underselling rule)

Scattered qualifiers are what actually weaken a paper — not honesty. Concentrate hedging:
1. **One scoping sentence per claim, in the right place** (limitations or the claim's own definition), never repeated defensively in abstract + intro + body.
2. **Hedge the quantity, not the contribution.** "ε_crit is conservative" (quantity) ✓; "our safety analysis is limited" (contribution) ✗.
3. **Every limitation gets a redirect.** Pair each honest caveat with what it *enables* or whose property it actually is (e.g., envelope breadth = surrogate's property; r_v non-certificate = the complement to certifiers, by design).
4. **Earn, don't disclaim.** Where a reviewer says "you didn't show X," prefer running X over conceding X. (Bulletproof > hand-waving.)

---

## Non-negotiables (where honesty does force a real change)

- **`ε_crit` worst-case form must be stated** (C-2). The measured-κ value cannot be presented as a *certificate* without the all-active qualifier. Fix is REFRAME, not CONCEDE — but it is mandatory.
- **The +0.62/+0.67 inconsistency and the PGD misattribution are factual errors** — fix verbatim.
- **The PF centrality null must be run or the cross-domain claim must be scoped.** You can choose FIX (run it) or REFRAME (scope to surrogate-agnostic attribution), but "ignore it" is not survivable against R4's CRITICAL.

Everything else is REFRAME/DEFEND with the hook retained.
