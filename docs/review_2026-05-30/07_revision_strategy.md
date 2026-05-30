# Strategic Revision Plan — Honest, Not Hedged

**Goal:** address every reviewer concern *without* under-hedging the contribution. The diagnosis is **mis-aimed claims, not a weak paper.** For each issue we choose **FIX** (kill it with a cheap experiment), **DEFEND** (reviewer/abstract is right, body already supports it), **REFRAME** (re-aim the claim at the axis we genuinely own), or **CONCEDE** (acknowledge — but only where the concession *creates* a niche). The default is: re-aim the headline so we claim *less of what we can't defend and more of what we own.*

---

## The re-aim thesis (one paragraph)

AEGIS's real, defensible contribution is **a single matrix-free object that yields three first-order vulnerability diagnostics at N=7,650 scale, plus a proved safety boundary with empirical margin.** It is *not* a state-of-the-art attacker, and it does not need to be. Every overclaim flagged by the panel comes from borrowing strength from an axis we don't own (raw attack damage, exact-LODF parity, fully-general ReLU proof). Drop those borrowed claims, foreground the four axes we *do* own, and the contribution is intact — sharper, even.

**Four axes we own (claim these confidently — do NOT hedge):**
1. **One-query / cost-normalized.** One matrix-free construction recovers the leading first-order direction a 50-step iterative attacker finds, at a fraction of the cost. (Genuine, by construction.)
2. **Proved safety boundary.** `ε_crit=(1−κ)/‖W‖₂` is a *sufficient* boundary that trained models satisfy with **2–4× margin** — a correct, useful result (R1 confirmed the resolvent bound).
3. **Model-agnostic structural ranking.** Binary-edge ranking beats admittance-based on the grid (P@10 **0.81 vs 0.27**, case118) without the physical model — the non-obvious result R3 praised.
4. **Matrix-free scale.** N=7,650 at 365 s / 5.5 GB, σ₁ matched to dense within **0.03%** — real systems contribution.

---

## Triage table

| # | Concern (reviewer) | Call | Action in one line | Claim you KEEP |
|---|--------------------|------|--------------------|----------------|
| C1 | Attacker headline circular (DA, R2) | **REFRAME + FIX** | Re-aim from "beats attackers" → "one-query diagnostic recovers the 50-step direction"; add cost-normalized axis + foreground breach-rate | "1/50th-cost recovery of the leading vulnerability direction" |
| C2 | Thm 1 over-scoped to ReLU (R1, DA) | **REFRAME (+ optional FIX)** | State exactly what's proved (boundary for 1-Lipschitz; sharp rate for all-active); ReLU as bounded empirical extension (η≤2.47). Optionally *prove* an η bound | A proved safety boundary + bounded ReLU slack |
| 3 | `L_J` finiteness bound can divide by ≤0 (R1) | **FIX** | Re-derive via `1/(1−κ)` using A3 only | Theorem becomes *more* correct |
| 4 | `τ=+0.996` global-vs-pairwise (R1, DA) | **REFRAME + FIX(debug)** | Theory = magnitude + sufficient pairwise order; report τ distribution; *debug the 4/33 failing cells* | "up to +0.996; 29/33 positive at p<10⁻⁵" + an *explanation* of the exceptions |
| 5 | "competitive with industry LODF" (R3) | **REFRAME + CONCEDE** | Lead with binary-beats-admittance; concede LODF is exact/faster — which *creates* the model-agnostic niche | Model-agnostic recovery without the admittance matrix |
| 6 | drug/fraud invoked, never tested (R3) | **FIX (preferred)** | Add ONE safety-relevant graph (molecular/fraud) to the transfer table; else reframe as motivation | Keep the safety-critical framing — earned |
| 7 | Statistics hygiene (R1, R2) | **FIX + DEFEND** | Holm/BH correction (strong cells survive trivially), label SD/SE, cite HMT rSVD bound; *define* "one-query" once and defend the term | "one-query" stays — now precisely defined |
| 8 | Originality = consolidation (EIC, DA) | **DEFEND** | Claim the *operationalization* axis confidently; trim only overreaching words ("spectrum") | A real method+systems contribution, unapologized-for |
| 9 | Numeric inconsistencies (R2, R3) | **FIX** | Reconcile AGNNCert cell, `r_cert/r_v` range, weighted-vs-binary, the two abstract τ's | Credibility protected |

---

## Item-by-item strategy

### C1 — Attacker headline (the highest-leverage move)
**Don't claim "we beat 50-step PGD."** You don't own raw-damage, and DA proved GR-BCD beats you at fair budgets (Cora k=5: 1.207 vs 0.643). Borrowing that claim is what invites the critical.
- **REFRAME the axis** from raw damage → **diagnostic-per-query**: "A single matrix-free query recovers the leading first-order direction that a 50-step iterative attacker converges to, at ~1/50th the cost and without labels." This is true *by construction* and is your actual value proposition.
- **FIX (cheap, bulletproof):** (a) add a **cost-normalized column** (damage per query / per FLOP) where one-query AEGIS dominates honestly; (b) **foreground the breach-rate test** (every breached node has ε>r_v) and report whether r_v predicts *which* nodes flip — converts the "flips are 0–1.8%" objection into evidence the diagnostic *works*.
- **CONCEDE, costlessly:** "AEGIS is a diagnostic, not a maximal attacker; dedicated budget-heavy attackers (GR-BCD) remain stronger on raw damage at large budgets." This sentence *removes* the attack surface without weakening anything you claim.
- **Result:** the critical evaporates and the one-query story gets *sharper*.

### C2 — Theorem 1 scope
**Don't gut it; state it precisely and proudly.** R1 confirmed Thm 1(b)'s resolvent bound is *unconditionally correct*. The only overclaim is the abstract verb implying a full-ReLU proof.
- **REFRAME:** "We prove `ε_crit` is a sufficient safety boundary (1-Lipschitz φ) and characterise three regimes sharply in the all-active case; trained ReLU models satisfy it empirically with bounded slack η≤2.47 and 2–4× margin." Still a *proved theorem* — and a margin result is a selling point.
- **Optional FIX (bulletproof, if time):** prove a *computable bound* on η via the activation-pattern nonnormality κ(V_W) (Rem `eta_relu` already hints η tracks κ(V_W)). Turning "empirical η" into "η ≤ [computable]" upgrades the empirical extension toward rigor and lets you claim more.

### 3 — `L_J` bound: pure FIX. Re-derive with `1/(1−κ)` (A3 only). Strictly strengthens the theorem.

### 4 — Transfer / τ
- **REFRAME** the *theory* claim to what Prop. transfer proves (magnitude bound + a pairwise-sufficient order holding for 47–62% of pairs) — keep the **empirical** τ as a strong empirical result.
- **FIX (debug, per your protocol):** the 4/33 non-positive cells (GCN-2 τ=−0.04) are almost certainly a *mechanism*, not noise — diagnose (normalization sign? assumption violation? near-zero true variance?). An explained exception is *stronger* than a buried one and lets you keep the headline honestly.
- **Report:** "transfer is positive in 29/33 architecture–dataset cells (p<10⁻⁵ after Holm correction), up to τ=+0.996 (Amazon Photo); the [N] exceptions occur when [mechanism]." Keeps the number, owns the exceptions.

### 5 — LODF
- **REFRAME + CONCEDE that builds the niche:** "LODF/PTDF are exact and faster *when the admittance matrix is available.* AEGIS recovers contingency rankings **model-agnostically from a learned surrogate** — binary-edge ranking beats admittance-based at P@10 0.81 vs 0.27 (case118)." Lead with the binary result; the LODF concession *defines when AEGIS is the right tool* (proprietary/unavailable physical model, learned-surrogate-only settings).

### 6 — Safety-critical motivation
- **FIX (preferred, bulletproof):** add **one** safety-relevant graph (e.g., OGB molecular, or a fraud graph) to the cross-domain transfer table. One row earns the entire drug/fraud framing and strengthens the cross-domain claim. This is the classic "run the experiment to keep the hook" — better than softening the intro.
- Fallback **REFRAME** only if infeasible: explicitly mark drug/fraud as motivating instances of safety-critical graph ML, with citation/co-purchase/power as the evaluated representatives.

### 7 — Statistics + "one-query"
- **FIX:** Holm/BH multiplicity correction (your p<10⁻⁵ cells survive correction with room to spare — so this *adds* credibility at zero narrative cost); label every `±` as SD or SE; cite Halko–Martinsson–Tropp for the rSVD error or state it empirically.
- **DEFEND "one-query":** define it once — "one matrix-free *diagnostic construction* (a single randomized SVD over the resolvent, ~600 JVPs), not three separate analyses." The term is legitimate; the body already says this. Keep it.

### 8 — Originality: **DEFEND.** The consolidation/operationalization *is* the contribution — no prior single object yields all three diagnostics matrix-free at this scale. Say so plainly. Trim only words that reach for conceptual novelty you didn't claim ("spectrum" → keep only if defined). Do **not** apologize for building on IFT — every method builds on something.

### 9 — Inconsistencies: **FIX.** Reconcile the AGNNCert cell (0.187 vs 0.163), `r_cert/r_v` ([4.4,15.0] vs [4.9,10.2]), weighted-vs-binary transfer, and the two abstract τ's (label them: structural-edge τ vs power-N-1 τ — different objects). These are credibility leaks; fixing them *protects* the strong claims.

---

## Re-aimed headline claims (before → after) — proof you lose nothing

| Before (flagged) | After (honest + strong) |
|---|---|
| "the one-query SVD direction matches 50-step PGD attackers" | "a single matrix-free query recovers the leading first-order vulnerability direction at ~1/50th the cost of a 50-step iterative attacker, without labels" |
| "we prove a closed-form three-regime characterisation" | "we prove a sufficient safety boundary `ε_crit` (with 2–4× empirical margin) and characterise three regimes — sharply for all-active networks, empirically (η≤2.47) for trained ReLU models" |
| "competitive with industry LODF" | "recovers N-1 contingency rankings model-agnostically from a learned surrogate, beating admittance-based ranking (P@10 0.81 vs 0.27) without access to the physical model" |
| "reaches τ=+0.996" (as typical) | "transfer is positive in 29/33 cells (p<10⁻⁵, corrected), up to τ=+0.996 on full-graph Amazon Photo (N=7,650)" |

Every "after" is *defensible against the Devil's Advocate* and still strong. Nothing here undersells.

---

## Cheap bulletproofing experiments (prioritized — "run, don't hedge")
1. **Cost-normalized attack table** (damage per query/FLOP) — closes C1 on an axis you win. *(hours)*
2. **Debug the 4/33 failing transfer cells** — explain the mechanism. *(hours–1 day)*
3. **Holm/BH correction** on the 33-cell transfer p-values — pure credibility gain. *(minutes)*
4. **One safety-relevant dataset row** (molecular/fraud) — earns the safety framing. *(1–2 days, optional)*
5. **(Optional, ambitious)** computable η bound via κ(V_W) — upgrades Thm 1's ReLU extension toward rigor. *(uncertain)*

## What NOT to hedge (defend these to the death)
- The one-query / matrix-free / N=7,650 scaling result.
- `ε_crit` as a proved sufficient boundary with margin.
- The binary-beats-admittance grid finding.
- The consolidation/operationalization as a genuine contribution.

> Net effect: ~3 cheap experiments + precise re-aiming of ~5 sentences. The paper concedes only what it never owned, and claims everything it does — a stronger, reviewer-proof submission, not a hedged one.
