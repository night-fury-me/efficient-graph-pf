# Adversarial Review of the Proposed Fix — `thm:phase_transition` regime (a), `thm:cf2s` lower side, framing relabels

**Date:** 2026-06-05
**Target:** `09_corrected_theory_draft.md` (PATCHES 1–5 + T1.3 note).
**Method:** Each fix claim re-derived from scratch and stress-tested with rng-seeded numerics
(seeds 20260605, 101, 7, 3141), independent of the draft's own framing. The fix is treated as a
hostile submission. Prior context: `07_theory_verification_ecrit.md` (the `ε_crit` gap + the
triangle-bound refutation) and `08_theory_verification_cf2s_zstar.md` (bracket sub-claims, `‖z*‖`).

**Bottom line up front:** the fix closes the blocking soundness gap (the `ε_glob` global certificate
is genuinely sound and mask-agnostic; the `ε_crit` relabel is honest; `ε_glob ≤ ε_crit` is correct).
**One sentence in PATCH 2 re-introduces a softer overclaim** — "the conservative IFT continues the
equilibrium uniquely for `ε<ε_crit`" is asserted as a guarantee but is only proved *locally* (per
region), and the global-to-`ε_crit` version is exactly the open gap `rem:obs_o1` itself flags. That
clause must be reworded. Everything else is PASS. **Ship after the item-2 one-line edit.**

---

## ITEM 1 — Is `ε_glob` a sound GLOBAL contraction radius? → **PASS**

**Claim.** For `ε<ε_glob := max(0, 1/‖W‖₂−‖Â‖₂)`, `F'(z)=φ(Â'zWᵀ+X_proj)` is a global contraction
in Frobenius norm for *every* activation pattern, hence a unique fixed point by Banach. The bound
`‖J_z'‖₂ ≤ ‖Â'‖₂‖W‖₂` is mask-independent.

**Verification.** The mechanism is correct and is the one finding 07 already isolated as the *only*
clean certificate: `J_z'=M'(Â'⊗W)` with `M'=diag(φ')` a 0/1 projection, so
`‖J_z'‖₂ = ‖M'(Â'⊗W)‖₂ ≤ ‖Â'⊗W‖₂ = ‖Â'‖₂‖W‖₂` **unconditionally** — submultiplicativity of the
spectral norm under a contraction (projection) on the left needs no assumption on `M'`. Then
`‖Â'‖₂ ≤ ‖Â‖₂+‖δÂ‖₂ ≤ ‖Â‖₂+‖δÂ‖_F = ‖Â‖₂+ε`, so `‖J_z'‖₂ < 1` whenever `ε < 1/‖W‖₂−‖Â‖₂ = ε_glob`.
This is a genuine *global* Lipschitz-`<1` map (the Lipschitz constant of `z↦φ(Â'zWᵀ+X_proj)` is
`≤‖Â'‖₂‖W‖₂` because `φ` is 1-Lipschitz and the linear part has operator norm `‖Â'‖₂‖W‖₂`), so Banach
gives a unique fixed point from *any* initialization — no per-region/IFT argument, no mask assumption.

**Numerics.**
- `‖Â‖₂ = 1.000000` exactly for the symmetric-normalized adjacency with self-loops, 200/200 graphs
  (`λ_max=1` is the Perron value of `D^{-1/2}(A+I)D^{-1/2}`), so `ε_glob = 1/‖W‖₂ − 1 = 1/c − 1` as
  the draft states.
- Adversarial-mask search inside `ε_glob` (4000 models, worst over all-active + 8 random masks per
  point): **0/4000** cases with `‖J_z'‖₂ ≥ 1`; worst `(‖J_z'‖₂ − (‖Â‖₂+ε)‖W‖₂) = −2.0e−5 ≤ 0`. The
  mask-agnostic bound is never violated.

**Corrected Step 2 is free of the original defect.** The original Step 2 fused this same algebra with
the *false* identity `1/‖W‖₂−‖Â‖₂ = (1−κ)/‖W‖₂` (which silently needs A4). PATCH 2 keeps the algebra,
correctly names its radius `ε_glob`, and stops there. **PASS, no edit.**

---

## ITEM 2 — THE KEY PROBE: does PATCH 2 re-introduce a softer overclaim? → **NEEDS-EDIT** (confirmed)

**The sentence under test (PATCH 2, last clause).**
> "The conservative IFT [bolte2021conservative] (one linear region at a time) continues the
> equilibrium uniquely for `ε<ε_crit` with the first-order bound of Step 1, and `ε_glob` supplies
> the stricter global guarantee."

**Why this is the same optimism in milder form.** The conservative-IFT machinery
(`bolte2021conservative`) the paper cites supplies continuation **on each linear region** — this is
exactly how the current source already phrases it (`A_preliminaries.tex:96` "an IFT on each linear
region"; `B_sensitivity.tex:30` "stable for small `‖δÂ‖` … measure-zero faces"; `theory.tex:12` "on
each linear region"). That is a **local** statement: a unique branch exists in *some* neighborhood of
the current `Â`, and persists as long as the active set is locally constant. It does **not** certify
that the branch stays unique across the **finitely many region crossings** that occur as `ε` grows to
the `O(1)` budget `ε_crit`. The draft's own PATCH-2 paragraph *concedes the equilibrium "crosses
finitely many ReLU regions before `ε_crit`"* two sentences earlier — and then asserts unique
continuation all the way to `ε_crit` anyway. Internally inconsistent: if region-crossing is what kills
the mask-fixed triangle bound (finding 07, and the draft says so), the *same* region-crossing is
uncertified for the IFT branch. The branch can in principle fail before `ε_crit` two ways the local
IFT does not exclude: (i) the true (mask-changed) `J_z'` at the continued equilibrium reaches `ρ=1`
along a newly-activated mode, or (ii) the branch bifurcates at a region boundary (two equilibria for
the perturbed `Â`).

**Decisive point — this is `rem:obs_o1`'s open gap, restated.** "A linearized eigenvalue reaching 1
⇒ the true nonlinear fixed point persists/destabilizes" is *precisely* the step `rem:obs_o1` labels
**open proof gap (2), "linear-to-nonlinear bifurcation (empirical only)"**, and the scaling of the
masked Jacobian's dominant eigenvalue is **open proof gap (1), "masked-operator spectral scaling
(unproved)."** PATCH 2's "continues uniquely for `ε<ε_crit`" asserts as a *guarantee* the very
implication the appendix elsewhere flags as unproved. A hostile reviewer who reads `rem:obs_o1` and
then PATCH 2 will catch the contradiction immediately.

**Numerics — the claim is empirically robust but NOT proved.** I hill-climbed the perturbation
direction to break the *continued* branch (start the ReLU iteration from the clean equilibrium `z*`,
push `ε → 0.99·ε_crit`), over 60 strict-partial models, ~30k continued-branch evaluations in the open
`(ε_glob, ε_crit)`:
- worst true `ρ(J_z')` on the continued branch = **0.8235** (`<1` everywhere);
- worst true `‖J_z'‖₂` on the continued branch = **0.9425** (`<1` everywhere);
- branch continuation failed to converge: **0** points; branch split (continued ≠ global re-solve):
  **0** points.

So *following the equilibrium by continuation*, the realized mask keeps `‖J_z'‖` tracking the
within-region linearization `κ+ε‖W‖` (which hits 1 only at `ε_crit`), not the adversarial
mask-injected worst case. **This explains why the claim is *true in practice* and why it is *not a
theorem*:** the favorable behavior is a property of the continued mask, which is governed by the
fixed-point equation, not by anything `κ` or the IFT certifies. Contrast the *free-mask* adversarial
test (finding 07, reproduced here: triangle bound `κ+ε‖W‖` violated **1909/4000**, injection up to
**+0.37**) — an arbitrary `M'` *does* push `‖J_z'‖` over the line well inside `ε_crit`; the continued
equilibrium's mask simply does not realize that worst case. Nothing in the proof rules it out; the
data does.

**Verdict.** PATCH 2's `ε_glob` half is a genuine guarantee (item 1). The `ε_crit` half is honestly
*weaker* than the original (it no longer claims contraction/safety on `(ε_glob, ε_crit)`, only unique
continuation), and it is empirically bulletproof — but **"continues uniquely for `ε<ε_crit`" is
asserted, not proved**, the only proof tool cited (`bolte2021conservative`) licenses the *local*
version only, and the *global*-to-`ε_crit` version is `rem:obs_o1`'s named open gap. This is exactly
the residual optimism the brief suspected: a linearized persistence claim dressed as a guarantee, in
milder form. **NEEDS-EDIT.**

**Strongest defensible persistence statement / corrected wording.** Replace the PATCH-2 last sentence
with:
> "On each linear region the conservative IFT [bolte2021conservative] continues the equilibrium
> uniquely with the first-order bound of Step 1; this local continuation persists while the active set
> is constant. For the global, region-independent guarantee we rely on `ε_glob`: for `ε<ε_glob` the
> map is a Frobenius-norm contraction under every activation pattern, so Banach gives a single
> equilibrium reached from any initialization. Whether the continued branch remains the *unique*
> equilibrium across the finitely many region crossings up to the larger linearized scale `ε_crit` is
> not certified by the contraction argument — it is the empirical regularity of `rem:obs_o1` (the
> measured break sits at `ε_reach>ε_crit`), whose two proof gaps (masked-operator spectral scaling;
> linear-to-nonlinear bifurcation) remain open."

This keeps `ε_crit` as the linearized scale that drives regime (b), keeps `ε_glob` as the only
*global* guarantee, and routes the "to `ε_crit`" persistence to where the paper already admits it is a
conjecture — removing the contradiction. (Optionally add one sentence: "Empirically the continued
branch stays the unique contraction up to `ε_crit` across our suite; we present this as observation,
not theorem.")

---

## ITEM 3 — Is the `ε_crit` relabel honest and the regime picture consistent? → **PASS**

**3a. `(1−κ)/‖W‖` as the linearized critical budget.** Correct. On a *fixed* activation region,
`J_z'=J_z+M(δÂ⊗W)` with `M` the (locally constant) clean-equilibrium mask, so
`‖J_z'‖₂ ≤ κ + ‖M(δÂ⊗W)‖₂ ≤ κ + ε‖W‖₂`, and the within-region first-order factor `κ+ε‖W‖₂` reaches
1 exactly at `ε=(1−κ)/‖W‖₂=ε_crit`. This is the budget at which the *linearization on the home region*
loses contraction, which is precisely what drives the regime-(b) resolvent divergence (Step 3's
`min_i|1−λ_i| = ‖W‖₂(ε_crit−ε)` uses this within-region linearization, and the draft's Step-3 fix
correctly relabels that equality as "the within-region linearization," not the mask-agnostic bound).
The relabel "linearized critical budget" is accurate.

**3b. `ε_glob ≤ ε_crit` always.** Proved and verified. `ε_glob ≤ ε_crit ⟺ 1/‖W‖₂−‖Â‖₂ ≤ (1−κ)/‖W‖₂
⟺ 1 − ‖Â‖₂‖W‖₂ ≤ 1 − κ ⟺ κ ≤ ‖Â‖₂‖W‖₂`, which is the unconditional Kronecker-submultiplicativity fact
`κ=‖M(Â⊗W)‖₂ ≤ ‖Â‖₂‖W‖₂` (A3/A1). Numerics: **0/3000** violations across partial-ReLU masks (2736 of
them strict-partial); `(ε_crit−ε_glob)` ranges `[0, 0.60]`, the `0` floor being the all-active equality
case. The draft's inline justification (eq:budgets, "with equality only when every unit is active
(A4)") is exactly right.

**3c. Regime (b) relabel consistent with `rem:obs_o1`.** The appended clause "this is a first-order
rate, and the true nonlinear divergence sits at `ε_reach>ε_crit` (`rem:obs_o1`)" is consistent:
`rem:obs_o1` gives `ε_reach/ε_crit = 2.17–8.72`, so `ε_reach>ε_crit`, and labels it empirical. No
overclaim. **PASS.** (Self-consistency bonus: with the item-2 reword, regime (b)'s "first-order rate"
and PATCH 2's "linearized scale, not certified to `ε_crit` globally" now tell the *same* story —
`ε_crit` is everywhere the linearized object, `ε_glob` the only global one, `ε_reach` the empirical
break. Coherent.)

---

## ITEM 4 — PATCH 3, bracket lower side → **PASS**

**Claim.** `ε_br ≥ ε_glob` under (A1)–(A3); under (A4) `ε_glob=ε_crit` so `ε_br^all ≥ ε_crit`; the
bracket `ε_crit ≤ ε_br^all ≤ (C/β)ε_crit` survives within its all-active scope.

**Verification.** PATCH 3 replaces the false "`<1` for `ε<ε_crit`" with "`<1` uniformly over masks for
`ε<ε_glob`," concluding `ε_br ≥ ε_glob` under (A1)–(A3) — sound by item 1 (mask-agnostic). It then
notes that the bracket theorem *already* assumes (A4) (the upper side (ii) needs the all-active
spectral law `ρ(J_z')=ρ(Â')ρ(W)`, confirmed in `D_boundary.tex` (ii); `thm:cf2s` is even titled
"all-active"). Under (A4), `κ=‖Â‖₂‖W‖₂` so `ε_glob=ε_crit` and `ε_br^all ≥ ε_crit` is recovered
*exactly* — no weakening. This matches finding 08 T1.1 ("the bracket structure survives the fix; one
fix closes both"). The `ε_crit ≤ ε_br^all` lower endpoint and the `≤ (C/β)ε_crit` upper endpoint are
both intact within the all-active scope the theorem already declares. **PASS.**

*One nicety to confirm in the actual edit (not a fault of the patch text):* the live theorem statement
`thm:cf2s_full` (i) (`D_boundary.tex:73–76`) currently reads "`ε_br ≥ ε_crit`, in particular
`ε_br^all ≥ ε_crit`" under "(any 1-Lipschitz φ; no (A4))." After PATCH 3 the honest in-statement
claim for the *no-A4* part is `ε_br ≥ ε_glob`; `ε_br ≥ ε_crit` is the *A4* corollary. When applying,
make sure the **theorem statement** (i), not only the proof, carries `ε_glob` for the no-A4 clause and
`ε_crit` only under A4 — otherwise the statement still over-reads. The patch's parenthetical handles
this, but the edit must touch `thm:cf2s_full` (i)'s statement line, not just the proof body.

---

## ITEM 5 — PATCHES 4–5, framing → **PASS**

**PATCH 4 (abstract).** OLD "a deterministic guarantee the empirical break exceeds by 2–9×" → NEW "a
deterministic certificate, which the *measured* nonlinear break exceeds by 2–9× (10 seeds)." This is
exactly finding 08 T1.2's recommended fix: the word "deterministic certificate/guarantee" now attaches
to the *radius* (which the certificate earns), and the 2–9× number is explicitly "measured" — pointing
at `ε_reach/ε_crit` (`rem:obs_o1`, conjecture), not at the bracket. Accurate. The bracket's own proven
`10–16×` stays out of the abstract (correct). **PASS.**

**PATCH 5 (intro contribution 2).** OLD "under-states the true break by 2–9×" → NEW "under-states the
*measured* nonlinear break by 2–9× (10 seeds; the all-active boundary itself is bracketed in closed
form, `thm:cf2s`)." Correctly: (a) "measured" tags 2–9× as empirical `ε_reach`; (b) the closed-form
bracket is attributed to the *all-active boundary* (`ε_br^all`), its true scope, not to the nonlinear
break. So "measured nonlinear break" (= `ε_reach`, empirical) and "deterministic certificate" (=
`ε_glob`/`ε_crit` radius + the all-active bracket, proven) now describe what is proven vs. empirical
correctly. **PASS.**

*Cross-check with item 2's reword:* PATCH 4/5 say the closed-form object is "deterministic." With the
item-2 fix, the deterministic radius is `ε_glob` (and the all-active bracket); `ε_crit` is the
linearized scale. The abstract/intro do not claim `ε_crit` is a *global* safe radius (they say "safe
radius" generically and route the number to "measured"), so no inconsistency is introduced. If the
author wants maximal safety, the abstract's "closed-form safe radius" could read "closed-form safe
radius (`ε_glob`)" — optional, not required.

---

## ITEM 6 — New error, circularity, or vacuity? Is two-budget presentation coherent? → **PASS** (with the item-2 caveat)

**Two budgets coherent?** Yes, and it is the *right* structure: `ε_glob` (global, mask-agnostic,
A1+A2+threat-bound only) is the rigorous contraction certificate; `ε_crit=(1−κ)/‖W‖₂ ≥ ε_glob` is the
linearized scale that (i) drives the regime-(b) divergence rate and (ii) is the lower bracket endpoint
*under A4* (where the two coincide). No circularity: `ε_glob` does not use `κ`; `ε_crit` uses `κ` for
the within-region rate, not for the global certificate. No new vacuity: `ε_glob = 1/c−1 > 0` for `c<1`
(e.g. `≈0.11` at `c=0.9`), non-trivial; `ε_glob ≤ ε_crit` keeps the bracket non-empty.

**Does anything the originals rely on `ε_crit` break?** Checked the downstream consumers:
- `prop:radius` / `r_v`, `prop:transfer` remainder, `‖z*‖`, AEGIS-Conformal `C_v`/coverage — all use
  Step 1 (the first-order shift `σ_1(S)`, resolvent at the *clean* equilibrium), **not** Step 2's
  radius. Finding 08 T2 already confirmed `‖z*‖ ≤ ‖X_proj‖/(1−κ)` is sound at the fixed point with the
  *local* `κ` (resolvent identity), independent of the perturbation-ball question. The draft's "Net
  effect / Untouched" list is correct: these are not disturbed.
- `tab:cross_domain` keeps its per-dataset `ε_crit` column (caption relabel only). Honest, since
  `ε_crit` remains a well-defined per-model linearized scale.

**Is there a cleaner correct-er fix?** The alternative (Option A: drop `ε_crit`, use only `ε_glob`)
is *more* conservative and fully avoids the item-2 issue, at the cost of a dataset-independent column
(`1/c−1`). Option B (this draft) is the minimal honest correction that *keeps the paper's features* —
acceptable, **provided the item-2 clause is reworded** so that no global persistence is claimed to
`ε_crit`. With that edit, the two-budget presentation is the cleanest version that retains the
per-dataset table. No hidden new error beyond item 2.

---

## Per-item verdicts

| Item | Subject | Verdict |
|---|---|---|
| 1 | `ε_glob` global mask-agnostic contraction radius | **PASS** |
| 2 | PATCH 2 "IFT continues uniquely for `ε<ε_crit`" | **NEEDS-EDIT** (residual overclaim; reword) |
| 3 | `ε_crit` relabel + `ε_glob≤ε_crit` + regime (b) | **PASS** |
| 4 | PATCH 3 bracket lower side | **PASS** (apply to the *statement* line, not just proof) |
| 5 | PATCHES 4–5 framing relabels | **PASS** |
| 6 | new error / coherence of two budgets | **PASS** (conditional on item 2) |

---

## Bottom line — is the draft sound and ready to apply?

**Almost.** The blocking soundness gap (the original `κ`-vs-`‖Â‖‖W‖` conflation) is genuinely closed:
`ε_glob` is a real, mask-agnostic, global contraction certificate, proved cleanly and verified with
zero adversarial violations. The relabels are honest and the framing patches accurately separate
proven (`ε_glob`, all-active bracket) from empirical (`ε_reach`, 2–9×).

**The single thing that must change before applying** is one clause in PATCH 2:
> "The conservative IFT … continues the equilibrium uniquely for `ε<ε_crit` …"

is an unproved global-persistence guarantee — the cited IFT licenses only the *local*/per-region
version, and the all-the-way-to-`ε_crit` version is exactly `rem:obs_o1`'s named open gap (and
contradicts the draft's own preceding sentence about region-crossing). It is empirically robust (0/30k
adversarial continued-branch breaks, 0 splits, 0 non-existence) but it is *not a theorem*. Replace it
with the corrected wording in item 2 (local IFT continuation + `ε_glob` for the global guarantee +
"to-`ε_crit` persistence is the empirical `rem:obs_o1` regularity, two open gaps"). Also ensure PATCH 3
edits the `thm:cf2s_full` (i) *statement* line so the no-A4 clause reads `ε_br ≥ ε_glob` (with `ε_crit`
only under A4), not just the proof body.

**Ship call: NO-SHIP as written; SHIP after the item-2 reword (one sentence) + the item-4 statement-line
check.** Both are mechanical, zero new math, and do not touch any downstream result. With them, the
subcritical certificate is sound, the bracket survives within its declared all-active scope, and the
front matter is honest.
