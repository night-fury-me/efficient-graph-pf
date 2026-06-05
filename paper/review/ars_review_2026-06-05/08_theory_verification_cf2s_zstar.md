# Theory Verification — `thm:cf2s` bracket + `‖z*‖` finiteness bound

**Date:** 2026-06-05
**Scope:** Independent adversarial verification of two suspected "local-κ vs global-spectral"
conflations: (T1) the two-sided bracket `thm:cf2s` / `thm:cf2s_full`, and (T2) the
`‖z*‖ ≤ ‖X_proj‖/(1−κ)` bound feeding the curvature constants `L_J`, `C_v`.
Verified line-by-line against primary LaTeX (`theory.tex`, `appendix/D_boundary.tex`,
`appendix/C_rankings.tex`, `appendix/E_conformal.tex`, `appendix/A_preliminaries.tex`,
`background.tex`, `abstract.tex`, `introduction.tex`) plus rng-seeded numerics
(seeds 20260605, 7, 101, 2718, 3141). This pass reaches its own verdict and **disagrees
with the suspected hypothesis on T2**, which the math does not support.

Companion to `07_theory_verification_ecrit.md` (the confirmed `ε_crit` subcritical gap).

---

## TARGET 1 — the bracket theorem `thm:cf2s` / `thm:cf2s_full`

The bracket is `ε_crit ≤ ε_br^all ≤ (C/β)·ε_crit`, `C = g_W(1+κ)/(1−κ)`, proved in
`D_boundary.tex` parts (i)–(iv). The **core algebra was already verified sound** in a prior
pass (R6, 0 violations / 6000) and re-confirmed here; the live questions are the three
sub-claims about inheritance, scoping, and constant-consistency.

### Sub-claim 1 — Lower side (i) inherits the EXACT `ε_crit` gap (CONFIRM; proof gap)

**Claim under test:** does part (i) actually prove only `ε_br ≥ ε_suff = 1/‖W‖₂ − ‖Â‖₂`,
with "= ε_crit" silently requiring (A4)?

**Analysis.** The proof of (i) (D_boundary.tex l.114–119) writes, citing `eq:Jzp-bound`,
`‖J_z'‖₂ ≤ (‖Â‖₂+ε)‖W‖₂`, "which is below 1 for ε<ε_crit," concluding `ε_br ≥ ε_crit`.
The chain `(‖Â‖₂+ε)‖W‖₂ < 1` is equivalent to `ε < 1/‖W‖₂ − ‖Â‖₂ =: ε_suff`. The asserted
identity `ε_suff = ε_crit = (1−κ)/‖W‖₂` rearranges to `κ = ‖Â‖₂‖W‖₂`, i.e. the all-active
condition (A4). Under the *deployed* partial-ReLU regime `κ < ‖Â‖₂‖W‖₂`, the identity is
false and the written step proves a strictly smaller radius than `ε_crit`.

**Numeric (seed/closed-form):** with `‖Â‖₂=1, ‖W‖₂=0.9`: `ε_suff = 0.111`. All-active
`κ = 0.900 → ε_crit = 0.111` (identity holds). Reported partial `κ = 0.538 → stated
ε_crit = 0.513`, which *exceeds* the proven radius `ε_suff = 0.111` by 0.402. On
`(ε_suff, ε_crit]` part (i) as written proves no contraction.

**Verdict: CONFIRM — genuine proof gap, identical in kind to the `ε_crit` subcritical
defect (`07_...ecrit.md`).** Part (i) re-uses `eq:Jzp-bound` and therefore inherits its
all-active dependence verbatim. The statement `ε_br ≥ ε_crit` (and `ε_br^all ≥ ε_crit`)
is only literally proved for the smaller `ε_suff` unless (A4) is invoked. Note (i) IS
genuinely the fully-nonlinear half (only needs `‖diag(φ')‖₂≤1`); the defect is purely the
`ε_suff = ε_crit` relabeling, not the contraction logic.

**Severity: MAJOR (shared with the `ε_crit` finding — not additive).** This is the *same*
root defect surfacing in a second location; fixing `ε_crit` once (mask-agnostic radius)
fixes (i) automatically. It does not independently raise the paper's risk beyond the
`ε_crit` gap already catalogued.

**Load-bearing:** YES for the lower endpoint as stated, but the corrected (smaller)
endpoint `ε_suff` still gives a non-trivial two-sided bracket, so the *bracket structure*
survives the fix.

**Minimal fix:** state the lower side as `ε_br ≥ ε_suff := max(0, 1/‖W‖₂ − ‖Â‖₂)`, and
either (a) note `ε_suff = ε_crit` holds under (A4), or (b) carry `ε_suff` through the
bracket. One-line change, propagated from the `ε_crit` fix.

### Sub-claim 2 — Upper side (ii) is a sound ALL-ACTIVE statement; the abstract "true break / 2–9×" is the EMPIRICAL `ε_reach`, which the bracket does NOT prove (REFINE; front-matter over-claim, not a proof gap)

**Part (ii) internal soundness (re-checked):** SOUND as an all-active statement. The
convexity step (l.126–128) — `t ↦ λ_max(Â+tB)` is a pointwise max of affine functionals,
hence convex, so it lies above its tangent: `ρ(Â+tB) ≥ ρ(Â)+βt` — is correct (this was the
N1 one-word fix from R6; the file now correctly says "convex," 0/42000 tangent-lower-bound
violations). The β-identity `β = ⟨u₁, Bu₁⟩ = σ_E` (`eq:beta-eq`, D_boundary.tex l.54–59)
is correct via `P_E` orthogonal projection. (ii) correctly bounds **`ε_br^all`** (the
all-active spectral break) and is in-statement scoped "all-active, requires (A4)."

**The real question — what is "the true break"?** Three distinct break budgets exist
(see `tab:notation`, `tab:constants`):
- `ε_br^all` = all-active contraction break — **what the bracket controls**;
- `ε_spec = 1/ρ(W) − ρ(Â)` = all-active spectral break (the bracket's right endpoint driver);
- `ε_reach` = measured *nonlinear* ReLU break — defined in `rem:obs_o1`, which is
  **explicitly labelled a conjecture with two named open gaps** (masked-operator spectral
  scaling; linear→nonlinear bifurcation).

**Classification of the front matter:**
- `theory.tex` l.51: "the norm certificate under-states the **true break** by 2–9× (10
  seeds)" — the "10 seeds" and the value 2–9× match `rem:obs_o1`'s
  `ε_reach/ε_crit ∈ [2.17, 8.72]`, i.e. this is the **empirical `ε_reach/ε_crit`**, NOT
  `ε_br^all`. `tab:constants` confirms: the row "2–4× operating ρ-margin," "**2–9×
  empirical `ε_reach/ε_crit` (10 seeds)** → `rem:obs_o1`," "10–16× proven bracket `C/β`
  → `thm:cf2s`" are listed as **three distinct quantities**.
- `abstract.tex` l.2: "a closed-form safe radius adds a deterministic guarantee the
  **empirical break** exceeds by 2–9×." Uses the word "empirical break" — correctly
  pointing at `ε_reach`, not the bracket.
- `introduction.tex` l.15 (contribution 2): "a constant-factor two-sided characterisation
  of the breaking point … whose norm certificate under-states the **true break** by 2–9×."

**Numeric confirmation:** the bracket `ε_spec ≤ C·ε_crit` is the *all-active* object;
`ε_reach > ε_spec` empirically (`rem:obs_o1`: `ε_reach/ε_spec = 1.41–1.51`). So 2–9× is
strictly downstream of, and larger than, what the bracket proves. The bracket's own proven
constant is `10–16×` (`C/β`), correctly kept **out of the abstract**.

**Verdict: REFINE — front-matter over-claim, NOT a proof gap.** The bracket theorem is
correctly scoped (all-active, in-statement). The defect is rhetorical: the abstract/intro
phrase "**deterministic guarantee** the empirical break exceeds by 2–9×" attaches the word
"deterministic guarantee" (which the *bracket* earns, for `ε_br^all`) to the **2–9× number,
which is the conjecture-backed empirical `ε_reach/ε_crit`** (`rem:obs_o1`, two open gaps).
The two true statements — (a) the bracket deterministically gives `ε_br^all ≤ (C/β)ε_crit ≈
10–16×`, and (b) empirically `ε_reach/ε_crit ≈ 2–9×` (conjecture) — are individually sound
but the front matter fuses the *adjective* of (a) with the *number* of (b).

**Severity: MINOR–MODERATE (hostile theory reviewer).** A referee who reads
`rem:obs_o1` will see the honest "this is a conjecture, two proof gaps remain" box and
the `tab:constants` reconciliation, which *defuses* the charge — the paper does disclose
that 2–9× is empirical. The exposure is that the **abstract**, read alone, presents a
conjecture-derived ratio with the word "guarantee." Severity is held below MAJOR only
because the appendix scoping is explicit and correct.

**Load-bearing:** NO for any sold *soundness* guarantee. `ε_reach` is not used to certify
anything; the certified objects are `ε_crit` (radius) and the conformal coverage. 2–9× is
a *motivational* conservatism figure. Mis-framing it inflates the narrative, not a bound.

**Minimal fix:** in abstract and intro, change "a **deterministic guarantee** the empirical
break exceeds by 2–9×" → "a deterministic safe radius; the **empirically measured** break
exceeds it by 2–9× (10 seeds, `rem:obs_o1`)" — i.e. move the word "empirical/measured"
onto the 2–9× and drop "guarantee" from that clause. Keep the proven `10–16×` bracket
attribution where it already is (appendix only). Zero math change.

### Sub-claim 3 — Constant `C` is evaluated with the DEPLOYED (partial) κ, while its derivation assumes κ=‖Â‖‖W‖ (all-active): internal κ-inconsistency in the NUMBER, but the bound stays valid (REFINE; conservative, not broken)

**Claim under test:** is `C/β ≲ 16` computed with partial κ (inconsistent with its
all-active derivation), and if so does that invalidate the number?

**Derivation audit (iii), D_boundary.tex l.133–137.** The only step using the all-active
identity is `‖Â‖₂ = κ/‖W‖₂` (l.135: "`‖Â‖₂ = κ/‖W‖₂ = κ(ε_crit+‖Â‖₂)` gives
`‖Â‖₂ = κ/(1−κ)·ε_crit`"). This sets `κ := ‖Â‖₂‖W‖₂`. The final constant
`C = g_W(1+κ)/(1−κ)` is therefore **derived with κ = ‖Â‖₂‖W‖₂** (all-active).

**The reported number.** `theory.tex`/`thm:cf2s_full` (iii) reports `C/β ≲ 16` on the
suite, citing `g_W ∈ [1.19, 2.47]` and (via `tab:cross_domain`) `κ ∈ [0.14, 0.59]` — the
**deployed partial** κ. Plugging these:
- partial κ ∈ [0.14, 0.59], g_W ∈ [1.19, 2.47], β=0.62 → **C/β ∈ [2.5, 15.5]** ✓ matches
  reported `≲16`.
- all-active κ (the value C's derivation assumes; `‖Â‖₂≈1` so `κ≈‖W‖₂≈0.85–0.95`),
  g_W=2.47, β=0.62 → **C/β ∈ [49, 155]**.

So the reported `10–16×` is obtained by substituting the **deployed partial κ into a
constant C whose derivation assumed all-active κ**. This is a genuine internal
κ-inconsistency (the same family as the `ε_crit` defect).

**Does it break the bound? NO — it is conservative.** Decisive numeric over 802 *real*
partial-ReLU fixed points (seed 2718): `ε_spec ≤ C(κ_dep)·ε_crit(κ_dep)` had **0
violations**. Reason: ε_crit and C share κ; lowering κ raises `ε_crit=(1−κ)/‖W‖` and lowers
`C=g_W(1+κ)/(1−κ)`, and the product stays `≥ ε_spec`. An honest re-derivation that avoids
the all-active identity, `ε_spec ≤ g_W(ε_crit + ‖Â‖₂^true)`, also had **0 violations** (613
models, seed 3141). So the inequality `ε_spec ≤ C·ε_crit` survives the partial-κ
substitution.

**Verdict: REFINE — internal κ-inconsistency in the reported CONSTANT, but not an unsound
bound.** The number `C/β ≲ 16` mixes a partial-κ evaluation with an all-active-κ
derivation. The *honest* C consistent with its own derivation is much larger
(`C/β ≈ 49–155` at all-active κ). Reporting the smaller partial-κ value **under-states the
worst-case bracket constant** — which is conservative *for the gap claim* ("the certificate
under-states the break by at most C/β") only if read as "the gap is at least ε_crit and at
most ...": a *smaller* reported C/β is an *over-tight* (i.e. potentially anti-conservative)
claim about how loose ε_crit is. But numerically `ε_spec ≤ C(κ_dep)ε_crit(κ_dep)` holds, so
the substituted value is still a valid upper bound on `ε_spec/ε_crit`. Net: the *number* is
internally inconsistent (derivation κ ≠ evaluation κ) but the *inequality it annotates* is
not violated.

**Severity: MINOR (hostile theory reviewer).** A referee who recomputes C from its own
derivation (all-active κ) and gets 49–155× instead of 16× will flag the κ-inconsistency,
and the "10–16×" headline shrinks. But since (a) the bracket inequality still holds with
the deployed κ (0/802), and (b) C is explicitly "a-posteriori computable per model," this
is a reporting/consistency nit, not a soundness break.

**Load-bearing:** NO for soundness; YES for the advertised tightness of the bracket. If
forced to all-active-consistent C, the "10–16×" selling point becomes "49–155×," i.e. the
bracket is looser than advertised — a *tightness* downgrade, not an invalidation.

**Minimal fix:** either (a) state explicitly that C is reported at the *deployed* κ and
note the all-active-consistent value is larger (honest, keeps 10–16× with a caveat), or
(b) re-derive (iii) without the `‖Â‖₂=κ/‖W‖₂` substitution, using the true `‖Â‖₂`:
`C := g_W(‖W‖₂ + ‖Â‖₂‖W‖₂)/(1) · ...` evaluated at deployed quantities — which the
802-model test confirms is valid. Recommend (b): it removes the all-active identity from
(iii) entirely and makes the constant self-consistent at the deployed operating point.

---

## TARGET 2 — the `‖z*‖ ≤ ‖X_proj‖/(1−κ)` bound (`eq:LJ-bound`, `eq:Cvdef`)

**Suspected hypothesis (from the brief):** the Banach bound `‖z*‖ ≤ ‖F(0)‖/(1−L)` needs the
**global** Lipschitz constant `L = ‖Â‖₂‖W‖₂ ≥ κ`, not the **local** `κ = ‖J_z(z*)‖₂`; since
κ ≤ ‖Â‖‖W‖, the paper's `‖X_proj‖/(1−κ)` would **under-estimate** `‖z*‖`, threatening the
soundness of `C_v` and hence robust coverage.

### Independent analysis — the hypothesis is REFUTED.

**The local-κ bound is sound, via a derivation the Banach route does not see.** At a ReLU
fixed point `z* = max(Â z* W^T + X_proj, 0)`, let `M = diag(φ')` be the activation mask
(1 on active pre-activations, 0 elsewhere). On active coordinates the ReLU is the identity,
so `z*` satisfies the **linear** equation exactly; inactive coordinates are 0. In row-major
vec form:

```
vec(z*) = D_M (Â ⊗ W) vec(z*) + D_M vec(X_proj) = J_z vec(z*) + D_M vec(X_proj)
        => vec(z*) = (I − J_z)^{-1} D_M vec(X_proj).
```

Since `D_M` is a 0/1 diagonal (`‖D_M‖₂ ≤ 1`) and `J_z = D_M(Â⊗W)` with `‖J_z‖₂ = κ < 1`,

```
‖z*‖ ≤ ‖(I − J_z)^{-1}‖₂ · ‖D_M vec(X_proj)‖ ≤ (1/(1−κ)) · ‖X_proj‖.
```

This uses the **local** κ and is rigorous. The Banach global-Lipschitz argument is
*sufficient but not necessary*; the fixed-point linear-region identity yields the
local-κ bound directly. The hypothesis inverts the inequality direction: a *smaller*
denominator `(1−κ)` with κ the *operator norm of the actual resolvent generator* is exactly
what the resolvent bound delivers — `‖(I−J_z)^{-1}‖₂ ≤ 1/(1−‖J_z‖₂)` — there is no
under-estimate.

### Numerics (decisive)

- **Identity:** `vec(z*) = (I−J_z)^{-1} D_M vec(X_proj)` held **5909/5909** real partial-ReLU
  fixed points (seed 7, row-major vec). Confirms the derivation mechanism.
- **Bound:** `‖z*‖ ≤ ‖X_proj‖/(1−κ_local)` held **5909/5909** (seed 7) and **0 violations /
  4000** in an independent adversarial counterexample search (seed 20260605). No
  partial-ReLU fixed point with `‖z*‖ > ‖X_proj‖/(1−κ)` exists.
- **The local bound is STRICTLY BETTER, not merely valid:** in **787** of the 5909 models the
  global product `‖Â‖₂‖W‖₂ > 1` (up to 1.184) while `J_z` was still contractive (κ < 0.97).
  There the suspected "correct" global bound `‖X_proj‖/(1−‖Â‖‖W‖)` is **negative / infinite
  (vacuous)**, yet the paper's local-κ bound is finite and holds (e.g. κ=0.846, global=1.184,
  `‖z*‖=3.30 ≤ paper_bound 38.5`). The global denominator is the one that breaks in the
  partial-ReLU regime; the local one is the only finite, valid bound.

### Verdict — REFUTE (no proof gap)

**The `‖z*‖ ≤ ‖X_proj‖/(1−κ)` bound with local κ = ‖J_z‖₂ is CORRECT and sound.** This is the
*opposite* of the `ε_crit` case: there, the local-κ *radius* was unsound because the perturbed
mask M' changes across the ball and κ (the clean-point mask norm) loses control of `‖J_z'‖₂`.
Here, `‖z*‖` is evaluated **at a single fixed point with its own fixed mask M**, the resolvent
`(I−J_z)^{-1}` is exactly the one whose norm is `≤ 1/(1−κ)`, and `‖D_M‖₂ ≤ 1` only helps. No
region-crossing issue arises because there is no perturbation — it is the clean equilibrium.

**Important distinction from prior catalogued findings.** Earlier passes (R2/R3/R4, ICDM
memory) flagged a `‖z*‖` *denominator* defect — but that defect was specifically the form
`‖X_proj‖/(1−‖Â‖₂‖W‖₂)` (the **global** denominator, which goes vacuous when `‖Â‖‖W‖ ≥ 1`
under partial ReLU). The **current** appendix text (`eq:LJ-bound`, C_rankings.tex l.60, and
`eq:Cvdef`, E_conformal.tex l.54–58) writes `‖z*‖ ≤ ‖X_proj‖/(1−κ)` with **κ = ‖J_z‖₂**
(the local/correct form). So the paper has **already adopted the R3/R4 recommended fix**:
the κ-Lipschitz / resolvent form replaced the global-denominator form. The suspected
hypothesis is testing the *old* (already-fixed) defect against the *current* (correct) text.

**Severity: NONE (no defect in current text).** Were the denominator still `(1−‖Â‖‖W‖)` this
would be MAJOR (vacuous under partial ReLU); as written with `(1−κ)` it is sound.

**Load-bearing:** the bound *is* load-bearing for `C_v` → `Δ = L₁ε + C_vε²` (`lem:score-shift`)
→ the inflated calibration threshold in `thm:robust-cov`, and for `R_k` in `prop:transfer`.
**Because the bound is sound (and tight via the resolvent), the curvature `C_v` is NOT
under-estimated**, so the threshold inflation is adequate and robust coverage is not
threatened on this account. `rem:conf-caveats` (κ≈0.68 on the conformal subgraph →
`(1−κ)^{-2}≈9.8`) correctly notes the bound *inflates* C_v (makes the certificate more
conservative), and the empirical gate (0.92–0.98 at ε=0.05, 0 breaches) is consistent with
an over-, not under-, estimate.

**Minimal fix:** none required for soundness. Optional rigor polish: add the one-line
justification actually used — "`z*` solves the linear fixed-point equation on its active set,
so `vec(z*) = (I−J_z)^{-1}D_M vec(X_proj)` with `‖D_M‖₂≤1`, giving `‖z*‖ ≤ ‖X_proj‖/(1−κ)`" —
in place of the current terse "the κ-contraction of F at its fixed point (A3)" (C_rankings.tex
l.63), which under-explains *why* the local κ suffices and invites exactly the reviewer
hypothesis tested here.

---

## Cross-target summary

| Target / sub-claim | Verdict | Type | Severity | Threatens a sold guarantee's soundness? |
|---|---|---|---|---|
| T1.1 lower side (i) inheritance | CONFIRM | proof gap (= `ε_crit` defect) | MAJOR (shared, not additive) | The `ε_br ≥ ε_crit` endpoint, fixed by the `ε_crit` repair |
| T1.2 "true break / 2–9×" framing | REFINE | front-matter over-claim | MINOR–MODERATE | NO — `ε_reach` certifies nothing |
| T1.3 constant C (partial vs all-active κ) | REFINE | internal κ-inconsistency, bound still valid | MINOR | NO — bracket holds (0/802); tightness only |
| T2 `‖z*‖ ≤ ‖X_proj‖/(1−κ)` | **REFUTE** | hypothesis wrong; current text sound | NONE | NO — bound sound, C_v not under-estimated |

**Which threaten a sold guarantee's SOUNDNESS:** none newly. T1.1 is the *same* defect as
the already-catalogued `ε_crit` subcritical gap (one fix closes both); it is MAJOR but not a
new front. T1.2 and T1.3 are framing/tightness. **T2 is REFUTED** — the local-κ `‖z*‖` bound
is correct (the paper already adopted the prior-recommended fix), so the AEGIS-Conformal
curvature term `C_v` is *not* under-estimated and robust coverage is not threatened on this
account.

**Pattern note for memory:** the "local-κ vs global-spectral" root cause does **not** repeat
uniformly. It is a *genuine* defect when a κ-based quantity must hold over a **perturbation
ball** where the activation mask changes (the `ε_crit` radius; `thm:cf2s` lower endpoint;
`thm:cf2s` upper-side scoping to A4). It is **not** a defect for a quantity evaluated at a
**single fixed point with a fixed mask** (the `‖z*‖` bound), where the resolvent identity
makes the local κ exactly the right constant. Test region-crossing vs single-point before
assuming the pattern repeats.
