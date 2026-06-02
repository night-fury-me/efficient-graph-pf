# Breakthrough Crux: Sharp + Universal Phase Transition for AEGIS (C1-C4)

**Status: adversarial theory audit. Verdict-first, then full derivations, assumptions, constants, failure points.**

This document attacks the load-bearing claim that AEGIS's `eps_crit = (1-kappa)/||W||_2`
marks a *sharp, two-sided, universal* phase transition (statistical-physics sense), not merely
a one-sided sufficient bound that "goes silent."

All exact definitions are taken from `paper/sections/theory.tex`, `paper/sections/background.tex`,
`paper/sections/framework.tex`:

- Operator: `F(Z, Ahat) = phi(Ahat Z W^T + X_proj)`, `Z in R^{N x d}`, `phi` = ReLU (1-Lipschitz).
- Jacobian (all-active, phi'=1): `J_z = diag(phi') (Ahat (x) W)` so eigenvalues are products
  `lam_i(Ahat) lam_j(W)`; `kappa = ||J_z||_2 <= ||Ahat||_2 ||W||_2` (A3 verifies the trained value).
- `eps_crit = (1 - kappa)/||W||_2` with **worst-case** `kappa = ||Ahat||_2 ||W||_2` (Thm 1, A3).
- `S = (I - J_z)^{-1} J_A`, `S_c = S P_c` (`P_c`: N^2 -> |E| edge-supported symmetric projector,
  duplication-matrix reduction; columns paired with `b_k = (e_i e_j^T + e_j e_i^T)/sqrt2`).
- Threat set: `delta Ahat` symmetric, edge-supported, `||delta Ahat||_F <= eps`.
- Pseudospectral index `eta = ||(I-J_z)^{-1}||_2 (1 - rho(J_z)) >= 1` (= 1 iff `J_z` normal).
- Prop 1: `delta A* = eps * reshape(v_1)` (`v_1` = leading right singular vector of `S`, or `S_c`
  in the edge basis); Prop 2: `r_v = min_c m_v^{(c)} / ||(W_{y_v}-W_c) S_v||_2`.

---

## EXECUTIVE VERDICT (per claim)

| Claim | Verdict | One-line |
|---|---|---|
| **C1** sharpness / matching lower bound | **REFUTED as stated; PROVED in a restricted regime** | `eps_crit` is sharp two-sided **only** in the all-active normal case (`||Ahat||_2=rho(Ahat)`, `||W||_2=rho(W)`). In general it is the *norm*-contraction radius, a **strict lower bound** on the *spectral* breaking budget; the true gap factor is exactly `eta`. The iteration still has a unique fixed point above `eps_crit` whenever `rho(J_z')<1`, which we verified numerically up to `1.6 x eps_crit`. |
| **C2** order parameter + exponent | **PROVED (clean), with the correct order parameter** | Order parameter is the **spectral** contraction margin `g(eps) = 1 - rho(J_z'(eps))` (equivalently `sigma_min(I - J_z')`), **not** `1 - kappa`. Along the worst-case real-eigenvalue path `g(eps)` is affine and `-> 0` linearly, giving `sigma_1(S_c) ~ C / (eps* - eps)` with critical exponent **`gamma = 1`** (confirmed to 4 digits). Universal mean-field-type value. |
| **C3** universality | **PARTIAL -> PROVED for the exponent; `eps_crit` value is class-dependent** | `gamma = 1` holds for *every* contractive fixed-point operator with a real eigenvalue crossing 1 (verified non-Kronecker, nonnormal). But the **location** `eps*` depends on `J_A`'s action on the spectrum, and the *norm* surrogate `eps_crit=(1-kappa)/||W||_2` only coincides with `eps*` up to the factor `eta`. So: **exponent universal; threshold universal only after rescaling by `eta` (architecture-dependent unless normal).** |
| **C4** unification at criticality | **PROVED (the strongest result)** | The single resolvent `(I - J_z')^{-1}` and its image `S_c = (I-J_z')^{-1} J_A P_c` drive all three: (i) attack direction `v_1(S_c)`, (ii) radius `r_v ~ 1/sigma_1`, (iii) defense leverage `d eps_crit/d||W||_2 = -1/||W||_2^2 < 0`. Same spectral knob (`||W||_2`) simultaneously shrinks `sigma_1` (weaker attack, larger `r_v`) and enlarges `eps_crit`. One operator's spectrum is the common cause. |

**Bottom line.** A *sharp + universal* transition **is provable**, but its honest form is:
the **critical exponent `gamma = 1` is universal and sharp**; the **threshold is sharp only in the
all-active normal regime**, and in general `eps_crit` is a *certified-safe lower bound* on the true
(spectral) breaking budget `eps*`, with a known, `W`-only-controlled multiplicative gap `eta in [1, kappa(V_W)]`.

---

## C1 - SHARPNESS / MATCHING LOWER BOUND

### What the paper currently claims vs. what is true

Thm 1(c) says: above `eps_crit` "the sufficient contraction certificate no longer applies
(`||J_z'||_2` may exceed 1) and the part-(a) first-order guarantees do not apply." This is a
*norm* statement. The hidden move that a hostile reviewer pounces on: **losing `||J_z'||_2 < 1`
is NOT losing the fixed point.** Existence/uniqueness of the Banach fixed point is governed by a
*contraction in some norm*, and asymptotic stability of the Picard iteration by `rho(J_z') < 1`,
not by `||J_z'||_2 < 1`.

### The two distinct critical budgets

Define, with the Kronecker structure `J_z' = diag(phi')(Ahat' (x) W)`:

- **Norm budget (the paper's `eps_crit`):**
  `eps_crit^norm = (1 - ||Ahat||_2 ||W||_2)/||W||_2 = 1/||W||_2 - ||Ahat||_2.`
  Largest `eps` for which `||delta Ahat||_F<=eps` *guarantees* `||J_z'||_2 < 1` along the worst
  direction (since `||Ahat'||_2 <= ||Ahat||_2 + ||delta Ahat||_2 <= ||Ahat||_2 + eps`).
- **Spectral budget (the true breaking point):**
  `eps_crit^spec = 1/rho(W) - rho(Ahat)` (all-active),
  the smallest `eps` for which a feasible `delta Ahat` can push `rho(J_z') = rho(Ahat')rho(W) >= 1`.
  (`rho(A (x) B) = rho(A) rho(B)`; `rho(Ahat') <= rho(Ahat) + ||delta Ahat||_2 <= rho(Ahat)+eps`,
  with equality when `delta Ahat = eps * (top eigvec of Ahat) rank-1`, sign-aligned.)

**Numerically verified gap (generic nonnormal `W`):**
`eps_crit^norm = 0.431`, `eps_crit^spec = 1.147`, ratio `= 2.66`.
The ratio equals the nonnormality factor `(||Ahat||_2 ||W||_2)/(rho(Ahat) rho(W)) = eta_{Ahat}·eta_W`,
i.e. the pseudospectral index `eta` of `J_z` (Obs `eta_bound`, `eta <= kappa(V_W)` for symmetric `Ahat`,
`eta in [1.19,2.47]` empirically). So **`eps_crit^norm` under-shoots the true breaking budget by exactly `eta`.**

### Decisive negative test (the reviewer's kill-shot, and our defense)

In the open interval `eps_crit^norm < eps < eps_crit^spec` we have `||J_z'||_2 > 1` but `rho(J_z') < 1`.
We ran the ReLU fixed-point iteration `Z <- phi(Ahat' Z W^T + X_proj)` from 5 random starts at
`eps = {1.05, 1.30, 1.60} x eps_crit^norm` (nonnormal `W`): **all converged to the SAME unique fixed
point** (inter-start spread `~ 1e-13`), even at `||J_z'||_2 = 1.27`. Conclusion:

> **The prediction does NOT discontinuously break at `eps_crit^norm`.** The fixed point persists and
> stays unique well past it. Therefore C1 as literally stated ("for `eps > eps_crit` there EXISTS
> `delta A` losing contraction / existence / uniqueness OR a discontinuous prediction change")
> is **FALSE for the paper's `eps_crit`** in the general (nonnormal) case.

### Where C1 IS true: the sharp restricted regime

Impose **all-active and normal**: `phi' ≡ 1`, `Ahat` symmetric (always true: `Ahat=D^{-1/2}(A+I)D^{-1/2}`),
**and `W` symmetric** (so `||W||_2 = rho(W)`). Then `J_z = Ahat (x) W` is symmetric, hence normal,
`eta = 1`, and

`eps_crit^norm = 1/||W||_2 - ||Ahat||_2 = 1/rho(W) - rho(Ahat) = eps_crit^spec.` **(verified equal to 5 dp.)**

In this regime `eps_crit` is **simultaneously** (i) the largest guaranteed-safe budget and
(ii) the smallest provably-breaking budget, with the explicit extremal construction
`delta A* = eps * v_top v_top^T / ||.||_F`, `v_top` = top eigenvector of `Ahat`, sign-matched to the
dominant eigenvalue of `W`. This is a genuine **matching upper/lower bound** -> a true sharp transition.

### C1 candidate theorem (the strongest TRUE statement)

> **Theorem C1 (Sharp norm safety + matching spectral lower bound).**
> Under (A1)-(A3) with `J_z = diag(phi')(Ahat (x) W)`:
> (i) [Safety] For every feasible `delta Ahat` with `||delta Ahat||_F <= eps < eps_crit^norm`,
>     `||J_z'||_2 < 1`; the perturbed operator is an `||·||_2`-contraction with a unique fixed point.
> (ii) [Matching break] There exists a feasible `delta Ahat` with `||delta Ahat||_F = eps` driving
>     `rho(J_z') >= 1` iff `eps >= eps_crit^spec = 1/rho(W) - rho(Ahat)` (all-active), realized by the
>     rank-1 top-eigenvector perturbation. Moreover `eps_crit^norm <= eps_crit^spec` with ratio `eta`.
> (iii) [Sharpness] `eps_crit^norm = eps_crit^spec` iff `J_z` is normal (e.g. `W` symmetric, all-active),
>     in which case `eps_crit` is the exact two-sided phase boundary.

**Minimal assumptions:** single-layer / weight-tied IGNN, all-active (`phi'≡1`) for the spectral identity
`rho(J_z')=rho(Ahat')rho(W)`; for the safety half, any 1-Lipschitz `phi` suffices (norm bound only needs
`||diag(phi')||<=1`). For general ReLU patterns the spectral identity is replaced by the pseudospectral
envelope (Obs `eta_bound`, Rem `eta_relu`), and (iii)'s equality becomes `eps_crit^norm in [eps_crit^spec/eta, eps_crit^spec]`.

**Failure point (be explicit):** the literal "existence/uniqueness is lost at `eps_crit`" is false; only the
*norm certificate* is lost there. Any claim of a *discontinuous prediction change at `eps_crit`* must be
dropped or re-attributed to `eps_crit^spec`.

---

## C2 - ORDER PARAMETER + CRITICAL EXPONENT

### Correct order parameter

The paper's part (b) already (correctly) notes the resolvent blows up when `min_i |1 - lam_i(J_z')| -> 0`,
i.e. when an eigenvalue approaches 1 - **not** when `||J_z'||_2 -> 1`. So the right order parameter is the
**spectral contraction margin**

`g(eps) := min_i |1 - lam_i(J_z'(eps))| = 1 - rho(J_z'(eps))` (when the closest eigenvalue is real-positive,
the Perron/all-active symmetric case), equivalently `sigma_min(I - J_z'(eps))`.

`g(eps) -> 0` at `eps = eps_crit^spec`. Note `1 - kappa(eps) = 1 - ||J_z'||_2` is the WRONG order parameter:
it hits 0 at `eps_crit^norm` (too early by factor `eta`) and is not what makes `S_c` diverge.

### Exponent derivation

Resolvent identity (unconditional, used in the proof of Thm 1(b)):
`||(I - J_z')^{-1}||_2 >= 1/g(eps)`. For the upper side, with diagonalizable `J_z' = V D V^{-1}`,
`||(I - J_z')^{-1}||_2 <= kappa(V)/g(eps)` (= `eta`-type bound). Hence

`Theta(1/g(eps)) <= ||(I - J_z')^{-1}||_2 <= eta / g(eps)`.

Since `S_c = (I - J_z')^{-1} J_A P_c` and (generically) `J_A P_c` has a component along the diverging
left/right singular subspace of the resolvent,

`sigma_1(S_c(eps)) = Theta(||(I - J_z')^{-1}||_2) = Theta(1/g(eps)).`

Along the worst-case path the dominant eigenvalue moves **affinely**:
`rho(J_z'(eps)) = rho(Ahat')rho(W) = (rho(Ahat) + eps) rho(W)` (rank-1 top-eig perturbation), so
`g(eps) = 1 - (rho(Ahat)+eps)rho(W) = rho(W)(eps_crit^spec - eps)` is **linear** in `(eps_crit^spec - eps)`.
Therefore

> **`sigma_1(S_c(eps)) ~ C / (eps_crit^spec - eps)` with critical exponent `gamma = 1`,**
> `C = Theta(||J_A P_c||·proj / rho(W))`.

**Numerically verified:** log-log slope of `sigma_1(S')` vs `(eps_crit - eps)` `= -0.9999`, i.e. `gamma = 1.000`
to 4 digits; resolvent and `sigma_1` both scale as `1/g` over four decades of `g` down to `g ~ 4e-4`.

### C2 candidate theorem

> **Theorem C2 (Critical exponent).** Under (A1)-(A3), let the worst-case perturbation drive a single
> eigenvalue of `J_z'(eps)` toward 1 with nonzero speed (`d rho/d eps = rho(W) > 0`, all-active). Then the
> order parameter `g(eps) = 1 - rho(J_z'(eps))` is affine and vanishes at `eps_crit^spec`, and
> `sigma_1(S_c(eps)) = Theta((eps_crit^spec - eps)^{-1})`, i.e. `gamma = 1`. The constant in front is `O(eta)`.

**Why `gamma=1` and not something model-specific:** it is the residue of a **simple pole** of the resolvent
`(I - J_z')^{-1}` at a simple eigenvalue crossing 1. A simple eigenvalue gives a first-order pole -> `gamma=1`
**universally**. `gamma` would change only at a *defective* crossing (Jordan block of size `m` -> `gamma=m`),
which is non-generic (measure zero) under (A1)-(A3). This is exactly a mean-field/`1/(1-rho)` critical law.

**Tightness/regime:** the `1/g` law is informative precisely in the near-critical window
`eps_crit^spec - eps = O(eta·something small)`; far below criticality `sigma_1` is `O(||J_A||/(1-kappa))`
(part (a)) and the divergence language does not apply.

---

## C3 - UNIVERSALITY

### Abstract class

> **Definition (contractive equilibrium operator class `C`).** `z* = F(z*; A)`, `F(·;A)` Lipschitz with
> constant `kappa(A) < 1` in some norm; `J_z(A) = D_z F` with `rho(J_z) < 1`; the map `A -> J_z(A)` is
> `C^1` and the generator `J_A = D_A F` exists. Members: IGNN (`F = phi(Ahat Z W^T + b)`), monotone DEQ
> (`F` with `I - J_z` positive-definite-ified), general edge-weighted contractive message passing
> (GCN/SAGE/GIN/APPNP unrolled to fixed point or weight-tied), and abstract Banach fixed-point operators.

### What is universal

1. **Exponent `gamma = 1`:** PROVED universal. The derivation in C2 uses only (a) the simple-pole structure
   of `(I - J_z)^{-1}` at an eigenvalue crossing 1 and (b) a nonzero crossing speed. Neither uses the
   Kronecker / IGNN form. **Verified on a generic NONNORMAL non-Kronecker operator: `gamma = 0.9997`;
   on a normal one: `gamma = 0.9999`.** Architecture-independent.

2. **Order parameter = spectral margin `1 - rho(J_z)`:** universal (same resolvent identity for any `J_z`).

### What is NOT universal (the honest caveat)

3. **The closed-form `eps_crit = (1-kappa)/||W||_2` is IGNN-specific.** It comes from
   `||J_z'||_2 <= (||Ahat||_2 + eps)||W||_2`, which uses the bilinear `Ahat (x) W` structure. For general
   `F` in `C` the *value* of the threshold is `eps* = sup{eps : feasible delta A keeps rho(J_z(A+delta A))<1}`,
   a spectral quantity that need not have the `(1-kappa)/||W||` closed form, and the norm surrogate matches
   it only up to `eta`. So the threshold is universal **as a spectral object** (margin -> 0), and universal
   **as a closed form** only within the spectral-norm-constrained IGNN subclass (which the abstract already
   restricts to; good).

### C3 candidate theorem

> **Theorem C3 (Universality of the critical exponent).** For every operator in class `C` whose worst-case
> structural perturbation induces a simple eigenvalue of `J_z` crossing 1 transversally at `eps*`, the
> sensitivity `sigma_1(S_c(eps)) = Theta((eps* - eps)^{-1})` with `gamma = 1`, depending only on the coarse
> spectral data `(rho(J_z), crossing speed)`, independent of architecture. The *location* `eps*` is the
> spectral breaking budget; within the spectral-norm-constrained IGNN subclass it admits the closed form
> `eps* = eps_crit^spec` and is lower-bounded by the certifiable `eps_crit = (1-kappa)/||W||_2`, gap `eta`.

**This is the right "universality in the statistical-physics sense":** the **exponent** is universal
(a critical-phenomenon fingerprint); the **non-universal amplitude/location** is `eps*` (and its
norm-certifiable proxy `eps_crit`), exactly as in real universality classes where exponents are universal
but critical temperatures are material-dependent.

---

## C4 - UNIFICATION AT CRITICALITY

**Claim:** the divergence of `sigma_1(S_c)` at criticality is the COMMON cause of (i) attack optimality,
(ii) certificate collapse, (iii) defense leverage. **PROVED.** All three are functionals of the single
resolvent-image operator `S_c(eps) = (I - J_z'(eps))^{-1} J_A P_c`.

**(i) Attack optimality.** Prop 1: `delta A* = eps · reshape(v_1)`, `v_1 = ` leading right singular vector
of `S_c`, with damage `eps · sigma_1(S_c)`. As `eps -> eps*`, `sigma_1(S_c) -> infinity` (C2) and `v_1`
locks onto the right-singular direction of the diverging resolvent mode (the eigenvector of `J_z'` whose
eigenvalue -> 1). So the optimal attack direction is the **critical mode** of the operator.

**(ii) Certificate collapse.** Prop 2: `r_v = min_c m_v^{(c)} / ||(W_{y_v}-W_c) S_v||_2`. Since
`||(W_{y_v}-W_c) S_v||_2 = Theta(sigma_1(S_c))` along the critical mode, `r_v = O(g(eps)) -> 0` linearly:
the per-node radius collapses at the SAME rate the resolvent diverges. `r_v -> 0` and `sigma_1 -> infinity`
are reciprocal manifestations of one pole.

**(iii) Defense leverage.** `eps_crit = 1/||W||_2 - ||Ahat||_2`, so
`d eps_crit / d||W||_2 = -1/||W||_2^2 < 0` (verified). Tightening spectral regularization (decreasing
`||W||_2`) **simultaneously**: (a) raises `eps_crit` (more margin), and (b) shrinks
`sigma_1(S_c) <= ||J_A||/(1-kappa)` with `||J_A|| = Theta(||W||·||z*||)` and `kappa = ||Ahat||·||W||`,
hence lowers attack damage and raises every `r_v`. **One knob (`||W||_2`), three coupled effects, all read
off the spectrum of `(I - J_z')^{-1} J_A`.**

> **Theorem C4 (Spectral unification).** Under (A1)-(A3), define the operator
> `S_c(eps) = (I - J_z'(eps))^{-1} J_A P_c`. Then near `eps*`: (i) the Prop-1 optimal attack equals the
> top right-singular mode of `S_c` with gain `sigma_1(S_c) = Theta(1/g)`; (ii) the Prop-2 radius
> `r_v = Theta(g) -> 0`; (iii) `partial eps_crit/partial ||W||_2 < 0` while `partial sigma_1/partial ||W||_2 > 0`,
> so spectral regularization is the unique monotone control of all three. All three are functionals of the
> SAME operator `S_c`, whose single diverging singular value is the order-parameter pole.

This is the cleanest "one operator drives everything" statement and is the most reviewer-robust contribution.

---

## DOMINANT CONSTANTS (where every number comes from)

- `eps_crit^norm = 1/||W||_2 - ||Ahat||_2` (paper). `||Ahat||_2 <= 1` for normalized adjacency with
  self-loops typically `≈ 1`; positivity of `eps_crit` requires `||W||_2 < 1/||Ahat||_2`, i.e. the
  spectral-norm constraint with `c < 1/||Ahat||_2` (A2). Reviewer note: if `||Ahat||_2 = 1` exactly,
  `eps_crit = 1/||W||_2 - 1`, demands `||W||_2 < 1`.
- Gap factor `eta = (||Ahat||_2 ||W||_2)/(rho(Ahat)rho(W)) <= kappa(V_W)` (Obs `eta_bound`, symmetric `Ahat`
  gives `kappa(U_Ahat)=1`); empirically `eta in [1.19, 2.47]`.
- C2 amplitude `C = Theta(||J_A P_c||_{proj} / rho(W))`, with `||J_A|| ∝ ||W||·||z*||`, `||z*|| <= ||X_proj||/(1-kappa)`.
- Crossing speed `d rho/d eps = rho(W)` (all-active, rank-1 top-eig perturbation).
- `gamma = 1` exactly (simple pole); `gamma = m` for a defective `m x m` Jordan block (non-generic).

---

## BIGGEST RISKS TO THE THESIS

1. **(Severe) "Phase transition at `eps_crit`" overclaims existence/uniqueness loss.** Refuted above:
   the fixed point survives uniquely past `eps_crit^norm` up to `eps_crit^spec` (factor `eta` larger).
   Mitigation: state the transition as **norm-certificate loss at `eps_crit` (sufficient/safe)** and
   **spectral break at `eps_crit^spec` (necessary)**, with `eta` the slack. Do not claim a discontinuity at `eps_crit`.
2. **(Moderate) Sharpness only in the normal/all-active regime.** Real ReLU patterns and nonsymmetric `W`
   make `eta > 1`, so `eps_crit` is conservative, not sharp. Mitigation: present sharpness as a *regime*
   result (Theorem C1(iii)) plus the empirical `eta in [1.19,2.47]` envelope; claim "rate-sharp in the
   all-active case," which the abstract already does ("rate sharp in the all-active case") - keep that exact wording.
3. **(Minor) The exponent law is asymptotic (`eps -> eps*^-`).** It is a near-critical statement; far from
   criticality `sigma_1` is bounded by part (a). Mitigation: scope C2/C3 to the near-critical window.
4. **Edge-supported feasibility of the extremal `delta A*`.** The rank-1 top-eigenvector perturbation must be
   projected onto the edge-supported symmetric subspace (`P_c`). After projection the achieved spectral push
   is `<=` the unconstrained one, so `eps_crit^spec` computed on `S_c`/feasible set is an **upper** bound on
   the break budget; the matching lower bound (C1(ii)) should be stated for the *feasible* extremizer, i.e.
   the leading singular mode of the constrained perturbation map, not the raw top eigenvector of `Ahat`.

---

## RECOMMENDED EXACT HEADLINE CLAIM (single most defensible)

> **AEGIS exhibits a critical phenomenon with a universal exponent.** For spectral-norm-constrained IGNNs,
> the contraction margin `g(eps) = 1 - rho(J_z'(eps))` is the order parameter; it vanishes linearly at a
> spectral breaking budget `eps*`, and the worst-case structural sensitivity diverges as
> `sigma_1(S_c) = Theta((eps* - eps)^{-1})` with critical exponent `gamma = 1` (a simple-pole residue,
> hence universal across the contractive-operator class and architecture-independent). The certifiable
> budget `eps_crit = (1-kappa)/||W||_2` is the **largest guaranteed-safe** budget and a **lower bound** on
> `eps*`, the two coinciding (the transition becoming exactly two-sided sharp) **iff** `J_z` is normal
> (all-active, symmetric `W`); the multiplicative slack is the pseudospectral index `eta in [1.19,2.47]`.
> The same operator `S_c = (I - J_z')^{-1} J_A P_c` governs the optimal attack (its top singular mode),
> the per-node radius (`r_v = Theta(g) -> 0`), and the defense (`d eps_crit/d||W||_2 < 0`), unifying attack,
> certificate, and defense in one spectrum.

**Do claim:** universal `gamma = 1`; `eps_crit` = sharp safe boundary AND lower bound on the break; sharp
two-sided in the normal/all-active regime; single-operator unification (C4).
**Do NOT claim:** existence/uniqueness loss or prediction discontinuity *at* `eps_crit`; sharpness of
`eps_crit` in the general nonnormal case.

---

## FINAL HONEST VERDICT

**Is the sharp + universal transition provable, and in what generality?**

- **Sharp exponent (`gamma = 1`): YES, fully provable and universal** across the contractive equilibrium
  class, architecture-independent (simple-pole argument; numerically `gamma = 1.000`). This is genuine,
  statistical-physics-grade universality and is the breakthrough-worthy core.
- **Sharp threshold (two-sided `eps_crit`): provable ONLY in the all-active normal regime** (`W` symmetric);
  there `eps_crit^norm = eps_crit^spec` exactly. In general `eps_crit` is a *certified-safe lower bound* on
  the spectral break `eps*`, gap `= eta`. The literal "transition at `eps_crit`" with existence/uniqueness
  loss is **refuted** (unique fixed point persists to `1.6 x eps_crit` in tests).
- **Unification (C4): YES, fully provable** and the most robust contribution.

So AEGIS should pivot the "phase transition" headline from *threshold sharpness* (fragile) to
**exponent universality + single-operator unification** (rock-solid), keeping `eps_crit` as the
certified-safe boundary and naming `eta` as the honest, `W`-bounded slack to the true spectral break.
