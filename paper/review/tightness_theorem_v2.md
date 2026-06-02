# Flagship-Supporting Theorem v2 — Constant-Factor Two-Sided Bracket for the IGNN Contraction Boundary

**Status.** Revised to its strongest *defensible* form per the adversarial review
(`tightness_theorem_review.md`). The mathematics of the previous draft (`tightness_theorem_draft.md`)
was sound; this revision corrects the framing and two referee-checkable slips (N1, N2), scopes the
upper bound honestly, demotes the nonlinear active-fraction story to an empirical box, repairs the
`\beta` hole, and re-aims the contribution onto the **C4 unification**. *Honesty over impact.* This is
a **supporting** result for the C4-unification narrative, not a standalone "tight certificate."

**Positioning note (the contribution, in two sentences).** *The paper's contribution is the C4
unification: a single audited operator `S_c=(I-J_z)^{-1}J_A` simultaneously generates the
critical-driving attack direction, the contraction certificate `\ecrit`, and the spectral-margin
defense, and along its leading mode the norm-vs-radius slack of the IGNN is shown to equal exactly the
nonnormality `\gW` of the trained weight `W` and nothing from the (symmetric) graph. Theorem~1 below is
the quantitative corollary of that unification: a constant-factor, dimension-free, two-sided enclosure
of the all-active contraction boundary, rate-sharp in the normal/edge-aligned regime, with the
honest `O(1)` slack `C` reported as an audited per-model constant rather than advertised as
"tightness."*

---

## 0. Notation and standing assumptions

Keep the paper's macros. `\Ahat = D^{-1/2}(A+I)D^{-1/2} \in \R^{N\times N}` is the **symmetric**
normalized adjacency fixed by the graph; `W\in\R^{d\times d}` is the tied weight; `\phi` is the
elementwise activation; `\zstar` the equilibrium; `\otimes` the Kronecker product; `\rho(\cdot)` the
spectral radius; `\norm{\cdot}_2` the spectral norm; `\norm{\cdot}_F` the Frobenius norm. The
all-active Jacobian is `J_z = \diag(\phi')\,(\Ahat\otimes W)`; `\kappa := \norm{J_z}_2`. Write
`\Ahat'=\Ahat+\delta\Ahat`, `J_z'=\diag(\phi')(\Ahat'\otimes W)`.

```latex
\begin{definition}[Operator and certified regime]\label{ass:tight-v2}
Let $F_\theta(Z,\Ahat)=\phi(\Ahat Z W^\top + X_{\mathrm{proj}})$ with:
\begin{enumerate}
\item[(A1)] $\phi$ is $1$-Lipschitz (ReLU or any such), so $\norm{\diag(\phi')}_2\le 1$ everywhere;
nonsmoothness at $0$ is handled by the conservative IFT~\cite{bolte2021conservative}.
\item[(A2)] $\norm{W}_2\le c$ via spectral normalization (controls the \emph{numerator} of $\gW$ only;
see the a-posteriori remark below).
\item[(A3)] The trained model is contractive: $\kappa=\norm{J_z}_2<1$, reported and audited
post-training ($\kappa\in[0.14,0.59]$ across our suite). Equivalently $\norm{\Ahat}_2\norm{W}_2<1$.
\item[(A4, all-active)] On the operating set $\phi'\equiv 1$, so $J_z=\Ahat\otimes W$. The
\emph{upper} bound (ii)--(iii) of Theorem~\ref{thm:cf2s} is proved \emph{for this operator}; the
lower bound (i) does \emph{not} use (A4).
\end{enumerate}
\end{definition}
```

**Audited (a-posteriori) gap quantities.** Define
```latex
\gW := \frac{\norm{W}_2}{\rho(W)}\ge 1,\qquad \gA := \frac{\norm{\Ahat}_2}{\rho(\Ahat)}=1\ (\Ahat\ \text{symmetric}).
```
`\gW` is the **nonnormality** of `W`. We state it as an *audited per-model constant*, **not** a
quantity controlled a priori by training. Spectral normalization (A2) bounds `\norm{W}_2` from above
but does **not** bound `\rho(W)` away from `0`; in principle `\rho(W)\to0` at fixed `\norm{W}_2` sends
`\gW\to\infty`. We therefore *measure* `\gW` on each trained model (`\gW\in[1.19,2.47]` in our suite,
with the a-posteriori certificate `\gW\le\kappa_2(V_W)`, the eigenvector conditioning of `W`) and
report `C` as a function of that measured value. Every constant in Theorem~\ref{thm:cf2s} is
**computable post-training from `\rho(W),\norm{W}_2,\norm{\Ahat}_2` and the edge set**; none is asserted
to be uniformly bounded over all trainable `W`.

**Two budgets and the boundary.**
```latex
\ecrit \ :=\ \frac{1-\kappa}{\norm{W}_2}\ =\ \frac{1}{\norm{W}_2}-\norm{\Ahat}_2
\quad(\text{norm certificate, certified-safe}),
\qquad
\espec \ :=\ \frac{1}{\rho(W)}-\rho(\Ahat)
\quad(\text{all-active spectral break budget}).
```
`\ebreakall :=` the infimal `\norm{\delta\Ahat}_F` over feasible (symmetric, edge-supported)
perturbations at which the **all-active** operator loses contraction,
`\rho(\Ahat'\otimes W)\ge 1`. We write `\ebreakall` (superscript "all-act") throughout to mark that
the upper-bounded object is the all-active linearization, *not* the deployed ReLU model's stability
boundary.

**Edge-support alignment, stated as the code's computed singular mode (β-fix).** Let `P_E` be the
orthogonal projection onto the symmetric, edge-supported perturbation subspace
`\mathcal{S}_E:=\{\,M=M^\top : M_{ij}=0\ \text{if}\ (i,j)\notin E\cup\{(i,i)\}\,\}`. The matching attack
the code actually computes is the leading singular mode of the *`P_E`-restricted* eigenvalue-driving
map. Concretely, with `u_1` a unit top eigenvector of `\Ahat`, define
```latex
g_E:=P_E(u_1u_1^\top)\in\mathcal{S}_E,\qquad
\sigma_E:=\norm{g_E}_F\in[0,1],\qquad
B:=g_E/\sigma_E\ \ (\text{unit feasible direction, defined when }\sigma_E>0),
```
```latex
\beta:=\langle u_1,\,B\,u_1\rangle=\frac{\langle u_1, P_E(u_1u_1^\top)\,u_1\rangle}{\sigma_E}
=\frac{\sigma_E^2}{\sigma_E}=\sigma_E\ \in(0,1].
```
(The middle identity uses `\langle u_1,P_E(u_1u_1^\top)u_1\rangle=\langle u_1u_1^\top, P_E(u_1u_1^\top)\rangle
=\norm{g_E}_F^2=\sigma_E^2` since `P_E` is an orthogonal projection and `u_1u_1^\top` is symmetric.)
Thus **the alignment `\beta` equals the edge-supported Frobenius mass `\sigma_E=\norm{P_E(u_1u_1^\top)}_F`
of the rank-one critical direction** — a constructively computable, strictly positive quantity whenever
the top Perron mode places any mass on the edge set (always true for a connected graph, since `u_1>0`
entrywise and `E\neq\emptyset`). This replaces the a-priori `\beta` with the singular mode the
implementation uses, and makes the threat-model hypothesis `\beta>0` an *observable* rather than an
assumption.

---

## 1. The supporting theorem

```latex
\begin{theorem}[Constant-factor two-sided bracket for the all-active IGNN contraction boundary]
\label{thm:cf2s}
Let Definition~\ref{ass:tight-v2} hold with $\ecrit>0$ (certified regime) and $\Ahat$ symmetric, and
let $\beta=\sigma_E\in(0,1]$ be the edge-supported mass of the critical mode (\S0). Then:
\begin{enumerate}
\item[\textnormal{(i)}] \textbf{Lower side --- sound certificate (any $1$-Lipschitz $\phi$; no (A4)).}
For every feasible $\delta\Ahat$ with $\norm{\delta\Ahat}_F\le\varepsilon<\ecrit$, the perturbed
operator $F_\theta(\cdot,\Ahat+\delta\Ahat)$ is an $\norm{\cdot}_2$-contraction
($\norm{J_z'}_2<1$) with a unique equilibrium and $\rho(J_z')<1$. Hence the deployed model's
contraction boundary satisfies $\ebreak\ge\ecrit$; in particular $\ebreakall\ge\ecrit$.

\item[\textnormal{(ii)}] \textbf{Upper side --- all-active matching attack (requires (A4)).}
There is a feasible rank-one $\delta\Ahat^\star=(\espec/\beta)\,B\in\mathcal{S}_E$ with
$\norm{\delta\Ahat^\star}_F=\espec/\beta$ that drives the \emph{all-active} spectral radius to $1$;
hence
\[
\ebreakall\ \le\ \espec/\beta .
\]
In the unconstrained-symmetric threat model ($P_E=I$, so $\beta=\sigma_E=1$) the construction is
\emph{exact}: $\ebreakall=\espec$ with extremizer $\espec\,u_1u_1^\top$.

\item[\textnormal{(iii)}] \textbf{Constant-factor enclosure.} $\espec\le C\,\ecrit$ with the explicit,
dimension-free, \emph{a-posteriori-computable} constant
\[
C\ :=\ \gW\,\frac{1+\kappa}{1-\kappa}\ =\ \frac{\norm{W}_2}{\rho(W)}\cdot\frac{1+\norm{\Ahat}_2\norm{W}_2}{1-\norm{\Ahat}_2\norm{W}_2}.
\]
Combining with (i)--(ii), the all-active contraction boundary obeys the two-sided bracket
\[
\boxed{\ \ecrit\ \le\ \ebreakall\ \le\ \frac{C}{\beta}\,\ecrit\ }.
\]
\emph{Non-vacuity (\S0, $\beta$-fix).} The upper constant $C/\beta$ is finite and the bracket
non-vacuous whenever $\beta=\sigma_E>0$, which holds for every connected graph; on the suite
$\beta\approx0.62$, giving $C/\beta\lesssim 16$. The clean $\beta$-free form $\ecrit\le\ebreakall\le C\,\ecrit$
holds \emph{exactly} for the unconstrained-symmetric model ($\beta=1$), and as an upper bound on the
edge-constrained boundary it requires $\beta\ge 1$; we therefore report $C/\beta$ as the certified
edge-feasible constant and $C$ as the interpretable $\beta=1$ specialization, never substituting one
for the other.

\item[\textnormal{(iv)}] \textbf{Exact two-sided boundary (rate-sharp regime), with separated equality
conditions.}
\begin{itemize}
  \item[(a)] \emph{Endpoints coincide:} $\ecrit=\espec\iff \gW=1$ ($W$ normal, e.g.\ symmetric).
  This depends on $\gW$ \emph{alone}; $\kappa$ and $\beta$ are irrelevant to whether the two budgets
  are equal.
  \item[(b)] \emph{All three quantities coincide:} $\ecrit=\ebreakall=\espec\iff \gW=1$ \emph{and}
  $\beta=1$ (so the exact all-active extremizer $\espec\,u_1u_1^\top$ is itself edge-feasible). In that
  regime $\ecrit$ is the exact all-active contraction boundary (matching upper and lower bound).
  \item[(c)] \emph{Slack constant equals one:} $C/\beta=1\iff \gW=1$, $\beta=1$, \emph{and}
  $\kappa\to0^+$. The extra condition $\kappa\to0^+$ kills the $\tfrac{1+\kappa}{1-\kappa}$ inflation
  and is needed for the \emph{constant} (not the endpoints) to be unity.
\end{itemize}
\end{enumerate}
\end{theorem}
```

**Honest descriptor (use verbatim; do not write "tight").**
> *A **constant-factor (dimension-free, `O(1)`) two-sided characterization** of the all-active IGNN
> contraction boundary, **rate-sharp in the normal/edge-aligned regime**, with an explicit,
> a-posteriori-computable enclosure constant `C/\beta` whose value on our suite reaches `~10`--`16×`.*

We do **not** call this "tight." In the COLT/ICLR sense "tight" requires matching up to `1+o(1)` or a
construction attaining the upper constant; here the two-sided ratio is the genuine `O(1)` slack
`C/\beta`, empirically in `[1.02,18.2]` (median `2.25`; the in-suite certified constant is the larger
endpoint), attained simultaneously in both directions only in the degenerate corner of (iv)(c). "Tight"
is reserved for the rate-sharp regime (iv)(b), where the bracket collapses to a true matching bound.

**Two structural identities the bracket rests on** (both numerically exact to machine precision over
`4{,}000`--`6{,}000` random certified-regime instances):
- **All-active spectral law (A4 only).** `\rho(J_z')=\rho(\Ahat')\rho(W)` and
  `\norm{J_z'}_2=\norm{\Ahat'}_2\norm{W}_2` (Kronecker spectral / operator-norm multiplicativity).
  *This identity is the precise content of (A4): off the all-active set the masked operator is
  `D_a(\Ahat'\otimes W)` with `D_a=\diag(\phi')`, whose spectral radius does **not** factor as
  `\rho(\Ahat')\rho(W)`; this is exactly why (ii)--(iii) are scoped to `\ebreakall`.*
- **Additive gap identity (any $\phi$).**
  `\espec-\ecrit=\underbrace{(\tfrac1{\rho(W)}-\tfrac1{\norm{W}_2})}_{\ge0,\ \text{nonnormality of }W}
  +\underbrace{(\norm{\Ahat}_2-\rho(\Ahat))}_{=0\ (\text{sym.})}\ \ge0`, so `\ecrit\le\espec`, the slack
  supplied **entirely** by `W`'s nonnormality when `\Ahat` is symmetric (verified to `2.7\times10^{-15}`).

---

## 2. Proof of Theorem~\ref{thm:cf2s}

### 2.0 Setup
Feasible perturbations lie in `\mathcal{S}_E` (symmetric, edge-supported), `\norm{\delta\Ahat}_F=\varepsilon`.
We use Weyl's inequality `\rho(\Ahat')\le\rho(\Ahat)+\norm{\delta\Ahat}_2`,
`\norm{\Ahat'}_2\le\norm{\Ahat}_2+\norm{\delta\Ahat}_2`, and the norm ordering
`\norm{\delta\Ahat}_2\le\norm{\delta\Ahat}_F=\varepsilon`. Since `\Ahat` and the extremal `\delta\Ahat`
are symmetric, `\Ahat'` is symmetric, hence `\rho(\Ahat')=\norm{\Ahat'}_2` and Weyl holds with equality
along an aligned rank-one mode.

### 2.1 Lower side (i) — sound certificate, fully nonlinear
`J_z'=\diag(\phi')(\Ahat'\otimes W)`. By (A1), `\norm{\diag(\phi')}_2\le1`; the spectral norm is
sub-multiplicative and Kronecker-multiplicative:
```latex
\norm{J_z'}_2 \le \norm{\diag(\phi')}_2\,\norm{\Ahat'\otimes W}_2
\le \norm{\Ahat'}_2\norm{W}_2
\le (\norm{\Ahat}_2+\varepsilon)\,\norm{W}_2 .
```
*Justification.* First inequality: sub-multiplicativity of `\norm{\cdot}_2` under the diagonal mask,
licensed by (A1) (`\norm{\diag(\phi')}_2\le1`). Second: `\norm{A\otimes B}_2=\norm{A}_2\norm{B}_2`
holds unconditionally for the spectral norm. Third: Weyl/triangle, then `\norm{\delta\Ahat}_2\le\varepsilon`.
For `\varepsilon<\ecrit=\tfrac1{\norm{W}_2}-\norm{\Ahat}_2` this yields `\norm{J_z'}_2<1`. A spectral-norm
contraction has a unique fixed point (Banach) and `\rho(J_z')\le\norm{J_z'}_2<1`, so well-posedness and
asymptotic stability hold. Hence no feasible `\varepsilon<\ecrit` breaks contraction of the **deployed**
model: `\ebreak\ge\ecrit`. No (A4) used. `\qed`(i)

### 2.2 Additive gap identity and `\ecrit\le\espec`
Both budgets are positive in the certified regime. Algebraically,
```latex
\espec-\ecrit
=\Big(\tfrac1{\rho(W)}-\tfrac1{\norm{W}_2}\Big)+\Big(\norm{\Ahat}_2-\rho(\Ahat)\Big)\ \ge 0,
```
using `\rho\le\norm{\cdot}_2` for both factors; with `\Ahat` symmetric the second bracket is `0`. So
`\ecrit\le\espec`, slack `=` nonnormality of `W`. `\qed`(2.2)

### 2.3 Upper side (ii) — explicit all-active matching attack (requires (A4))
**Unconstrained extremizer.** Let `u_1` be a unit top eigenvector of symmetric `\Ahat`
(`\Ahat u_1=\rho(\Ahat)u_1`, sign chosen so the push is positive). With the rank-one symmetric
direction `t\,u_1u_1^\top` (note `\norm{u_1u_1^\top}_F=\norm{u_1}_2^2=1`), since `u_1u_1^\top` shares
the eigenbasis at `u_1` we have **exactly** (not merely first order)
```latex
\rho(\Ahat+t\,u_1u_1^\top)=\rho(\Ahat)+t .
```
Under (A4), `\rho(J_z')=\rho(\Ahat')\rho(W)=(\rho(\Ahat)+t)\rho(W)`, which reaches `1` at
`t=\espec=\tfrac1{\rho(W)}-\rho(\Ahat)`. So the **unconstrained** all-active break budget is exactly
`\espec`, extremizer `\espec\,u_1u_1^\top` (the rank-one critical-driving direction; crux C1).

**Feasibility (edge support), with the convexity argument (N1-fix).** `u_1u_1^\top` is dense; project
to `\mathcal{S}_E` via `B=P_E(u_1u_1^\top)/\sigma_E` (`\S0`), so `\beta=\sigma_E=u_1^\top B u_1>0`. By
Rayleigh first-order perturbation of the simple top eigenvalue of symmetric `\Ahat`,
```latex
\frac{d}{dt}\,\rho(\Ahat+tB)\Big|_{t=0}=u_1^\top B u_1=\beta .
```
**The required increment is a *lower* bound on `\rho`, and it follows from convexity, not concavity.**
The map `t\mapsto\lambda_{\max}(\Ahat+tB)=\max_{\norm{v}=1} v^\top(\Ahat+tB)v` is a pointwise maximum
of affine functions of `t`, hence **convex**. A convex function lies **above** its tangent line at any
point; at `t=0` the tangent is `\rho(\Ahat)+\beta t`, so
```latex
\rho(\Ahat+tB)\ \ge\ \rho(\Ahat)+t\,u_1^\top B u_1\ =\ \rho(\Ahat)+\beta t\qquad(\forall t\ge0).
```
(The earlier draft's appeal to "concavity" was a sign error: the largest eigenvalue is convex, and the
inequality the proof needs is precisely the tangent-below-the-curve consequence of *convexity*. This
was verified numerically: the lower bound holds with `0/42000` violations; the concave version is
violated `42000/42000`.) Choosing `\delta\Ahat^\star=(\espec/\beta)\,B\in\mathcal{S}_E` therefore drives
```latex
\rho(\Ahat')\ \ge\ \rho(\Ahat)+\beta\cdot(\espec/\beta)\ =\ \rho(\Ahat)+\espec\ =\ \tfrac1{\rho(W)},
```
and under (A4) `\rho(J_z')=\rho(\Ahat')\rho(W)\ge1`. Hence the feasible all-active budget satisfies
`\ebreakall\le\norm{\delta\Ahat^\star}_F=\espec/\beta`. When `\beta=1` the bound is exact and attained by
`\espec\,u_1u_1^\top`. `\qed`(ii)

### 2.4 Closing the bracket (iii) — bounding `\espec` by a computable multiple of `\ecrit`
Drop the non-positive `-\rho(\Ahat)` and use `\rho(\Ahat)=\norm{\Ahat}_2` (symmetric):
```latex
\espec=\tfrac1{\rho(W)}-\rho(\Ahat)\ \le\ \tfrac1{\rho(W)}=\gW\,\tfrac1{\norm{W}_2}=\gW\big(\ecrit+\norm{\Ahat}_2\big),
```
using `\tfrac1{\rho(W)}=\gW\tfrac1{\norm{W}_2}` and `\tfrac1{\norm{W}_2}=\ecrit+\norm{\Ahat}_2`. Since
`\norm{\Ahat}_2=\kappa/\norm{W}_2=\kappa(\ecrit+\norm{\Ahat}_2)`, we get
`\norm{\Ahat}_2=\tfrac{\kappa}{1-\kappa}\ecrit`. Substituting (verified, zero violations over `6{,}000`
certified instances):
```latex
\espec\ \le\ \gW\Big(\ecrit+\tfrac{\kappa}{1-\kappa}\ecrit\Big)=\gW\tfrac{1}{1-\kappa}\ecrit
\ \le\ \gW\tfrac{1+\kappa}{1-\kappa}\ecrit=C\,\ecrit .
```
With (i) (`\ecrit\le\ebreakall`) and (ii) (`\ebreakall\le\espec/\beta`), the boxed bracket
`\ecrit\le\ebreakall\le (C/\beta)\ecrit` follows. The `\beta=1` specialization gives the clean
`\ecrit\le\ebreakall\le C\,\ecrit`. `\qed`(iii)

### 2.5 Equality conditions (iv) — three separated claims (N2-fix)
- **(a) Endpoints `\ecrit=\espec`.** By §2.2 the gap is `(\tfrac1{\rho(W)}-\tfrac1{\norm{W}_2})\ge0`,
  zero **iff `\norm{W}_2=\rho(W)`, i.e. `\gW=1`** (`W` normal). Neither `\kappa` nor `\beta` appears in
  this gap, so endpoint equality is governed by `\gW` alone.
- **(b) All three coincide `\ecrit=\ebreakall=\espec`.** Add to (a) the requirement that the exact
  all-active extremizer `\espec\,u_1u_1^\top` be itself edge-feasible, i.e. `u_1u_1^\top\in\mathcal{S}_E`,
  equivalently `\beta=\sigma_E=1`. Then §2.3 attains `\rho(J_z')=1` at `\norm{\delta\Ahat^\star}_F=\espec=\ecrit`,
  so `\ecrit=\ebreakall`. Conditions: `\gW=1` **and** `\beta=1`. (`\kappa` still irrelevant.)
- **(c) Slack constant `C/\beta=1`.** Beyond (b), the inflation `\tfrac{1+\kappa}{1-\kappa}` must equal
  `1`, i.e. `\kappa\to0^+` (dilute Jacobian spectrum). Conditions: `\gW=1`, `\beta=1`, **and**
  `\kappa\to0^+`. This is the only regime in which the *constant* (not just the endpoints) is unity; it
  is the rate-sharp corner. `\qed`(iv)

This recovers and sharpens Thm.~\ref{thm:phase_transition}: the old result is the lower side (i); (ii)
supplies the matching all-active upper side; (iv) separates the three distinct equality conditions the
draft had conflated.

---

## 3. Empirical Observation (NOT part of Theorem~\ref{thm:cf2s}): nonlinear break budget and the active fraction

```latex
\begin{tcolorbox}[title={Observation~O1 (empirical / conjecture, seed-42 and suite measurements;
                  two named proof gaps; \emph{not} a theorem)}]
```
The deployed model is ReLU, not all-active; its measured nonlinear break budget `\ereach` exceeds the
all-active `\espec`. We report the **empirical** relation and state honestly that a proof is open.

**Measured (seed-42 reachability, 50-node ego subgraph; `\norm{\Ahat}_F=1.77`):**

| `\kappa_0` | `\rho_0` (clean) | `\ecrit` | `\espec` | `\ereach` | `\ereach/\espec` | `\ereach/\ecrit` | `\gamma` (resolvent) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.219 | 1.000 | 1.518 | 2.273 | 1.50 | 2.27 | 1.02 |
| 0.9 | 0.390 | 0.111 | 0.629 | 1.060 | 1.69 | 9.54 | 1.02 |

Empirically `\ereach\approx(1/a)\,\espec` with `a` the ReLU **active fraction**
(`a:=\tfrac1{Nd}\,\E\,\norm{\diag(\phi')}_0\in(0,1]`); with `a\approx0.6`, `1/a\approx1.5`--`1.7`, matching
`\ereach/\espec\in[1.5,1.7]`. Composing with the bracket gives the *observed* envelope
`\ereach/\ecrit\lesssim C/a=\tfrac1a\gW\tfrac{1+\kappa}{1-\kappa}`, spanning roughly `[2.3,10]` and
matching the measured `\ereach/\ecrit\in[2.3,9.5]`.

**This is a conjecture; two independent proof gaps remain open:**
1. **Masked-operator spectral scaling (unproved).** The relation `\ereach\approx\espec/a` assumes the
   dominant eigenvalue of the masked Jacobian `D_a(\Ahat'\otimes W)` scales as
   `\approx a\cdot\rho(\Ahat')\rho(W)`. **No such lemma holds in general**: the active set is correlated
   with the equilibrium and with `u_1`, so the active sub-block's Perron value is not `a` times the full
   one. This is a mean-field heuristic calibrated to two operating points (`\kappa_0\in\{0.5,0.9\}`).
2. **Linear-to-nonlinear bifurcation (empirical only).** The jump from "an eigenvalue of the *linearized*
   Jacobian reaches `1`" to "the *true nonlinear* fixed point destabilizes (50/50 prediction flips,
   reconverged `\rho\to\infty`)" is a bifurcation claim about the nonlinear fixed-point map, supported
   here only by the seed-42 run (which does exhibit a clean simple-pole divergence, `\gamma=1.02` on both
   margins).

**Honest framing.** We present `\ereach\approx(1/a)\,\espec` as an empirical regularity that the same two
knobs — nonnormality `\gW` and active fraction `a` — organize, **not** as a proven extension of
Theorem~\ref{thm:cf2s}. A proof of either gap is left open.

**Practical caveat (criticality is distant).** `\ereach` is `60`--`128\%` of `\norm{\Ahat}_F`, i.e.
`5`--`11×` the largest realistic budget (`\varepsilon\le0.2`); destabilization requires rewriting a
constant fraction of the graph. Realistic-budget robustness is governed by the first-order sensitivity
`\sigma_1(S_c)` at the operating point (the original AEGIS object), not by proximity to criticality.
```latex
\end{tcolorbox}
```

---

## 4. Scope, assumptions, and reviewer-facing honesty (self-review §7)

- **Lower side (i):** fully rigorous, weakest assumptions ((A1)–(A3); any `1`-Lipschitz `\phi`).
  Bounds the **deployed** model. No overclaim. This is the certificate the paper sells.
- **Upper side (ii)–(iii):** rigorous **for the all-active operator** (A4). Scoped to `\ebreakall` *in
  the theorem statement*. The matching construction bounds the all-active linearization `J_z=\Ahat\otimes W`,
  **not** the deployed ReLU model; the gap to the deployed model is exactly the subject of Observation O1
  and is left as conjecture.
- **`\beta` / non-vacuity:** the upper constant is `C/\beta` with `\beta=\sigma_E=\norm{P_E(u_1u_1^\top)}_F>0`
  for any connected graph; the bound is non-vacuous wherever `\sigma_E>0` and we report `C/\beta` (not the
  `\beta`-free `C`) as the certified edge-feasible constant. We do **not** claim `\beta` is bounded away
  from `0` uniformly as `N\to\infty`; if a graph family has `\sigma_E\to0` the constant degrades, and we
  state this rather than assume `N`-independence.
- **`\gW`, `C`:** a-posteriori audited per-model constants, computed from `\rho(W),\norm{W}_2,\norm{\Ahat}_2`.
  Training (A2) bounds `\norm{W}_2` only; `\gW` is *measured*, not *controlled*, and `C` is reported as a
  function of the measured `\gW`. Honest descriptor: **constant-factor, not tight** (in-suite `C/\beta`
  reaches `~10`–`16×`; random-instance `\espec/\ecrit\in[1.02,18.2]`, median `2.25`).
- **Descriptor discipline:** the word "tight" appears only for the rate-sharp regime (iv)(b). Elsewhere:
  "constant-factor two-sided," "dimension-free `O(1)` enclosure," "rate-sharp in the normal/edge-aligned
  regime."
- **Novelty placement (C4 re-aim):** the rank-one Perron bump and Kronecker multiplicativity are textbook;
  the lower side is the existing Thm.~1(a). The genuine, non-obvious content is (1) the **C4 unification**
  (one audited operator `S_c` yields attack, certificate `\ecrit`, and defense), and (2) the **exact
  attribution** of the norm-vs-radius slack to `W`'s nonnormality `\gW` and nothing from the symmetric
  graph. Theorem~\ref{thm:cf2s} is the *quantitative corollary* that makes C4 numeric; it is framed as
  supporting, not headline.

---

## 5. Intuition

The implicit GNN is a Kronecker contraction: its equilibrium Jacobian factors (all-active) as
`\Ahat\otimes W`, so stability is governed by the *product* of the graph and weight spectra. Two distinct
questions get two distinct answers. *"When does my safety proof stop?"* is the spectral **norm**: the
certificate uses `\norm{J_z'}_2\le\norm{\Ahat'}_2\norm{W}_2`, crossing `1` at `\ecrit`. *"When does the
all-active linearization actually destabilize?"* is the spectral **radius**: an eigenvalue of
`\Ahat'\otimes W` reaches `1` at `\espec`. The gap between "proof stops" and "all-active model breaks" is
exactly the norm-vs-radius gap, which (graph being symmetric) is precisely `W`'s nonnormality
`\gW=\norm{W}_2/\rho(W)`: a non-normal `W` has singular values overstating its eigenvalues, so the norm
certificate is conservative by the audited factor `\gW`. Theorem~\ref{thm:cf2s} says this conservatism is
*bounded and computable*: the all-active breaking budget is at most `C/\beta` times the certified one,
and the attack achieving it pushes the graph along its own Perron mode `u_1u_1^\top` (projected to edges)
until the dominant Kronecker eigenvalue hits `1`. The further empirical slack to the *deployed* ReLU
model (`1.5`–`1.7×`) is the active fraction — fewer live neurons mean weaker effective contraction, so
the real system tolerates a larger perturbation — but this last step is Observation O1, not the theorem.
The reassuring headline for robustness survives: the well-posedness margin is genuinely large (order
`\norm{\Ahat}_2`), the cheap norm certificate never *understates* it, and it overstates safety by no more
than the explicit, audited factor `C/\beta`.

---

## Confirmation: all 8 required fixes applied

1. **"tight" → "constant-factor two-sided."** Done. Honest descriptor stated; "tight" used only for the
   rate-sharp corner (iv)(b). Slack `~10`–`18×` reported explicitly (§1, §4).
2. **Upper bound scoped to A4 in the statement.** Done. Object renamed `\ebreakall`; (ii)–(iii) say
   "all-active matching attack (requires (A4))" inside the theorem; the deployed model gets only (i).
3. **Nonlinear `\ereach`/`(1/a)` demoted to a labelled empirical/conjecture box.** Done. §3 Observation O1
   in a `tcolorbox`, "NOT part of Theorem," with both proof gaps (masked-operator scaling; linear→nonlinear
   bifurcation) named, and the relation stated as `\ereach\approx(1/a)\,\espec`, "proof open."
4. **`\beta` hole fixed.** Done. `\beta` redefined as the code's computed `P_E`-restricted singular-mode
   mass `\sigma_E=\norm{P_E(u_1u_1^\top)}_F`, constructively `>0` for connected graphs; certified constant
   is `C/\beta`; no uniform `N`-independence assumed; degradation stated honestly.
5. **N1 fixed.** Done. §2.3 now argues from **convexity** of `\lambda_{\max}` (tangent-below-the-curve),
   with the `0/42000` vs `42000/42000` verification noted; "concavity" removed.
6. **N2 fixed.** Done. (iv) split into (a) endpoints `\ecrit=\espec\iff\gW=1` alone; (b) all-three coincide
   needs `\gW=1,\beta=1`; (c) constant `C/\beta=1` additionally needs `\kappa\to0^+`.
7. **`\gW`, `C` stated a-posteriori / audited.** Done. §0 audited-quantities paragraph: normalization
   bounds `\norm{W}_2`, not `\rho(W)` from below; `\gW` measured per model; `C` reported as a function of
   the measured value.
8. **Contribution re-aimed onto C4 unification.** Done. Positioning note (top) + §4 novelty paragraph frame
   the C4 unification (one operator → attack, certificate, defense) as the contribution, with the bracket
   as its quantitative corollary, not a standalone "tight certificate."
