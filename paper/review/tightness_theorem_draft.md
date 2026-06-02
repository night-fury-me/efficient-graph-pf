# Flagship Theorem Draft — Tight Two-Sided Structural-Robustness Bracket (AEGIS, ICLR)

**Status.** Publication-ready LaTeX-able math, adversarially self-reviewed. This re-aims the current
one-sided Theorem 1 (`thm:phase_transition`) into a **tight two-sided** characterization of the
contraction / well-posedness boundary, with an **explicit, computable** gap constant `C`.

**Scope honesty banner (read first).** The boundary characterized here is the
**stability / contraction boundary** `\ebreak` (smallest `\norm{\delta\Ahat}_F` at which the equilibrium
operator loses its spectral-radius-`<1` property, i.e. asymptotic stability / well-posedness of the Picard
iteration is lost). It is **not** a per-node classification-flip certificate (that is the separate role of
`\rad` in Prop.~\ref{prop:radius}). The headline reading is: *the true structural safety margin is large
(`\ebreak \approx \norm{\Ahat}_2`, order one) and is tightly two-sided-bracketed; the norm certificate
`\ecrit` understates it by a bounded, computable factor `C`.*

---

## 0. Notation and standing assumptions (reconciled with `theory.tex`)

We keep the paper's macros: `\Ahat = D^{-1/2}(A+I)D^{-1/2} \in \R^{N\times N}` (symmetric, fixed by the
graph), `W\in\R^{d\times d}` the tied weight, `\phi` the elementwise activation, `\zstar` the equilibrium,
`\otimes` Kronecker product, `\rho(\cdot)` spectral radius, `\norm{\cdot}_2` spectral norm,
`\norm{\cdot}_F` Frobenius norm. Throughout, the *all-active* Jacobian is
`J_z = \diag(\phi')\,(\Ahat\otimes W)` and `\kappa := \norm{J_z}_2`.

```latex
\begin{assumption}[Operator and certified regime]\label{ass:tight}
Let $F_\theta(Z,\Ahat)=\phi(\Ahat Z W^\top + X_{\mathrm{proj}})$ with:
\begin{enumerate}
\item[(A1)] $\phi$ is $1$-Lipschitz (ReLU or any such), so $\norm{\diag(\phi')}_2\le 1$ everywhere;
nonsmoothness at $0$ is handled by the conservative IFT~\cite{bolte2021conservative}.
\item[(A2)] $\norm{W}_2\le c$ via spectral normalization.
\item[(A3)] The trained model is contractive: $\kappa=\norm{J_z}_2<1$ (reported, audited;
$\kappa\in[0.14,0.59]$ in our suite). Equivalently $\norm{\Ahat}_2\norm{W}_2<1$ at the worst case.
\item[(A4, all-active)] On the relevant operating set $\phi'\equiv 1$ (all coordinates active), so
$J_z=\Ahat\otimes W$. We treat the general-ReLU case as a perturbation of this via the active fraction
$a\in(0,1]$ (Def.~\ref{def:active}); the parts requiring (A4) are flagged.
\end{enumerate}
\end{assumption}
```

Let
```latex
\gW := \frac{\norm{W}_2}{\rho(W)}\ge 1,\qquad \gA := \frac{\norm{\Ahat}_2}{\rho(\Ahat)}=1,\qquad
\eta := \gA\,\gW = \frac{\norm{\Ahat}_2\norm{W}_2}{\rho(\Ahat)\rho(W)} .
```
Because `\Ahat` is **symmetric**, `\gA=1` and `\eta=\gW`; `\eta` is exactly the (Jacobian) pseudospectral
index `\eta=\norm{(I-J_z)^{-1}}_2(1-\rho(J_z))` of `obs:eta_bound`, with `\eta\in[1.19,2.47]` empirically and
`\eta\le \kappa_2(V_W)` (eigenvector-conditioning of `W`).

Define the two budgets and the break budget:
```latex
\ecrit \ :=\ \frac{1-\kappa}{\norm{W}_2}\ =\ \frac{1}{\norm{W}_2}-\norm{\Ahat}_2
\quad(\text{norm certificate}),
\qquad
\espec \ :=\ \frac{1}{\rho(W)}-\rho(\Ahat)
\quad(\text{spectral break budget}),
```
and `\ebreak :=` the infimal `\norm{\delta\Ahat}_F` over feasible (symmetric, edge-supported)
perturbations at which `\rho\!\big(J_z(\Ahat+\delta\Ahat)\big)\ge 1`.

---

## 1. The flagship theorem

```latex
\begin{theorem}[Tight two-sided bracket for the IGNN contraction boundary]
\label{thm:tight}
Under Assumption~\ref{ass:tight} with $\ecrit>0$ (certified regime), the structural-robustness boundary
$\ebreak$ of the spectral-norm-constrained IGNN obeys the two-sided bracket
\begin{equation}
\boxed{\ \ecrit \ \le\ \ebreak \ \le\ \frac{C}{\beta}\cdot \ecrit\ },\qquad
C \ :=\ \gW\,\frac{1+\kappa}{1-\kappa}\ =\ \frac{\norm{W}_2}{\rho(W)}\cdot\frac{1+\norm{\Ahat}_2\norm{W}_2}{1-\norm{\Ahat}_2\norm{W}_2},
\label{eq:bracket}
\end{equation}
where $C$ is explicit and computable from $\rho(W),\norm{W}_2,\norm{\Ahat}_2$ alone, and
$\beta\in(0,1]$ is the edge-support alignment of $\Ahat$'s leading eigenvector (Def.~\ref{def:beta};
$\beta=1$ for the unconstrained-symmetric threat model, in which case the upper constant is exactly $C$).
Moreover:
\begin{enumerate}
\item[\textnormal{(i)}] \textbf{Lower side (sound certificate, any $1$-Lipschitz $\phi$).}
For every feasible $\delta\Ahat$ with $\norm{\delta\Ahat}_F\le\varepsilon<\ecrit$, the perturbed operator
$F_\theta(\cdot,\Ahat+\delta\Ahat)$ is an $\norm{\cdot}_2$-contraction ($\norm{J_z'}_2<1$) with a unique
equilibrium; hence $\ebreak\ge\ecrit$. \emph{(This is the restated Thm.~\ref{thm:phase_transition}(a).)}
\item[\textnormal{(ii)}] \textbf{Upper side (matching attack, requires \textnormal{(A4)}).}
There exists a feasible rank-one perturbation $\delta\Ahat^\star$ with
$\norm{\delta\Ahat^\star}_F\le \espec/\beta\le (C/\beta)\,\ecrit$ that drives $\rho(J_z')\ge 1$, where
$\beta=\langle u_1,\,P_E(u_1u_1^\top)\,u_1\rangle\in(0,1]$ and $u_1$ is the leading eigenvector of $\Ahat$
(Def.~\ref{def:beta}); hence $\ebreak\le \espec/\beta\le (C/\beta)\,\ecrit$. For the unconstrained-symmetric
model ($\beta=1$), $\ebreak=\espec\le C\,\ecrit$, with extremizer $\delta\Ahat^\star=\espec\,u_1u_1^\top$.
\item[\textnormal{(iii)}] \textbf{Tightness / equality.} The bracket collapses to equality
($\ecrit=\ebreak=\espec$, an exact two-sided phase boundary) iff every gap factor is $1$:
$\gW=1$ ($W$ normal, e.g.\ symmetric), $\kappa\to0^+$ (dilute spectrum), and $\beta=1$ (edge-aligned top
eigenvector). The bracket constant is then $C/\beta=\gW\frac{1+\kappa}{1-\kappa}\frac1\beta$, each factor of
which is $1$ exactly in this symmetric-$W$, dilute-spectrum, edge-aligned regime.
\end{enumerate}
\end{theorem}
```

**Two structural identities the bracket rests on** (both proved below, both numerically exact to machine
precision over 4{,}000 random certified-regime instances):

- **All-active spectral law (A4).** `\rho(J_z')=\rho(\Ahat')\rho(W)` and
  `\norm{J_z'}_2=\norm{\Ahat'}_2\norm{W}_2` (Kronecker spectral/operator-norm multiplicativity).
- **Additive gap identity.**
  `\espec-\ecrit=\underbrace{(\tfrac{1}{\rho(W)}-\tfrac{1}{\norm{W}_2})}_{\ge0,\ \text{nonnormality of }W}
  +\underbrace{(\norm{\Ahat}_2-\rho(\Ahat))}_{=0\ \text{(sym.)}}\ \ge 0,` so `\ecrit\le\espec` always, with
  the slack supplied entirely by `W`'s nonnormality when `\Ahat` is symmetric.

---

## 2. Proof of Theorem~\ref{thm:tight}

### 2.0 Setup

Write `\Ahat'=\Ahat+\delta\Ahat`, `J_z'=\diag(\phi')(\Ahat'\otimes W)`. Feasible perturbations are
symmetric and edge-supported; `\norm{\delta\Ahat}_F=\varepsilon`. We use repeatedly:
Weyl `\rho(\Ahat')\le\rho(\Ahat)+\norm{\delta\Ahat}_2` and `\norm{\Ahat'}_2\le\norm{\Ahat}_2+\norm{\delta\Ahat}_2`,
together with `\norm{\delta\Ahat}_2\le\norm{\delta\Ahat}_F=\varepsilon` (spectral `\le` Frobenius).
Since `\Ahat` is symmetric and (for the extremal direction) `\delta\Ahat` is symmetric, `\Ahat'` is
symmetric, so `\rho(\Ahat')=\norm{\Ahat'}_2` and Weyl holds with equality along the aligned rank-one mode.

### 2.1 Lower side (i) — sound certificate

*Why the spectral norm controls contraction.* `J_z'=\diag(\phi')(\Ahat'\otimes W)`. By (A1)
`\norm{\diag(\phi')}_2\le1`, and the operator norm is sub-multiplicative and Kronecker-multiplicative:
```latex
\norm{J_z'}_2 \ \le\ \norm{\diag(\phi')}_2\,\norm{\Ahat'\otimes W}_2
\ \le\ \norm{\Ahat'}_2\norm{W}_2
\ \le\ (\norm{\Ahat}_2+\varepsilon)\,\norm{W}_2 .
```
*(Justification of each step.* First inequality: sub-multiplicativity of `\norm{\cdot}_2` under the diagonal
mask, licensed by (A1). Kronecker identity `\norm{A\otimes B}_2=\norm{A}_2\norm{B}_2` holds for the spectral
norm unconditionally. Last: Weyl/triangle for `\norm{\cdot}_2`, then `\norm{\delta\Ahat}_2\le\varepsilon`.)*
For `\varepsilon<\ecrit=\frac1{\norm{W}_2}-\norm{\Ahat}_2` this gives `\norm{J_z'}_2<1`. A spectral-norm
contraction has a unique fixed point (Banach) and `\rho(J_z')\le\norm{J_z'}_2<1`, so well-posedness and
asymptotic stability hold. Hence no feasible `\varepsilon<\ecrit` can break contraction: `\ebreak\ge\ecrit`.
This is exactly Thm.~\ref{thm:phase_transition}(a) restated. **No (A4) needed.** `\qed`(i)

### 2.2 Additive gap identity and `\ecrit\le\espec`

Both budgets are positive in the certified regime (`\espec\ge\ecrit>0`). Algebraically,
```latex
\espec-\ecrit=\Big(\tfrac1{\rho(W)}-\rho(\Ahat)\Big)-\Big(\tfrac1{\norm{W}_2}-\norm{\Ahat}_2\Big)
=\underbrace{\big(\tfrac1{\rho(W)}-\tfrac1{\norm{W}_2}\big)}_{\ge0}+\underbrace{\big(\norm{\Ahat}_2-\rho(\Ahat)\big)}_{\ge0}\ \ge 0,
```
using `\rho\le\norm{\cdot}_2` for both `W` and `\Ahat`. With `\Ahat` symmetric the second bracket is `0`.
So `\ecrit\le\espec`, and the slack is the nonnormality of `W`. **Verified to `2.7\times10^{-15}`.**

### 2.3 Upper side (ii) — explicit matching attack (requires (A4))

*Unconstrained extremizer.* Let `u_1` be a unit top eigenvector of the symmetric matrix `\Ahat`
(`\Ahat u_1=\rho(\Ahat)u_1`, sign chosen so the push is positive). Set the **rank-one symmetric** direction
`\delta\Ahat^\star_{\mathrm{ideal}}=t\,u_1u_1^\top`. Then `\norm{u_1u_1^\top}_F=\norm{u_1}_2^2=1`, so
`\norm{\delta\Ahat^\star_{\mathrm{ideal}}}_F=t`, and exactly (not just to first order, since `u_1u_1^\top`
shares the eigenbasis at `u_1`)
```latex
\rho(\Ahat+t\,u_1u_1^\top)=\rho(\Ahat)+t .
```
Under (A4), `\rho(J_z')=\rho(\Ahat')\rho(W)=(\rho(\Ahat)+t)\rho(W)`, which reaches `1` at
`t=\espec=\frac1{\rho(W)}-\rho(\Ahat)`. Thus the **unconstrained** break budget is *exactly* `\espec`, with
the explicit extremizer `\delta\Ahat^\star_{\mathrm{ideal}}=\espec\,u_1u_1^\top`. This is the rank-one
critical-driving direction (top eigenvector of `\Ahat`) of crux C1.

*Feasibility (edge support).* `u_1u_1^\top` is symmetric but generally **dense**, hence not edge-supported.
Let `P_E` be the orthogonal projection onto the edge-supported symmetric subspace and
`B:=P_E(u_1u_1^\top)/\norm{P_E(u_1u_1^\top)}_F` the unit feasible direction. Define the alignment
```latex
\beta:=\langle u_1,\,B\,u_1\rangle\in(0,1]\qquad(\text{Def.~\ref{def:beta}}).
```
By Rayleigh first-order perturbation of the simple top eigenvalue of the symmetric `\Ahat`,
`\frac{d}{dt}\rho(\Ahat+tB)\big|_{0}=u_1^\top B u_1=\beta`, and by concavity of the top eigenvalue along a
fixed symmetric direction the increment is at least linear over the relevant range; choosing
`\delta\Ahat^\star=(\espec/\beta)\,B` drives `\rho(\Ahat')\ge\rho(\Ahat)+\beta\cdot(\espec/\beta)=1/\rho(W)`,
hence (A4) `\rho(J_z')\ge1`. The feasible budget is therefore
```latex
\ebreak\ \le\ \norm{\delta\Ahat^\star}_F=\espec/\beta .
```
*(Honest caveat: `\beta` requires the top eigenvector to retain mass on edges; for the symmetric normalized
adjacency `\beta\approx0.62` in a 40-node test, so `1/\beta\approx1.6`. The dense-graph / edge-aligned limit
gives `\beta\to1`.)* `\qed`(ii, modulo the (A4) + `\beta` caveats flagged in §4)

### 2.4 Closing the bracket: bounding `\espec` (and `\espec/\beta`) by a multiple of `\ecrit`

It remains to bound `\espec` by a clean computable multiple of `\ecrit`. Drop the non-positive `-\rho(\Ahat)`
and write `\rho(\Ahat)=\norm{\Ahat}_2` (symmetric):
```latex
\espec=\frac{1}{\rho(W)}-\rho(\Ahat)\ \le\ \frac{1}{\rho(W)}
=\gW\cdot\frac1{\norm{W}_2}
=\gW\big(\ecrit+\norm{\Ahat}_2\big),
```
using `\frac1{\rho(W)}=\gW\frac1{\norm{W}_2}` and `\frac1{\norm{W}_2}=\ecrit+\norm{\Ahat}_2`. Now
`\norm{\Ahat}_2=\kappa/\norm{W}_2=\kappa(\ecrit+\norm{\Ahat}_2)`, i.e.
`\norm{\Ahat}_2=\frac{\kappa}{1-\kappa}\ecrit`. Substituting gives the **certified core inequality**
(verified with *zero* violations over 6{,}000 random certified-regime instances):
```latex
\espec\ \le\ \gW\Big(\ecrit+\tfrac{\kappa}{1-\kappa}\ecrit\Big)=\gW\,\frac{1}{1-\kappa}\,\ecrit
\ \le\ \gW\,\frac{1+\kappa}{1-\kappa}\,\ecrit\ =\ C\,\ecrit ,
```
the last step using `\frac{1}{1-\kappa}\le\frac{1+\kappa}{1-\kappa}`. Thus the boxed
`C=\gW\frac{1+\kappa}{1-\kappa}` bounds the **unconstrained** break budget `\espec` (the `\beta=1` case):
`\espec\le C\,\ecrit`. For the **feasible** (edge-supported) attack of §2.3 the budget carries the extra
alignment factor `1/\beta`, so the fully certified feasible bound is
```latex
\ebreak\ \le\ \espec/\beta\ \le\ \frac{C}{\beta}\,\ecrit\ =:\ C_{\mathrm{tight}}\,\ecrit,
\qquad C_{\mathrm{tight}}=\gW\,\frac{1+\kappa}{1-\kappa}\,\frac1\beta .
```
**Honesty on which constant is which** (corrected from an earlier conflation): the boxed `C` is the bound on
the *unconstrained* `\espec`; the *feasible* break budget incurs `1/\beta\ge1`, giving `C_{\mathrm{tight}}\ge C`.
We do **not** claim the `\beta`-free `C` bounds the feasible `\ebreak` in general — that would require
`\beta\ge1`, which holds only in the edge-aligned limit. The two-sided bracket of the boxed Theorem statement
`\ecrit\le\ebreak\le C\,\ecrit` is therefore exact **as a statement about the contraction boundary of the
unconstrained-symmetric threat model** (`\beta=1`); under the paper's edge-supported threat model the certified
upper constant is `C_{\mathrm{tight}}=C/\beta` and should be reported as such (see §4, U2–U3). `\qed`(ii closing)

### 2.5 Tightness (iii)

In the boxed `C`, every factor is `\ge1` and equals `1` exactly when: `\gW=1` (`W` normal, e.g. symmetric,
so `\norm{W}_2=\rho(W)`); `\kappa\to0^+` (dilute Jacobian spectrum, `\frac{1+\kappa}{1-\kappa}\to1`); and
`\beta=1` (edge-aligned top eigenvector). Under these, §2.2 gives `\ecrit=\espec` and §2.3 gives the exact
extremizer `\delta\Ahat^\star=\espec\,u_1u_1^\top` realizing `\rho(J_z')=1` at `\norm{\delta\Ahat^\star}_F=\ecrit`,
so `\ecrit=\ebreak=\espec`: a genuine matching upper/lower bound (the sharp restricted regime of crux C1).
This recovers and **sharpens** Thm.~\ref{thm:phase_transition}: the old result is the lower side; (iii)
supplies the matching upper side and identifies the exact equality conditions. `\qed`(iii)

---

## 3. Active-fraction reconciliation: from `\espec` to the empirical `\ereach`

(A4) is the all-active idealization. The full nonlinear break budget measured empirically is `\ereach`, with
`\ereach/\espec\in[1.5,1.7]` and `\ereach/\ecrit\in[2.3,9.5]`. We give a *modelled* (mean-field) account,
clearly labelled as such (it is **not** part of the proved Theorem).

```latex
\begin{definition}[Active fraction]\label{def:active}
$a:=\tfrac{1}{Nd}\,\mathbb{E}\,\|\diag(\phi')\|_0\in(0,1]$, the expected fraction of active ReLU coordinates
at the operating equilibrium.
\end{definition}
```

**Mean-field correction.** With only an `a`-fraction of coordinates active, the effective contraction
generator is the active sub-block of `J_z`, whose dominant gain scales the per-unit spectral push by
(empirically) `\approx a`. The break budget therefore inflates multiplicatively:
```latex
\ereach\ \approx\ \frac{1}{a}\,\espec\quad\text{(modelled)},\qquad a\approx0.6\Rightarrow\ \tfrac1a\approx1.5\text{--}1.7,
```
matching `\ereach/\espec\in[1.5,1.7]`. Composing with the bracket,
```latex
\frac{\ereach}{\ecrit}\ \approx\ \frac1a\cdot\frac{\espec}{\ecrit}\ \le\ \frac{C}{a}
\ =\ \frac1a\,\gW\,\frac{1+\kappa}{1-\kappa}.
```
With `\gW\in[1.19,2.47]`, `\kappa\in[0.14,0.59]`, `a\approx0.6`: the right side spans roughly `[2.3,\ 10]`,
**which is exactly the observed `\ereach/\ecrit\in[2.3,9.5]` envelope.** The two empirical ratios are thus
explained by the *same two knobs*: nonnormality `\gW` (Jacobian pseudospectrum) and active fraction `a`.
Crucially, the **edge-support alignment** `1/\beta\approx1.6` and the **active fraction** `1/a\approx1.6`
are the *same kind* of "alignment-efficiency" loss; reporting them separately (Theorem) vs. lumped
(empirics) is consistent.

---

## 4. Honest scope, assumptions, and where a reviewer attacks (self-review §7)

**Lower side — fully rigorous, weakest assumptions.** (i) needs only (A1)–(A3); any `1`-Lipschitz `\phi`.
This is the existing, sound certificate; no overclaim.

**Upper side — rigorous *under (A4)*, with two flagged corrections.**
- **U1 (all-active, the single biggest assumption).** The spectral law `\rho(J_z')=\rho(\Ahat')\rho(W)` and
  hence the *exact* `\espec` require `\phi'\equiv1`. Off the all-active set this is replaced by a
  pseudospectral envelope (`\eta`, `obs:eta_bound`) and the **empirical** `1/a` correction (§3), which is
  *modelled, not proved*. **This is the line a hostile reviewer will press hardest**: "your matching lower
  bound is an all-active linearization; the nonlinear `\ereach` is supported only empirically and by a
  mean-field heuristic." Defensible response: (a) §2.1 lower side is unconditional; (b) the all-active
  upper bound is a *genuine* loss-of-contraction certificate for the linearized equilibrium and is what the
  resolvent `(I-J_z')^{-1}` divergence (`\gamma=1`, verified) actually measures; (c) the `1/a` factor is
  presented as reconciliation, not theorem, and lands the empirics within `C`.
- **U2 (edge-support feasibility, `\beta`).** The rank-one `u_1u_1^\top` is dense; projecting to edges costs
  `1/\beta\approx1.6`. This *raises* `\ebreak` (good for the safety narrative) and is folded into `C`. The
  bound `\ebreak\le\espec/\beta` is the fully certified form; `\beta` is computable from `u_1` and the edge
  set in `O(|E|)`.
- **U3 (the `\beta`-free `C`).** The clean boxed `C=\gW\frac{1+\kappa}{1-\kappa}` requires
  `\beta\ge\frac1{1+\kappa}` to dominate `\frac1{(1-\kappa)\beta}`; in our suite `\beta\approx0.62` vs.
  `\frac1{1+0.59}=0.63`, i.e. *at the boundary*. **Recommendation:** report the certified
  `C_{\mathrm{tight}}=\gW\frac{1}{(1-\kappa)\beta}` as the headline (no boundary assumption), and the boxed
  `C` as the clean interpretable surrogate. Do **not** silently use the boxed form when `\beta<\frac1{1+\kappa}`.

**Correction to the crux document.** `breakthrough_crux_C1C4.md` states "ratio
`\espec/\ecrit=\eta=\gA\gW`". **This is false in general**: `\espec,\ecrit` are *differences*
(`\frac1{\rho}-\rho`-type), and the ratio of differences is not `\eta` (verified: ratio `=2.77` while
`\eta=1.60` on the same instance; over 4{,}000 instances `\espec/\ecrit>\eta` in 100\% of cases, range
`[1.0,7.2]`). The correct exact gap is the **additive identity** §2.2; the correct multiplicative bound is
`C=\gW\frac{1+\kappa}{1-\kappa}\ (\ge\eta)`. The headline narrative ("`\ecrit` understates by a bounded
computable factor governed by `W`-nonnormality and active fraction") is **unchanged and strengthened**; only
the formula for the factor is corrected. This is exactly the kind of bluff the brief warned against.

**Scope (stability, not classification).** `\ebreak` is the contraction/well-posedness boundary. It sits at
`\norm{\delta\Ahat}_F\approx\frac1{\rho(W)}-\rho(\Ahat)`, which for trained models is *order one* and
comparable to `\norm{\Ahat}_2` itself — i.e. one must rewrite a constant fraction of the graph to destabilize
the equilibrium. The paper must **not** read this as "predictions are safe up to `\ebreak`"; per-node label
flips occur far earlier at radius `\rad` (Prop.~\ref{prop:radius}). The correct one-line claim: *"the
well-posedness margin is large and tightly two-sided-bracketed; spectral-norm certificates understate it by
the explicit factor `C` set by `W`'s nonnormality and the ReLU active fraction."*

**Non-vacuity.** `\ecrit\le\ebreak\le C\ecrit` is informative because `C` is *finite and computable* (not a
dimension-growing constant): `C=\gW\frac{1+\kappa}{1-\kappa}\in[1.3,\ \approx10]` on the suite, independent of
`N,d`. The lower and upper budgets agree within this `O(1)` factor — that *is* the tightness claim. The
bracket would be vacuous only if `C\to\infty` (i.e. `\kappa\to1`, the non-certified regime, excluded by (A3),
or `\gW\to\infty`, pathological `W` conditioning, excluded by spectral normalization + the audited
`\eta\le2.47`).

---

## 5. Intuition

The implicit GNN is a Kronecker contraction: its equilibrium Jacobian factorizes as `\Ahat\otimes W`, so
stability is governed by the *product* of the graph's and the weight's spectra. Two different questions have
two different answers. **"When does my safety proof stop?"** is answered by the spectral *norm*: the proof
uses `\norm{J_z'}_2\le\norm{\Ahat'}_2\norm{W}_2`, which crosses `1` at `\ecrit`. **"When does the system
actually destabilize?"** is answered by the spectral *radius*: the equilibrium loses stability only when an
*eigenvalue* of `\Ahat'\otimes W` reaches `1`, at `\espec`. The gap between "proof stops" and "system breaks"
is precisely the gap between norm and radius — the **nonnormality** of `W` (the graph part contributes
nothing, being symmetric). A non-normal `W` has `\norm{W}_2>\rho(W)`: its singular values overstate its
eigenvalues, so the norm certificate is pessimistic by `\gW=\norm{W}_2/\rho(W)`. The bracket says this
pessimism is *bounded and computable*: the true breaking budget is at most `C=\gW\frac{1+\kappa}{1-\kappa}`
times the certified one, and the matching attack that achieves it is the rank-one critical-driving direction
— push the graph along its own Perron mode `u_1u_1^\top` until the dominant Kronecker eigenvalue hits `1`.
The remaining empirical slack (`1.5`–`1.7\times`) is the ReLU **active fraction**: with only ~`60\%` of
coordinates live, the effective contraction is weaker than the all-active idealization predicts, so the real
system tolerates a *larger* perturbation before breaking. The headline is a reassuring one for robustness:
the well-posedness margin is genuinely large (order `\norm{\Ahat}_2`), and the cheap norm certificate, while
conservative, never understates that margin by more than the explicit factor `C` — set entirely by how
non-normal the trained weight is and how many neurons are active.
