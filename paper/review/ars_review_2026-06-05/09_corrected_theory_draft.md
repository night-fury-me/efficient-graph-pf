# Drop-in Corrections — `thm:phase_transition`, bracket lower side, framing relabels

**Derived from** `07_theory_verification_ecrit.md` + `08_theory_verification_cf2s_zstar.md`.
**What it fixes:** the subcritical certificate's `κ`-vs-`‖Â‖‖W‖` conflation (MAJOR, blocking) and `thm:cf2s` lower side (same defect), plus the two framing relabels (T1.2, T1.3).
**Review status:** adversarially reviewed by `ml-theory-reviewer` (`10_fix_review.md`). Verdict: items 1, 3, 4, 5, 6 PASS; item 2 NEEDS-EDIT (the IFT-to-`ε_crit` clause was a residual overclaim). **Both required edits — the PATCH 2 item-2 reword and the PATCH 3a statement-line fix — are now incorporated below, so this draft is SHIP-READY** (Option B). `ε_glob` verified a sound global certificate (0/4000 adversarial violations); `ε_glob ≤ ε_crit` proved (0/3000); no downstream result touched.

---
## The consequential choice (read first)

For these models `‖Â‖₂=1` (symmetric-normalized adjacency with self-loops, `λ_max=1`) and `‖W‖₂=c` (spectral cap). Two honest framings:

- **Option A (verifier's literal fix).** Replace `ε_crit` everywhere by the mask-agnostic global radius `ε_glob = max(0, 1/‖W‖−‖Â‖)`. Fully sound *global* contraction certificate. **Cost:** `ε_glob = 1/c − 1` is dataset-independent, so the per-dataset `ε_crit` column of `tab:cross_domain` collapses to one constant (≈0.11 at c=0.9). Loses a main-text feature.

- **Option B (recommended).** Keep `ε_crit=(1−κ)/‖W‖` but relabel it the **linearized critical budget** (the per-dataset scale where the *first-order* contraction factor `κ+ε‖W‖` hits 1, which is what drives the regime-(b) divergence), and *additionally* state the rigorous **global contraction radius** `ε_glob ≤ ε_crit`. Keeps the per-dataset table (caption changes one phrase), stays honest (the true nonlinear break is `ε_reach`, 2–9× larger, already in `rem:obs_o1`), and adds a genuine global certificate. **This is what the patches below implement.**

Either is defensible; B preserves more of the paper. If you prefer A, say so and I'll regenerate the patches (mechanical).

> **Preamble:** add `\newcommand{\eglob}{\varepsilon_{\mathrm{glob}}}` next to the other `\e...` macros in `aaai_aegis.tex`.

---
## PATCH 1 — `sections/theory.tex` : critical-budget definition + regime (a)

**OLD**
```latex
Define the critical budget
\begin{equation}
\ecrit = \frac{1 - \kappa}{\norm{W}_2}.
\end{equation}
Then structural perturbations $\delta \Ahat$ with $\norm{\delta \Ahat}_F = \varepsilon$ exhibit three regimes:

\noindent\textbf{(a) Subcritical} ($\varepsilon < \ecrit$): The perturbed operator remains contractive with a unique fixed point satisfying
\begin{equation}
\norm{\Delta \zstar}_F \leq \sigma_1(S) \cdot \varepsilon + O(\varepsilon^2),
\label{eq:shift_bound}
\end{equation}
where $S = (I - J_z)^{-1} J_A$ is the structural sensitivity matrix and $\sigma_1(S) \leq \norm{J_A}_{\mathrm{op}} / (1 - \kappa)$.
```

**NEW**
```latex
Define the \emph{linearized critical budget} and the \emph{global contraction radius}
\begin{equation}
\ecrit = \frac{1 - \kappa}{\norm{W}_2},
\qquad
\eglob = \max\!\Big(0,\ \tfrac{1}{\norm{W}_2} - \norm{\Ahat}_2\Big)\ \le\ \ecrit .
\label{eq:budgets}
\end{equation}
Since $\kappa=\norm{J_z}_2\le\norm{\Ahat}_2\norm{W}_2$, with equality only when every unit is active
(A4), $\eglob\le\ecrit$ and the two coincide in the all-active case. The first is the per-model scale
at which the first-order contraction factor $\kappa+\varepsilon\norm{W}_2$ reaches $1$; the second is
the budget below which the perturbed operator contracts under \emph{every} activation pattern
(\cref{app:proof_phase}). Structural perturbations $\delta \Ahat$ with $\norm{\delta \Ahat}_F = \varepsilon$
then exhibit three regimes:

\noindent\textbf{(a) Subcritical} ($\varepsilon < \ecrit$): The equilibrium is continued uniquely by the
implicit function theorem ($I-J_z$ is invertible since $\kappa<1$) and moves boundedly,
\begin{equation}
\norm{\Delta \zstar}_F \leq \sigma_1(S) \cdot \varepsilon + O(\varepsilon^2),
\label{eq:shift_bound}
\end{equation}
with $S = (I - J_z)^{-1} J_A$ and $\sigma_1(S) \leq \norm{J_A}_{\mathrm{op}} / (1 - \kappa)$. For the
stricter budget $\varepsilon<\eglob$ the perturbed operator is moreover a \emph{global} contraction with
a unique fixed point (\cref{app:proof_phase}).
```

**Regime (b)** — one clause for honesty (optional but recommended). In the sentence ending
``$\ecrit$ \emph{lower-bounds} the divergence threshold, the slack being the nonnormality $\eta$ \dots'',
append: ``; this is a first-order rate, and the true nonlinear divergence sits at $\ereach>\ecrit$
(\cref{rem:obs_o1}).''

**Table caption** (`tab:cross_domain`, `sections/experiments.tex`): change ``$\ecrit$ is the local
critical budget'' → ``$\ecrit$ is the linearized critical budget (\cref{eq:budgets}); the rigorous
global radius is $\eglob\le\ecrit$''.

---
## PATCH 2 — `sections/appendix/B_sensitivity.tex` : proof Step 2 (the actual bug)

**OLD**
```latex
\emph{Step 2: contraction survives below $\ecrit$.}
Perturbing $\Ahat$ by $\delta\Ahat$ changes the state Jacobian to $J_z'=\diag(\phi')(\Ahat'\otimes W)$.
Because $\norm{\diag(\phi')}_2\le1$ (A1) and the spectral norm is Kronecker-multiplicative,
\begin{equation}
\norm{J_z'}_2\le\norm{\Ahat'}_2\norm{W}_2\le(\norm{\Ahat}_2+\varepsilon)\,\norm{W}_2,
\label{eq:Jzp-bound}
\end{equation}
and the right-hand side is below $1$ precisely when $\varepsilon<1/\norm{W}_2-\norm{\Ahat}_2=\ecrit$.
A spectral-norm contraction has a unique fixed point by the Banach theorem, so for every feasible
$\varepsilon<\ecrit$ the perturbed model still has a single well-defined equilibrium: this is the
subcritical regime, and $\ecrit$ is a sufficient safety boundary.
```

**NEW**
```latex
\emph{Step 2: a global certificate and a linearized scale.}
Perturbing $\Ahat$ by $\delta\Ahat$ changes the state Jacobian to $J_z'=\diag(\phi')(\Ahat'\otimes W)$,
where $\diag(\phi')$ is the activation mask at the \emph{perturbed} equilibrium. Because
$\norm{\diag(\phi')}_2\le1$ (A1) holds for \emph{every} mask and the spectral norm is
Kronecker-multiplicative,
\begin{equation}
\norm{J_z'}_2\le\norm{\Ahat'}_2\norm{W}_2\le(\norm{\Ahat}_2+\varepsilon)\,\norm{W}_2,
\label{eq:Jzp-bound}
\end{equation}
whose right-hand side is below $1$ exactly when $\varepsilon<\eglob=1/\norm{W}_2-\norm{\Ahat}_2$. This
bound is mask-agnostic, hence valid uniformly as the perturbation crosses activation regions; a
spectral-norm contraction has a unique fixed point by the Banach theorem, so for every feasible
$\varepsilon<\eglob$ the perturbed model has a single globally well-defined equilibrium. This is the
rigorous global contraction radius.
The linearized budget $\ecrit=(1-\kappa)/\norm{W}_2\ge\eglob$ is the corresponding first-order quantity:
on a fixed activation region $J_z'=J_z+\diag(\phi')(\delta\Ahat\otimes W)$, so
$\norm{J_z'}_2\le\kappa+\varepsilon\norm{W}_2$, which reaches $1$ at $\varepsilon=\ecrit$. It is
\emph{not} a global Banach radius under (A1)--(A3) alone, because the equilibrium crosses finitely many
ReLU regions before $\ecrit$ (the mask is constant only locally, not across the whole ball); the
identity $\ecrit=\eglob$ holds only in the all-active case (A4), where $\kappa=\norm{\Ahat}_2\norm{W}_2$.
On each linear region the conservative IFT~\cite{bolte2021conservative} continues the equilibrium
uniquely with the first-order bound of Step~1, and this local continuation persists while the active
set is constant. The global, region-independent guarantee is $\eglob$: for $\varepsilon<\eglob$ the map
is a Frobenius-norm contraction under every activation pattern, so the Banach theorem gives a single
equilibrium reached from any initialization. Whether the continued branch stays the \emph{unique}
equilibrium across the finitely many region crossings up to the larger linearized scale $\ecrit$ is not
certified by the contraction argument; it is the empirical regularity of \cref{rem:obs_o1} (the measured
break sits at $\ereach>\ecrit$), whose two proof gaps---masked-operator spectral scaling and
linear-to-nonlinear bifurcation---remain open. Empirically the continued branch stays the unique
contraction up to $\ecrit$ across our suite; we present this as observation, not theorem.
```

> **Item-2 reword (added after `ml-theory-reviewer` review `10`):** the original draft's last clause
> ("continues the equilibrium uniquely for $\varepsilon<\ecrit$") was a residual overclaim — the cited
> conservative IFT licenses only *local* (per-region) continuation, and global persistence to $\ecrit$
> is exactly `rem:obs_o1`'s two open gaps. The wording above (verified 0/30k adversarial
> continued-branch breaks, but *not a theorem*) routes the to-$\ecrit$ persistence to the empirical
> regularity and keeps $\eglob$ as the only global guarantee.

**Step 3 reference fix** — in the same proof, the clause
``$\min_i\abs{1-\lambda_i}=1-\norm{J_z'}_2=\norm{W}_2(\ecrit-\varepsilon)$ by \eqref{eq:Jzp-bound} with
equality'' should read ``$=\norm{W}_2(\ecrit-\varepsilon)$ using the within-region linearization
$\norm{J_z'}_2=\kappa+\varepsilon\norm{W}_2$'' (the equality is the linearized, not the mask-agnostic,
bound).

---
## PATCH 3 — `sections/appendix/D_boundary.tex` : bracket lower side (i)

### PATCH 3a — the THEOREM STATEMENT line `thm:cf2s_full` (i) *(added after review `10`, item 4: the statement, not only the proof, over-reads)*

**OLD**
```latex
\item[\textnormal{(i)}] \textbf{Lower side (any $1$-Lipschitz $\phi$; no (A4)).} For every feasible
$\delta\Ahat$ with $\norm{\delta\Ahat}_F\le\varepsilon<\ecrit$, the perturbed operator is an
$\norm{\cdot}_2$-contraction with a unique equilibrium and $\rho(J_z')<1$. Hence
$\ebreak\ge\ecrit$, and in particular $\ebreakall\ge\ecrit$.
```
**NEW**
```latex
\item[\textnormal{(i)}] \textbf{Lower side (any $1$-Lipschitz $\phi$).} For every feasible
$\delta\Ahat$ with $\norm{\delta\Ahat}_F\le\varepsilon<\eglob$, the perturbed operator is an
$\norm{\cdot}_2$-contraction with a unique equilibrium and $\rho(J_z')<1$, with no use of (A4); hence
$\ebreak\ge\eglob$. Under (A4), $\kappa=\norm{\Ahat}_2\norm{W}_2$ gives $\eglob=\ecrit$, so
$\ebreak\ge\ecrit$ and in particular $\ebreakall\ge\ecrit$.
```
*(The bracket (iii) endpoint `ε_crit ≤ ε_br^all` is then the (A4) corollary — unchanged, since the bracket already assumes (A4).)*

### PATCH 3b — the proof body of (i)

**OLD**
```latex
\emph{(i) Lower side.} By (A1), $\norm{\diag(\phi')}_2\le1$; the spectral norm is sub-multiplicative
and Kronecker-multiplicative, so as in \eqref{eq:Jzp-bound},
$\norm{J_z'}_2\le(\norm{\Ahat}_2+\varepsilon)\norm{W}_2$, which is below $1$ for
$\varepsilon<\ecrit$. A spectral-norm contraction has a unique fixed point and $\rho(J_z')\le\norm{J_z'}_2<1$,
so no feasible $\varepsilon<\ecrit$ breaks contraction of the deployed model: $\ebreak\ge\ecrit$. This
step uses (A1)--(A3) only. The additive gap \eqref{eq:additive-gap} then gives $\ecrit\le\espec$.
```

**NEW**
```latex
\emph{(i) Lower side.} By (A1), $\norm{\diag(\phi')}_2\le1$ for every activation mask; the spectral
norm is sub-multiplicative and Kronecker-multiplicative, so as in \eqref{eq:Jzp-bound},
$\norm{J_z'}_2\le(\norm{\Ahat}_2+\varepsilon)\norm{W}_2<1$ uniformly over masks for
$\varepsilon<\eglob=1/\norm{W}_2-\norm{\Ahat}_2$. A spectral-norm contraction has a unique fixed point
and $\rho(J_z')\le\norm{J_z'}_2<1$, so no feasible $\varepsilon<\eglob$ breaks contraction:
$\ebreak\ge\eglob$ under (A1)--(A3) alone. Under the all-active operating point (A4) assumed here,
$\kappa=\norm{\Ahat}_2\norm{W}_2$, so $\eglob=\ecrit$ and $\ebreakall\ge\ecrit$. The additive gap
\eqref{eq:additive-gap} then gives $\ecrit\le\espec$.
```
*(The bracket as a whole already assumes (A4) for the upper side and the spectral law, so within its
all-active scope `ε_glob=ε_crit` and the lower side `ε_br^all ≥ ε_crit` is recovered — no weakening of
the bracket.)*

---
## PATCH 4 — `sections/abstract.tex` : T1.2 relabel (one sentence)

**OLD**
```latex
for contractive models a closed-form safe radius adds a deterministic guarantee the empirical break exceeds by $\mathbf{2}$--$\mathbf{9\times}$.
```
**NEW**
```latex
for contractive models a closed-form safe radius adds a deterministic certificate, which the \emph{measured} nonlinear break exceeds by $\mathbf{2}$--$\mathbf{9\times}$ ($10$ seeds).
```

---
## PATCH 5 — `sections/introduction.tex` : T1.2 relabel (contribution 2)

**OLD**
```latex
whose norm certificate under-states the true break by $2$--$9\times$ ($10$ seeds).
```
**NEW**
```latex
whose norm certificate under-states the \emph{measured} nonlinear break by $2$--$9\times$ ($10$ seeds; the all-active boundary itself is bracketed in closed form, \cref{thm:cf2s}).
```

---
## T1.3 note (no hard patch — pick one) — `theory.tex` / `D_boundary.tex`, the `C/β` constant

The reported `C/β≲16×` evaluates `C=g_W(1+κ)/(1−κ)` at the **deployed partial** `κ∈[0.14,0.59]`, but
`C` is *derived* (Appendix D (iii)) under the all-active identity `κ=‖Â‖‖W‖`. Evaluated self-consistently
(all-active `κ`) the constant is `≈49–155×`. The bound `ε_spec ≤ C·ε_crit` was never violated on real
models (0/802), so this is a *reporting* inconsistency, not a broken bound. Fix one of:
1. Report `C/β` at the all-active `κ` (consistent with the derivation): state ``$C/\beta$ reaches
   $\approx 50$--$155\times$ when evaluated at the all-active $\kappa=\norm{\Ahat}_2\norm{W}_2$ its
   derivation assumes; at the deployed partial $\kappa$ the realised ratio is $\le 16\times$''.
2. Or keep `≲16×` but add ``(evaluated at the deployed $\kappa$; the constant $C$ is derived all-active,
   so this is the realised, not the worst-case, enclosure)''.

Option 1 is the more defensible against a hostile reviewer.

---
## Net effect
- **Soundness:** the blocking gap is closed — `ε_glob` is a genuine global contraction certificate (A1+A2+threat-bound only), `ε_crit` is honestly the linearized scale, and the IFT gives local persistence. `thm:cf2s` lower side recovered within its all-active scope.
- **Narrative preserved (Option B):** `tab:cross_domain` keeps its per-dataset `ε_crit` column (caption relabel only); `rem:obs_o1`'s `ε_reach` is now the consistent "measured break."
- **Untouched (verified sound in `08`):** `prop:radius`, `prop:transfer` remainder, `C_v`/`‖z*‖`, AEGIS-Conformal coverage — they use Step 1, not Step 2.
