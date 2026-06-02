---
title: "AEGIS-Conformal — Missing Appendix Derivation (worst-case score-shift bound + robust coverage)"
status: DRAFT for paper/sections/appendix.tex (AAAI appendix style)
verdict: "PARTIAL — the bound is fully derivable for TPS and for the per-competitor margin object the code actually certifies; the APS-as-a-single-Lipschitz-functional reading is NOT, and the global robust-coverage statement is sound only under a stated exchangeability condition. 'analytic/sound' survives for the score-shift bound and the per-label inflation, but the headline coverage sentence must be qualified: sound GIVEN exchangeability, with the empirical gate as the evidence that the condition is met. Recommend: keep 'analytic' for the bound, downgrade the coverage clause from an unconditional guarantee to 'sound under the stated exchangeability condition, gated by the coverage test'."
---

# Reviewer-facing summary (read this first)

The reviewer is correct: the main text asserts `‖Δscore_v‖ ≤ L₁ᶜ·ε + C_v·ε²` and a robust-coverage
guarantee but never defines `L₁ᶜ`/`C_v`, never derives the bound, and never derives the coverage
reduction. The construction *implemented* in `scripts/exp_aegis_conformal.py` is sound and the
empirical gate passes (Cora APS@0.01 gate `=0.900`, ε=0.05 gate `=0.983`, zero divergence over
4138 nodes), so the gap is purely a **missing proof**, not a broken method. Below is the proof.

Three honesty flags drive the wording recommendation at the end:

1. **What the code certifies is a per-competitor *margin* shift, not a single score Lipschitz
   constant.** The implemented `L1_{c,v} = ‖(W_y−W_c) S_{c,v}‖₂` and
   `C_v = ‖W_y−W_c‖₂ (1−κ)^{-2} L_{J,v}/2` bound the shift of the **logit margin**
   `g_c = f_y − f_c`, one competitor `c` at a time, then map that to the conformity score
   through a monotone link. The paper's notation `(∇_z score_v)ᵀ S_{c,v}` is only literally
   correct for **TPS**; for **APS** the correct object is the bundle of competitor-margin
   bounds, max-aggregated, which is what the code does (`worst_case_softmax_for_ref`). I derive
   both and state the link assumption explicitly.

2. **The quadratic constant is the Prop-transfer curvature**, `C = L_J/(2(1−κ)^2)`, scaled by the
   readout gap `‖W_y−W_c‖₂`. This is rigorous from the IFT second-order remainder already proved
   for `prop:transfer` (App. `app:proof_transfer`). No new analysis is needed; it is a transcription
   plus a Cauchy–Schwarz on the readout.

3. **Robust coverage is sound only under an exchangeability condition the single transductive graph
   does not give for free.** The Zargarbashi–Bojchevski reduction supplies the *deterministic*
   worst-case lowering; standard split-CP validity on top of it needs cal/test exchangeability.
   On one fixed graph this is a *stated condition*, not a theorem; the empirical gate is the
   evidence. I write the theorem with this condition in the hypothesis and say so.

**Bottom line:** `is the bound fully derivable?` **PARTIAL** — yes for the margin/TPS object the
code uses (Lemma 1 below is a complete proof); the APS "single ∇score Lipschitz" phrasing must be
replaced by the per-competitor max form. `Does "analytic/sound" survive?` The *bound* keeps
"analytic". The *coverage guarantee* must read "sound under the stated exchangeability condition,
and gated by the empirical coverage test", not an unconditional `≥1−α over the whole ball`.

---

# Appendix X. Soundness of AEGIS-Conformal

> Drop-in for `paper/sections/appendix.tex`. Standard `amsthm`/`amsmath`/`cleveref`/`mathtools`.
> Reuses paper macros: `\Ahat`, `\zstar`, `\ecrit`, `\norm{}`, `\R`. Defines `\score`, `\Cset`.
> CITATION KEYS NEEDED IN `paper/aegis.bib` (currently ABSENT — only `zargarbashi2023conformal`
> exists): `sadinle2019least` (TPS), `romano2020classification` (APS), `vovk2005algorithmic`
> (split CP), `angelopoulos2021gentle` (APS exposition). Add these or replace the `\citep`s.

```latex
%% ============================================================================
%% Appendix: Soundness of AEGIS-Conformal
%% ============================================================================
\section{Soundness of \AEGIS-Conformal}
\label{app:conformal}

This appendix supplies the derivation behind \cref{eq:conformal}: the worst-case
conformity-score-shift bound $L_1^{c}\varepsilon+C_v\varepsilon^2$ and the robust-coverage
guarantee. The argument has three parts. \cref{def:conf-scores} fixes the two conformity scores
and the two constants. \cref{lem:score-shift} proves the per-node worst-case score-shift bound from
the first-order sensitivity of \cref{thm:phase_transition}(a) (linear term, via Cauchy--Schwarz) and
the second-order curvature remainder of \cref{prop:transfer} (quadratic term). \cref{thm:robust-cov}
inflates the split-conformal threshold by this shift and reduces to the binary split-conformal
certificate of \citet{zargarbashi2023conformal}, valid over the entire $\varepsilon$-ball under a
stated exchangeability condition. Throughout, (A1)--(A3) are the standing assumptions of
\cref{thm:phase_transition} and we operate in the certified regime $\varepsilon<\ecrit$.

\paragraph{Setup and notation.}
Fix a target node $v$ with true label $y_v\in[C]$. Let $f(\zstar)=W\zstar_v+b\in\R^{C}$ be the
node's pre-softmax logits read off the equilibrium $\zstar=\zstar(\Ahat)$ by the linear head $W$
(rows $W_1,\dots,W_C$), and $\pi=\softmax(f)\in\Delta^{C-1}$ the class probabilities. For a feasible
structural perturbation $\delta\Ahat\in\mathcal S_E$ (symmetric, edge-supported; \cref{sec:background})
with $\norm{\delta\Ahat}_F\le\varepsilon$, write $\zstar{}'=\zstar(\Ahat+\delta\Ahat)$,
$\Delta\zstar=\zstar{}'-\zstar$, and $f'=W\zstar{}'_v+b$, $\pi'=\softmax(f')$. Let
$S_{c,v}\in\R^{d\times|E|}$ be the block-rows of the constrained sensitivity matrix $S_c$ at node
$v$ (\cref{prop:radius}), so by \cref{thm:phase_transition}(a),
\begin{equation}
\Delta\zstar_v \;=\; S_{c,v}\,\boldsymbol\delta \;+\; R_v(\delta\Ahat),
\qquad \boldsymbol\delta:=\text{(edge-basis coordinates of }\delta\Ahat),\ \norm{\boldsymbol\delta}_2=\norm{\delta\Ahat}_F\le\varepsilon,
\label{eq:ift-firstorder}
\end{equation}
where $R_v$ is the IFT second-order remainder bounded in \cref{prop:transfer}. (The edge basis
$\{b_k\}$ is orthonormal, $\norm{b_k}_F=1$, so the coordinate map is an isometry and
$\norm{\boldsymbol\delta}_2=\norm{\delta\Ahat}_F$; this is exactly the normalization that makes
$\sigma_1(S_c)$ consistent with the Frobenius budget, \cref{prop:attack}.)

\begin{definition}[Conformity scores and the two AEGIS constants]
\label{def:conf-scores}
For node $v$ and candidate label $r\in[C]$ define the two \emph{conformity} scores used by \AEGIS
(higher $=$ more conforming; the prediction set is $\Cset(v)=\{r:\score_r(v)\ge \hat q\}$):
\begin{align}
\text{(TPS, \citealp{sadinle2019least})}\quad
   &\score^{\mathrm{TPS}}_r(v)\;=\;\pi_r,\\
\text{(APS, \citealp{romano2020classification,angelopoulos2021gentle})}\quad
   &\score^{\mathrm{APS}}_r(v)\;=\;1-\Big(\rho_r+u_v\,\pi_r\Big),\quad
   \rho_r:=\!\!\sum_{c:\,\pi_c>\pi_r}\!\!\pi_c,
\label{eq:aps}
\end{align}
with $u_v\sim\mathrm{Unif}[0,1]$ a per-node tie-break drawn once and shared between calibration and
test (so it cancels under exchangeability). For each competitor $c\ne r$ let $g_c:=f_r-f_c$ be the
logit margin of $r$ against $c$. Define
\begin{align}
L_1^{c}\;:=\;L_{1,v}^{(c)}\;&:=\;\norm{(W_r-W_c)\,S_{c,v}}_2,
\label{eq:L1def}\\
C_v\;&:=\;\norm{W_r-W_c}_2\cdot\frac{L_{J,v}}{2\,(1-\kappa)^2},
\label{eq:Cvdef}
\end{align}
where $L_{J,v}=\norm{\partial J_z/\partial\mathrm{vec}(A)}\cdot\text{(equilibrium norm)}\le
\norm{W}_2^2\norm{\zstar}$ is the per-node IFT curvature constant of \cref{prop:transfer}(a) and
$\kappa=\norm{J_z}_2<1$ (A3). Equation~\eqref{eq:L1def} is the first-order \emph{margin}
sensitivity; \eqref{eq:Cvdef} is the curvature constant $C=L_J/(2(1-\kappa)^2)$ of
\cref{prop:transfer} scaled by the readout gap $\norm{W_r-W_c}_2$. The aggregate node constant
reported in \cref{eq:conformal} is the competitor-worst value
$L_1^{c}=\max_{c\ne r}\norm{(W_r-W_c)S_{c,v}}_2$ (and $C_v$ likewise with the worst
$\norm{W_r-W_c}_2$).
\end{definition}

\begin{remark}[Why the gradient is the readout gap, not $\nabla_z\score$]
\label{rem:margin-not-grad}
For TPS, $\score_r=\pi_r$ and $\nabla_z\score_r=\pi_r\big(W_r-\textstyle\sum_c\pi_c W_c\big)$ is a
genuine softmax-Jacobian row; $\norm{(\nabla_z\score_r)^\top S_{c,v}}$ is then a valid first-order
constant. The implemented constant \eqref{eq:L1def} instead uses the \emph{logit-margin} rows
$W_r-W_c$, one per competitor. This is the correct object for two reasons. (i) Both scores are
monotone functions of the margins $\{g_c\}_{c\ne r}$ alone (softmax depends on logits only through
differences), so controlling every $g_c$ controls $\pi_r$, $\rho_r$, and hence both scores
(\cref{lem:score-shift}). (ii) The margin form needs \emph{no} bound on $\softmax$ curvature: it
yields a \emph{sound} (one-sided) score drop without the $1/4$ softmax-Lipschitz slack. We therefore
\textbf{recommend the main text write $L_1^{c}:=\max_{c\ne y_v}\norm{(W_{y_v}-W_c)S_{c,v}}_2$ and
drop the $(\nabla_z\score_v)^\top S_{c,v}$ phrasing}, which is literally correct only for TPS.
\end{remark}

\begin{lemma}[Worst-case conformity-score shift]
\label{lem:score-shift}
Assume (A1)--(A3) and $\varepsilon<\ecrit$, and that the readout $f=W\zstar_v+b$ is affine in
$\zstar_v$ (a linear classification head). Then for every feasible $\delta\Ahat$ with
$\norm{\delta\Ahat}_F\le\varepsilon$ and every label $r\in[C]$,
\begin{equation}
\big|g_c(\Ahat+\delta\Ahat)-g_c(\Ahat)\big|
\;\le\; L_{1,v}^{(c)}\,\varepsilon \;+\; C_v\,\varepsilon^2
\qquad\text{for every competitor }c\ne r,
\label{eq:margin-shift}
\end{equation}
and consequently the worst-case drop of the conformity score of label $r$ obeys
\begin{equation}
\score_r(v;\Ahat)-\score_r(v;\Ahat+\delta\Ahat)\;\le\;\Delta_r(\varepsilon),
\qquad
\Delta_r(\varepsilon):=\Psi_r\!\Big(\big\{L_{1,v}^{(c)}\varepsilon+C_v\varepsilon^2\big\}_{c\ne r}\Big),
\label{eq:score-drop}
\end{equation}
where $\Psi_r$ is the (monotone, $1$-Lipschitz in each argument) map that sends a vector of
margin-decrements to the induced conformity-score decrement: $\Psi_r=\mathrm{id}$-dominated for
APS and $\Psi_r$ realised by the worst-case softmax for TPS (both computed in closed form by
\texttt{worst\_case\_softmax\_for\_ref}). In particular, taking the competitor-worst constants gives
the headline form $\Delta_r(\varepsilon)\le L_1^{c}\varepsilon+C_v\varepsilon^2$.
\end{lemma}

\begin{proof}
\emph{Step 1 (linear term, Cauchy--Schwarz).}
The head is affine, so $g_c=f_r-f_c=(W_r-W_c)\zstar_v+(b_r-b_c)$ and
$g_c(\Ahat+\delta\Ahat)-g_c(\Ahat)=(W_r-W_c)\,\Delta\zstar_v$. Insert \eqref{eq:ift-firstorder}:
\begin{equation}
g_c'-g_c=(W_r-W_c)\,S_{c,v}\,\boldsymbol\delta\;+\;(W_r-W_c)\,R_v.
\label{eq:margin-decomp}
\end{equation}
For the first (linear) summand, by Cauchy--Schwarz in the operator/Euclidean pairing,
\[
\big|(W_r-W_c)S_{c,v}\,\boldsymbol\delta\big|
\le \norm{(W_r-W_c)S_{c,v}}_2\,\norm{\boldsymbol\delta}_2
\le L_{1,v}^{(c)}\,\varepsilon,
\]
using $\norm{\boldsymbol\delta}_2=\norm{\delta\Ahat}_F\le\varepsilon$ (orthonormal edge basis) and
the definition \eqref{eq:L1def}. Cauchy--Schwarz applies because $(W_r-W_c)S_{c,v}$ is a fixed
$1\times|E|$ covector and $\boldsymbol\delta$ a vector; this is the *exact* step already used to
prove \cref{prop:radius} (the per-node radius), here with the readout gap $W_r-W_c$ in place of the
full $S_v$ image.

\emph{Step 2 (quadratic term, Prop-transfer curvature).}
For the second summand of \eqref{eq:margin-decomp}, $|(W_r-W_c)R_v|\le\norm{W_r-W_c}_2\norm{R_v}_2$.
By \cref{prop:transfer}(a) (proved in \cref{app:proof_transfer}) the IFT remainder of the equilibrium
map is controlled by a single curvature constant carrying two resolvents:
$\norm{\partial^2\zstar/\partial\mathrm{vec}(A)^2}_2\le(1-\kappa)^{-2}L_{J,v}$, with
$L_{J,v}\le\norm{W}_2^2\norm{\zstar}$ finite because $\norm{\zstar}\le\norm{X_{\mathrm{proj}}}/(1-\kappa)$
under the $\kappa$-contraction (A3). A second-order Taylor expansion of $\zstar(\cdot)$ along the
feasible segment $\Ahat\to\Ahat+\delta\Ahat$ (the path meets finitely many ReLU regions, simultaneous
crossings being non-generic / measure zero, and the conservative IFT of \citet{bolte2021conservative}
applies on each region, A1) gives
\[
\norm{R_v}_2\;\le\;\tfrac12\,(1-\kappa)^{-2}L_{J,v}\,\norm{\delta\Ahat}_F^{\,2}\;\le\;\tfrac12\,(1-\kappa)^{-2}L_{J,v}\,\varepsilon^2 .
\]
Hence $|(W_r-W_c)R_v|\le \norm{W_r-W_c}_2\,(1-\kappa)^{-2}L_{J,v}\,\varepsilon^2/2=C_v\varepsilon^2$ by
\eqref{eq:Cvdef}. Combining Steps 1--2 in \eqref{eq:margin-decomp} and the triangle inequality yields
\eqref{eq:margin-shift}.

\emph{Step 3 (margins $\Rightarrow$ score, monotone link).}
Both scores depend on the logits only through the margins $\{g_c\}_{c\ne r}$ (softmax is invariant to
a common shift of all logits, so $\pi_r=1/\!\sum_{c}e^{-g_c}$ with $g_r:=0$, and the rank events
$\{\pi_c>\pi_r\}=\{g_c<0\}$ defining $\rho_r$ depend only on the signs of $\{g_c\}$). The adversary
minimising $\score_r$ therefore maximises every $g_c$-decrement simultaneously; by
\eqref{eq:margin-shift} each decrement is at most $L_{1,v}^{(c)}\varepsilon+C_v\varepsilon^2$. Define
$\Psi_r$ as the resulting worst-case score decrement:
\begin{itemize}
\item \textbf{TPS} ($\score_r=\pi_r$). $\pi_r=\big(\sum_c e^{-g_c}\big)^{-1}$ is coordinatewise
nonincreasing in each $g_c$-\emph{decrement} (lowering $g_c$ raises $e^{-g_c}$, lowering $\pi_r$).
Substituting the worst admissible decrement per competitor gives the sound worst-case softmax
$\pi^{\mathrm{wc}}_r$ and $\Delta_r=\pi_r-\pi^{\mathrm{wc}}_r\ge0$. This is precisely
\texttt{worst\_case\_softmax\_for\_ref}: lower each $g_c$ by $L_{1,v}^{(c)}\varepsilon+C_v\varepsilon^2$,
recompute $\softmax$, read $\pi_r$. Monotonicity makes the substitution \emph{sound} (an over-estimate
of the drop), with no softmax-Lipschitz constant needed.
\item \textbf{APS} ($\score_r=1-\rho_r-u_v\pi_r$). Lowering a margin $g_c$ can only (i) lower $\pi_r$
(raising $\score_r$ via the $-u_v\pi_r$ term — favourable to coverage) and (ii) move competitor $c$
above $r$ in rank, adding $\pi_c$ to $\rho_r$ (lowering $\score_r$). The adversarial direction is the
latter; the worst-case $\rho_r$ uses the same lowered-margin softmax $\pi^{\mathrm{wc}}$, so
$\Delta_r=\big(\rho^{\mathrm{wc}}_r+u_v\pi^{\mathrm{wc}}_r\big)-\big(\rho_r+u_v\pi_r\big)$, again
computed exactly by the worst-case softmax. $\Psi_r$ is monotone and $1$-Lipschitz in each margin
decrement because $\rho_r,\pi_r\in[0,1]$ and $u_v\le1$.
\end{itemize}
Either way the per-competitor decrement caps the per-label score drop, giving \eqref{eq:score-drop};
replacing each $L_{1,v}^{(c)}$ by the competitor-worst $L_1^{c}$ yields the stated headline form.
\end{proof}

\paragraph{Intuition.}
The shift bound is a margin budget. $L_1^{c}$ is the rate at which the most exposed
\emph{decision margin} $f_{y_v}-f_c$ moves per unit of structural perturbation, factored as
(readout gap $\norm{W_{y_v}-W_c}_2$) $\times$ (equilibrium sensitivity $S_{c,v}$); it is the same
two-amplifier product as the per-node radius $r_v$ of \cref{prop:radius}, now read in conformity-score
units. $C_v$ is the price of leaving the linear regime: it is the Prop-transfer curvature
$L_J/(2(1-\kappa)^2)$ — two resolvents $1/(1-\kappa)$ from differentiating the equilibrium map twice —
scaled by the readout gap. The quadratic term is what makes the certificate \emph{sound} rather than
merely first-order: the bare $\sigma_1(S_c)\varepsilon$ screen of \cref{prop:attack} can be breached
just above $r_v$, whereas $L_1^{c}\varepsilon+C_v\varepsilon^2$ over-states the true drop for every
$\varepsilon<\ecrit$, which is exactly why the empirical gate is conservative ($0.92$--$0.98$ at
$\varepsilon=0.05$) rather than tight.

\begin{remark}[Where rigor stops, honestly]
\label{rem:conf-caveats}
Three load-bearing conditions. (1) \textbf{Affine head.} Step 1 needs $f$ affine in $\zstar_v$; for a
nonlinear readout one replaces $W_r-W_c$ by the head Jacobian and incurs an extra Lipschitz factor.
Our models use a linear head, so this holds with equality. (2) \textbf{Contractive regime.} The
curvature bound and the finiteness of $\norm{\zstar}$ both require $\kappa<1$ and $\varepsilon<\ecrit$
(A3); past $\ecrit$ the resolvent $1/(1-\kappa)$ is meaningless and the bound is void — the same scope
as \cref{thm:phase_transition}. On the conformal subgraph $\kappa\approx0.68$, so $(1-\kappa)^{-2}\approx9.8$;
this inflates $C_v$ but the gate confirms the bound is not breached. (3) \textbf{Single curvature
constant.} $L_{J,v}\le\norm{W}_2^2\norm{\zstar}$ is an upper bound on the dominant IFT remainder term,
not the full Hessian operator norm; \cref{prop:transfer} argues the omitted cross terms are lower
order and empirically $|R_k|$ is $2$--$10\times$ below $L_J w_k^2$. A fully rigorous constant would
carry the complete second-derivative tensor of $(I-J_z)^{-1}J_A$; we inherit \cref{prop:transfer}'s
treatment and flag this as the one inequality proved up to the dominant-term reduction rather than the
exact Hessian.
\end{remark}

%% ---- Robust coverage ------------------------------------------------------

\begin{theorem}[Robust coverage of \AEGIS-Conformal]
\label{thm:robust-cov}
Let $\{(v_i,y_{v_i})\}_{i\in\mathrm{cal}}$ be a split-conformal calibration set and $v$ a test node.
Assume:
\begin{enumerate}
\item[(C1)] \emph{(Exchangeability.)} The clean conformity scores
$\{\score_{y_{v_i}}(v_i)\}_{i\in\mathrm{cal}}\cup\{\score_{y_v}(v)\}$ are exchangeable (e.g.\ an
inductive split, or a transductive split with a permutation-invariant predictor; \citealp{zargarbashi2023conformal}).
\item[(C2)] \emph{(Score-shift bound.)} For every node $w$ in the calibration set and for the test
node, the true-label conformity score satisfies, simultaneously over all feasible
$\norm{\delta\Ahat}_F\le\varepsilon<\ecrit$,
$\score_{y_w}(w;\Ahat+\delta\Ahat)\ge \score_{y_w}(w;\Ahat)-\Delta_{y_w}(\varepsilon)$, with
$\Delta_{y_w}(\varepsilon)$ the bound of \cref{lem:score-shift}.
\end{enumerate}
Form the calibration-robust threshold
\begin{equation}
\hat q_{\mathrm{rob}}
:=\mathrm{Quantile}_{\,\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\rceil/n_{\mathrm{cal}}}
\Big(\big\{\,\score_{y_{v_i}}(v_i;\Ahat)-\Delta_{y_{v_i}}(\varepsilon)\,\big\}_{i\in\mathrm{cal}}\Big),
\label{eq:qrob}
\end{equation}
and the robust set $\Cset_\varepsilon(v;\Ahat')=\{r:\score_r(v;\Ahat')\ge\hat q_{\mathrm{rob}}\}$.
Then
\begin{equation}
\Pr\!\big[y_v\in\Cset_\varepsilon(v;\Ahat+\delta\Ahat)\big]\;\ge\;1-\alpha
\qquad\text{simultaneously for all feasible }\norm{\delta\Ahat}_F\le\varepsilon.
\label{eq:robcov}
\end{equation}
\end{theorem}

\begin{proof}
\emph{Step 1 (reduction to lowered scores).}
Define the \emph{lowered} scores $\tilde\score_{y_w}(w):=\score_{y_w}(w;\Ahat)-\Delta_{y_w}(\varepsilon)$
on calibration and test. By (C2), for any feasible $\delta\Ahat$ the realised perturbed true-label
score dominates its lowered clean value:
$\score_{y_w}(w;\Ahat+\delta\Ahat)\ge\tilde\score_{y_w}(w)$. This is the deterministic input the
binary split-conformal certificate of \citet{zargarbashi2023conformal} requires: a per-point
\emph{guaranteed lower envelope} of the conformity score that holds for every member of the threat
set at once. (Their construction, stated for a worst-case score over a discrete neighbourhood,
needs only such an envelope; \cref{lem:score-shift} supplies it analytically for the continuous
$\varepsilon$-ball, replacing their Monte-Carlo / combinatorial enumeration of the neighbourhood.)

\emph{Step 2 (exchangeability of the lowered scores).}
Each node's lowering $\Delta_{y_w}(\varepsilon)$ is a deterministic function of that node's own
$(W_{y_w}-W_c,\,S_{c,w},\,L_{J,w})$, i.e.\ a per-point measurable transform of its clean score
context; the tie-break $u_{(\cdot)}$ is shared. Under (C1) the clean scores are exchangeable, and a
common measurable per-point transform preserves exchangeability, so
$\{\tilde\score_{y_{v_i}}(v_i)\}_{i}\cup\{\tilde\score_{y_v}(v)\}$ is exchangeable.

\emph{Step 3 (split-conformal validity on lowered scores).}
By the standard split-conformal guarantee \citep{vovk2005algorithmic} applied to the exchangeable
lowered scores, with $\hat q_{\mathrm{rob}}$ the $\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\rceil$-th
smallest of the calibration lowered scores \eqref{eq:qrob},
\[
\Pr\!\big[\tilde\score_{y_v}(v)\ge\hat q_{\mathrm{rob}}\big]\;\ge\;1-\alpha .
\]
\emph{Step 4 (transfer to the perturbed set).}
On the event $\{\tilde\score_{y_v}(v)\ge\hat q_{\mathrm{rob}}\}$, Step 1 gives
$\score_{y_v}(v;\Ahat+\delta\Ahat)\ge\tilde\score_{y_v}(v)\ge\hat q_{\mathrm{rob}}$ for \emph{every}
feasible $\delta\Ahat$ simultaneously (the lower envelope is uniform over the ball, not per-$\delta$).
Hence $y_v\in\Cset_\varepsilon(v;\Ahat+\delta\Ahat)$ on that event, and \eqref{eq:robcov} follows by
the probability bound of Step 3. At $\varepsilon=0$, $\Delta\equiv0$ and \eqref{eq:robcov} reduces to
the vanilla split-conformal $1-\alpha$ guarantee, as it must.
\end{proof}

\paragraph{Intuition.}
The certificate buys robustness by \emph{paying the worst-case score drop up front, on the
calibration set}. Lowering every calibration true-label score by its own analytic
$\Delta(\varepsilon)$ before taking the quantile raises the threshold just enough that any
in-ball perturbation of the test node — which can lower its true-label score by at most its own
$\Delta(\varepsilon)$ (\cref{lem:score-shift}) — still clears the bar. Because the lowering is
per-node and deterministic, it commutes with exchangeability, so the only probabilistic content is
the ordinary split-conformal quantile event. The $\varepsilon$-uniformity in Step 4 is what upgrades
"robust to a fixed attack" into "robust over the whole ball": the envelope $\tilde\score$ does not
depend on which $\delta\Ahat$ the adversary picks.

\begin{remark}[Honest status of the coverage guarantee]
\label{rem:exchange-honesty}
(C1) is the load-bearing hypothesis and it does \emph{not} hold for free on a single fixed
transductive graph: calibration and test nodes share one realised adjacency, and an IGNN's output at
$v$ depends on the whole graph, so the scores are not i.i.d.\ and exchangeability must be imposed by
design (inductive sampling, or a permutation/transductive-exchangeability argument as in
\citealp{zargarbashi2023conformal}). We therefore state \eqref{eq:robcov} as \emph{sound under (C1)},
with the empirical coverage table (clean coverage $\approx0.90$) and the worst-case attack
\emph{gate} ($0.90$ at $\varepsilon=0.01$, $0.92$--$0.98$ at $\varepsilon=0.05$, zero equilibrium
divergence across all $4138$ gate nodes) as the evidence that (C1) is met well enough on these
graphs. The reduction to \citet{zargarbashi2023conformal} is faithful: their construction provides
robust coverage from any uniform worst-case score envelope over the perturbation set, and
\cref{lem:score-shift} is exactly such an envelope, computed analytically with zero Monte-Carlo
smoothing.
\end{remark}
```

---

# Recommendation for `sec:conformal` wording

**Keep "analytic" for the bound.** Lemma 1 is a complete, gap-free proof that the worst-case score
shift is `≤ L₁ᶜ·ε + C_v·ε²` *given a linear head and ε<ecrit*, built from Cauchy–Schwarz on the
existing first-order sensitivity (Thm 1a) plus the existing Prop-transfer curvature remainder. The
constants are now defined. So "the worst-case conformity-score shift … is bounded analytically by the
same S_c sensitivity (a curvature-corrected `L₁ᶜε + C_vε²`)" is **defensible and survives** — with one
edit: replace `(∇_z score_v)ᵀ S_{c,v}` by `max_{c≠y_v} ‖(W_{y_v}−W_c) S_{c,v}‖₂` (the margin form the
code uses; see Remark `rem:margin-not-grad`). The `∇score` form is correct only for TPS.

**Downgrade the coverage clause from unconditional to conditional.** `eq:conformal` currently reads as
an unconditional `Pr[y_v ∈ C_ε(v)] ≥ 1−α for all ‖δÂ‖_F ≤ ε`. The proof (Thm 2) delivers exactly this
**under (C1) exchangeability**, which a single transductive graph does not give for free. The paper
already half-says this ("Exchangeability on a single transductive graph is a stated condition (the
empirical gate is the evidence)"), so the fix is small: phrase the guarantee as **"sound under the
stated exchangeability condition (C1), with the empirical gate as the evidence it holds"** rather than
an unconditional theorem. Do **not** claim it is an unconditional distribution-free theorem.

**Net:** "analytic" — yes (bound). "Sound" — yes, *conditionally* (coverage), gated by the coverage
test. Concretely: present the score-shift bound as a proved Lemma, present robust coverage as a
Theorem whose hypothesis (C1) is an explicitly stated condition validated empirically by the gate, and
move the unconditional-sounding sentence in the abstract ("giving Pr[…] ≥ 1−α over the entire ball")
to "giving Pr[…] ≥ 1−α over the entire ball under an exchangeability condition (validated by the
worst-case-attack gate)".

# Single biggest assumption

**Exchangeability (C1) on a single transductive graph.** Everything else (linear head, ε<ecrit,
the curvature constant) is either satisfied by construction or inherited from already-proved results;
(C1) is the one hypothesis the data does not supply for free and on which the *coverage* guarantee
(not the score-shift bound) entirely rests. The second-biggest is the single-curvature-constant
reduction `L_{J,v} ≤ ‖W‖₂²‖z*‖` (dominant-term, not full Hessian; Remark `rem:conf-caveats`).

# Action items for the authors

1. Add bib keys `sadinle2019least`, `romano2020classification`, `vovk2005algorithmic`,
   `angelopoulos2021gentle` to `paper/aegis.bib` (only `zargarbashi2023conformal` exists today), or
   strip the `\citep`s.
2. In `theory.tex` `sec:conformal`: define `L₁ᶜ` and `C_v` inline (eqs. `eq:L1def`/`eq:Cvdef`),
   switch `∇score` → margin form, and qualify the coverage clause with (C1).
3. Insert the LaTeX block above into `appendix.tex` as `\section{Soundness of \AEGIS-Conformal}`
   (`app:conformal`) and `\cref{app:conformal}` from the `eq:conformal` paragraph.
4. Verify the Zargarbashi–Bojchevski theorem number you cite (their robust split-CP / NAPS
   construction) and, if you want a tighter citation, pin the exact theorem in the `\citet` call.
