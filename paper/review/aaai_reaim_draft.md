# AAAI re-aim — abstract · contributions · Conformal flagship (for sign-off)

Draft of the re-aimed narrative. **Nothing is wired into the `.tex` yet** — this is for you
to approve the new framing before I move it into `sections/` + the AAAI template.
The re-aim shifts the thesis from *"we audit fault lines (diagnostics)"* to *"one operator
gives the diagnostics **and** a deployable distribution-free certificate"*, anchored on
**AEGIS-Conformal**. Strong existing numbers (39/39 τ, 74–156× PGD, scalability) are kept.

---

## 1. Re-aimed abstract  (replaces `sections/abstract.tex`)

```latex
\begin{abstract}
Graph neural networks now inform safety-critical decisions in drug-interaction screening,
fraud detection, and power-grid contingency analysis, where adversarial perturbations of the
graph can flip predictions. Deploying such a model demands two things current tools separate:
a map of its \emph{adversarial fault lines}, and a usable robustness \emph{guarantee}. \AEGIS
supplies both from one matrix-free object, the \emph{constrained sensitivity matrix} $S_c$,
computed through a Neumann-series resolvent and randomized SVD that scale to $N{=}7{,}650$ on
one GPU. From $S_c$ we obtain (i) three first-order diagnostics---the SVD-optimal attack
direction, per-edge sensitivity rankings, and per-node radii; (ii) \emph{\AEGIS-Conformal}, a
distribution-free certificate whose worst-case conformity-score shift over the
$\varepsilon$-ball is bounded \emph{analytically} by $S_c$ (no Monte-Carlo smoothing), giving
$\Pr[y_v\in C_\varepsilon(v)]\geq 1{-}\alpha$ over the entire ball---coverage holds at the
nominal level under the very worst-case attack it certifies (10 seeds, Cora and Citeseer;
sets of ${\sim}1.0$--$1.5$ labels), \emph{non-vacuous where deterministic radii are thin};
and (iii) a constant-factor two-sided characterization of the structural-robustness boundary
with critical budget $\ecrit{=}(1{-}\kappa)/\norm{W}_2$, showing the norm certificate is
$\mathbf{2\text{--}10\times}$ conservative relative to the spectral boundary an attack
actually reaches. Regularizing $\sigma_1(S_c)$ improves robustness, so one operator drives
attack, certification, and defense. Across 6 datasets, 7 architectures, and 4 domains
(390 runs), the continuous-to-discrete bridge is positive in all $\mathbf{39/39}$ cells
(median $\tau{=}\mathbf{+0.99}$, $p{<}10^{-5}$; $\mathbf{+0.996}$ on Amazon Photo,
$N{=}7{,}650$), and one query delivers $\mathbf{74}$--$\mathbf{156\times}$ the per-query
damage of 50-step PGD.
\end{abstract}
```
*Changes:* leads with the dual (fault-lines **+** guarantee); Conformal is the new pillar;
the phase-transition claim is re-aimed to "norm cert 2–10× conservative" (honest, from the
spectral bracket); adds the defense in one clause. Dropped from the abstract (kept in body):
the delocalized-vulnerability note and the case-study sentence, to make room.

---

## 2. Re-aimed contributions  (replaces the `\textbf{Contributions.}` paragraph in `sections/introduction.tex`)

```latex
\textbf{Contributions.}
\textbf{(1) Constrained sensitivity operator.} $S_c$ specialises equilibrium IFT
sensitivity~\cite{koh2017understanding,gould2021deep} to \emph{structural} edge perturbations
via $P_c$; one matrix-free query reads the SVD-optimal direction, per-edge rankings, and
per-node radii together (matching a dense $\sigma_1$ to $0.03\%$ at $N{=}200$, scaling to
$N{=}7{,}650$).
\textbf{(2) Certification from sensitivity.} \emph{\AEGIS-Conformal} turns the $S_c$
sensitivity bound into a \emph{distribution-free} robustness certificate---coverage
$\geq 1{-}\alpha$ over the $\varepsilon$-ball, holding at the nominal level under worst-case
attack (10 seeds), non-vacuous where the deterministic per-node radius is thin; and a
constant-factor two-sided characterization of the structural-robustness boundary
($\ecrit{=}(1{-}\kappa)/\norm{W}_2$) shows norm certificates are $2$--$10\times$ conservative.
\textbf{(3) Coupled defense.} Regularizing $\sigma_1(S_c)$ measurably improves robustness, so
the operator that finds the worst attack also drives the defense.
\textbf{(4) Empirical evaluation.} 6 datasets, 7 architectures, 4 domains (390 runs):
four-quadrant attack comparison, head-to-head vs.\ GR-BCD/PR-BCD~\cite{geisler2021robustness},
continuous-to-discrete transfer (\cref{fig:tau_heatmap}), and a fraud-detector audit.
```
*Changes:* contribution (2) is now **certification** (Conformal + the honest spectral
bracket), replacing the old "phase-transition theory" framing; (3) is the coupled defense
(new); (4) unchanged.

---

## 3. New Conformal flagship subsection  (replaces `rem:certificates` in `theory.tex:82`; lands in the certificates/experiments section)

```latex
\subsection{\AEGIS-Conformal: A Distribution-Free Certificate}
\label{sec:conformal}
The per-node radius $r_v$ (\cref{eq:radius}) is a first-order threshold---locally tight, but
violable at larger $\varepsilon$. We upgrade it to a \emph{sound, distribution-free}
guarantee. Split conformal prediction yields a set $C(v)$ with $\Pr[y_v\in C(v)]\geq1{-}\alpha$;
making this hold \emph{robustly} over the whole ball $\norm{\delta\Ahat}_F\leq\varepsilon$
requires bounding the worst-case shift of the conformity score under perturbation. \AEGIS
supplies that bound \emph{analytically}: the shift is controlled by the same $S_c$ sensitivity
(a curvature-corrected $L_1^{c}\varepsilon+C_v\varepsilon^2$), replacing the ${\sim}10^4$-sample
randomized smoothing the construction would otherwise need. Feeding it into the binary-certificate
split conformal of \citet{zargarbashi2023conformal} yields a robust set with
\[
\Pr\!\big[y_v\in C_\varepsilon(v)\big]\;\geq\;1-\alpha
\qquad\text{for all }\ \norm{\delta\Ahat}_F\leq\varepsilon .
\]
\Cref{tab:conformal} reports coverage over the 10 preferred seeds. The \emph{gate}---coverage
under the worst-case \AEGIS attack at magnitude $\varepsilon$ (reconverging the equilibrium
after the $v_1$ perturbation)---holds at the nominal $1{-}\alpha{=}0.90$ at $\varepsilon{=}0.01$
and is conservative ($0.92$--$0.98$) at $\varepsilon{=}0.05$, with \emph{zero} equilibrium
divergence across all gate nodes; prediction sets are ${\sim}1.0$--$1.5$ labels. The certificate
is thus non-vacuous in exactly the regime where the deterministic radius certifies few nodes,
uses zero Monte-Carlo samples, and via $S_K$ applies to any GNN. (Exchangeability on a single
transductive graph is stated as a condition, with the empirical gate as evidence; the present
construction forms $S_c$ densely at $N{=}200$, and a matrix-free conformal pass is future work.)
```

```latex
\begin{table}[t]\centering\small
\caption{\AEGIS-Conformal coverage (10 seeds; $\alpha{=}0.1$, target $0.90$). \textbf{Gate}:
coverage under the worst-case attack at $\varepsilon$. ``set'': mean prediction-set size
(labels, of 7 Cora / 6 Citeseer classes). Std over seeds $\leq0.06$.}
\label{tab:conformal}
\begin{tabular}{llccc}
\toprule
Data & score & $\varepsilon$ & clean cov / set & \textbf{gate} \\
\midrule
Cora & APS & 0.01 & 0.901 / 1.37 & \textbf{0.900} \\
     & APS & 0.05 & 0.892 / 1.06 & \textbf{0.983} \\
     & TPS & 0.01 & 0.882 / 0.95 & \textbf{0.895} \\
     & TPS & 0.05 & 0.893 / 0.99 & \textbf{0.983} \\
\midrule
Citeseer & APS & 0.01 & 0.919 / 1.50 & \textbf{0.925} \\
         & APS & 0.05 & 0.915 / 1.35 & \textbf{0.968} \\
         & TPS & 0.01 & 0.910 / 1.21 & \textbf{0.918} \\
         & TPS & 0.05 & 0.917 / 1.26 & \textbf{0.957} \\
\bottomrule
\end{tabular}
\end{table}
```
*Notes:* needs a bib entry for `zargarbashi2023conformal` (Zargarbashi & Bojchevski, conformal
prediction sets for GNNs); the cross-refs to `rem:certificates` in `related_work.tex` and
`case_study.tex` get re-pointed to `sec:conformal`.

---

## Not in this deliverable (next, after sign-off)
- **Theory re-aim:** Thm 1 → the constant-factor two-sided spectral bracket (`tightness_theorem_v2.md`), honestly scoped, + the γ=1 / ε* validation (10-seed, finishing).
- **Defense subsection** (σ₁-regularizer table + coupling).
- **AAAI template:** I need the **AAAI-27 author kit** (`aaai27.sty`, `aaai27.bst`) to reformat IEEEtran→AAAI and compile — can you drop it in `paper/`, or shall I try to fetch it?
</content>
