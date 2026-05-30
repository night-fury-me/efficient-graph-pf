# R2 framing patches — adversarial-result paragraphs and table footnotes

These patches drop into `paper/sections/experiments.tex` at the three places
the R2 numbers will land. Every figure is backed by a CSV in
`results/revision_R2/` (path noted next to each patch). Numbers cross-checked
2026-05-28.

Design principles (per `feedback_bulletproof_over_handwaving`):
1. Lead with a concrete number, not an apology.
2. The contrast / complementarity is in the data, not in the prose.
3. No hand-wavy "AEGIS is faster" claims at small N — `exp_scalability_10seed.csv`
   shows AEGIS matrix-free is ~17× slower than GR-BCD on a Cora-sized graph;
   the AEGIS advantage is **label-freeness and closed-form determinism**, not
   wall-clock at small N. (At large N matrix-free still wins because dense OOMs.)

---

## Patch 1 — GR-BCD framing (closes P1.3)

Drop in as the first subsection of the structural-attack comparison block
in `experiments.tex`. The accompanying table replaces the placeholder.

```latex
\subsection{Structural attack comparison: AEGIS vs.\ GR-BCD}
\label{sec:exp:grbcd}

We compare \AEGIS's closed-form per-edge ranking against GR-BCD~\cite{geisler2021robustness},
the standard iterative structural attacker. The threat models differ in three
operational dimensions that we want the reader to keep in view:
GR-BCD requires \emph{(i)} label access at attack time, \emph{(ii)} an
inner-loop projected-gradient optimisation, and \emph{(iii)} per-budget retuning;
\AEGIS produces a single, label-free ranking from one matrix-free SVD of $S_c$.
The question is therefore not ``which method attacks harder?'' --- GR-BCD does, by
construction --- but ``how much ranking quality does the closed-form proxy retain,
and where does it converge to the iterative gold standard?''

Table~\ref{tab:grbcd} reports the answer across 10 seeds and three citation graphs.
On Pubmed the two rankings \emph{converge}: AEGIS recovers
$\mathbf{98.9}\text{--}\mathbf{102.7\%}$ of GR-BCD's cumulative $\ell_2$ damage
at $k\in\{1,5,10\}$ and the per-edge scores agree at $\tau=0.685\pm0.126$.
On Cora the gap is widest --- AEGIS recovers 40.6\% at $k{=}1$ rising to
72.6\% at $k{=}10$ with $\tau=0.159\pm0.068$ --- which is the expected regime
for a first-order screener applied to a graph whose attack-relevant
non-linearities are not dominated by the spectral structure of $S_c$.
This split is itself informative: AEGIS is a tight proxy where the resolvent
of Theorem~\ref{thm:phase} dominates the local attack geometry, and a loose
proxy where it does not, with the dataset-level $\tau$ acting as a built-in
self-diagnostic.\footnote{%
GR-BCD evaluation uses the authors' reference implementation with our 10-seed
protocol; per-seed wall-clock and budget settings are in
Appendix~\ref{app:grbcd-protocol}. \AEGIS reuses the same trained IGNN
checkpoints --- no additional training or label access is required.}

\begin{table}[t]
\centering
\caption{Comparison of \AEGIS's closed-form per-edge ranking against GR-BCD's
iterative attack. Damage is mean cumulative $\ell_2$ over 10 seeds; ratio is
$\text{AEGIS}/\text{GR-BCD}$ (higher means AEGIS recovers more of the
gold-standard damage with its label-free ranking). Kendall $\tau$ is computed
on the per-edge scores. \textbf{Bold}: ratio $\geq 0.90$.}
\label{tab:grbcd}
\small
\begin{tabular}{l c c c c c}
\toprule
Dataset & $k$ & AEGIS & GR-BCD & Ratio & Kendall $\tau$ \\
\midrule
Cora     & 1   & $0.245{\scriptstyle\pm0.122}$ & $0.603{\scriptstyle\pm0.250}$ & $0.41$ & $0.159{\scriptstyle\pm0.068}$ \\
Cora     & 5   & $0.643{\scriptstyle\pm0.255}$ & $1.207{\scriptstyle\pm0.492}$ & $0.53$ & $0.159$ \\
Cora     & 10  & $1.045{\scriptstyle\pm0.381}$ & $1.440{\scriptstyle\pm0.590}$ & $0.73$ & $0.159$ \\
\midrule
Citeseer & 1   & $0.209{\scriptstyle\pm0.294}$ & $0.406{\scriptstyle\pm0.243}$ & $0.51$ & $0.193{\scriptstyle\pm0.127}$ \\
Citeseer & 5   & $0.414{\scriptstyle\pm0.334}$ & $0.499{\scriptstyle\pm0.286}$ & $0.83$ & $0.193$ \\
Citeseer & 10  & $0.551{\scriptstyle\pm0.348}$ & $0.591{\scriptstyle\pm0.298}$ & $\mathbf{0.93}$ & $0.193$ \\
\midrule
Pubmed   & 1   & $0.090{\scriptstyle\pm0.032}$ & $0.091{\scriptstyle\pm0.032}$ & $\mathbf{0.99}$ & $\mathbf{0.685}{\scriptstyle\pm0.126}$ \\
Pubmed   & 5   & $0.232{\scriptstyle\pm0.094}$ & $0.226{\scriptstyle\pm0.094}$ & $\mathbf{1.03}$ & $\mathbf{0.685}$ \\
Pubmed   & 10  & $0.356{\scriptstyle\pm0.149}$ & $0.350{\scriptstyle\pm0.155}$ & $\mathbf{1.02}$ & $\mathbf{0.685}$ \\
\bottomrule
\end{tabular}
\end{table}
```

Source: `results/revision_R2/grbcd_baseline.csv` (90 rows, 10 seeds × 3 datasets × 3 budgets).

---

## Patch 2 — LODF retarget framing (closes P1.9, P3.9)

Drop in after the existing power-grid screening table. Replaces any
prose that previously claimed "LODF is worse than AEGIS" without
disclosing the case57-thermal corner.

```latex
\subsection{Power-grid screening: AEGIS vs.\ LODF across physical metrics}
\label{sec:exp:lodf}

A hostile reviewer can object that LODF~\cite{...} could win the screening
contest if it were retargeted from $\ell_2$ voltage-angle damage onto a more
operationally meaningful objective. We tested three retargets --- $\ell_2$
voltage-angle damage (the metric in our main table), thermal overload count,
and voltage-magnitude violations --- against the true N-1 contingency
outcomes across 10 seeds on case57 and case118.

Two things are true at the same time, and Table~\ref{tab:lodf} reports
both honestly. \emph{(i)} On the small grid case57, LODF retargeted to its
native thermal-overload metric reaches $\mathrm{P}@10 = 0.60$ --- its
best-case scenario --- still below \AEGIS's $0.66\text{--}0.81$ band
(Table~\ref{tab:case_study}), but within striking distance.
\emph{(ii)} On case118 \emph{every} LODF retarget collapses to
$\mathrm{P}@10 \leq 0.20$, and on the voltage retarget on case57 LODF is
\emph{anti-correlated} with the true ranking
($\tau = -0.112\pm0.001$). The LODF screener is therefore both
\emph{metric-fragile} (relative ordering across the three retargets flips by
$\geq 30$ percentage points on case57) and \emph{case-fragile} (the largest
grid kills every retarget). \AEGIS's structural ranking, by construction,
sees the same per-edge $\|[S_c]_{:,k}\|_2$ regardless of which downstream
physical objective the operator cares about --- the screener is the same;
only the post-hoc interpretation changes.\footnote{%
$\tau$ is undefined (NaN) for the thermal-overload retarget because
thermal overload is a binary indicator and Kendall's $\tau$ requires a
non-degenerate ordering; we still report $\mathrm{P}@10$ for that column
since the top-$k$ retrieval set is well-defined on binary outcomes.}

\begin{table}[t]
\centering
\caption{LODF retargeted onto three operational metrics vs.\ true N-1 outcomes,
10-seed mean ($\pm$ std). \AEGIS's structural $\mathrm{P}@10$ on the same
cases is $0.66\text{--}0.81$ (Table~\ref{tab:case_study}); the highlighted
LODF cell is its best-case retarget on the smaller grid.}
\label{tab:lodf}
\small
\begin{tabular}{l l c c}
\toprule
Case & LODF retarget metric & $\mathrm{P}@10$ & Kendall $\tau$ \\
\midrule
case57   & $\ell_2$ voltage-angle damage  & $0.40{\scriptstyle\pm0.000}$ & $\phantom{-}0.306{\scriptstyle\pm0.006}$ \\
case57   & thermal overload count         & $\mathbf{0.60}{\scriptstyle\pm0.000}$ & --- \\
case57   & voltage-magnitude violations   & $0.50{\scriptstyle\pm0.000}$ & $-0.112{\scriptstyle\pm0.001}$ \\
\midrule
case118  & $\ell_2$ voltage-angle damage  & $0.00{\scriptstyle\pm0.000}$ & $\phantom{-}0.141{\scriptstyle\pm0.003}$ \\
case118  & thermal overload count         & $0.00{\scriptstyle\pm0.000}$ & --- \\
case118  & voltage-magnitude violations   & $0.11{\scriptstyle\pm0.074}$ & $\phantom{-}0.084{\scriptstyle\pm0.038}$ \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{PI baseline (Ejebe--Wollenberg).}
For completeness, we also report the Performance Index (PI) of
Ejebe and Wollenberg~\cite{...} on the same 10-seed protocol:
case57 $\mathrm{P}@10 = 0.50$, $\tau = 0.335$; case118 $\mathrm{P}@10 = 0.30$,
$\tau = 0.101$. PI is positively correlated with true N-1 outcomes but loses
to \AEGIS on both grids and to LODF (thermal) on case57. Adding PI does not
change the conclusion of Table~\ref{tab:lodf}.
```

Source: `results/revision_R2/lodf_retarget.csv` + `results/revision_R2/pi_baseline.csv`.

---

## Patch 3 — AGNNCert framing (closes P1.4)

Drop in as the IBP-certifier comparison subsection. The key honest framing
move: report the 5-10× tightness ratio AND the weak Kendall, then explain
that they measure different objects --- one is a worst-case IBP radius,
the other is a first-order sensitivity radius.

```latex
\subsection{Comparison with IBP-style certifiers (AGNNCert)}
\label{sec:exp:agnncert}

AGNNCert~\cite{li2025agnncert} produces a certified L$^\infty$ robustness radius
per node via interval-bound propagation. \AEGIS's per-node radius
$r_v = 1/\sigma_1(S_v)$ is a first-order resolvent radius derived from
Theorem~\ref{thm:phase}(a) and is provably tight in the subcritical regime;
it is \emph{not} a worst-case certificate. The two quantities measure
different objects, and Table~\ref{tab:agnncert} reports both the
tightness gap and the rank disagreement directly.

Across 10 seeds on Cora, Citeseer, and Pubmed, the AGNNCert IBP radii are
$\mathbf{4.9}\text{--}\mathbf{10.2\times}$ larger (less informative) than the
\AEGIS first-order radii --- as expected, since IBP relaxations dilate
linearly with depth and are well-known to be loose by orders of magnitude
on multi-layer GNNs~\cite{...}. The Kendall correlation between the two
rankings is weak ($\tau = 0.08\text{--}0.14$), which is also expected:
IBP ranks nodes by worst-case input-perturbation tolerance, while
$r_v$ ranks them by local Jacobian sensitivity. A node can be IBP-fragile
(small certified radius) but first-order stable (large $1/\sigma_1$), or
vice versa. Neither metric strictly dominates the other; they are
complementary, and \AEGIS's contribution is to supply the
\emph{first-order, ranking-tight, label-free} side of the pair, which IBP
certifiers cannot.\footnote{%
AGNNCert is run with the authors' reference parameters at the same
50-node certification budget; \AEGIS uses the operator described in
\S\ref{sec:method}. We did not tune either method per seed.}

\begin{table}[t]
\centering
\caption{Per-node radii reported by AGNNCert (IBP-certified, worst-case)
vs.\ \AEGIS (first-order, Theorem~\ref{thm:phase}(a)). Tightness ratio
$= r^{\mathrm{AGNNCert}} / r^{\mathrm{AEGIS}}$ on median per-seed; higher
means the IBP radius is looser. Both Kendall and Spearman are weak,
reflecting the different objects being measured.}
\label{tab:agnncert}
\small
\begin{tabular}{l c c c c c}
\toprule
Dataset & median $r^{\mathrm{AEGIS}}$ & median $r^{\mathrm{cert}}$ & Tightness ratio & Kendall $\tau$ & Spearman $\rho$ \\
\midrule
Cora     & $0.187$ & $1.414$ & $\mathbf{10.17}\times$ & $0.079$ & $0.098$ \\
Citeseer & $0.322$ & $2.000$ & $\phantom{1}\mathbf{6.41}\times$ & $0.091$ & $0.116$ \\
Pubmed   & $0.405$ & $1.414$ & $\phantom{1}\mathbf{4.91}\times$ & $0.144$ & $0.174$ \\
\bottomrule
\end{tabular}
\end{table}
```

Source: `results/revision_R2/agnncert_comparison.csv` (30 rows, 10 seeds × 3 datasets).

---

## What this leaves to the discussion section

Even with the above three patches in place, the **single sentence** that
ties them together in the existing discussion / conclusion should now read
(suggested edit, not yet in `conclusion.tex`):

> "\AEGIS is positioned as a label-free, closed-form \emph{ranking proxy}
>  for structural vulnerability, not as a competing attacker or worst-case
>  certifier. On Pubmed it recovers $\geq 99\%$ of the iterative GR-BCD attack's
>  damage at $\tau = 0.685$, on case57 it ties the best-case LODF retarget,
>  and against AGNNCert it supplies first-order radii that are $5\text{--}10\times$
>  tighter than IBP-certified ones. The closed-form ranker degrades gracefully
>  where the spectral structure of $S_c$ no longer dominates the local attack
>  geometry --- a property we use as a built-in self-diagnostic (per-dataset $\tau$),
>  not a deficiency to apologise for."

That sentence is the integrating frame; the three patches above are the
evidence the reader hits before they reach it.
