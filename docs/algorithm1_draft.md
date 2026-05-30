# Algorithm 1 draft — AEGIS Vulnerability Analysis (matrix-free pipeline)

Drop-in for `paper/sections/framework.tex`. Faithful to
`iem/scalable.py::scalable_adversarial_analysis` and the
`ScalableSensitivity` operator (matvec / rmatvec / top_k_svd /
edge_vulnerability), with the dense fallback `adversarial.py::
full_adversarial_analysis` covered by the early branch.

---

## 1. Required preamble patch

`paper/aegis.tex` currently loads only `\usepackage{algorithmic}` (body
keywords). Add the float wrapper:

```latex
\usepackage{algorithm}    % float environment + caption for algorithms
```

Place this on the line immediately after `\usepackage{algorithmic}`
(currently L4 of `aegis.tex`). The two packages are designed to be used
together.

---

## 2. Insertion point in `framework.tex`

Insert **after line 15** ("The four stages are illustrated in
\cref{fig:pipeline}.") and **before** the current `\subsection{Stage 1}`
heading.  Recommended: keep the four `\subsection{Stage…}` paragraphs as
expanded narrative, with each one opening by referring to the
corresponding line range of Algorithm 1 (e.g. "Stage 1 implements
Alg.~\ref{alg:aegis}, lines 1–3."). That keeps the algorithm box as the
authoritative summary while the prose continues to carry the
justification and implementation notes.

If column space is tight, replace the four subsection paragraphs with
one short paragraph per stage; the algorithm box then carries the
procedural detail.

---

## 3. LaTeX (paste into `framework.tex`)

```latex
\begin{algorithm*}[t]
\caption{\AEGIS Vulnerability Analysis (matrix-free pipeline).
Dense fallback ($N \leq 200$) replaces lines 4--11 with explicit
formation of $J_z$, $J_A$, and $S = (I - J_z)^{-1} J_A$, followed by a
deterministic SVD.}
\label{alg:aegis}
\begin{algorithmic}[1]
\REQUIRE Trained GNN operator $F(\,\cdot\,,\mathrm{ctx})$ with
  equilibrium $\zstar = F(\zstar,\mathrm{ctx})$; context
  $\mathrm{ctx}\!\ni\!\Ahat\in\R^{N\times N}$; perturbation budget
  $\varepsilon>0$; SVD rank $k$, oversampling $p$, power iterations
  $n_{\mathrm{iter}}$; tolerance $\tau$ for Neumann truncation.
\REQUIRE \textit{(optional)} classifier head $h$, logits $Y$, labels
  $y$ for per-node radii; weight $W$ for $\ecrit$.
\ENSURE Leading singular value $\sigma_1(S_c)$ and maximally
  sensitive direction $\delta\Ahat^{\star}$; per-edge vulnerability
  spectrum $\{v_{ij}\}_{(i,j)\in E}$; per-node first-order radii
  $\{r_v\}$; critical budget $\ecrit$.
\STATE \COMMENT{\textit{Stage 1: subgraph extraction \& forward pass}}
\STATE Extract BFS ego-subgraph around the target node if in
  subgraph mode (\cref{sec:subgraph_ablation}); else use the full
  graph (\cref{sec:scalability}).
\STATE Iterate $z \leftarrow F(z,\mathrm{ctx})$ to convergence; let
  $\zstar$ be the fixed point and
  $E\leftarrow\{(i,j)\,{:}\,i{<}j,\;\Ahat_{ij}\neq 0\}$.
\STATE \COMMENT{\textit{Stage 2: matrix-free $S_c$ operator}}
\STATE Estimate $\kappa\leftarrow\norm{J_z}_2$ by power iteration on
  the JVP of $F_z$; set Neumann depth
  $K\leftarrow\lceil\log(1/\tau)/\log(1/\kappa)\rceil$.
\STATE \textbf{define} $\textsc{matvec}(v)\colon \R^{|E|}\to\R^{Nd}$:
  \STATE \quad $\delta A \leftarrow P_c(v)$ \COMMENT{symmetric
    scatter $v_k$ to entries $(i,j),(j,i)$; Stage 3 of \cref{sec:framework}}
  \STATE \quad $b \leftarrow J_A\,\mathrm{vec}(\delta A)$ via
    forward-mode JVP of $F$ w.r.t.\ $\Ahat$ at $\zstar$.
  \STATE \quad \textbf{return} $\sum_{j=0}^{K} J_z^{\,j} b$ with
    early stop when $\norm{J_z^{j}b}<\tau\,\norm{b}$
    \COMMENT{truncated Neumann series}
\STATE \textbf{define} $\textsc{rmatvec}(u)\colon \R^{Nd}\to\R^{|E|}$:
  \STATE \quad $\bar b \leftarrow \sum_{j=0}^{K}(J_z^{\top})^{j} u$
    \COMMENT{adjoint Neumann}
  \STATE \quad $g \leftarrow J_A^{\top}\bar b$ via reverse-mode VJP;
    \textbf{return} $P_c^{\top}(g)$.
\STATE \COMMENT{\textit{Stage 4: vulnerability outputs}}
\STATE \COMMENT{\textit{(a) maximally sensitive direction --- randomized SVD~\cite{halko2011finding}}}
\STATE Draw $\Omega\sim\mathcal{N}(0,I)\in\R^{|E|\times(k+p)}$;
  $Y\leftarrow \textsc{matvec}(\Omega)$ column-wise.
\FOR{$j = 1$ \TO $n_{\mathrm{iter}}$}
  \STATE $Y\leftarrow \textsc{matvec}\bigl(\textsc{rmatvec}(Y)\bigr)$;
    re-orthonormalise $Y$ via QR.
\ENDFOR
\STATE $(Q,\_)\leftarrow\mathrm{QR}(Y)$;\quad
  $B\leftarrow Q^{\top} \textsc{matvec}^{\star}$ assembled column-wise;
  $(U_B,\Sigma,V_B^{\top})\leftarrow\mathrm{SVD}(B)$.
\STATE $\sigma_1 \leftarrow \Sigma_{1,1}$;\;
  $\delta\Ahat^{\star} \leftarrow
  \varepsilon \cdot \mathrm{sym}\bigl(P_c(V_B[:,1])\bigr)/
  \norm{\mathrm{sym}(P_c(V_B[:,1]))}$
  \COMMENT{Prop.~\ref{prop:opt_attack}}
\STATE \COMMENT{\textit{(b) per-edge vulnerability spectrum}}
\FOR{$(i,j)\in E$}
  \STATE $v_{ij}\leftarrow \norm{\textsc{matvec}(e_{ij})}_2$
    \COMMENT{column norm of $S_c$}
\ENDFOR
\STATE \COMMENT{\textit{(c) per-node first-order radii ---
  computed only if $h, Y, y$ are supplied}}
\IF{$h, Y, y$ provided}
  \FOR{$v = 1$ \TO $N$}
    \STATE Margin $m_v\leftarrow Y_{v,y_v}-\max_{c\neq y_v}Y_{v,c}$;
      runner-up class $c^{\star}$.
    \STATE Estimate $\norm{S_{c,v}}_2$ by $10$ steps of randomized
      power iteration on $\textsc{rmatvec}$ restricted to node~$v$'s
      block-rows.
    \STATE $r_v \leftarrow m_v\,/\,\bigl(\norm{W_{y_v}-W_{c^{\star}}}_2
      \cdot \norm{S_{c,v}}_2\bigr)$ \COMMENT{Prop.~\ref{prop:radius}}
  \ENDFOR
\ENDIF
\STATE \COMMENT{\textit{(d) critical budget for IGNN-class operators}}
\IF{$\norm{W}_2$ extractable}
  \STATE $\ecrit\leftarrow(1-\kappa)/\norm{W}_2$
    \COMMENT{Thm.~\ref{thm:phase_transition}}
\ENDIF
\STATE \textbf{return} $\bigl(\sigma_1,\,\delta\Ahat^{\star},\,
  \{v_{ij}\}_{(i,j)\in E},\,\{r_v\}_{v=1}^{N},\,\ecrit\bigr)$.
\end{algorithmic}
\end{algorithm*}
```

---

## 4. Design notes (read before pasting)

- **`algorithm*` vs `algorithm`.** The body has ~35 numbered lines; in
  IEEEtran two-column it will overflow a single column. `algorithm*`
  spans both columns and reads cleanly. If the editor prefers a
  one-column box, collapse Stages 2 and 4 sub-blocks
  (`matvec`/`rmatvec` definitions, randomized SVD) into a single
  `\STATE` each citing scalable.py — this drops the box to ~18 lines.

- **Cross-references.** The draft cites four labels that already exist
  in the manuscript: `prop:radius` (theory.tex:81), `prop:explicit`
  (theory.tex:157), `thm:phase_transition` (theory.tex:9), and the
  section labels `sec:subgraph_ablation`, `sec:scalability`,
  `sec:framework`. **`prop:opt_attack` does not yet exist** — either
  add this label to the optimal-attack proposition in theory.tex, or
  replace the reference with an inline phrase ("the leading right
  singular vector of $S_c$, as in our optimal-attack proposition").

- **Notation consistency.** Uses paper macros throughout (`\AEGIS`,
  `\Ahat`, `\R`, `\norm`, `\zstar`, `\ecrit`). The symbol $\kappa$ for
  the contraction constant matches the paper's existing convention
  (\textit{cf.} `framework.tex` L25, experiments §
  \emph{Notation} paragraph) — do NOT use $\rho$ here; the paper draws
  the $\kappa$/$\rho$ distinction explicitly (operator-norm vs.
  spectral-radius) and Algorithm 1 must report $\kappa$.

- **Faithfulness to code.** Line numbers below should be visible to a
  reviewer who reads the code release alongside the paper:
  - Stage 1, line 3: `ScalableSensitivity.__init__` edge enumeration
    (`scalable.py` L56–63).
  - Stage 2, line 5: `_estimate_rho` (L≈87) + `_adaptive_neumann_depth`
    (L82).
  - Stage 2, lines 6–9: `matvec` (L≈190).
  - Stage 2, lines 10–12: `rmatvec` (L≈196).
  - Stage 4(a), lines 15–20: `top_k_svd` (L≈208).
  - Stage 4(b), line 22: `edge_vulnerability` method
    (`ScalableSensitivity.edge_vulnerability`).
  - Stage 4(c): `_scalable_node_radii` (L≈ end of file).
  - Stage 4(d): `extract_W_spectral_norm` +
    `critical_perturbation_budget` (`adversarial.py` L335, L302).

- **What deliberately stays out of the algorithm box.**
  The non-normality diagnostic (`nonnormality_index`,
  `adversarial.py` L436) and the empirical tightness check
  (`validate_bound_tightness`, L467) are post-hoc diagnostics, not
  parts of the analysis pipeline — they belong in the experiments
  prose, not Algorithm 1.

---

## 5. Suggested companion text edit

After the algorithm box, add a one-sentence pointer paragraph (replaces
nothing; complements the existing Stage 1–4 subsections):

```latex
\noindent\textit{Reading guide.}\;
The four stages of Algorithm~\ref{alg:aegis} correspond one-to-one to
the subsections below: Stage~1 (lines~1--3, \cref{sec:framework} Stage~1),
Stage~2 (lines~4--12, Stage~2), Stage~3 / $P_c$ (line~6, Stage~3), and
Stage~4 (lines~13--32, Stage~4).
```

This keeps the original prose untouched on first pass; if reviewers find
the prose redundant after the box is in place, it can be trimmed in a
later revision without restructuring.
