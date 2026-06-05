# Introduction — minimal-edit version

Same philosophy as the abstract (`12`): keep the current opening, structure, and style; change only what's
necessary for the coupling/cost reframe + the honesty corrections. The intro already states the unification
and the coupling, so the footprint is small: **4 necessary edits + 1 optional**. Supersedes the fuller
intro rewrite in `11` (which restructured the opening).

---
## EDIT 1 — Radar reframe (Para 2 prose) — *positioning, NECESSARY*

Replaces the weak "nonzero mass on all seven" (a "does ε of everything" claim, flagged DA-m1) with the
unification.

**BEFORE**
```latex
\Cref{fig:positioning} makes this concrete on seven capability axes: structural attacks and certifiers
each cover only their own corner, whereas \AEGIS is the only method with nonzero mass on all seven. No
prior method audits, certifies, and defends in a single matrix-free pass.
```
**AFTER**
```latex
\Cref{fig:positioning} makes this concrete on seven capability axes: structural attacks and certifiers
each cover only their own corner, whereas \AEGIS spans all seven, reading an attack direction, an edge
ranking, and a per-node certificate from one object. No prior method audits, certifies, and defends in a
single matrix-free pass.
```

---
## EDIT 2 — Fig. 1 (radar) caption — *positioning, NECESSARY (matches EDIT 1)*

**BEFORE**
```latex
... \AEGIS wins no axis outright but is the only method with nonzero mass on all seven. Defense ...
```
**AFTER**
```latex
... \AEGIS wins no axis outright but is the only method that spans all seven from one object---reading an
attack, an edge ranking, and a per-node certificate in one pass. Defense ...
```
*(Keeps the honest "wins no axis outright"; reframes "nonzero mass" → "spans all seven from one object.")*

---
## EDIT 3 — Contribution (2), certification — *honesty, NECESSARY*

Two small fixes: add the exchangeability qualifier; "true break" → "measured break" (consistent with the
abstract and `rem:obs_o1`). This contribution **already carries** the deterministic-radius content the
abstract dropped, so the relocation is automatic — no new sentence needed.

**BEFORE**
```latex
\textbf{(2) Certification from sensitivity.} The same bound yields two guarantees: \AEGIS-Conformal, a
\emph{distribution-free} certificate with coverage $\geq 1{-}\alpha$ over the $\varepsilon$-ball that
holds at the nominal level under worst-case attack and stays non-vacuous where the deterministic radius
is thin (\cref{sec:conformal}); and, for contractive models, a constant-factor two-sided characterisation
of the breaking point $\ecrit{=}(1{-}\kappa)/\norm{W}_2$ in the contraction factor $\kappa{=}\norm{J_z}_2$,
whose norm certificate under-states the true break by $2$--$9\times$ ($10$ seeds).
```
**AFTER**
```latex
\textbf{(2) Certification from sensitivity.} The same bound yields two guarantees: \AEGIS-Conformal, a
\emph{distribution-free} certificate \emph{(sound under exchangeability)} with coverage $\geq 1{-}\alpha$
over the $\varepsilon$-ball, holding at the nominal level under worst-case attack and non-vacuous where
the deterministic radius is thin (\cref{sec:conformal}); and, for contractive models, a constant-factor
two-sided characterisation of the breaking point $\ecrit{=}(1{-}\kappa)/\norm{W}_2$ in the contraction
factor $\kappa{=}\norm{J_z}_2$, whose certificate the \emph{measured} break exceeds by
$\mathbf{2}$--$\mathbf{9\times}$ ($10$ seeds).
```
*(Note: `ε_crit=(1−κ)/‖W‖` is kept here as a summary — the precise `ε_glob`/`ε_crit` "linearized vs global"
distinction from PATCH 1 lives in `theory.tex`, not the intro. The contribution already frames `ε_crit` as
conservative, so this stays consistent.)*

---
## EDIT 4 — Contribution (3), coupled defense — *coupling evidence, NECESSARY*

The coupling is already stated; add the quantitative evidence (the anticorrelation) that makes it a
*finding*, not a claim — the positioning payload, mirroring the abstract.

**BEFORE**
```latex
\textbf{(3) Coupled defense.} Penalizing $\sigma_1(S_c)$ trades clean accuracy for certified robustness,
so the operator that finds the worst attack also tunes the defense (\cref{sec:defense}).
```
**AFTER**
```latex
\textbf{(3) Coupled defense.} Penalizing $\sigma_1(S_c)$ trades clean accuracy for certified robustness,
so the operator that finds the worst attack also tunes the defense---attack magnitude and certified radius
anticorrelate ($-0.65$, $10/10$ seeds; \cref{sec:defense}).
```

---
## EDIT 5 — Para 1, power-grid example — ✅ CHOSEN: light scoping half-clause (2026-06-05)

No change is strictly required: Para 1 is motivation, and the fraud case is the demonstrated one. But the
conclusion concedes the contractive surrogate "cannot model voltage collapse," so leading with power grids
and disclaiming them later is the self-inflicted wound R3 flags. If you want to preempt a power-systems
reviewer, the lightest fix keeps fraud (demonstrated) + drug, and either softens or drops the power-grid
clause, *or* adds a half-clause scoping it:

**Option (light) — add a scoping half-clause:**
```latex
... and power grids can miss contingencies when their graph is structurally fragile~\cite{nakiganda2023graph}
(we audit the model's structural sensitivity, not the grid's physics; \cref{sec:conclusion}). A practitioner
faces a question current tools leave unanswered: ...
```
**Option (cut) — drop power-grid, keep two demonstrated/clean domains.** Your call; not necessary.

---
## What does NOT change (kept verbatim)
- Para 1 opening sentence + the practitioner question (the hook).
- The "adversarial fault lines" framing and the three-thread breakdown in Para 2.
- The `S_c` description ("records how strongly each edge can push the equilibrium … from one randomized SVD").
- The two scope notes (contractive-IGNN vs general `S_c`; threat model).
- Fig. 2 (pipeline) + caption.
- Contribution (1) and Contribution (4) — untouched.

## Net
Four one-line edits (radar prose + caption, conformal qualifier + true→measured, defense anticorrelation),
all length-neutral or near it; + the chosen power-grid light scoping. The intro's structure, hook, and
voice are otherwise unchanged.

---
## FINAL intro — all edits applied (✅ CHOSEN 2026-06-05; drop-in for `sections/introduction.tex`)

Changed spans relative to current: power-grid scoping clause (Para 1); radar prose (Para 2); "(sound under
exchangeability)" + "measured" in Contribution (2); the `$-0.65$` anticorrelation in Contribution (3).
Fig. 1 caption (separate float) updated per EDIT 2.

```latex
Graph neural networks are now deployed where structural errors carry real consequences: fraud accounts
evade detection via perturbed transaction edges~\cite{zugner2018adversarial}, drug-interaction models
misfire under perturbed molecular graphs~\cite{dai2018adversarial}, and power grids can miss contingencies
when their graph is structurally fragile~\cite{nakiganda2023graph} (we audit the model's structural
sensitivity, not the grid's physics; \cref{sec:conclusion}). A practitioner faces a question current tools
leave unanswered: \emph{which edges, if perturbed, would cause predictions to fail, and by how much?}

% --- Figure 1 (radar) here; caption updated per EDIT 2 ---

Each existing thread maps only part of these \emph{adversarial fault lines}: structural attacks rank edges
by label-driven gradient search but return no per-node budget or certificate~\cite{zugner2018adversarial,zugner2019adversarial,geisler2021robustness,wu2019adversarial};
smoothing and certifiers return per-node or collective certificates but neither an edge ranking nor a
direction~\cite{bojchevski2020efficient,bojchevski2019certifiable,schuchardt2023localized,schuchardt2021collective,li2025agnncert,zugner2019certifiable};
and robust-architecture defenses harden models without surfacing which edges are
vulnerable~\cite{zhu2019robust,zhang2020gnnguard,jin2020graph}. \Cref{fig:positioning} makes this concrete
on seven capability axes: structural attacks and certifiers each cover only their own corner, whereas
\AEGIS spans all seven, reading an attack direction, an edge ranking, and a per-node certificate from one
object. No prior method audits, certifies, and defends in a single matrix-free pass. \AEGIS (Adversarial
Evaluation of Graph Integrity via Sensitivity) does, from one object: the constrained sensitivity matrix
$S_c$ (\cref{sec:framework}). $S_c$ records how strongly each edge can push the equilibrium prediction, so
its leading singular direction, column norms, and per-node margins deliver the three diagnostics from one
randomized SVD, and the same bound upgrades each per-node radius into a distribution-free certificate
(\cref{sec:conformal}). Two scope notes: the closed-form break characterisation (\cref{thm:phase_transition})
holds for contractive implicit GNNs, while $S_c$ extends to any GNN with continuous edge-weight message
passing (\cref{sec:explicit_extension}); the threat model is edge deletion and weight perturbation
(\cref{sec:background}).

% --- Figure 2 (pipeline) here; unchanged ---

\textbf{Contributions.} \textbf{(1) Constrained sensitivity operator.} $S_c$ specialises equilibrium IFT
sensitivity~\cite{koh2017understanding,gould2021deep} to \emph{structural} edge perturbations via $P_c$;
one matrix-free query reads the SVD-optimal direction, per-edge rankings, and per-node radii together,
where no prior object yields all three (matching a dense $\sigma_1$ to $0.03\%$ at $N{=}200$, scaling to
$N{=}7{,}650$). \textbf{(2) Certification from sensitivity.} The same bound yields two guarantees:
\AEGIS-Conformal, a \emph{distribution-free} certificate \emph{(sound under exchangeability)} with coverage
$\geq 1{-}\alpha$ over the $\varepsilon$-ball, holding at the nominal level under worst-case attack and
non-vacuous where the deterministic radius is thin (\cref{sec:conformal}); and, for contractive models, a
constant-factor two-sided characterisation of the breaking point $\ecrit{=}(1{-}\kappa)/\norm{W}_2$ in the
contraction factor $\kappa{=}\norm{J_z}_2$, whose certificate the \emph{measured} break exceeds by
$\mathbf{2}$--$\mathbf{9\times}$ ($10$ seeds). \textbf{(3) Coupled defense.} Penalizing $\sigma_1(S_c)$
trades clean accuracy for certified robustness, so the operator that finds the worst attack also tunes the
defense---attack magnitude and certified radius anticorrelate ($-0.65$, $10/10$ seeds; \cref{sec:defense}).
\textbf{(4) Empirical evaluation} (390 runs): four-quadrant attack comparison, head-to-head vs.\
GR-BCD/PR-BCD~\cite{geisler2021robustness} and complementary positioning against
AGNNCert~\cite{li2025agnncert} (\cref{app:baselines}), continuous-to-discrete transfer
(\cref{fig:tau_heatmap}), and a fraud audit (\cref{sec:fraud_case}).
```

**Fig. 1 caption (drop-in, per EDIT 2):**
```latex
Seven-axis capability radar (frontier semantics; $0$ at the centre, $1$ at the rim): \AEGIS wins no axis
outright but is the only method that spans all seven from one object---reading an attack, an edge ranking,
and a per-node certificate in one pass. Defense (the $\sigma_1(S_c)$ penalty) is benchmarked separately
(\cref{sec:defense}); axis definitions and per-method scoring are in \cref{app:baselines}.
```
