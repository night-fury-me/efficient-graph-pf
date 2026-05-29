# E5 — Experimental-coverage table: scope

**Closes:** R02-Rec7 ("headline counts vs per-table coverage mismatch").
**Constraint:** paper is at **10 pages, zero slack** (a 12-word caption addition tipped it to 11pp this session). Any table needs an equal offset.
**Status:** scoped, not built. Coverage audited against the live `.tex` + `results/` CSVs on 2026-05-30.

---

## 1. The discrepancy E5 resolves

The Setup claims "**9 datasets across 4 domains**" and the paper reports "**7 GNN architectures**." A reviewer reading per-table captions sees neither number reproduced in any single experiment, which reads as inflation. The truth is that the headline numbers are the **union** across experiments:

- **7 architectures** appear in exactly **2** results: `tab:explicit` (Cora) and `fig:tau_heatmap` (the 33-cell transfer map). Every other study uses **IGNN**, the implicit-GNN workhorse.
- **9 datasets** = **5 graph** (Cora, Citeseer, Pubmed, Amazon Photo, WikiCS) + **4 power** (IEEE case14/30/57/118). These two groups are disjoint and never co-occur in one table.
- Many classification studies run on **Cora only**, on a **50-node subgraph**.

E5 makes this coverage structure explicit so the union counts are obviously honest.

## 2. Verified coverage (all counts locked against data)

| # | Result (label) | Datasets | Arch. | Scale | Runs |
|---|----------------|----------|-------|-------|------|
| T1 | `tab:cross_domain` κ/ε_crit | 5 graph | IGNN | full | 50 |
| T2 | `tab:tightness_eps` envelope ratio | 5 graph | IGNN | 50-node | 200 |
| T3 | `tab:attack_full` attack methods | Cora,Citeseer,WikiCS (3) | IGNN | 50-node | 120 |
| F3 | `fig:greedy_topk` discrete removal | Cora | IGNN | 50-node | — |
| F4 | `fig:breach` breach rate | 5 graph | IGNN | sub+full | 300 |
| F5 | `fig:scalability` dense vs matrix-free | Cora→Amazon | IGNN | full | — |
| T4 | `tab:baselines` GR-BCD/AGNNCert/LODF | Pubmed,Cora,case118 | IGNN | mixed | — |
| F6 | `fig:phase_transition` κ-sweep | Cora | IGNN | 50-node | 110 |
| T5 | `tab:explicit` arch sweep | Cora | **7** | full | 70 |
| F7 | `fig:tau_heatmap` transfer τ | 5 graph | **7** | mixed | **330** |
| T6 | `tab:ieee` power flow | **4 IEEE** | GCN-PF | full | 40 |
| F8 | `tab:ieee14`/`fig:ieee14` ranking | case14 | GCN-PF | full | 10 |

**Locked facts:**
- 7 architectures = GCN-2, GCN-4, SAGE-2, GIN-2, APPNP, GAT-2, IGNN.
- `fig:tau_heatmap` = **33 of 35 cells** (GAT-2 only on Cora/Citeseer/Amazon → missing 2 cells), 33×10 = 330 runs. Matches the abstract's "29/33 positive."
- **Correction to the revision plan:** PF is **4 IEEE cases, not 5** (case300 was dropped under P1). "tightness=5" and "four-quadrant=3" both verified correct.

## 3. Option A — the full coverage table (`tab:coverage`)

Drop-in, `\footnotesize`, 11 rows (merge T6+F8 into one "power flow" row):

```latex
\begin{table}[t]\centering
\caption{Experimental coverage. ``5 graph'' $=$ Cora, Citeseer, Pubmed, Amazon Photo, WikiCS; ``4 IEEE'' $=$ case14/30/57/118; ``7 arch'' $=$ GCN-2/4, SAGE-2, GIN-2, APPNP, GAT-2, IGNN (GAT-2 on 3 datasets, so $33$ of $35$ cells). All 10 seeds.}
\label{tab:coverage}\footnotesize\setlength{\tabcolsep}{4pt}
\begin{tabular*}{\columnwidth}{@{\extracolsep{\fill}}l l c l@{}}
\toprule
Result & Datasets & Arch. & Scale \\\midrule
\cref{tab:cross_domain}     & 5 graph              & IGNN  & full     \\
\cref{tab:tightness_eps}    & 5 graph              & IGNN  & subgraph \\
\cref{tab:attack_full}      & Cora/Cite/WikiCS     & IGNN  & 50-node  \\
\cref{fig:greedy_topk}      & Cora                 & IGNN  & 50-node  \\
\cref{fig:breach}           & 5 graph              & IGNN  & sub+full \\
\cref{fig:scalability}      & Cora--Amazon         & IGNN  & full     \\
\cref{tab:baselines}        & Pubmed/Cora/c118     & IGNN  & mixed    \\
\cref{fig:phase_transition} & Cora                 & IGNN  & 50-node  \\
\cref{tab:explicit}         & Cora                 & 7     & full     \\
\cref{fig:tau_heatmap}      & 5 graph              & 7     & 33 cells \\
power flow (\cref{tab:ieee})& 4 IEEE               & GCN-PF& full     \\
\bottomrule
\end{tabular*}\end{table}
```

**Space:** ~14–18 lines (table) + caption ~3 lines ≈ **a third of a column**. **Needs an offset.**

**Offset (revision plan §6 lever):** demote `tab:tightness_eps` (5×5 numeric grid) to the released repo, replace with one sentence:
> "First-order tightness stays in $[1.00, 1.16]$ across all 5 graph datasets for $\varepsilon\le0.20$ (full grid in the released code)."

That frees ≈ the coverage table's footprint → net ~0 pages.

## 4. Option B — clarifying sentences (no table, ~2–3 lines)

Append to `\textbf{Setup.}` (resolves the count concern at ~5% of Option A's cost, no offset needed beyond a possible 1-line trim):

```latex
No single study spans all $9\times7$ combinations: the seven architectures
appear in \cref{tab:explicit} (Cora) and the 33-cell map of
\cref{fig:tau_heatmap}; the remaining studies use IGNN, our implicit-GNN
workhorse, on 1--5 of the five graph datasets, while power flow covers the
four IEEE cases with a contractive GCN surrogate.
```

## 5. Recommendation

| | Reviewer value | Space | Risk |
|---|---|---|---|
| **A (table)** | High — scannable, definitive | +⅓ column, **needs `tab:tightness_eps` demotion** | medium (table-for-table swap, rebuild near 10pp edge) |
| **B (sentences)** | ~85% — states the coverage structure explicitly | ~2–3 lines, maybe a 1-line trim | low |

**Default: Option B.** E5 is a clarity/REFRAME item, not a missing result; the sentence version closes R02-Rec7 directly and is page-safe. Reach for **Option A** only if a reviewer explicitly wants a coverage *table*, in which case the `tab:tightness_eps`→repo demotion is the offset.

**Effort:** B ≈ 15 min (edit + rebuild + verify). A ≈ 1 hr (table + demotion + repo note + rebuild + page-count verify).
