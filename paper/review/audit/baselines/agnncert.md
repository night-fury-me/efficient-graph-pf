# Baseline Faithfulness Audit — AGNNCert (Li & Wang 2025)

**Audited:** 2026-06-05
**Cite key:** `li2025agnncert`
**Verdict:** **UNFAITHFUL-PROXY** (impl) / paper positioning is **DEFENSIBLE after fixes** (see end)

---

## 1. Implementation location

| Item | Path |
|------|------|
| Impl under audit | `scripts/revision_R2/R2_02_agnncert_comparison.py` — `agnncert_radii()` (~L75) |
| Output it produced | `results/revision_R2/agnncert_comparison.csv` (30 rows: 3 datasets × 10 seeds) |
| Log | `results/revision_R2/logs/R2_02_agnncert_comparison.log` |
| Prior team scoping (independent) | `paper/review/agnncert_scoping.md` — already reached the same conclusion |

**Official vs. hand-rolled:** **Hand-rolled.** The impl is a bespoke single-edge IBP/brute-force probe, explicitly *not* AGNNCert. The docstring states it implements "a deterministic radius via single-edge interval-bound propagation" because "Li 2025 does not yet have a publicly mature codebase."

**Official code DOES exist.** The paper (USENIX Security 2025 camera-ready, "Summary of Major Changes" §5) commits to a public GitHub release, and AGNNCert is a USENIX'25 artifact. The docstring's "no mature codebase" premise is **stale/false** as of submission. (Authors: **Jiate Li, Binghui Wang**; arXiv:2502.00765.)

---

## 2. What AGNNCert actually does (from the fetched paper, arXiv:2502.00765 / USENIX Sec'25)

Source: `ctx_search(source: "agnncert_li2025")` over the indexed PDF.

AGNNCert is a **voting / divide-and-vote (hash-partition) deterministic certifier** — NOT IBP, NOT smoothing. Algorithm:

1. **Graph division into T sub-graphs.** A hash function maps the graph's components into `T` disjoint sub-graphs. Two strategies:
   - **Edge-centric (AGNNCert-E):** hash *edges* so edge sets are disjoint across sub-graphs. Deleting/injecting any one edge perturbs exactly **one** sub-graph (Thm 2).
   - **Node-centric (AGNNCert-N):** treat each undirected edge as two directed edges; hash each *node* by its outgoing edges so all of a node's edges land in one directed sub-graph. Manipulating any one node/feature perturbs exactly **one** sub-graph (Thm 7); guarantees an unbounded number of perturbed edges per node.
2. **Run the (base) GNN on each of the T sub-graphs** → T predictions per target. For graph classification, isolated nodes get a zero-feature dummy node to preserve the global-pooling count.
3. **Aggregate by majority vote** across the T sub-graph predictions → final label `h(G) = argmax` vote count.
4. **Deterministic certified perturbation size.** Robustness is governed by the **gap between the most-voted (correct) label and the runner-up** (paper §6: "determined by the gap between the most votes ... and second-most votes"). A perturbation of size `m` (total # of manipulated **edges + nodes + feature-changed nodes**) can flip at most a bounded number `M` of sub-graph predictions (Thm 1 sufficient condition; per-strategy bounds in Eqns 13 / 15). The certified size is the largest `m` for which votes cannot be swung past the runner-up. **It is a count (graph-edit / edit-count), discrete and deterministic, requiring no probability.**

**Experiments:** node-classification certified up to ~200 edits, graph-classification up to ~25; certified accuracy reported as a function of perturbation size `m` swept over `T`. Treats the GNN as a **black box** (trained with the partition ensemble). Compared against Bi-RS (probabilistic, node-injection) and GNNCert (the deterministic predecessor it generalizes).

---

## 3. What OUR impl does (`agnncert_radii`)

For each node `v` in a ≤50-node sub-graph of the trained **IGNN** (equilibrium model):
1. Enumerate edges incident to `v`.
2. Rank them by single-edge **brute-force margin impact** (re-run the full IGNN with each one edge zeroed).
3. Greedily remove edges in worst-first order, re-running the IGNN after each removal, counting the largest `k_safe` for which `argmax` is unchanged (capped at `max_perturb=20`).
4. Report `r_cert[v] = sqrt(2 * k_safe)` — a Frobenius-norm embedding of `k` discrete edge removals.

Output scale (CSV): `median_r_cert` is **√2 ≈ 1.414 on Cora/Pubmed, 2.0 on Citeseer** for *every* seed → i.e. `k_safe` is essentially always **1 (occasionally 2)**. `n_nodes_certified` ≈ 50/50. `τ` vs. AEGIS ≈ +0.11 (weak).

---

## 4. GAPS

| # | Gap | Severity | Location | Fix |
|---|-----|----------|----------|-----|
| G1 | No graph division. No `T` sub-graphs, no hash partition — the defining mechanism of AGNNCert is entirely absent. | **Critical** | `R2_02_agnncert_comparison.py:75` (`agnncert_radii` body) | Implement edge/node-centric hashing into T sub-graphs, or use the official artifact; do not call a single-edge IBP probe "AGNNCert". |
| G2 | No voting / no vote margin. Certificate is NOT derived from a most-vs-runner-up vote gap (the actual AGNNCert guarantee, §6, Eqns 13/15). | **Critical** | same | Aggregate T sub-graph predictions by majority vote; derive certified `m` from the vote gap. |
| G3 | Wrong threat quantity. Greedy worst-first deletion of edges *incident to v only* is a **local, edge-deletion-only, non-certified** heuristic (it can miss the true worst case and ignores edge/node injection + feature perturb that AGNNCert covers). Not a sound certificate at all. | **Critical** | `R2_02...py` incident-edge loop | A real certificate must lower-bound over ALL size-`m` perturbations, not greedily probe one node's incident edges. |
| G4 | `T=30–80` (claimed in `F_experiments.tex` L116) is **not** present or used anywhere in the impl. There is no `T`; the only knob is `max_perturb=20` (an unrelated cap). | **High** | `R2_02...py` (`max_perturb`), `F_experiments.tex:116` | Either run real AGNNCert with `T∈[30,80]`, or stop attributing a specific `T` range to our code. |
| G5 | Stale premise. Docstring claims AGNNCert has "no publicly mature codebase" — false; official USENIX'25 artifact / GitHub exists. | **Medium** | `R2_02...py` docstring (top) | Update comment; prefer official code. |
| G6 | Mislabel propagation. AGNNCert called a "sound IBP certifier / IBP-style" in `experiments.tex` & `introduction.tex`; it is divide-and-vote, not IBP. | **Medium** | `paper/sections/experiments.tex` (~L97,107–109), `introduction.tex` | Relabel "deterministic partition/voting certifier." (Already flagged in `agnncert_scoping.md`.) |
| G7 | Bib error. `@li2025agnncert` lists wrong author/venue (was "Li, Yuning … ICLR 2025"). | **Medium** | `paper/aegis.bib` | Correct to "Li, Jiate and Wang, Binghui", USENIX Security 2025, arXiv:2502.00765. |
| G8 | Hardcoded-looking output. `median_r_cert` is a constant √2/2.0 across all seeds; if any `tab:baselines` cell shows `1.414` it is the `√(2k)` embedding of k=1, **not** an AGNNCert number. | **High (if cited as a number)** | `agnncert_comparison.csv`; any `tab:baselines` AGNNCert cell | Never present `1.414` as an AGNNCert result. |

---

## 5. Protocol checklist

1. **Division-based deterministic certificate (T sub-graphs, voting, margin → radius)?** **NO.** It is a per-node single-edge greedy IBP/brute-force probe. None of {division, T, voting, vote-margin} present.
2. **Is T=30–80 the # of division parts, used for the bound?** **NO.** No `T` exists in the impl; the appendix claim `T=30–80` describes the *real* AGNNCert (correctly, as a property of the method), but our code does not realize it.
3. **Same certified notion (edit-count / # edge changes)?** **Partially in units only.** Our `k_safe` is an edge-*deletion* count embedded as `√(2k)`; AGNNCert certifies total edits over edges **+ nodes + features** via a sound voting bound. Our quantity is neither sound nor the same scope, but it does live on a "# discrete edits" axis.
4. **Is the capability-axis scoring in `app:baselines` defensible?** **YES, and it does NOT rest on our impl.** Crucially, the paper does **not** put our proxy number on the radar. The radar (`fig_positioning_radar.tex`) and threat table (`F_experiments.tex` L97) score AGNNCert as the **certifier-thread frontier** using **AGNNCert's own published capabilities** (deterministic ✓, bounded edit count, per-node certificate, query/retraining cost), explicitly framing it as **complementary, not a head-to-head loss** (`F_experiments.tex` L71–74, L83–86). `agnncert_scoping.md` already concluded a like-for-like certified-radius comparison is **incommensurable** and dropped it. So the *positioning* is sound **provided** the proxy CSV/`1.414` is not surfaced as data and the IBP/bib/labeling errors (G5–G7) are fixed.

---

## 6. VERDICT

**Impl faithfulness: UNFAITHFUL-PROXY.** `agnncert_radii` shares only the output *units* (a √(2k) edit-count embedding) with AGNNCert. It implements none of the method's three defining pieces — graph division into T sub-graphs, majority voting, and the vote-margin → certified-edit-count bound — and is not even a sound certificate (greedy, incident-edges-only, deletion-only). The script's own docstring concedes it is "AGNNCert-style," and its justifying premise ("no mature codebase") is now false.

**Paper element at risk:**
- `tab:threat_model` (`F_experiments.tex` L97: "AGNNCert — bounded edit count, Cert ✓") — **row content is CORRECT** (it describes real AGNNCert), so the table stands once the "IBP" mislabel (G6) and bib (G7) are fixed.
- `fig:positioning` / `fig:positioning_radar` (`app:baselines`) — radar scores AGNNCert on its *published* capabilities, **not** our proxy, so the axis scoring is **DEFENSIBLE**.
- **Genuine exposure = G8:** any table/figure that prints the proxy's `1.414` (or our `τ`) as an "AGNNCert" measurement. That number must not appear as an AGNNCert result; the comparison is correctly positioned as categorical/complementary, not numeric.

**Net:** Because AEGIS claims no numeric head-to-head win over AGNNCert, the *positioning* survives. But the **implementation must not be described or cited as AGNNCert** anywhere, and the IBP/bib/codebase-availability text errors should be corrected before submission.

---

## 7. Sources

- AGNNCert paper: Li, Jiate & Wang, Binghui. *AGNNCert: Defending Graph Neural Networks against Arbitrary Perturbations with Deterministic Certification.* USENIX Security 2025. arXiv:2502.00765. (Indexed KB source: `agnncert_li2025` / "AGNNCert paper (Li 2025, arXiv 2502.00765)".)
- USENIX presentation page: https://www.usenix.org/conference/usenixsecurity25/presentation/li-jiate
- Prior internal scoping (independent, concordant): `paper/review/agnncert_scoping.md`.
