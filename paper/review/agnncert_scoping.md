# AGNNCert scoping — fair head-to-head feasibility for AEGIS-Certify (ICDM)

**Task:** decide whether a FAIR numerical head-to-head between AEGIS-Certify
(our sound, closed-form, matrix-free Frobenius certified radius `rho_v` for an
implicit/equilibrium IGNN) and the external baseline **AGNNCert** is feasible,
on what common axis/model, or whether to pivot.

**Bottom line up front:** a like-for-like *certified-radius* comparison is
**NOT feasible** (incommensurable axes + AGNNCert cannot run on our IGNN as a
white-box certifier; it treats the GNN as a black box and certifies a *discrete
edit count*, not a Frobenius radius). A fair *certified-accuracy-vs-budget* curve
**is feasible on an explicit-GNN proxy** if we accept AGNNCert's discrete-edit
axis and embed our continuous `rho_v` into it via `‖δÂ‖_F = w·√(2k)`. The
strongest publishable move is the **categorical positioning claim** ("first
*sound* structural certifier for *equilibrium* GNNs"), anchored by AGNNCert's
own published numbers, plus the optional explicit-GNN curve as supporting
evidence. **Also: fix two factual errors in the current draft (below).**

---

## Facts table (1–6) — CONFIRMED vs INFERRED

| # | Question | Answer | Status | Source |
|---|----------|--------|--------|--------|
| **1** | **Identity** | **AGNNCert: Defending Graph Neural Networks against Arbitrary Perturbations with Deterministic Certification.** Authors **Jiate Li, Binghui Wang** (Illinois Inst. of Technology). **Accepted at USENIX Security 2025.** **arXiv:2502.00765** (cs.CR), submitted 2 Feb 2025. | **CONFIRMED** | arXiv abstract page; arXiv comment field "Accepted by Usenix Security 2025"; official GitHub README. |
| **2** | **Threat model / budget units** | Certifies **arbitrary discrete perturbations**: edge insert/delete, **node** insert/delete, and **node-feature** modification — at *test time* (pretrained, clean classifier). The certificate is a **certified perturbation size `M`** measured in a **count** (# edges, # nodes/features perturbed), i.e. an **L0 / graph-edit-distance** budget. **NOT** a continuous norm; **NOT** Frobenius. `M` is the *largest* perturbation count it can guarantee per instance (computed, not pre-set). | **CONFIRMED** | Abstract; rebuttal "Summary of Major Changes" §1.2–1.3 (test-time; `M` from Eq.(13)/(15) is the max certified perturbation size for edge/node variants). |
| **3** | **Mechanism / soundness** | **Hash-based graph partition + majority voting** — deterministic. A `HashAgent` hashes each edge (`md5/sha1/sha256(V·u+v) mod T`) or node into one of **T** disjoint subgraphs (default `T=30`; `T=300` in the node-classification script), runs the base GNN per subgraph, and **majority-votes**. `M` follows from the top-1 vs runner-up vote-bin gap. It is the **deterministic descendant of hash/bagging smoothing** (a *voting* certifier), **NOT IBP / interval-bound propagation** and **NOT randomized smoothing** (no sampling, no abstention, 100% sound). | **CONFIRMED** | Official code `edge_hash.py`, `node_hash.py` (`HashAgent`, `RobustNodeClassifier`); abstract ("deterministic … encompass existing certified defenses as special cases"). |
| **4** | **Model class** | Treats the base GNN as a **black box** (`NodeGCN, NodeGAT, NodeGSAGE`; node + graph classification). Because it never propagates intervals *through* the model, it is **architecture-agnostic and would in principle run on our IGNN too** — but only as a *voting wrapper around the implicit solver*, **never** as a white-box equilibrium-aware certifier. It does **not** model the fixed-point / IFT structure; there is **no interval propagation through an equilibrium**. So AGNNCert and AEGIS-Certify certify *different objects*: AGNNCert = robustness of a *voted ensemble of subgraph classifiers*; AEGIS = robustness of the *actual single IGNN equilibrium map*. | **CONFIRMED** (black-box, GCN/GAT/SAGE) / **INFERRED** (could wrap IGNN, but semantics differ — never run on equilibrium models in the paper) | Code (`gnn.py`, `AGNNCert-E_Node.py`); abstract. |
| **5** | **Official code** | **Yes — runnable.** `github.com/JetRichardLee/AGNNCert` (official; created 2025-01-17, last push **2025-02-10**, ~113 KB, pure **Python / PyTorch + PyTorch-Geometric**). Scripts split by perturbation×task: `AGNNCert-{E,N}_{Node,Graph,Amazon}.py`. Datasets on **Zenodo (records/14737141)**. **No LICENSE file** (caveat: usage rights unstated). Mirror: `1000fishcn/agnncert` (MIT). No open issues. Feasible to run on standard Cora/PyG setup; **no GPU strictly required** (small subgraphs, but voting over T=30–300 reruns is the cost). | **CONFIRMED** | GitHub API tree + meta; README. | 
| **6** | **Published numbers (Cora/Citeseer)** | Reports **node & graph certified accuracy vs. certified perturbation size** on Cora-ML, CiteSeer, PubMed, Computers (node) and protein/graph sets, over GCN/GAT/GraphSAGE; claims **superiority over SOTA edge-only and node-only certified baselines**. **Exact per-budget table cells (e.g. "Cora certified acc = X% at k edge edits") could NOT be extracted programmatically** — the arXiv HTML render exposes only the rebuttal page and the PDF is image-based. **Must be read off the PDF / USENIX camera-ready / Zenodo to quote.** | **PARTIAL — structure CONFIRMED, exact cell values UNCONFIRMED** | Abstract (claims); arXiv PDF not machine-parseable in this environment. |

### Two factual errors in the current AEGIS draft (fix before submission)
1. `paper/sections/experiments.tex` (L97, L107–L109) and the body call AGNNCert
   **"a sound IBP certifier"** / **"sound IBP certificate."** **Wrong.** AGNNCert
   is a **hash-partition voting** certifier (divide-and-vote), not IBP. The
   intro (`introduction.tex`) also lumps it under "IBP-style certifiers
   `\cite{li2025agnncert,zugner2019certifiable}`" — only the second is IBP-style.
   → relabel AGNNCert as **"deterministic partition/voting certifier."**
   (`fig_positioning.tex` row 4 "per-node radius (deterministic)" is fine.)
2. `paper/aegis.bib` `@li2025agnncert` has **wrong author and venue**:
   currently "Li, Yuning and others … ICLR 2025." **Correct:** `Li, Jiate and
   Wang, Binghui`, **USENIX Security 2025**, arXiv:2502.00765. Fix the bib.
3. The `tab:baselines` "AGNNCert" column value **`1.414`** (= √2) is **NOT a real
   AGNNCert number.** It comes from the repo's *home-grown IBP proxy*
   (`scripts/revision_R2/R2_02_agnncert_comparison.py`), which the script's own
   docstring admits is an **"AGNNCert-style"** single-edge IBP probe because
   "Li 2025 does not yet have a publicly mature codebase" — a statement that is
   now **false** (official code exists). The cached
   `results/revision_R2/agnncert_comparison.csv` shows `median_r_cert`
   hardcoded at √2≈1.414 (Cora/Pubmed) or 2.0 (Citeseer) — i.e. just the
   `√(2·k)` embedding of k=1–2 edges, **not** AGNNCert's voting certificate.
   → either run the *real* AGNNCert, or stop attributing this number to it.

---

## FEASIBILITY VERDICT

**A like-for-like certified-RADIUS comparison is NOT feasible**, for three
independent reasons:

1. **Incommensurable axes.** AEGIS-Certify emits a **continuous Frobenius radius
   `rho_v`** on the normalized weighted adjacency; AGNNCert emits a **discrete
   edit-count `M`** (L0 / graph-edit-distance). A radius in ‖·‖_F and a count of
   edge flips are not the same quantity; any direct "0.163 vs 1.414" row (as
   currently in `tab:baselines`) is a **category error**.
2. **Different certified object.** AGNNCert certifies a **voted ensemble of T
   subgraph classifiers**; AEGIS certifies the **single IGNN equilibrium map**.
   Even on the same graph they guarantee different functions.
3. **Model mismatch.** AGNNCert never certifies an equilibrium/DEQ model
   white-box; it would only *wrap* our IGNN as a black-box voting ensemble,
   which is a different (and weaker, smoothed) model than the one AEGIS
   certifies. There is **no equilibrium-aware certificate** in AGNNCert to
   compare against AEGIS's IFT-resolvent bound.

**A common-AXIS comparison IS feasible** if we adopt AGNNCert's native axis
(**certified accuracy vs. #edge-edits k**) and *project AEGIS onto it* via the
embedding already used in the repo:

> A single undirected edge toggle of weight `w` changes two symmetric entries of
> Â, so `‖δÂ‖_F = w·√(2k)` for `k` independent edge edits (`w=1` on the binary
> graph → `√2·√k`). Hence **AEGIS certifies node `v` robust to any k-edge edit
> whenever `rho_v ≥ w·√(2k)`, i.e. `k ≤ rho_v² / (2w²)`.** This converts our
> continuous `rho_v` into a **sound discrete-edit guarantee** on AGNNCert's axis.

This is a *sound lower bound* (worst-case over the k chosen edges sits inside the
`rho_v`-ball), so the curve is fair and conservative in our favor's-disfavor (we
under-claim, never over-claim).

**Common MODEL:** the two methods do not share a native model.
- **Where both apply:** an **explicit GNN proxy** (GCN/GraphSAGE on Cora/CiteSeer).
  AGNNCert runs natively; AEGIS runs via its explicit-GNN extension
  (`prop:explicit`, finite-difference `S_K` → `S_c`; already in the paper,
  `tab:explicit`). This is the only place a *numerical* curve is apples-to-apples.
- **Where only AEGIS applies:** the **IGNN** (our headline model). AGNNCert can
  only black-box-wrap it (voting ensemble), which is a different model — report
  this as a **scope advantage**, not a number.

---

## RECOMMENDED PATH (primary = positioning, secondary = optional curve)

### PRIMARY — categorical positioning claim (low-risk, no GPU, ship this)
Frame AEGIS-Certify as the **first *sound, deterministic, white-box* structural
certifier for *equilibrium/implicit* GNNs**, and use AGNNCert as the
**positioning anchor**, NOT a head-to-head number:

- **Claim:** every prior *sound* structural-GNN certifier is either
  **smoothing** (probabilistic, abstains: Bojchevski; localized smoothing) or
  **partition/voting** (AGNNCert — deterministic but a *voted ensemble*, L0/GED
  budget, black-box base model); every prior *implicit-net* certifier is
  **input-only** (Jafarpour L4DC'22). The cell "sound + structural + equilibrium
  + closed-form Frobenius radius from the IFT resolvent" is **unoccupied.**
- **Positioning figure** (`fig_positioning.tex` already exists): keep AGNNCert in
  the "deterministic per-node certificate" row, but **relabel its budget axis as
  *discrete L0/GED* and its model as *black-box voting ensemble*** so the reader
  sees the two axes never coincide. This kills the "you don't beat AGNNCert"
  reviewer prong by **scope**, not by a contestable number.
- **Honest one-liner** (replaces the broken `tab:baselines` AGNNCert row): "AGNNCert
  (USENIX'25) gives a *deterministic L0/GED* certificate for *black-box* GNNs via
  hash-partition voting; AEGIS-Certify gives a *deterministic Frobenius* radius
  for the *equilibrium map itself*. The budgets (edit-count vs ‖δÂ‖_F) and the
  certified object (voted ensemble vs single fixed point) differ, so we compare
  on the shared discrete-edit axis (Fig. X) rather than as interchangeable radii."

### SECONDARY — optional certified-accuracy-vs-k curve on the explicit-GNN proxy
Only if a reviewer demands a number. **No GPU needed** (Cora/CiteSeer, GCN, T≈30).

- **Data/model:** Cora and CiteSeer, **GCN** (and optionally GraphSAGE), the
  explicit-GNN setting where both methods are native.
- **Compute:**
  - **AGNNCert:** run the official `AGNNCert-E_Node.py` (edge perturbation, node
    classification) → its native **certified accuracy vs. certified edge-edit
    size k** curve. (Pin T, hash, seed; cite their reported numbers as a sanity
    check on our run.)
  - **AEGIS-Certify:** compute `rho_v` per node (existing `iem.certify` /
    `per_node_robust_radius`), then **certified-accuracy(k) = fraction of
    correctly-classified nodes with `rho_v ≥ √(2k)`** (the `w=1` embedding).
    This reuses the existing 0%-breach pipeline; it is "re-plot the breach
    experiment as certified-accuracy-vs-budget" (cheap, no retraining).
- **Plot:** one panel per dataset — **x = #edge edits k (0…~5)**, **y = certified
  node accuracy (%)**, two step-curves (AGNNCert vs AEGIS-Certify-projected),
  with a caption stating the `‖δÂ‖_F=√(2k)` embedding and that AEGIS's curve is a
  *sound lower bound*. Expected, honest outcome: **AGNNCert's curve sits higher**
  (it is a voting certifier tuned for L0; our Frobenius→L0 projection is
  conservative — `rho_v` is `4.4–15×` tighter than even our own IBP proxy). **Do
  not spin this as "we win on tightness."** The win is **categorical** (sound +
  equilibrium + closed-form + matrix-free, where AGNNCert offers none of those on
  the implicit model), plus AEGIS additionally emits an **edge ranking + attack
  direction** from the same `S_c` pass, which AGNNCert does not.

### What to FLAG / could NOT confirm
- **Exact AGNNCert Cora/CiteSeer certified-accuracy cell values** (fact 6) — the
  arXiv PDF is image-only and the HTML render is paywalled here. **Read them off
  the PDF / USENIX'25 camera-ready / Zenodo before quoting any number.**
- **License** of the official repo is **absent** — if we *run* their code or ship
  derived numbers, confirm usage rights (the `1000fishcn/agnncert` mirror is MIT
  but provenance is the unlicensed original).
- Whether AGNNCert's voting wrapper *can* be put around our IGNN solver is
  INFERRED (architecturally yes); it would certify a *different* (smoothed)
  model, so it is a scope statement, not a comparison.
