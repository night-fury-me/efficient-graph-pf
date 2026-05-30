# Revision R4 — Transfer / τ=+0.996 (DATA ANALYSIS — decision needed before editing)

**Concern (R1 W3 + DA M3):** the proposition proves a magnitude bridge + a *sufficient pairwise* order (47–62% of pairs), not a global rank; the headline τ=+0.996 reads as typical. **Debugging the data surfaced more than a wording issue.**

## What the data actually shows

**The heatmap (`results/tau_all_datasets.csv`, 33 cells, 50-node subgraphs, unweighted v_ij):**
- mean **+0.355**, median **+0.348**, min **−0.276**, max **+0.894** (GCN-4/Pubmed), IQR [0.21, 0.54].
- 29/33 positive; one-sided sign test **p=5.46e-6** (matches the paper's p<10⁻⁵). ✓ honest, defensible.
- 4 non-positive cells: GCN-2/Citeseer **−0.276**, IGNN/Amazon Photo **−0.147**, GCN-4/Amazon Photo −0.038, GCN-2/Cora −0.032.

**The +0.996 is a DIFFERENT estimator** (`results/revision_R2/amazon_fullgraph_stratified.csv`, full-graph Amazon Photo): it is `tau_strat_weighted` = the **edge-weighted A_ij·v_ij** ranking under **stratified top-v_ij ground-truth sampling**. On the *same* data:
- `tau_strat_weighted` (weighted) = **+0.996**
- `tau_strat_raw` (v_ij alone) = **−0.223**  ← the unweighted AEGIS score *anti*-correlates
- `tau_sub` (50-node) = −0.147 ; `tau_top100_only` = −0.006

So +0.996 is **not** a heatmap cell, **not** the typical transfer (median +0.35), and is achieved only by the edge-weighted score under stratified sampling. The body discloses "edge-weighted A_ij·v_ij … stratified" (L172); the **abstract does not** and juxtaposes it with the 29/33 as if one result.

## The load-bearing question (missing control)
The theory (Prop. transfer a) predicts `d_k ≈ w_k·v_k`, so using the weight w_k=A_ij is *correct*. **But:** because raw v_ij is −0.223 and only A_ij·v_ij is +0.996, the correlation may be carried by the edge weight w_k, not the AEGIS score v_k. The decisive control — **τ(w_k alone, d_k) vs τ(w_k·v_k, d_k)** — is not reported. If weight-only ≈ +0.99, v_k adds no rank information and +0.996 is largely a weight artifact; if weight-only is much lower, v_k is doing real work and +0.996 is a genuine, strong result.
- Per-edge `d_k` is **not saved** → this control needs a (GPU) rerun of stratified N-1 on Amazon Photo.

**Additional flag:** the subagent reports `kappa_full ≈ 1.0006` on full-graph Amazon Photo — i.e. A3 (κ<1) is *marginally violated* exactly where +0.996 lives, so Prop. transfer's stated assumptions don't strictly hold at that scale.

## Failing-cell mechanism (for the honest disclosure, computable now)
- GCN-2/Cora & GCN-4/Amazon Photo: **noise-dominated** (|mean τ|/seed-std < 1.2 → τ≈0, no signal).
- GCN-2/Citeseer & IGNN/Amazon Photo: **stably anti-correlated** (|mean|/std = 4.3, 5.7). GCN-2 matches the paper's own L182 mechanism: shallow 2-hop aggregation → near-uniform first-order shifts, outside Prop. transfer's sufficient regime.

## Options (honest, not undersold)
- **A (bulletproof, needs your GPU rerun):** compute the weight-only control τ(A_ij, d_k) on Amazon Photo. If A_ij·v_ij beats A_ij alone, *keep +0.996* and report the control — turns the concern into a strength. (I'll write the script; you run it.)
- **B (reframe now, no rerun):** lead the transfer claim with the honest heatmap distribution (median +0.35, 29/33 positive, p=5.46e-6); present +0.996 in the body only, fully qualified (edge-weighted, stratified, full-graph) and **not** in the abstract; disclose + explain the 4 failing cells.
- **Recommended:** B now (strictly more honest, compatible with any later control) + A as the bulletproofing that lets +0.996 stand prominently.

Files: `scripts/exp_tau_all_datasets.py`, `results/tau_all_datasets.csv`, `results/revision_R2/amazon_fullgraph_stratified.csv`. Prose: `experiments.tex` L172/L182, abstract L2.

---

## Update — experiment prepared (chosen path: report theory-weighted ranking in the heatmap)

Per the author's call ("if edge-weighted works, report *it* in the heatmap"), `scripts/exp_tau_all_datasets.py` is patched (verified aligned) to emit three τ per cell:
- `tau` — unweighted `v_k` (existing)
- `tau_weighted` — theory's `w_k·v_k` (Prop. transfer predictor)
- `tau_weight_only` — `w_k` alone (the control: does `v_k` add rank info?)

**Run (author, GPU; ~1–2 h):** `.venv/bin/python scripts/exp_tau_all_datasets.py` → overwrites `results/tau_all_datasets.csv` with the 3 columns (deterministic given seeds).

**Decision tree once results land:**
- If `tau_weighted` > `tau` AND `tau_weighted` ≫ `tau_weight_only` across cells → **switch the heatmap to the weighted ranking**, restore τ=+0.996 as its high-end, report the weight-only control as evidence `v_k` matters. Bulletproof + consistent with the abstract.
- If `tau_weighted` ≈ `tau_weight_only` → weight carries it; report the unweighted heatmap honestly (median +0.35) and keep +0.996 demoted/qualified.

R4 prose edits are blocked on this rerun.

---

## RESULTS (complete — 330 runs / 33 cells; `results/tau_all_datasets.csv` now has `tau`, `tau_weighted`, `tau_weight_only`)

| ranking | positive | mean | median | min | max |
|---|---|---|---|---|---|
| `v_k` unweighted (old heatmap) | 29/33 | +0.355 | +0.348 | −0.276 | +0.894 |
| **`w·v` weighted (Prop. transfer predictor)** | **33/33** | +0.919 | **+0.987** | +0.232 | +0.999 |
| `w` weight-only (control) | 33/33 | +0.736 | +0.810 | +0.123 | +0.974 |

- Unweighted reproduces the published heatmap exactly (mean +0.355 / median +0.348) → rerun is consistent.
- **Weighted = the estimator the theory predicts → 33/33 positive, median +0.987.** Rescues every cold cell: Citeseer/GCN-2 −0.276→+0.990; Amazon/IGNN −0.147→**+0.997** (= the +0.996 full-graph headline; now the Amazon cell of one consistent estimator, not a separate number); Cora/GCN-2 −0.032→+0.989; WikiCS/GCN-2 +0.045→+0.988; Amazon/GCN-4 −0.038→+0.997.
- **Control PASSES — `v_k` adds value beyond edge weight.** Per-arch `v_k`-lift (tauW−tauWo): APPNP +0.226, GIN-2 +0.218, GCN-4 +0.209, IGNN +0.185, SAGE-2 +0.155, GAT-2 +0.146, GCN-2 +0.122 (mean +0.18). Edge weight alone is +0.74; v_k lifts to +0.92.
- **GAT-2 reversal (honest exception):** unweighted +0.471 > weighted +0.349 (weighting loses ~0.12–0.22 on Cora/Citeseer GAT). Mechanism: attention dynamically reweights edges, so the static normalized `w_k` mis-predicts. → report GAT with the unweighted ranking.

## Reframe plan (honest + strong; pending sign-off)
1. **fig:tau_heatmap → theory-weighted `w·v` ranking** (33/33 positive, median +0.99). Regenerate `figures/fig_tau_heatmap.pdf` from the new CSV column (update `scripts/figures/make_fig_tau_heatmap.py` to use `tau_weighted`; GAT row uses `tau`).
2. **Add the control sentence:** weighted beats a pure-edge-weight baseline by Δτ≈+0.18 → `v_k` adds rank information (pre-empts "it's just the weight").
3. **GAT-2** reported with unweighted ranking + 1-line attention-reweighting mechanism.
4. **Abstract:** replace "positive in 29/33 … and reaches +0.996" with one consistent claim — "the theory-weighted ranking transfers in 33/33 cells (median +0.99, up to +0.996 on full-graph Amazon Photo), beating an edge-weight baseline by Δτ≈+0.18."
5. **Prop. transfer** keeps its scoping (magnitude bridge + sufficient pairwise order; global τ empirical) — now validated by 33/33 under the theory's own predictor.

## EXECUTED (2026-05-30) ✓ — builds at 10pp, 0 overfull
- `make_fig_tau_heatmap.py` → uniform `tau_weighted`; `fig_tau_heatmap.pdf` regenerated (all 33 cells positive).
- abstract: 33/33 + median +0.99 + Δτ≈+0.18 control + +0.996 (one consistent estimator).
- experiments.tex: transfer paragraph (33/33, Δτ control, GAT exception), caption (weighted ranking), Prop.-transfer scoping ("global τ empirical, not implied").
- **Caveat for the record:** the Δτ≈+0.18 control is the 50-node heatmap mean; the full-graph +0.996's *own* weight-only control was not separately computed, but the Amazon 50-node cell (weighted +0.997 vs weight-only +0.803) strongly implies it. Optional further bulletproofing: add `tau_weight_only` to the full-graph stratified run if a reviewer presses.
