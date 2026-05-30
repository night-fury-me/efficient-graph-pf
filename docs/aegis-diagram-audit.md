# AEGIS Overview Diagram — Equation/Notation Audit (merged)

Diagram: `paper/figures/aegis.drawio` → `figures/aegis--overview--diagram.pdf` (theory.tex:112).
Refs: `theory.tex`, `framework.tex` (`alg:aegis`), `experiments.tex`, `abstract.tex`, `conclusion.tex`.
Merges: this-session audit + cross-session review. Date 2026-05-29.
Provenance tags: [both] · [rev] reviewer-only · [me] this-session-only. Status: ✅ done · ⬜ to-do.

---

## Resolution (applied 2026-05-29)

**Applied to `aegis.drawio` + re-exported PDF/PNG (page size unchanged, 1140.96×773.04 pts):**
P0: #1 τ-range ✅, #2 "central technical contribution"→"One analytical object, three diagnostics" ✅, #3 radius→`min_{c≠y_v} m_v^{(c)}/‖(W_{y_v}−W_c)S_v‖₂` ✅, #4 5 datasets ✅.
P1: #5 PyTorch ✅, #6 typos ✅ (already in source).
P2: #7 all `D`/`N_d`/`N·d`→`Nd` ✅, #8 `δÂ⋆` ✅, #9 `k=10` ✅, #10 Explicit-GNN→"S_K: unrolled K-layer sensitivity" ✅, #11 `\mathcal{F}`→`F` ✅, #12 `κ=‖J_z‖₂` ✅, #13 `K∈[20,50]` ✅, #14 "Per-node sensitivity radius" ✅.
P3: #17 Forward Pass→`O(N_sub·d)` ✅.

**Documented, intentionally NOT changed (no-op or against "no clutter" preference):**
#15 `v_k`/`v_ij` — paper uses BOTH (`v_k`=theory.tex:86, `v_ij`=alg/abstract); already paper-consistent.
#16 `K`/`k`/`S_K` overload — inherent to the paper's own notation; renaming would diverge from the paper.
#18 `/√2` — current column formula matches theory.tex:63 exactly (the `/√2` lives in basis `b_k`, not the column); adding it would diverge.
#19 BFS `N_sub≤200` — correct (dense regime); experiments' 50-node default is a separate parameter.
#20 `σ₁`/`ε_crit` outputs omitted — by design (abstract's "three diagnostics"); not adding per your "results don't belong in the overview" call.

---

## P0 — Content out of sync with revised paper (highest priority)

| # | Block | Diagram | Target (paper) | Tag | Status |
|---|-------|---------|----------------|-----|--------|
| 1 | Power Flow N-1 | `τ = 0.37–0.72` | `τ = +0.37 to +0.62` (0.72 = dropped case300) — case_study.tex:6 | [both] | ✅ fixed this session |
| 2 | Constrained Projection (on P_c) | "The central technical contribution" | De-overclaimed: P_c = "standard duplication-matrix reduction" (theory.tex:63); operational contribution = matrix-free routing. → "matrix-free routing N²→\|E\|" or drop superlative | [rev] | ⬜ **still present** |
| 3 | Per-Node Radius | `r_v = m_v/(‖∇f‖₂·‖S_v‖₂)` | eq:radius (theory.tex:69): `r_v = min_{c≠y_v} m_v^{(c)} / ‖(W_{y_v}−W_c)S_v‖₂` (min-over-classes, composed norm). Current form is the superseded surrogate AND uses a bogus `‖∇f‖` | [both] | ⬜ |
| 4 | Cross-Domain Transfer | `7 archs × 9 datasets` | `7 archs × 5 datasets` (33-cell grid = 5×7−2; 9 is the full suite) — experiments.tex:186 | [both] | ✅ fixed this session |

## P1 — Typos / capitalization

| # | Block | Fix | Tag | Status |
|---|-------|-----|-----|--------|
| 5 | Matrix-free row | `pyTorch` → `PyTorch` | [rev] | ⬜ **still present** |
| 6 | Outputs / Defense | `Vulnarability→Vulnerability`, `perturnbation→perturbation` | [rev] | ✅ already correct in source (0 occ.); re-exported |

## P2 — Notation / dimensions

| # | Block | Diagram | Target | Tag | Status |
|---|-------|---------|--------|-----|--------|
| 7 | Sensitivity + Projection (all complexities & matrix dims) | mixes `D`, `N_d`, `N·d` | unify to **`Nd`** (paper): `O(D²)→O((Nd)²)`, `O(D×N²)→O(Nd·N²)`, `O(D³)→O((Nd)³)`, `O(K·D)→O(K·Nd)`, `N_d→Nd`; `D×N²→Nd×N²`, `D×\|E\|→Nd×\|E\|` | [me] | ⬜ |
| 8 | Outputs → Optimal Attack | `δA*` | `δÂ⋆` (`\delta\Ahat^\star`) — constrained-pipeline attack, abstract/framework/alg | [me] | ⬜ |
| 9 | Vuln. → Randomized SVD | `rSVD(S_c), k=6` | `k=10` (framework.tex:34 `k=p=10, n_iter=2`; no `k=6` in paper) | [me] | ⬜ |
| 10 | App. Ext. → Explicit GNN Extension | garbled "S_K — Per-node tolerance budget" | "unrolled K-layer sensitivity (S_K replaces S)" — theory.tex:123/133 | [both] | ⬜ |
| 11 | Graph Proc / Sensitivity | script `\mathcal{F}` | plain `F` (paper: `F(Z)=ReLU(…)`, `G=z−F`) | [me] | ⬜ |
| 12 | Matrix-free → Power Iter | `κ(J_z)` | `κ = ‖J_z‖₂` (alg.1) | [me] | ⬜ |
| 13 | Matrix-free → Neumann | `K ≈ 20–100` | `K ∈ [20,50]` (framework.tex:36) | [me] | ⬜ |
| 14 | Outputs → Sensitivity Radii | "Per-node tolerance budget" | paper term "per-node sensitivity radii"; pick ONE term (also vs intro "robustness budget") | [both] | ⬜ |
| 15 | Vuln. → Column Norms vs Outputs | `v_k` vs `v_ij` | unify (k indexes edge (i,j)) | [both] | ⬜ |
| 16 | Symbol overload | `K` = Neumann depth AND `S_K` layer count; `k`(=rank) vs `K` visually confusable | use distinct glyphs | [rev] | ⬜ |

## P3 — Verify / optional

| # | Block | Note | Tag |
|---|-------|------|-----|
| 17 | Graph Proc → Forward Pass | `O(T·N_sub)` has `T` (belongs to fixed-point) + drops `d`; likely `O(N_sub·d)` — confirm | [me] |
| 18 | Constrained Projection | `[S_c]_{:,k}=S_{:,iN+j}+S_{:,jN+i}` omits `/√2` of unit-norm edge basis `b_k` — matches theory.tex:63 column formula; acceptable, note if exactness wanted | [rev] |
| 19 | Graph Proc → BFS | `N_sub ≤ 200` is dense regime; experiments default = 50-node BFS — optional | [me] |
| 20 | Outputs | alg returns 5 (`σ₁, δÂ⋆, {v_ij}, {r_v}, ε_crit`); diagram shows 3 "diagnostics" — by design; optional add `ε_crit=(1−κ)/‖W‖₂` (IGNN-only) | [me] |

## Verified correct (no change)
`[S_c]_{:,k}` indexing · Neumann `Σ_{k=0}^K J_z^k b` · Linear Solve `S=(I−J_z)⁻¹J_A` · `v_k=‖[S_c]_{:,k}‖₂` · `Z=f_θ(X,A)` · `F(Z*,A)=Z*` (modulo #11) · `X∈R^{N×d}` · N≤200/N>200 split · 29/33 cells.
