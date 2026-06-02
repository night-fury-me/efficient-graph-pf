# Interp gating — SSSP shortest-path-tree resolvent decoding (seed 42)

**VERDICT: EXPRESSIVITY-FLOOR (architecture, not the κ<1 contraction wall). PIVOT DEAD on SSSP.**

The decisive discriminating gate for the interpretability pivot. Asks whether the
equilibrium resolvent `(I − J_z)⁻¹` / constrained sensitivity `S_c` mechanistically
**decodes the shortest-path TREE** — a strict weighted subset of edges that is NOT
reducible to adjacency support or graph spectrum (the opposite of the connected-
components gate, whose causal structure WAS adjacency support and therefore tied all
baselines).

- Script: `scripts/exp_interp_sssp.py`
- CSV: `results/exp_interp_sssp.csv` (per-graph G1/C1'/C3'/C2')
- Runtime: 77 s on one RTX 4090, seed 42.

## Headline numbers (held-out graphs, seed 42)

| Quantity | Value |
|---|---|
| κ = ρ(J_z) | **0.3306 ± 0.0080** (≪ 1; contraction holds — but the model chose this, far below the 0.95 ceiling) |
| **G1** κ<1 IGNN SSSP corr (held-out graphs) | **0.533 ± 0.045** (train R2 0.209, val-node R2 0.200) — FAILS |
| **G1-control** UNCONSTRAINED same-arch (no projection, ‖W‖·‖A‖=6.9) | train_corr 0.879, **held-out corr 0.453** — ALSO FAILS (memorizes train, no generalization) |
| **C1'** SP-tree recovery AUC: resolvent | 0.852 ± 0.039 |
| C1' inputgrad baseline | 0.848 ± 0.039 |
| C1' hop-distance baseline | **0.859 ± 0.031** (best baseline) |
| C1' adjacency-incidence baseline | 0.692 ± 0.013 |
| **C1' resolvent − best-baseline margin** | **−0.007** (TIES; does NOT beat) |
| **C3'** tree/non-tree separation (resolvent-pred / re-solved) | 1.09× / 1.09× (no separation in the model) |
| C3' true-Dijkstra tree vs non-tree | tree |dd|=3.14e-2, non-tree |dd|=0.0 (algorithm separates perfectly; model does not) |
| **C2'** eigen-mode alignment with SP-tree depth (resolvent vs adjacency) | 0.284 vs 0.254, margin +0.030 (negligible) |

## What this means

**G1 is the load-bearing result and it FAILS — but not as a κ<1 contraction wall.**
The decisive distinction comes from the unconstrained same-architecture control:

- κ<1 IGNN: held-out distance corr **0.533**, and ρ(J_z) collapses to **0.33** — far
  below the 0.95 ceiling it was allowed. The trained operator chooses to be *strongly*
  contractive and only smooths LOCAL distance structure.
- UNCONSTRAINED control (no `_project_W`, ‖W‖·‖A‖ = 6.9, expansive): train_corr 0.879
  but held-out corr **0.453** (val R2 went negative in sweeps). It MEMORIZES training
  nodes and does not generalize SSSP either.

Because **neither** the contractive nor the expansive variant of `z* = relu(A·Wz + Ux)`
generalizes SSSP, the failure is an **architectural expressivity floor**: a
linear-aggregation ReLU operator lacks the **min-plus** inductive bias that shortest
paths require. (DEQ algorithmic-reasoning papers that learn SSSP use min / attention
aggregators, not `A·Wz`.) Under κ<1 this architecture additionally collapses to a strong
contraction that captures only a weak linear-walk / resistance-like proxy for distance.

**Honest caveat on framing the AEGIS-criticality tie.** This run does NOT cleanly
isolate "κ<1 cannot do long-range min-plus while unconstrained can" — both fail here, so
the headline cannot be "contraction wall." The mechanism observed (ρ(J_z) collapsing to
0.33 and capturing only local distance) is *consistent* with the criticality intuition
that long-range min-plus computation needs ρ→1, but the clean wall would require an
architecture that the unconstrained variant CAN fit and κ<1 CANNOT. With the mandated
`IGNN_Kappa` operator that separation did not materialize. Report the result as an
architecture/expressivity floor, not as a proven contraction wall.

**C1'/C3'/C2' are formally MOOT** (you cannot decode an algorithm the model never
executed) but were computed and reported in full for completeness and to substantiate
the verdict. They reinforce it:

- **C1' (the crux): the resolvent does NOT beat baselines.** Resolvent AUC 0.852 vs
  inputgrad 0.848 (tie) and hop-distance 0.859 (the resolvent actually loses to a pure
  topology heuristic). This is the SAME ties-baselines pattern as the CC gate. The only
  baseline the resolvent beats is bare adjacency-incidence (0.692) — which is not a
  meaningful win. So even setting aside G1, C1' is WEAK.
- **C3': the model has no tree/non-tree causal separation** (1.09×), even though the
  TRUE algorithm separates perfectly (tree edge changes d(s,u) by ~δ; the "losing"
  non-tree edge changes it by exactly 0). The resolvent faithfully predicts the *model's*
  equilibrium response (cos 0.999–1.000 vs the re-solved nonlinear equilibrium — the
  resolvent math is correct), but the model's response is flat across tree/non-tree
  because it did not learn shortest paths.
- **C2': resolvent eigen-modes align with SP-tree depth no better than adjacency**
  (+0.030).

## Non-definitional design (the point of choosing SSSP)

The CC gate was weak because "which edges reach u" = the connected edges by definition,
so every method tied. SSSP avoids this ONLY if weighted distance is not collinear with
hop-count. A first attempt with random **geometric** graphs was **rejected** in
debugging: there corr(weighted-dist, hop-dist) ≈ 0.92, so SSSP degenerates back into
BFS/reachability (the definitional trap in disguise; the hop-distance baseline then
trivially wins). The final design uses **connected graphs with log-uniform random
weights decoupled from topology** (avg degree 4):

- `[design] corr(weighted-dist, hop-dist) = 0.571` — paths genuinely REORDER; SSSP is
  not reducible to BFS.
- Shortest-path tree edges are **binding**: raising a leaf's last tree-edge weight by δ
  moves the true d(s,u) by ~δ (true-Dijkstra tree |dd| = 3.14e-2 vs non-tree 0.0), so
  C3' is a meaningful causal test — the algorithm separates cleanly even though the
  trained model does not.

## Correctness / self-checks (ALL PASS)

- **S1** Dijkstra: networkx `single_source_dijkstra` vs `scipy.sparse.csgraph.dijkstra`
  agree exactly on distances AND shortest-path-tree edge sets (|tree| = 39 on the
  40-node check). Ground truth is independently cross-verified.
- **S2** resolvent input block: `(I − J_z)⁻¹ J_x[:,src]` vs autograd `∂z*/∂x_s` through
  the full differentiable fixed-point solve — **rel-err 1.9e-16, cos 1.000000**
  (bit-exact; the reused, previously-validated CC path, now on the weighted operator).
- **S3** (the NEW critical check — the C1' edge-weight path): `S_c` column
  `(I − J_z)⁻¹(J_A[:,iN+j]+J_A[:,jN+i])` vs autograd `∂z*/∂w_e` for a single physical
  edge weight — **rel-err 9.8e-11, cos 1.000000**. Confirms the edge-weight sensitivity
  the C1' ranking is built on is exact (the looser tol vs S2 is the finite-difference
  `J_A`, as expected; still 1e-10).

The use of an **unnormalized** weighted operator `A_w[i,j] = w_ij` is deliberate so that
`∂z*/∂w_e` is a *literal* `S_c` column: GCN renormalization would smear one physical
weight across many `A_hat` entries and muddy "edge-weight sensitivity." Weights are
rescaled by a constant (spectral radius → 1) which preserves the shortest-path tree
exactly (argmin is scale-invariant) and keeps the κ<1 budget meaningful. Double
precision throughout the linear algebra.

## Debug-before-accepting trail

The WEAK/floor result is genuine, not a bug:
1. The resolvent–vs–re-solved cosine is 0.999–1.000 (C3'), so the resolvent machinery is
   correct; the flat response is the model's, not the math's.
2. G1 failure was stress-tested before acceptance: swept κ∈{0.95, 0.99, 0.999},
   hidden∈{64, 128, 256}, epochs up to 1000, depth∈{15, 30}, 24 training graphs, and two
   weight ranges. Held-out corr never exceeded ~0.53. Narrowing weights to [0.5,2.0]
   (closer to hop-distance) *lowered* held-out corr to 0.40, confirming the model
   captures even less genuine weighted structure than hop-count.
3. The unconstrained control rules out "under-trained / under-capacity" and "κ too tight"
   — it has 6.9× expansive gain and still fails to generalize (memorizes train only).

## Recommendation

Do **not** pivot the AEGIS line to "the resolvent mechanistically decodes graph
algorithms." Two clean, hostile algorithms (connected-components, shortest-path) have now
both returned WEAK on the discriminating recovery test, for different reasons:
CC tied baselines definitionally; SSSP is not even executable by the mandated
`relu(A·Wz)` IGNN, and where measurable the resolvent still tied/lost to baselines.

If the interpretability angle is pursued further, it requires a fundamentally different
ingredient that this gate has now scoped out:
- an architecture with the right inductive bias (min-plus / attention aggregator) so the
  model actually executes the algorithm — only then is "does the resolvent decode it"
  a well-posed question; and
- a target whose causal structure beats BOTH adjacency support (CC's failure) AND a
  topology heuristic like hop-distance (SSSP's failure here).

The constructive science remains: `S_c` is exact (S2/S3 bit-level), faithfully predicts
the model's own nonlinear equilibrium response (C3' cos ≈ 1.0), and its analytic
foundation is intact — but on these two algorithms it decodes the *model's* computation,
which simply is not the graph algorithm. That keeps `S_c` valid as a **sensitivity /
model-auditing** tool (the existing AEGIS scope) and closes the "mechanistic algorithm
decoder" pivot.
