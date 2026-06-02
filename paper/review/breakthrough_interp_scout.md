# Breakthrough scout — Mechanistic interpretability of equilibrium computation via the resolvent

**Date:** 2026-06-02 · **Scope:** brutal viability check on reframing the AEGIS line as
*"mechanistic interpretability of fixed-point / equilibrium computation"* using the IFT
resolvent `(I−J_z)⁻¹` and the constrained sensitivity `S_c = (I−J_z)⁻¹ J_A P_c` as **causal**
objects that reveal *what algorithm an equilibrium model actually computes.*

**One-line verdict.** **Breakthrough-viable, but ONLY if it ships the killer demo** (recover a
known graph algorithm's causal structure from the resolvent on a DEQ-GNN trained to execute it).
Without that demo it is **interesting-but-unvalidatable** and collapses into "relabeled
sensitivity." It is **not already-crowded** — the two halves it would fuse exist separately and
the intersection cell is genuinely empty. This is the highest-ceiling pivot scouted so far, and
the only one whose central claim is *empirically falsifiable* rather than a slogan.

---

## 1. Prior art — where the gap actually is

Three literatures border the idea. None occupies the target cell.

### (A) Fixed-point / Jacobian interpretability of *recurrent* dynamics — the real methodological precedent
- **Maheswaranathan, Williams, Golub, Ganguli, Sussillo, "Reverse engineering recurrent
  networks … reveals line attractor dynamics," NeurIPS 2019 (arXiv:1906.10720).** THE canonical
  move the proposal wants to make: *find fixed points → linearize → read the Jacobian's
  eigen/topological structure → recover the algorithm* (here, a line attractor implementing
  integration for sentiment). Universality across LSTM/GRU/vanilla shown.
- Lineage: Sussillo & Barak (2013, fixed-point topology of trained RNNs); **"Reverse engineering
  RNNs with Jacobian switching linear dynamical systems," 2021 (arXiv:2111.01256)**;
  **"Mechanistic interpretability of RNNs emulating HMMs," 2025 (arXiv:2510.25674)**.
- **Why this is the load-bearing prior art AND why it does NOT close the gap:** it is
  *feature-space* dynamics `h_{t+1}=F(h_t,x_t)`, fixed points found *numerically* (the network
  is not defined by its fixed point), and the analyzed Jacobian is `∂F/∂h` (the *recurrence*
  Jacobian `J_z`). It NEVER (i) treats the fixed point as the model's defining object (DEQ/IGNN),
  (ii) uses the **input/structure** Jacobian `J_A`, (iii) forms the **resolvent** `(I−J_z)⁻¹` as
  the causal transfer operator, or (iv) works on **graphs / structural (edge) perturbations**.
  → AEGIS's `S_c` is precisely the object the Sussillo program lacks: the *closed-form,
  matrix-free, input→equilibrium causal map*, not a numerically-traced state-space portrait.

### (B) Equilibrium models trained on KNOWN algorithms — the demo substrate already exists (no-go risk)
- **Georgiev, Wilson, Buffelli, Liò, "Deep Equilibrium Algorithmic Reasoning," 2024
  (arXiv:2410.15059).** Trains **DEQ-GNNs to execute CLRS-30 algorithms** (Bellman-Ford,
  shortest-path, etc.) by *directly solving the equilibrium* — explicitly because an algorithm's
  output is a fixed point. **This builds exactly the models the killer demo needs.**
- **CRITICAL — it does NOT interpret them.** Its entire contribution set (per its abstract) is:
  equilibrium-solving formulation + performance + regularisations to stabilize the solve. The
  Jacobian/resolvent is used (if at all) only as an *implicit-diff training mechanism*, never as
  an interpretability lens. No eigen-decomposition, no causal decomposition, no algorithm
  recovery. → The substrate is public; the *interpretation* is unclaimed. This is the single most
  important finding: someone already trained known-algorithm equilibrium GNNs and **left the
  resolvent-interpretability question on the table.**
- Adjacent NAR theory that gives the *ground-truth causal structure* to validate against:
  **Veličković et al., "Neural Algorithmic Reasoning" (2105.02761)**; **"GNNs are Dynamic
  Programmers," ICLR 2022 (2203.15544)**; **Xu et al., "What Can Neural Networks Reason About?"
  (algorithmic alignment, 1905.13211)**. These hand you the *correct* causal graph (the DP
  recurrence's dependency structure / Bellman update support) the resolvent must reproduce.

### (C) GNN explainability & influence/Jacobian attribution — the crowded-but-orthogonal neighbor
- GNNExplainer / PGExplainer / subgraph-attribution: explain *which subgraph* drives a
  prediction, but treat the GNN as a **black-box feed-forward map**; no fixed point, no operator
  decomposition, no "what computation." Saliency, not mechanism.
- **Influence functions (Koh-Liang 2017)** and **"Jacobian Scopes: token-level causal
  attributions in LLMs," 2026 (2601.16407)** — Jacobian-as-causal-attribution is an active 2025–26
  theme, but for *training-point* or *token-level* attributions, never the IFT resolvent of an
  equilibrium operator, never eigen-mode *primitives*. AEGIS's own related-work already
  distinguishes `S_c` from influence functions (structure not training set; equilibrium resolvent
  not training Hessian; full geometry not a scalar).
- **Caveat to respect — "Computational Complexity of Circuit Discovery for Inner
  Interpretability," 2024 (2410.08025):** faithful *discrete* circuit discovery is hard in
  general. The resolvent route *sidesteps* this (it is a continuous, closed-form linear-response
  object, not a combinatorial search over sub-circuits) — but the report must not over-claim a
  full discrete "circuit"; it yields a **causal linear-response decomposition**, which is weaker
  than a Boolean circuit and should be sold as such.

### The gap, pinned
> No prior work uses the **IFT resolvent `(I−J_z)⁻¹` / `S_c`** as a **causal circuit-like
> decomposition** of an **equilibrium GNN** to **recover and validate a known graph algorithm's
> causal structure.** (A) does fixed-point-Jacobian interpretability but on RNN feature dynamics,
> not the structural resolvent, not graphs. (B) builds known-algorithm equilibrium GNNs but never
> interprets them. (C) attributes but does not decompose computation. The synthesis A×B×C is open.

---

## 2. The mechanism — is there a principled construction, or just relabeled sensitivity?

This is where honesty matters most. There is a **genuine, non-trivial construction**, but it
needs *new theory* to be more than `S_c` with an interpretability sticker.

**What the resolvent legitimately is.** The IGNN equilibrium `z* = φ(Âz*Wᵀ + X)` is the
fixed point, and the *converged linear response* of node states to a perturbation `Δ` injected at
the operator is `Δz* = (I−J_z)⁻¹ J Δ`, with `J_z = diag(φ′)(Â⊗W)`. So `(I−J_z)⁻¹ = Σ_k J_zᵏ`
(Neumann) is **literally the sum over all message-passing path-lengths of the converged
computation** — `[(I−J_z)⁻¹]_{ij}` is the total causal gain from node/feature `j` to node `i`
*integrated over the infinite unrolling the equilibrium represents*. This is a real causal object:
it is the equilibrium analogue of a "circuit" (paths × gains), and it is *exactly* what layer-wise
circuit tracing cannot reach because a DEQ has no layers. That part is sound and is the genuine
hook ("DEQs have no layers, so the resolvent IS the trace").

**Three candidate constructions, ranked by how much is real vs. relabeling:**

1. **Path/edge causal decomposition (REAL, but ≈ existing `S_c` per-edge columns).** Column
   `[S_c]_{:,k}` already *is* "how the equilibrium responds to edge `k`." Calling it "which edges
   causally implement the output" is **mostly relabeling** of the per-edge transfer the paper
   already reports (the τ≈0.99 backbone). Necessary but not sufficient for breakthrough.

2. **Eigen-modes of `(I−J_z)⁻¹` as "computational primitives" (REAL *if validated*, otherwise
   hand-wavy).** The eigvecs of `J_z` (hence of the resolvent, same eigenvectors, eigenvalues
   `1/(1−λ)`) are the natural "modes" of the converged computation; the near-critical modes
   (`λ→1`, huge resolvent gain) are the directions the computation *amplifies most*. The claim
   "each dominant eigen-mode corresponds to a sub-routine of the algorithm" is **physically
   motivated but is the swamp risk**: without a ground-truth algorithm to map modes onto, "mode =
   primitive" is unfalsifiable narrative. It becomes rigorous ONLY under the §3 demo, where modes
   can be matched to known DP sub-structures (e.g., a Bellman-Ford relaxation front, a
   connected-component indicator). This is the make-or-break construction.

3. **Causal *intervention* test (REAL, NEW, and the rigor-anchor).** The resolvent predicts a
   *first-order* causal effect; the equilibrium model lets you *actually intervene*
   (perturb edge/feature, re-solve, measure true `Δz*`) and **verify the resolvent's predicted
   causal pathway matches the realized one.** This do-calculus-style check (predict from
   `(I−J_z)⁻¹`, then *do* the intervention and re-converge) is what upgrades the whole thing from
   "sensitivity" to "**validated causal mechanism**." The machinery for this *already exists in the
   repo* — the reachability experiment already perturbs `Â`, re-solves the nonlinear equilibrium,
   and compares to the linear-response prediction (it found the linearization is faithful below
   criticality, γ≈1.02). That is precisely the predict-then-intervene loop, already shown to hold.

**Verdict on the mechanism:** there is a principled construction (path-integrated causal gain +
eigen-mode primitives + predict-then-intervene validation). Constructions (1) alone is relabeling;
(2)+(3) together, *anchored to a known algorithm*, are a genuine new contribution. The dividing
line between "rigorous" and "swamp" is **whether there is ground truth to validate the mode↔
primitive map.** That is entirely supplied by §3.

---

## 3. The killer result — the one demo that makes it a breakthrough

**Demo (the make-or-break):** *Resolvent recovery of a known graph algorithm.*

- **Train** an implicit/equilibrium GNN (the existing IGNN operator, or a MonDEQ-GNN — both
  already run in this repo, §MonDEQ probe) to execute a graph algorithm whose causal structure is
  **known in closed form**. Best targets, easiest→hardest:
  1. **Single-source shortest path / Bellman-Ford** (CLRS-30; ground-truth causal structure =
     the DP relaxation dependency graph; equilibrium = converged distances). *Substrate already
     built by arXiv:2410.15059 — directly reusable.*
  2. **Connected components / reachability** (ground truth = transitive closure; eigen-structure
     of the resolvent should expose component-indicator modes — clean spectral prediction).
  3. **A symbolic logic rule / message-passing fixpoint** (e.g., a Datalog-style 1-step rule whose
     least fixpoint is exactly the equilibrium) — cleanest possible ground truth.
- **Claim to verify (all falsifiable):**
  - **(C1) Causal support recovery.** The edges/nodes with large resolvent gain
    `|[(I−J_z)⁻¹ J_A]|` coincide with the algorithm's true data-dependency support (e.g., for SSSP,
    the resolvent's causal pathway from source→target tracks the *actual shortest path edges*,
    not just adjacency). **Metric:** rank-correlation / AUROC of resolvent gain vs. ground-truth
    dependency, *beating* a black-box saliency / GNNExplainer baseline. (AEGIS already has the
    τ-style ranking harness.)
  - **(C2) Mode ↔ primitive.** Dominant eigen-modes of `(I−J_z)⁻¹` map to named algorithmic
    sub-structures (relaxation front / component indicator). **Metric:** alignment of each mode's
    support with the labeled sub-structure.
  - **(C3) Predict-then-intervene (the causal clincher).** Resolvent-predicted effect of deleting
    a "critical" edge (one on the shortest path) vs. an "inert" edge matches the *re-solved*
    equilibrium's true change, and the predicted-critical edge is the one whose removal actually
    changes the algorithm's output. **Metric:** predicted vs. realized `Δz*` correlation
    (the reachability experiment already showed this holds below criticality) AND output-flip
    accuracy. This is the do-intervention validation that defeats "it's just saliency."
- **Why this is a breakthrough if it lands:** it would be the **first demonstration that an
  equilibrium model's converged computation can be mechanistically read off a closed-form causal
  operator, validated against ground-truth algorithmic structure** — i.e., mechanistic
  interpretability for the layer-less (DEQ/implicit) regime where transformer-circuit tooling does
  not apply. It converts AEGIS's `S_c` from "an audit heuristic" into "the causal trace of
  equilibrium computation," with an unimpeachable validation (known algorithm).

**Feasibility with EXISTING machinery — HIGH.**
- Matrix-free resolvent (truncated Neumann), randomized SVD, `J_z`/`J_A` builders, per-edge
  ranking (τ), and **predict-vs-re-solve intervention** are *all already implemented and
  bug-audited* (S_c machinery 0.00% vs dense SVD; reachability predict-then-intervene shown
  faithful below criticality with γ≈1.02).
- The MonDEQ probe shows the pipeline runs on a *second* equilibrium family (breadth for free).
- The ONLY genuinely new engineering: (i) train the IGNN/MonDEQ on a CLRS task (small graphs,
  cheap; substrate exists in 2410.15059's setup), (ii) extract ground-truth dependency labels
  (trivial for SSSP/CC), (iii) the mode↔primitive alignment metric (new but straightforward).
- **Risk flags (be honest):** (a) IGNN may not *cleanly* learn the algorithm (NAR is finicky;
  may need the DEQ-AR regularisations) → start with CC/Datalog where the fixpoint is exact, not
  the hardest CLRS tasks. (b) **The contraction `κ<1` that makes `S_c` well-defined may conflict
  with the expressivity needed to execute an algorithm** (a relaxation front is near-critical) —
  this is the deepest scientific risk and also potentially the *most interesting result*: it would
  tie "the algorithm lives near criticality" to AEGIS's own `ε_crit`/resolvent-divergence story,
  unifying the interp pivot with the salvaged criticality finding. (c) Eigen-modes may be
  *delocalized* (AEGIS already found the attack mode spans tens of edges) → modes may not map
  one-to-one to crisp primitives; mitigate by choosing tasks with low-dimensional solution
  structure (CC, line-attractor-like SSSP), exactly as Sussillo'19 found low-D structure.

---

## 4. Honest verdict

**BREAKTHROUGH-VIABLE, conditional on the §3 killer demo.** Tiered:

- **If C1+C3 hold on even ONE known algorithm (e.g., shortest-path or connected-components):**
  genuine breakthrough — "mechanistic interpretability of equilibrium computation, validated
  against ground truth," a clean new cell (A×B×C). High-ceiling, defensible at a top venue.
- **If only C1 holds (causal support recovery, no clean mode↔primitive map):** solid, novel
  paper ("the equilibrium resolvent is a faithful causal attributor for implicit GNNs, beating
  black-box explainers"), but *incremental over influence/attribution* — good, not breakthrough.
- **If the model can't learn the algorithm with `κ<1`, OR modes are hopelessly delocalized:**
  the *interpretability* claim is unvalidatable → fall back to the criticality-unification reading
  (algorithm-execution forces near-criticality), which is interesting but is the swamp the
  reachability finding already warned about.

**Why it beats the prior pivots scouted in this repo.** The criticality "law" was *refuted*
(distant large-ε limit), Fisher/Certify were *guarantee-upgrades* of `S_c` (no new falsifiable
phenomenon). This pivot is the first with a **central claim that is concretely falsifiable by a
single experiment using machinery already built and audited** — and whose *failure modes are
themselves publishable* (the κ<1 ↔ expressivity tension unifies with AEGIS's own physics). The
swamp risk is real and is fully localized to one question (is there ground truth for the
mode↔primitive map?), which §3 resolves by construction.

**Recommended next action (one experiment, per the project's one-at-a-time protocol):** the
**connected-components OR shortest-path resolvent-recovery probe (C1+C3 only, seed 42 smoke)** —
train the existing IGNN on the smallest CLRS-style instance, check (a) it converges with κ<1,
(b) resolvent gain ranks true dependency edges above chance, (c) predict-then-intervene matches
re-solved Δz*. That single smoke test *gates the entire pivot*: GO if all three clear, else fall
back. Do not write theory before this smoke passes (debug-before-accepting; bulletproof-over-
handwaving).

---

### Key references (for the eventual related-work)
- Maheswaranathan/Sussillo, NeurIPS 2019 — arXiv:1906.10720 (fixed-point Jacobian reverse-eng; THE precedent to differentiate from)
- Sussillo & Barak 2013; arXiv:2111.01256 (Jacobian switching LDS); arXiv:2510.25674 (RNN-as-HMM mech-interp)
- Georgiev/Liò 2024 — arXiv:2410.15059 (Deep Equilibrium Algorithmic Reasoning; the demo SUBSTRATE, leaves interpretation open)
- Veličković, Neural Algorithmic Reasoning — arXiv:2105.02761; GNNs as DP — arXiv:2203.15544; algorithmic alignment — arXiv:1905.13211 (ground-truth causal structure)
- Gu et al., Implicit GNNs — arXiv:2009.06211; Bai et al., DEQ — arXiv:1909.01377 (model class)
- Circuit-discovery complexity caveat — arXiv:2410.08025; Jacobian-causal-attribution trend — arXiv:2601.16407
