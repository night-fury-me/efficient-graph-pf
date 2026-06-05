# Baseline Faithfulness Audit — RobustGCN-lite & GNNGuard-lite

**Scope:** verify the two "-lite" robust-GNN backbones against their official methods; assess whether the simplification is disclosed and whether the paper's "match or exceed IGNN" claim is fair.

**Impl location:** `scripts/revision_R2/R2_10_robust_arch.py`
- `RobustGCNLite` — class at **L71-108**; forward L101-108.
- `GNNGuardLite` — class at **L111-141**; forward L132-141.
- Spectral cap helper `spectral_normalize_` — **L56-68** (σ₁(W)≤0.9, applied to BOTH backbones at init + every train step).
- Train loop (per-step cap) — **L144-169**.
- AEGIS τ ranking — **L172-187**.

**Paper claim at stake:** `paper/sections/appendix/F_experiments.tex:323-324`
> "Under the σ₁(W)≤0.9 cap, RobustGCN-lite [zhu2019robust] and GNNGuard-lite [zhang2020gnnguard] match or exceed IGNN; the cap is a precondition (without it κ>1 voids Thm 1)."

Also cited in `introduction.tex:15` and `related_work.tex:8` as exemplars of "robust-architecture defenses [that] harden models without surfacing which edges are vulnerable" — i.e. the paper leans on their *identity as robustness mechanisms* to motivate AEGIS.

**Reported numbers (10 seeds, `results/revision_R2/robust_arch.csv`):**

| dataset | arch | acc | κ(J_z) | τ vs brute |
|---|---|---|---|---|
| Cora | RobustGCN-lite | 0.721 | 0.454 | +0.367 |
| Cora | GNNGuard-lite | 0.808 | 0.450 | +0.099 |
| Citeseer | RobustGCN-lite | 0.661 | 0.603 | +0.537 |
| Citeseer | GNNGuard-lite | 0.680 | 0.603 | +0.532 |

The "match or exceed IGNN" is about **τ (AEGIS-ranking fidelity vs brute-force edge-removal damage)**, NOT accuracy or adversarial robustness. (Round-2 unnormalized run: κ drifts to 0.89/1.98, τ collapses to ~0 — this is the basis of the "cap is a precondition" sub-claim, and that part checks out.)

---

## Method 1 — RobustGCN (Zhu et al., KDD 2019)

### Official defining mechanism (verified against DeepRobust source `deeprobust/graph/defense/r_gcn.py`)
1. **Gaussian latent representations** — every hidden node state is a distribution with **mean μ AND variance σ²** (`GGCL_F`/`GGCL_D` layers each carry `weight_miu` + `weight_sigma`).
2. **Variance-based attention** — `Att = torch.exp(-gamma * self.sigma)`; high-variance (adversarially-perturbed) neighbors are **down-weighted** before aggregation: `miu_out = adj_norm1 @ (miu * Att)`, `sigma_out = adj_norm2 @ (sigma * Att * Att)`. This attention IS the defense.
3. **Distribution-matching regularizer** — loss adds a KL term pulling the latent Gaussian toward a prior `N(0, I)` (`self.gc1.kl_loss(...)` / `-0.5*sum(1+2σ-μ²-σ²)`) plus a **reparameterized sample** at prediction (`self.miu + eps*σ`).

### Our "-lite" version (`RobustGCNLite`, L71-108)
| Official component | Kept? | Where |
|---|---|---|
| μ and σ projections exist | partial | `W_mu`, `W_sigma` declared L91-92 |
| σ actually used in output | **NO** | L104: `sigma` computed then explicitly labeled `# variational scale (unused for forward output)`; L105 output `Z` uses only `mu`, never `sigma` |
| variance-based attention `exp(-γσ)` down-weighting neighbors | **NO** | absent entirely; aggregation is plain `A_hat @ mu @ W_hiddenᵀ` |
| KL distribution-matching regularizer | **NO** | train loop L158-162 is plain `cross_entropy`; no KL term |
| reparameterized sampling at prediction | **NO** | deterministic forward |
| **σ₁(W)≤0.9 spectral cap (NOT part of RobustGCN)** | ADDED | L95, L164 — an IGNN-style constraint grafted on |

**What it actually is:** a 1-layer-recurrence GCN (input proj `W_mu` + hidden map `W_hidden`) with a spectral cap. `W_sigma` is a dead parameter path — its output is computed and discarded. **The defining mechanism (Gaussian variance attention + KL regularizer) is DROPPED, not approximated.** The σ computation is a no-op decoration: removing L104 would not change a single logit, gradient, or reported number.

---

## Method 2 — GNNGuard (Zhang & Zitnik, NeurIPS 2020)

### Official defining mechanism (verified against authors' README, `mims-harvard/GNNGuard`)
1. **Cosine-similarity neighbor importance** — for each edge, compute cosine similarity of the two endpoints' embeddings (`att_coef`).
2. **Edge pruning AND reweighting** — "assign higher weights to edges connecting similar nodes while pruning edges between unrelated nodes." Pruning drops low-similarity edges below characteristic threshold P₀; the *surviving* edges are **continuously reweighted by normalized similarity** (a per-edge soft attention `a_ij`), not a binary keep mask.
3. **Layer-wise graph memory** — across layers the edge weights are smoothed by a learned/fixed keep-rate β: `w_t = β·w_{t-1} + (1-β)·w_t` ("graph memory"), stabilizing the pruned structure across message-passing rounds. This is the second defining component.

### Our "-lite" version (`GNNGuardLite`, L111-141)
| Official component | Kept? | Where |
|---|---|---|
| cosine similarity of embeddings | yes (mechanically) | L135-136: `H_norm = normalize(H); sim = H_norm @ H_normᵀ` |
| edge pruning by similarity threshold | partial / weak | L137-138: `prune_mask = (sim > 0.1); A_pruned = A_hat * prune_mask` |
| similarity-based **reweighting** of surviving edges | **NO** | mask is binary {0,1}; surviving edges keep their original `A_hat` weight, NOT a similarity weight |
| layer-wise graph **memory** smoothing (β keep-rate) | **NO** | single layer, no cross-layer memory |
| **σ₁(W)≤0.9 spectral cap (NOT part of GNNGuard)** | ADDED | L126, L164 |

**Threshold concern:** ReLU embeddings (L134) are non-negative, so pairwise cosine similarity is almost always ≥0 and the `sim > 0.1` cut prunes very few edges — the "guard" is largely inert on these graphs. So even the one retained component (pruning) is weak in effect.

**What it actually is:** a 1-layer GCN with a near-trivial binary edge mask and a spectral cap. **Cosine similarity is present, but reweighting and the layer-wise memory — half the defining mechanism — are DROPPED**, and the surviving pruning is mostly a no-op at threshold 0.1.

---

## GAPS

| # | Gap | Severity | Location | Fix |
|---|---|---|---|---|
| G1 | RobustGCN-lite computes σ but **never uses it**; no variance attention, no KL regularizer. Not RobustGCN's defense at all. | **HIGH** | `R2_10_robust_arch.py:104-105`, train L158-162 | Either (a) implement `Att=exp(-γσ)` aggregation + KL loss (real RobustGCN), or (b) rename to e.g. "capped-GCN (RobustGCN-inspired backbone)" and stop citing it as RobustGCN's mechanism. |
| G2 | GNNGuard-lite drops similarity **reweighting** and **layer-wise memory**; keeps only a binary prune mask. | **HIGH** | `R2_10_robust_arch.py:137-141` | Reweight surviving edges by normalized similarity + add the β-memory across ≥2 layers; or rename/relabel. |
| G3 | GNNGuard prune threshold 0.1 on ReLU (non-neg) embeddings prunes almost nothing → mechanism inert. | MED | `R2_10_robust_arch.py:134-138` | Use signed embeddings (pre-ReLU) or a data-calibrated P₀; verify prune fraction >0. |
| G4 | Simplification NOT disclosed in the paper. Only the "-lite" suffix signals it; no sentence states what was dropped. The R2 internal report describes the impls but the appendix/related-work does not. | **HIGH** | `F_experiments.tex:323-324`; `introduction.tex:15`; `related_work.tex:8` | Add one sentence in App. F: "RobustGCN-lite / GNNGuard-lite are spectral-capped GCN backbones that retain [X] but omit [Gaussian variance attention / similarity reweighting + memory] for S_c compatibility." |
| G5 | "match or exceed IGNN" risks reading as a *robustness/accuracy* parity claim; it is only a τ (ranking-fidelity) claim, and both baselines are spectral-capped (so by construction near-IGNN in operator structure). | MED | `F_experiments.tex:323-324` | State explicitly that the comparison is AEGIS-ranking τ under a shared σ₁≤0.9 cap, not a defense-strength comparison. |
| G6 | Official code was available and unused: **DeepRobust** ships RobustGCN (`deeprobust/graph/defense/r_gcn.py`); **mims-harvard/GNNGuard** is the authors' official repo. Neither was used or cited as the impl source. | MED (process) | n/a | Note in App. F that backbones are minimal re-implementations, not the official DeepRobust/GNNGuard code, with rationale (matrix-free S_c / equilibrium-operator compatibility). |

---

## VERDICTS

### RobustGCN-lite → **MISLEADING-LABEL**
The defining mechanism of RobustGCN is the Gaussian-latent **variance-based attention** plus the **KL distribution-matching regularizer**. Our impl computes σ and then discards it (L104 comment: "unused for forward output"), has no `exp(-γσ)` down-weighting, and no KL term. It is a spectral-capped GCN with a dead σ branch. Calling it "RobustGCN-lite" and citing `zhu2019robust` for it is not faithful — none of the paper's distinguishing machinery survives. The σ scaffolding gives a false impression of fidelity.

### GNNGuard-lite → **MISLEADING-LABEL** (borderline SIMPLIFIED)
It does keep the cosine-similarity step, so it is closer than RobustGCN-lite. But GNNGuard's defense is *reweight-and-prune + layer-wise memory*; our version keeps only a binary prune mask (no reweighting, no memory), and at threshold 0.1 on ReLU embeddings the prune is nearly inert. Two of the method's defining components are absent and the third is weak in effect. The "guard" essentially does nothing on these graphs, so labeling it GNNGuard overstates fidelity. Tip toward MISLEADING-LABEL; at best SIMPLIFIED-but-UNDISCLOSED.

### Is "match or exceed IGNN" fair? → **NOT (as currently worded)**
Two problems: (1) **scope** — it is a τ ranking-fidelity claim, but reads as defense/accuracy parity; (2) **circularity/triviality** — both backbones are forced to σ₁(W)≤0.9, the same contraction constraint that defines IGNN's regime, so they are *capped-GCN operators* by construction. Matching IGNN on τ is then close to tautological (similar contractive operator ⇒ similar S_c ⇒ similar ranking), and does NOT demonstrate that AEGIS handles *robust architectures* — because the robustness mechanisms were removed. The claim is technically true of the code as written but not of the methods it names, and the disclosure needed to make it honest (G4/G5) is missing.

---

## PAPER-NUMBER-AT-RISK
- **The "robust backbones" paragraph, `F_experiments.tex:323-324`** — the entire sentence and its τ values (RobustGCN-lite +0.367/+0.537; GNNGuard-lite +0.099/+0.532, Cora/Citeseer). If a reviewer checks the cited methods, the τ numbers stand but their *attribution to RobustGCN/GNNGuard* does not.
- **Motivating framing in `introduction.tex:15` and `related_work.tex:8`** — these cite zhu2019robust/zhang2020gnnguard as robustness defenses AEGIS complements; the appendix is the only place the paper claims to *run* them, so a faithfulness challenge lands squarely on App. F.
- No headline-table number depends on these; the risk is a **credibility/faithfulness flag on a supporting "generalizes to robust architectures" claim**, not a main result. Cheapest fix is disclosure (G4+G5: ~2 sentences) rather than re-implementation; bulletproof fix is to implement the real mechanisms (G1+G2) and rerun the 10-seed τ.

## OFFICIAL CODE REFERENCES (for fix / re-impl)
- RobustGCN: DeepRobust — `deeprobust/graph/defense/r_gcn.py` (`GGCL_F`/`GGCL_D`, `Att=exp(-γσ)`, `kl_loss`). Verified.
- GNNGuard: `github.com/mims-harvard/GNNGuard` (authors' repo; README verified — "higher weights to similar edges while pruning unrelated", graph memory). Source `.py` paths under that repo (gnn_misg / defense modules) host `att_coef` + memory.
