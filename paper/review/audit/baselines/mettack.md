# Baseline Faithfulness Audit — Mettack / Metattack (Zügner & Günnemann, ICLR 2019)

**Cite key:** `zugner2019adversarial`
**Official code:** DeepRobust `deeprobust.graph.global_attack.Metattack` / `MetaApprox`
(author TF: `danielzuegner/gnn-meta-attack`).
**Auditor protocol:** team baseline-verification (precedent: GR-BCD / PR-BCD failures).
**Date:** 2026-06-05.

---

## (a) Implementation location + official-vs-hand-rolled

| File | Role |
|------|------|
| `iem/examples/mettack_comparison.py` | Defines `SurrogateGCN` (L45), `mettack_edge_scores` (L62), `evaluate_attack` (L122), single-run driver. **This is the attack.** |
| `scripts/exp_mettack_10seed.py` | 10-seed × 3-dataset wrapper. Imports `mettack_edge_scores` + `evaluate_attack` from the file above (L33-35). Produces the 150 cells. |
| `scripts/revision_R2/R2_03_stats_reanalysis.py` | `mettack_sign_test` (L173) → the `149/150, p<10^-43` claim. |

**Verdict on provenance: HAND-ROLLED.** `deeprobust` is **not installed** (`import deeprobust` raises `ModuleNotFoundError` in the project venv) and is **never imported** anywhere in the three files. No `Metattack`, `MetaApprox`, `BaseMeta`, `inner_train`, `get_meta_grad`, or `self_training_label` symbol appears in our code. The implementation is a from-scratch re-creation that *names itself* "Meta-Self" in its docstring (L73) but does not implement the meta-learning algorithm.

---

## (b) Official algorithm's defining steps (DeepRobust `Metattack`)

Confirmed against `deeprobust/graph/global_attack/mettack.py` (fetched & indexed this session, source label `deeprobust_mettack_source`):

1. **Surrogate** = linearized/standard 2-layer GCN, symmetric-normalized `Â = D^{-1/2}(A+I)D^{-1/2}` (`utils.normalize_adj_tensor`).
2. **Self-training labels** (`self_training_label`, the *defining* unlabeled-node step): take surrogate output `argmax(1)` on the **unlabeled** nodes, overwrite train positions with true labels → `labels_self_training`.
3. **Bi-level meta-gradient** (the *defining* feature):
   - `inner_train` unrolls **`train_iters` SGD steps** over the surrogate weights, each via
     `weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)` with **momentum velocities** `w_velocities = momentum*v + g`. The *entire training trajectory* is retained in the autograd graph.
   - `get_meta_grad` then differentiates the **attacker loss** — `loss = lambda_*loss_labeled + (1-lambda_)*loss_unlabeled`, where `loss_unlabeled = NLL(output[idx_unlabeled], labels_self_training)` — **back through that unrolled trajectory** to obtain `adj_grad = ∂L_atk / ∂A` (a *meta*-gradient: gradient of post-training loss w.r.t. a structural hyperparameter).
4. **Score** (`get_adj_score`): `adj_meta_grad = adj_grad * (-2*modified_adj + 1)` (sign-flip so connected pairs score *removal*), then singleton filter + log-likelihood-ratio cutoff (`filter_potential_singletons`, `ll_constraint`).
5. **Greedy flip with re-computation** (the *defining* loop): in `attack(...)` repeat **`n_perturbations`** times — `inner_train` (re-train through the *current* perturbed graph) → meta-grad → pick the single `argmax` edge → flip it in `adj_changes` → continue. Budget is typically **5–25 % of |E|** (paper uses ptb_rate sweeps), **symmetric** add-or-remove (`undirected=True` mirrors `adj_changes`; flips can both add and delete edges).
   - `MetaApprox` is the cheaper variant: accumulates an approximate meta-gradient during training instead of full unrolling — still differentiates *through* the loop, still greedy-per-step.

---

## (c) Our implementation's steps (`mettack_edge_scores`, L62-115)

1. Build `SurrogateGCN` (L45-55): `forward = Â @ W2 @ relu(Â @ W1 @ X)`. **The `A_hat` passed in is the precomputed symmetric-normalized matrix** — *however* the gradient is taken w.r.t. that **already-normalized `A_sub`**, not w.r.t. raw `A` through the normalization (see Gap M2).
2. Train surrogate **once** for `train_epochs=100` on `cross_entropy(logits, pseudo_labels)` (L85-91). `pseudo_labels` = `IGNN.head` argmax on **all** subgraph nodes (`exp_mettack_10seed.py` L94, `mettack_comparison.py` L196) — i.e. there is **no train/unlabeled split**; every node is treated as a supervised target.
3. **Single backward pass on the final weights** (L94-99): `A_diff = A_sub.detach().requires_grad_(True)`; `loss_atk = -cross_entropy(logits, pseudo_labels)`; `loss_atk.backward()`. **No `create_graph=True`, no unrolled SGD, no momentum, no `autograd.grad` chained through training.** `A_diff` is detached from the trained weights, so this is `∂L / ∂A` **at fixed θ\*** — a plain static surrogate-input gradient, mathematically *not* a meta-gradient.
4. Score (L101-114): `grad = -(A_diff.grad)`, zero diagonal, symmetrize `(grad+grad.T)/2`, keep only entries where `A_sub[i,j] != 0` → score existing edges for **removal only**. Sort once.
5. Budget application (`exp_mettack_10seed.py` L127-129, `mettack_comparison.py` L264-271): `mettack_edges[:k]` for `k = 1..5`. **One-shot top-k slice of a single static ranking — no re-computation, no re-training, no greedy re-selection after each flip.**
6. `evaluate_attack` (L122-145): only ever **deletes** edges (`A[i,j]=A[j,i]=0`), then reconverges the IGNN. Add-edge perturbations are impossible on either side.
7. `mettack_sign_test` (`R2_03` L173-185): `n_total=150`, `n_wins=149` are **hard-coded integer literals**, not read from any results CSV (every *other* table in that file reads a CSV via `reanalyse_csv`). The `p<10^-43` is `binomtest(149,150,0.5,'greater')` on those literals.

---

## (d) GAPS

| # | Gap | Severity | File:line | Fix |
|---|-----|----------|-----------|-----|
| C1 | **No meta-gradient / no bi-level differentiation.** Single `loss_atk.backward()` on *fixed* trained weights with `A_diff` detached from θ — this is `∂L/∂A` at fixed θ\*, i.e. a plain static surrogate-gradient edge score (the same class as a one-shot FGSM-on-structure). The *defining* Metattack mechanism — `autograd.grad(..., create_graph=True)` through `train_iters` momentum-SGD steps, then differentiating the post-training loss — is absent. This is **exactly the failure mode of the old GR-BCD/PR-BCD audits**: a gradient sweep mislabeled as the named meta-attack. | **Critical** | `mettack_comparison.py:84-99` | Call DeepRobust `Metattack(model=surrogate, nnodes, attack_structure=True, undirected=True)` `.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations)`; or re-implement `inner_train` with `create_graph=True` + momentum velocities and meta-grad back through it. |
| C2 | **No greedy re-computation.** Single static ranking, then `mettack_edges[:k]`. Official attack re-runs `inner_train`+meta-grad and flips the single argmax edge **once per perturbation** through the evolving graph. Top-k slicing a one-shot gradient is a documented *weaker* attack (gradients decorrelate from true damage after the first flip). | **Critical** | `exp_mettack_10seed.py:127-129`; `mettack_comparison.py:264-271` | Loop `for step in range(k): recompute meta-grad on current modified_adj; flip argmax edge`. |
| C3 | **No self-training objective.** Official loss is `λ·loss_labeled + (1−λ)·NLL(output[unlabeled], self_training_labels)` with a real train/unlabeled split. Ours uses `−CE(logits, pseudo_labels)` over **all** nodes with no split and no labeled/unlabeled weighting; `λ`, `idx_unlabeled`, and `self_training_label` do not exist. Docstring calls this "Meta-Self," which it is not. | **Major** | `mettack_comparison.py:84-99`, `:194-196` | Introduce `idx_train`/`idx_unlabeled`, self-training labels, and the two-term `lambda_`-weighted attacker loss (Eq. 10). |
| C4 | **149/150 and p<10^-43 are hard-coded literals**, not derived from the experiment's per-cell win counts (contrast: every other table in `R2_03` is recomputed from a CSV). The headline stat is therefore unverifiable from data and could silently diverge from `greedy_topk_attack.csv` / the 10-seed run. | **Major** | `R2_03_stats_reanalysis.py:175-176` | Read the 150 per-(seed,dataset,k) IFT-vs-Mettack comparisons from the results CSV and *count* wins; binomtest on the counted values. |
| M1 | **Removal-only, both sides.** `evaluate_attack` can only delete edges, and the ranking keeps only existing edges. Real Metattack flips symmetric entries and **adds** edges (often its strongest move). The comparison silently strips Mettack's edge-addition capability. | Major | `mettack_comparison.py:107-112, 132-134` | Allow both add and remove (score all node pairs; let `evaluate_attack` set 0→1 and 1→0). |
| M2 | **Gradient taken w.r.t. pre-normalized `Â`, not raw `A`.** `A_diff` is the symmetric-normalized matrix; official meta-grad flows through `normalize_adj_tensor`. Differentiating the post-normalization matrix mis-weights high-degree node pairs and changes the sign-flip score semantics (`-2a+1` assumes raw 0/1 entries, but `A_sub` entries are normalized weights, not 0/1). | Major | `mettack_comparison.py:53-55, 95, 110-111` | Make raw `A` the leaf; normalize inside `forward`; apply `(-2·A_raw+1)` on raw entries. |
| M3 | **Surrogate trained on the *victim's* (IGNN) pseudo-labels** rather than a true black-box surrogate. Couples attacker to victim predictions; Metattack's threat model assumes no victim output access (uses its own surrogate's self-training labels). Mildly *over*-strengthens transfer but is still off-spec. | Minor | `exp_mettack_10seed.py:94`; `mettack_comparison.py:196` | Train surrogate to convergence and self-train from *its own* output. |
| F1 | **Budget regime is cherry-picked against the opponent.** `k ∈ {1..5}` edge *deletions* on a ~50-node subgraph is a tiny-budget regime where a one-shot static gradient is weakest and a single SVD direction is strongest. Standard Metattack budget is 5–25 % of |E| with greedy re-optimization; the paper never reports Mettack at its native budget. The "early-warning" label reframes the one regime where AEGIS wins as the headline. | Major (fairness) | `experiments.tex:54`; `exp_mettack_10seed.py` MAX_K=5 | Report a budget sweep to ptb_rate ≈0.25 with the *faithful* greedy attack; keep early-warning as a secondary panel, not the headline number. |

---

## (e) VERDICT

**UNFAITHFUL.** The implementation is a hand-rolled one-shot static surrogate-gradient edge score (single `backward()` on fixed weights, no bi-level differentiation, no greedy re-computation, no self-training objective) — it is a plain gradient attack mislabeled "Metattack," the same class of error as the previously-caught GR-BCD/PR-BCD baselines, so the head-to-head does not test the cited method.

## (f) Paper number at risk

`experiments.tex:54` — AEGIS "inflicts **3–10× more equilibrium damage than Mettack** … (**149/150 wins, p<10^-43**)" in the early-warning regime k∈{1..5}. Both the 3–10× multiplier and the 149/150 / p<10^-43 sign-test rest on this unfaithful opponent; additionally the 149/150 figure is a hard-coded literal (R2_03:175-176), not recomputed from data. The abstract/intro one-liner ("matches 50-step PGD … beats Mettack") inherits the same risk.

### Recommended remediation order
1. Drop in DeepRobust `Metattack` (greedy, meta-grad, self-training, add+remove) as the opponent → fixes C1–C3, M1–M3 at once.
2. Recompute 149/150 and the p-value from the actual per-cell CSV → fixes C4.
3. Add a budget sweep; demote "early-warning k≤5" from headline to a labeled regime → fixes F1.
