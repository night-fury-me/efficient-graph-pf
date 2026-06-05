# Baseline-implementation faithfulness audit — rollup

Date: 2026-06-05. Protocol: `feedback_verify_baselines_vs_official` — verify every competing-method
impl against the official code/paper before trusting results. One subagent per baseline; per-method
reports in this directory. GR-BCD / PR-BCD were already verified (use the official PyG
`GRBCDAttack`/`PRBCDAttack` in `scripts/exp3_sota_attack_sweep.py`) and are **FAITHFUL**.

Verdict scale: 🟥 UNFAITHFUL · 🟧 MISLABELED / WEAK (impl ≠ its name, or strawman) · 🟨 MINOR / PROVENANCE · 🟩 FAITHFUL.

| # | Baseline | Verdict | Core problem | Paper number at risk |
|---|----------|---------|--------------|----------------------|
| 1 | **Mettack** | 🟥 UNFAITHFUL | Hand-rolled **one-shot static surrogate-gradient**, not Metattack: no meta-gradient through training, no greedy re-rank, no self-training. **AND `149/150` is hard-coded** (`R2_03_stats_reanalysis.py:175`), p<10⁻⁴³ is a binomtest on literals. | `experiments.tex:54` "3–10× more than Mettack (149/150, p<10⁻⁴³)" + abstract/intro one-liner |
| 2 | **PGD — "IFT-gradient"** | 🟥 UNFAITHFUL (label) | Claims IFT/implicit gradient `(I−J_z)⁻¹`; actually **plain backprop through a truncated 50-step solve**. Numbers likely close (κ≤0.9) but the claim is false. | `experiments.tex:39` caption; IFT-PGD "72–92%" |
| 3 | **PGD — Cls-PGD** | 🟨 MINOR (under-powered) | `grad.sign()` L∞ step on an L2/Frobenius ball; spurious clamp; **no random restarts**; inner solve 50 iters vs victim's 100. All bias toward AEGIS. | "74–156×" (`experiments.tex:35`, abstract); Cls-PGD "15–70% / 1.5–3.1×" |
| 4 | PGD — Critical-PGD (reachability) | 🟩 FAITHFUL | Proper tangent-space projected ascent, multi-LR, restarts. | — |
| 5 | **Greedy (label-aware)** | 🟧 MISLABELED | A **correct, strong sequential oracle** — but **NOT label-aware**: it optimizes the *label-free* equilibrium shift, same signal as AEGIS. "Greedy uses labels / AEGIS has no label access" is fictional (neither uses labels). Quantitative 54–67% survives. | `experiments.tex:30` caption ("label-aware", "no label access") |
| 6 | **512-query black-box** | 🟧 WEAK (strawman) | Plain **i.i.d. uniform random search**, 512 tries, zero adaptive feedback — not NES/SimBA/Square. 44% is a high-dim artifact (uniform sphere vs rank-1 optimum), measures weakness not query-hardness. Confirms R2 §2a. | `experiments.tex:35` "44±4%" (the 99%/cos=0.99 transfer is **safe**) |
| 7 | **RobustGCN-lite** | 🟧 MISLEADING-LABEL | The **entire defining mechanism is stripped**: σ branch is dead code (`R2_10:104` "unused"), no variance-attention, no KL reg → just a spectral-capped GCN. Not disclosed. | `F_experiments.tex:323` "match or exceed IGNN" |
| 8 | **GNNGuard-lite** | 🟧 MISLEADING-LABEL | Cosine kept only as a **binary prune mask** (no reweight, no memory); `sim>0.1` on ReLU embeddings prunes ~nothing → inert. Not disclosed. | `F_experiments.tex:323` |
| 9 | **AGNNCert** | 🟥 UNFAITHFUL-PROXY (positioning OK) | Our `agnncert_radii` is a per-node greedy single-edge probe (constant √2≈1.414); implements **none** of AGNNCert's division/voting/margin. BUT paper only positions it on its **own published** capabilities (no numeric head-to-head) → defensible. Also: bib `@li2025agnncert` **wrong author/venue** (correct: Li & Wang, USENIX'25) and mislabeled "IBP". | `tab:threat_model` + radar (positioning safe); bib + "IBP" label are real errors |
| 10 | **Randomized smoothing** | 🟨 MINOR (provenance) | Cert itself is a **faithful Cohen-2019** impl (correct CP bound, radius, abstention; frob "Cert 0.00" is real). BUT table per-variant rows + Wall@10⁴ (linear extrapolation) + **"11,700–57,000×"** are **not reproducible from the committed CSV**. | `tab:smoothing` rows; "11,700–57,000×" |
| 11 | Degree / edge-betweenness / spectral | 🟩 FAITHFUL | Standard centralities (official `nx.edge_betweenness_centrality`; leading-eigenvector spectral). "+6–148%" claim NOT at risk. | — |
| 12 | **Random** (AtkAdv denominator) | 🟨 MINOR | Budget/normalization fair, but **single draw per seed** (vs the repo's own 5-shuffle greedy script). Published AtkAdv (3.6/4.1/3.8) appears to come from the 5-shuffle pipeline — **confirm provenance**. | tab:cross_domain / tab:explicit AtkAdv (likely safe) |
| — | GR-BCD / PR-BCD | 🟩 FAITHFUL | Official PyG `GRBCDAttack`/`PRBCDAttack`. (already verified) | — |

## The pattern
Most failures are **label mismatches** — the implementation does not do what its name/caption claims:
Mettack (≠ meta-learning), "IFT-gradient" PGD (≠ implicit gradient), "label-aware" Greedy (≠ label-aware),
RobustGCN/GNNGuard-lite (≠ their defenses), AGNNCert proxy (≠ division-voting). This is exactly the
failure class the protocol exists to catch (cf. the old GR-BCD/PR-BCD).

## Remediation priority
1. **🟥 Mettack — fix before submission.** Swap in DeepRobust `Metattack` (meta-grad + greedy + self-training + add/remove); **recompute 149/150 from the actual paired CSV** (the hard-coded literal is the most urgent integrity item).
2. **🟧 Wording fixes (cheap, do now):** rename "IFT-gradient PGD" → e.g. "unrolled-gradient PGD" or implement the true resolvent gradient (it already exists at `iem/adversarial.py:170`); fix Greedy "label-aware/no label access" → "shift-oracle"; add 2 disclosure sentences for RobustGCN/GNNGuard-lite (or implement the real mechanisms via the official code, which exists); fix the AGNNCert bib entry + drop the "IBP" label; never print the 1.414 as an AGNNCert number.
3. **🟨 Strengthen / reproduce:** add NES or SimBA at 512 queries for the black-box contrast; give Cls-PGD `grad/‖grad‖` steps + random restarts + 100-iter inner solve and re-run "74–156×"; re-run the smoothing script at true M=10⁴, commit the CSV, regenerate `tab:smoothing` + the multiplier; add 5-draw averaging to Random (or confirm the published AtkAdv already used it).

Per-baseline detail: `mettack.md`, `pgd.md`, `structural_rankings.md`, `greedy.md`, `blackbox_query.md`, `randomized_smoothing.md`, `agnncert.md`, `robust_backbones.md`.
