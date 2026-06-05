# Claims Ledger — Appendix F (Detailed Experimental Results)

Source file: `paper/sections/appendix/F_experiments.tex` (376 lines)
Scope: every number in the DETAILED EXPERIMENTS appendix. Ground-truth for the body's experimental claims.
HARD RULE reminder: every result must use the 10 preferred seeds `[42,137,271,314,1729,2718,3141,5772,6561,9999]`. Flag any result not at 10 seeds.

## Ledger (one row per number / cell)

| id | quantity_key | value | setting | n_seeds | location | exact_quote |
|----|--------------|-------|---------|---------|-----------|-------------|
| F001 | n_seeds | 10 | global; all appendix results | 10 | F_experiments.tex:15 | "All results use the 10 fixed seeds of \cref{app:repro}." |
| F002 | transfer_cos | 0.99 | transfer surrogate recovers one-query damage; tab:attack_full quadrant | 10 | F_experiments.tex:22-23 | "recovers $99\%$ of the one-query damage ($\cos{=}0.99$)" |
| F003 | blackbox_recovery | 44±4% | 512-query black-box search vs one-query damage | 10 | F_experiments.tex:23 | "versus $44{\pm}4\%$ for a $512$-query black-box search" |
| F004 | query_count | 512 | black-box search budget | 10 | F_experiments.tex:23 | "a $512$-query black-box search" |
| F005 | damage_vs_pgd_per_query | 74–156× | AEGIS per-query damage vs 50-step PGD | 10 | F_experiments.tex:24-25 | "it delivers $74$--$156\times$ the damage of a $50$-step PGD" |
| F006 | pgd_steps | 50 | PGD step count | 10 | F_experiments.tex:24-25 | "a $50$-step PGD" |
| F007 | cls_pgd_deficit | 15–70% less | classification-loss PGD damage deficit at 50x cost | 10 | F_experiments.tex:25-26 | "classification-loss PGD inflicts $15$--$70\%$ less at $50\times$ the cost" |
| F008 | cls_pgd_cost_factor | 50× | cost multiplier of Cls-PGD | 10 | F_experiments.tex:26 | "$15$--$70\%$ less at $50\times$ the cost" |
| F009 | ift_pgd_recovery | 72–92% | IFT-gradient PGD (solver validation) recovery | 10 | F_experiments.tex:26-27 | "IFT-gradient PGD (solver validation, not an independent baseline) $72$--$92\%$" |
| F010 | prediction_flips | 0–1.8% | prediction flips for all methods | 10 | F_experiments.tex:27 | "prediction flips stay $0$--$1.8\%$ for all methods" |
| F011 | damage_dZstar_epsilon | 0.10 | ε for equilibrium damage fig (fig:attack_comparison) | 10 | F_experiments.tex:33 | "Equilibrium damage $\norm{\Delta Z^\star}_2$ at $\varepsilon{=}0.10$ across 10 seeds" |
| F012 | aegis_vs_clspgd_fig | 1.5–3.1× | one AEGIS query vs 50-step Cls-PGD damage (fig) | 10 | F_experiments.tex:36 | "one analytic \AEGIS query inflicts $1.5$--$3.1\times$ the damage of a $50$-step Cls-PGD run per dataset" |
| F013 | smoothing_base_radius_rs | 0.123 | randomized smoothing base radius at σ=0.05 | 10 | F_experiments.tex:42 | "constant per-node base radii ($0.123$ vs.\ \AEGIS's $0.078$ at $\sigma{=}0.05$)" |
| F014 | smoothing_base_radius_aegis | 0.078 | AEGIS radius at σ=0.05 | 10 | F_experiments.tex:42 | "vs.\ \AEGIS's $0.078$ at $\sigma{=}0.05$" |
| F015 | smoothing_sigma | 0.05 | σ for radius comparison | 10 | F_experiments.tex:42 | "at $\sigma{=}0.05$" |
| F016 | localized_smoothing_samples | ~10^5 | MC samples per node for localized smoothing | n/a | F_experiments.tex:43 | "localized smoothing~\cite{schuchardt2023localized} needs ${\sim}10^5$ MC samples per node" |
| F017 | aegis_radius_dense | r_v≈0.10 | AEGIS radius in dense regions | 10 | F_experiments.tex:44 | "$r_v{\approx}0.10$ in dense regions" |
| F018 | aegis_radius_boundary | 0.01 | AEGIS radius at boundaries | 10 | F_experiments.tex:44-45 | "$0.01$ at boundaries" |
| F019 | cls_grad_tau_cora | -0.20±0.05 | per-edge cls-gradient vs ground-truth corr, Cora | 10 | F_experiments.tex:48 | "anti-correlate ... on Cora ($\tau{=}\ms{-0.20}{0.05}$" |
| F020 | aegis_tau_cora | +0.32±0.04 | AEGIS per-edge corr vs ground-truth, Cora | 10 | F_experiments.tex:48 | "vs.\ \AEGIS's $\ms{+0.32}{0.04}$" |
| F021 | aegis_p10_advantage | 2–6× | AEGIS higher P@10 over cls-gradient | 10 | F_experiments.tex:49 | "\AEGIS achieves $2$--$6\times$ higher P@10" |
| F022 | finite_diff_tau_Sc | 0.999 | finite-diff reproduces S_c column-norm ranking | 10 | F_experiments.tex:50-51 | "reproduce the $S_c$ column-norm ranking ($\tau{=}0.999$)" |
| F023 | static_greedy_gap | ≤3 pp | iterative S_c recomputation closes ≤3pp of static-vs-greedy gap | 10 | F_experiments.tex:51-52 | "iterative $S_c$ recomputation closes $\le3$\,pp of the static-versus-greedy gap" |
| F024 | grbcd_epochs | 125 | GR-BCD/PR-BCD optimizer epochs | 10 | F_experiments.tex:61-62 | "what their $125$-epoch optimizer converges to" |
| F025 | n_datasets | 6 | datasets in budget sweep | 10 | F_experiments.tex:63 | "across all six datasets and the full budget sweep" |
| F026 | epsilon_budget | k∈{1,...,50} | full budget sweep range | 10 | F_experiments.tex:63 | "the full budget sweep $k\!\in\!\{1,\ldots,50\}$" |
| F027 | damage_match_pct | ~8% | AEGIS one-query matches attacker damage to within ~8% | 10 | F_experiments.tex:64 | "matches the attacker's equilibrium damage to within ${\sim}8\%$ and frequently exceeds it" |
| F028 | kappa_band / rank_agreement_grbcd | τ=0.35–0.81, mean≈0.5 | rank agreement range across datasets (vs GR-BCD) | 10 | F_experiments.tex:67-68 | "Rank agreement is moderate and dataset-dependent ($\tau{=}0.35$--$0.81$, mean ${\approx}0.5$)" |
| F029 | citeseer_edgeweight_tau | τ≈0 | Citeseer edge-weight-only ranking | 10 | F_experiments.tex:70 | "Citeseer's edge-weight ranking alone gives $\tau{\approx}0$" |
| F030 | atkadv_vs_structural | +6–148% | AEGIS AtkAdv over degree/betweenness/spectral at ε=0.01 | 10 | F_experiments.tex:75-76 | "beats degree-proportional, edge-betweenness, and spectral rankings at $\varepsilon{=}0.01$ ($+6$--$148\%$ AtkAdv" |
| F031 | atkadv_epsilon | 0.01 | ε for structural-baseline comparison | 10 | F_experiments.tex:75-76 | "at $\varepsilon{=}0.01$" |
| F032 | wilcoxon_p_structural | p<0.001 | Wilcoxon p for structural-baseline beat | 10 | F_experiments.tex:76 | "Wilcoxon $p{<}0.001$" |
| F033 | mettack_damage_ratio | 3–10× | AEGIS vs Mettack equilibrium damage, early-warning | 10 | F_experiments.tex:76-77 | "inflicts $3$--$10\times$ more equilibrium damage than Mettack" |
| F034 | mettack_regime | k∈{1,...,5} | early-warning regime | 10 | F_experiments.tex:77 | "the early-warning regime $k\!\in\!\{1,\ldots,5\}$" |
| F035 | mettack_paired_wins | 149/150 | paired wins vs Mettack | 10 | F_experiments.tex:78 | "($\mathbf{149/150}$ paired wins, $p{<}10^{-43}$)" |
| F036 | mettack_p | p<10^-43 | p-value for Mettack paired wins | 10 | F_experiments.tex:78 | "$p{<}10^{-43}$" |
| F037 | positioning_radar_axes | 7 | capability radar axis count | n/a | F_experiments.tex:105 | "a seven-axis capability radar" |
| F038 | rank_agreement_grbcd_range | τ=0.16–0.99 | white-box gradient attribution oracle AEGIS proxies | 10 | F_experiments.tex:108-109 | "the attribution oracle that \AEGIS only proxies ($\tau{=}0.16$--$0.99$)" |
| F039 | agnncert_subgraph_T | T=30–80 | AGNNCert deterministic subgraph evaluations | n/a | F_experiments.tex:116 | "AGNNCert ... is deterministic ($T{=}30$--$80$ subgraph evaluations)" |
| F040 | localized_smoothing_samples2 | ~10^5 samples/node | localized smoothing collective certificate cost | n/a | F_experiments.tex:116 | "localized smoothing ... at ${\sim}10^5$ samples/node" |
| F041 | grbcd_epochs_fig | 125 | GR-BCD/PR-BCD epochs (fig caption) | 10 | F_experiments.tex:122-123 | "against faithful $125$-epoch GR-BCD and PR-BCD attackers" |
| **tab:baselines (AEGIS one query vs GR-BCD/PR-BCD, 10 seeds)** | | | | | F_experiments.tex:129-153 | caption ln131-135 |
| F042 | edges_cora | 61 | Cora |E| (subgraph) | 10 | F_experiments.tex:145 | "Cora & $61$ & $+0.44$ & $1.01$ & $0.99$ & $1.01$ & $0.99$" |
| F043 | rank_agreement_grbcd_cora | +0.44 | Cora τ (AEGIS A_ij·v_ij vs GR-BCD selection) | 10 | F_experiments.tex:145 | "Cora & $61$ & $+0.44$ ..." |
| F044 | damage_ratio_grbcd_cora_k1 | 1.01 | Cora AEGIS/GR-BCD damage ratio at k=1 | 10 | F_experiments.tex:145 | "... $1.01$ & $0.99$ & $1.01$ & $0.99$" |
| F045 | damage_ratio_grbcd_cora_k50 | 0.99 | Cora AEGIS/GR-BCD damage ratio at k=50 | 10 | F_experiments.tex:145 | "... $1.01$ & $0.99$ ..." |
| F046 | damage_ratio_prbcd_cora_k1 | 1.01 | Cora AEGIS/PR-BCD damage ratio at k=1 | 10 | F_experiments.tex:145 | "... $1.01$ & $0.99$" |
| F047 | damage_ratio_prbcd_cora_k50 | 0.99 | Cora AEGIS/PR-BCD damage ratio at k=50 | 10 | F_experiments.tex:145 | "... $1.01$ & $0.99$" |
| F048 | edges_citeseer | 73 | Citeseer |E| (subgraph) | 10 | F_experiments.tex:146 | "Citeseer & $73$ & $+0.37$ ..." |
| F049 | rank_agreement_grbcd_citeseer | +0.37 | Citeseer τ | 10 | F_experiments.tex:146 | "Citeseer & $73$ & $+0.37$ ..." |
| F050 | damage_ratio_grbcd_citeseer_k1 | 1.00 | Citeseer AEGIS/GR-BCD k=1 | 10 | F_experiments.tex:146 | "... $1.00$ & $0.97$ & $1.00$ & $0.96$" |
| F051 | damage_ratio_grbcd_citeseer_k50 | 0.97 | Citeseer AEGIS/GR-BCD k=50 | 10 | F_experiments.tex:146 | "... $1.00$ & $0.97$ ..." |
| F052 | damage_ratio_prbcd_citeseer_k1 | 1.00 | Citeseer AEGIS/PR-BCD k=1 | 10 | F_experiments.tex:146 | "... $1.00$ & $0.96$" |
| F053 | damage_ratio_prbcd_citeseer_k50 | 0.96 | Citeseer AEGIS/PR-BCD k=50 | 10 | F_experiments.tex:146 | "... $1.00$ & $0.96$" |
| F054 | edges_pubmed | 53 | Pubmed |E| (subgraph) | 10 | F_experiments.tex:147 | "Pubmed & $53$ & $+0.81$ ..." |
| F055 | rank_agreement_grbcd_pubmed | +0.81 | Pubmed τ | 10 | F_experiments.tex:147 | "Pubmed & $53$ & $+0.81$ ..." |
| F056 | damage_ratio_grbcd_pubmed_k1 | 1.00 | Pubmed AEGIS/GR-BCD k=1 | 10 | F_experiments.tex:147 | "... $1.00$ & $0.99$ & $1.00$ & $0.99$" |
| F057 | damage_ratio_grbcd_pubmed_k50 | 0.99 | Pubmed AEGIS/GR-BCD k=50 | 10 | F_experiments.tex:147 | "... $1.00$ & $0.99$ ..." |
| F058 | damage_ratio_prbcd_pubmed_k1 | 1.00 | Pubmed AEGIS/PR-BCD k=1 | 10 | F_experiments.tex:147 | "... $1.00$ & $0.99$" |
| F059 | damage_ratio_prbcd_pubmed_k50 | 0.99 | Pubmed AEGIS/PR-BCD k=50 | 10 | F_experiments.tex:147 | "... $1.00$ & $0.99$" |
| F060 | edges_wikics | 55 | WikiCS |E| (subgraph) | 10 | F_experiments.tex:148 | "WikiCS & $55$ & $+0.35$ ..." |
| F061 | rank_agreement_grbcd_wikics | +0.35 | WikiCS τ | 10 | F_experiments.tex:148 | "WikiCS & $55$ & $+0.35$ ..." |
| F062 | damage_ratio_grbcd_wikics_k1 | 1.33 | WikiCS AEGIS/GR-BCD k=1 (AEGIS exceeds attacker) | 10 | F_experiments.tex:148 | "... $1.33$ & $1.00$ & $1.33$ & $0.99$" |
| F063 | damage_ratio_grbcd_wikics_k50 | 1.00 | WikiCS AEGIS/GR-BCD k=50 | 10 | F_experiments.tex:148 | "... $1.33$ & $1.00$ ..." |
| F064 | damage_ratio_prbcd_wikics_k1 | 1.33 | WikiCS AEGIS/PR-BCD k=1 | 10 | F_experiments.tex:148 | "... $1.33$ & $0.99$" |
| F065 | damage_ratio_prbcd_wikics_k50 | 0.99 | WikiCS AEGIS/PR-BCD k=50 | 10 | F_experiments.tex:148 | "... $1.33$ & $0.99$" |
| F066 | edges_amazon | 85 | Amazon |E| (subgraph) | 10 | F_experiments.tex:149 | "Amazon & $85$ & $+0.60$ ..." |
| F067 | rank_agreement_grbcd_amazon | +0.60 | Amazon τ | 10 | F_experiments.tex:149 | "Amazon & $85$ & $+0.60$ ..." |
| F068 | damage_ratio_grbcd_amazon_k1 | 1.00 | Amazon AEGIS/GR-BCD k=1 | 10 | F_experiments.tex:149 | "... $1.00$ & $0.99$ & $1.00$ & $0.99$" |
| F069 | damage_ratio_grbcd_amazon_k50 | 0.99 | Amazon AEGIS/GR-BCD k=50 | 10 | F_experiments.tex:149 | "... $1.00$ & $0.99$ ..." |
| F070 | damage_ratio_prbcd_amazon_k1 | 1.00 | Amazon AEGIS/PR-BCD k=1 | 10 | F_experiments.tex:149 | "... $1.00$ & $0.99$" |
| F071 | damage_ratio_prbcd_amazon_k50 | 0.99 | Amazon AEGIS/PR-BCD k=50 | 10 | F_experiments.tex:149 | "... $1.00$ & $0.99$" |
| F072 | edges_amazonfraud | 78 | Amazon Fraud |E| (subgraph) | 10 | F_experiments.tex:150 | "Amazon Fraud & $78$ & $+0.46$ ..." |
| F073 | rank_agreement_grbcd_amazonfraud | +0.46 | Amazon Fraud τ | 10 | F_experiments.tex:150 | "Amazon Fraud & $78$ & $+0.46$ ..." |
| F074 | damage_ratio_grbcd_amazonfraud_k1 | 1.00 | Amazon Fraud AEGIS/GR-BCD k=1 | 10 | F_experiments.tex:150 | "... $1.00$ & $1.01$ & $1.00$ & $0.99$" |
| F075 | damage_ratio_grbcd_amazonfraud_k50 | 1.01 | Amazon Fraud AEGIS/GR-BCD k=50 (AEGIS exceeds) | 10 | F_experiments.tex:150 | "... $1.00$ & $1.01$ ..." |
| F076 | damage_ratio_prbcd_amazonfraud_k1 | 1.00 | Amazon Fraud AEGIS/PR-BCD k=1 | 10 | F_experiments.tex:150 | "... $1.00$ & $0.99$" |
| F077 | damage_ratio_prbcd_amazonfraud_k50 | 0.99 | Amazon Fraud AEGIS/PR-BCD k=50 | 10 | F_experiments.tex:150 | "... $1.00$ & $0.99$" |
| **tab:da_decomp (edge weight vs sensitivity, rank agreement w/ GR-BCD)** | | | | | F_experiments.tex:155-178 | caption ln156-162 |
| F078 | da_tau_Av_cora | +0.44 | Cora τ_Av (full A_ij·v_ij ranking vs GR-BCD) | 10 | F_experiments.tex:170 | "Cora & $+0.44$ & $+0.29$ & $+0.56$ & $+0.14$ & $1.07$ & $1.59$" |
| F079 | da_tau_A_cora | +0.29 | Cora τ_A (edge-weight-only) | 10 | F_experiments.tex:170 | "Cora ... $+0.29$ ..." |
| F080 | da_tau_v_cora | +0.56 | Cora τ_v (sensitivity-only) | 10 | F_experiments.tex:170 | "Cora ... $+0.56$ ..." |
| F081 | da_dtau_cora | +0.14 | Cora Δτ = τ_Av − τ_A | 10 | F_experiments.tex:170 | "Cora ... $+0.14$ ..." |
| F082 | da_D_AvA_cora | 1.07 | Cora D_Av/D_A (damage of full ranking vs edge-weight) | 10 | F_experiments.tex:170 | "Cora ... $1.07$ & $1.59$" |
| F083 | da_D_Avv_cora | 1.59 | Cora D_Av/D_v | 10 | F_experiments.tex:170 | "Cora ... $1.59$" |
| F084 | da_tau_Av_citeseer | +0.37 | Citeseer τ_Av | 10 | F_experiments.tex:171 | "Citeseer & $+0.37$ & $-0.01$ & $+0.76$ & $+0.39$ & $1.15$ & $1.26$" |
| F085 | da_tau_A_citeseer | -0.01 | Citeseer τ_A (≈0, sensitivity decisive) | 10 | F_experiments.tex:171 | "Citeseer ... $-0.01$ ..." |
| F086 | da_tau_v_citeseer | +0.76 | Citeseer τ_v | 10 | F_experiments.tex:171 | "Citeseer ... $+0.76$ ..." |
| F087 | da_dtau_citeseer | +0.39 | Citeseer Δτ | 10 | F_experiments.tex:171 | "Citeseer ... $+0.39$ ..." |
| F088 | da_D_AvA_citeseer | 1.15 | Citeseer D_Av/D_A | 10 | F_experiments.tex:171 | "Citeseer ... $1.15$ & $1.26$" |
| F089 | da_D_Avv_citeseer | 1.26 | Citeseer D_Av/D_v | 10 | F_experiments.tex:171 | "Citeseer ... $1.26$" |
| F090 | da_tau_Av_pubmed | +0.81 | Pubmed τ_Av | 10 | F_experiments.tex:172 | "Pubmed & $+0.81$ & $+0.63$ & $+0.94$ & $+0.18$ & $1.45$ & $1.00$" |
| F091 | da_tau_A_pubmed | +0.63 | Pubmed τ_A | 10 | F_experiments.tex:172 | "Pubmed ... $+0.63$ ..." |
| F092 | da_tau_v_pubmed | +0.94 | Pubmed τ_v | 10 | F_experiments.tex:172 | "Pubmed ... $+0.94$ ..." |
| F093 | da_dtau_pubmed | +0.18 | Pubmed Δτ | 10 | F_experiments.tex:172 | "Pubmed ... $+0.18$ ..." |
| F094 | da_D_AvA_pubmed | 1.45 | Pubmed D_Av/D_A | 10 | F_experiments.tex:172 | "Pubmed ... $1.45$ & $1.00$" |
| F095 | da_D_Avv_pubmed | 1.00 | Pubmed D_Av/D_v | 10 | F_experiments.tex:172 | "Pubmed ... $1.00$" |
| F096 | da_tau_Av_wikics | +0.35 | WikiCS τ_Av | 10 | F_experiments.tex:173 | "WikiCS & $+0.35$ & $+0.28$ & $+0.35$ & $+0.07$ & $1.03$ & $1.11$" |
| F097 | da_tau_A_wikics | +0.28 | WikiCS τ_A | 10 | F_experiments.tex:173 | "WikiCS ... $+0.28$ ..." |
| F098 | da_tau_v_wikics | +0.35 | WikiCS τ_v | 10 | F_experiments.tex:173 | "WikiCS ... $+0.35$ ..." |
| F099 | da_dtau_wikics | +0.07 | WikiCS Δτ | 10 | F_experiments.tex:173 | "WikiCS ... $+0.07$ ..." |
| F100 | da_D_AvA_wikics | 1.03 | WikiCS D_Av/D_A | 10 | F_experiments.tex:173 | "WikiCS ... $1.03$ & $1.11$" |
| F101 | da_D_Avv_wikics | 1.11 | WikiCS D_Av/D_v | 10 | F_experiments.tex:173 | "WikiCS ... $1.11$" |
| F102 | da_tau_Av_amazon | +0.60 | Amazon τ_Av | 10 | F_experiments.tex:174 | "Amazon & $+0.60$ & $+0.41$ & $+0.59$ & $+0.19$ & $1.05$ & $1.82$" |
| F103 | da_tau_A_amazon | +0.41 | Amazon τ_A | 10 | F_experiments.tex:174 | "Amazon ... $+0.41$ ..." |
| F104 | da_tau_v_amazon | +0.59 | Amazon τ_v | 10 | F_experiments.tex:174 | "Amazon ... $+0.59$ ..." |
| F105 | da_dtau_amazon | +0.19 | Amazon Δτ | 10 | F_experiments.tex:174 | "Amazon ... $+0.19$ ..." |
| F106 | da_D_AvA_amazon | 1.05 | Amazon D_Av/D_A | 10 | F_experiments.tex:174 | "Amazon ... $1.05$ & $1.82$" |
| F107 | da_D_Avv_amazon | 1.82 | Amazon D_Av/D_v | 10 | F_experiments.tex:174 | "Amazon ... $1.82$" |
| F108 | da_tau_Av_amazonfraud | +0.46 | Amazon Fraud τ_Av | 10 | F_experiments.tex:175 | "Amazon Fraud & $+0.46$ & $+0.19$ & $+0.24$ & $+0.27$ & $1.77$ & $1.80$" |
| F109 | da_tau_A_amazonfraud | +0.19 | Amazon Fraud τ_A | 10 | F_experiments.tex:175 | "Amazon Fraud ... $+0.19$ ..." |
| F110 | da_tau_v_amazonfraud | +0.24 | Amazon Fraud τ_v | 10 | F_experiments.tex:175 | "Amazon Fraud ... $+0.24$ ..." |
| F111 | da_dtau_amazonfraud | +0.27 | Amazon Fraud Δτ (vs GR-BCD; brute-force reaches +0.90) | 10 | F_experiments.tex:175 | "Amazon Fraud ... $+0.27$ ..." |
| F112 | da_D_AvA_amazonfraud | 1.77 | Amazon Fraud D_Av/D_A | 10 | F_experiments.tex:175 | "Amazon Fraud ... $1.77$ & $1.80$" |
| F113 | da_D_Avv_amazonfraud | 1.80 | Amazon Fraud D_Av/D_v | 10 | F_experiments.tex:175 | "Amazon Fraud ... $1.80$" |
| F114 | brute_force_overlap_dtau_max | +0.90 | Δτ vs brute-force single-edge removal (Amazon Fraud), fig:tau_heatmap | 10 | F_experiments.tex:162 | "where the Amazon\,Fraud increment reaches $+0.90$" |
| F115 | smoothing_matched_sigma | σ=ε/√(2|E|) | matched Frobenius-ball smoothing scale | 10 | F_experiments.tex:182 | "The matched smoothing $\sigma{=}\varepsilon/\sqrt{2|E|}$" |
| F116 | smoothing_cert_fraction | 0.77–0.96 | per-coordinate ball (σ=ε) certified node fraction | 10 | F_experiments.tex:184 | "does it certify ($0.77$--$0.96$ of nodes)" |
| F117 | smoothing_speedup_percoord | 23,000–57,000× | Sc bound speedup vs per-coordinate smoothing | 10 | F_experiments.tex:184-185 | "at $\mathbf{23{,}000}$--$\mathbf{57{,}000}\times$ the wall-clock of the zero-sample $S_c$ bound" |
| F118 | smoothing_speedup_matched | 11,700–16,700× | Sc bound speedup vs matched-ball smoothing | 10 | F_experiments.tex:185-186 | "$\mathbf{11{,}700}$--$\mathbf{16{,}700}\times$ faster" |
| F119 | kappa_band / kappa_sweep_range | [0.30, 0.99] | κ_max sweep on Cora (fig:phase_transition) | 10 | F_experiments.tex:192 | "sweeps $\kappa_{\max}{\in}[0.30,0.99]$ on Cora (10 seeds)" |
| F120 | kappa_cap_max | 0.99 | spectral-normalization cap pushed to | 10 | F_experiments.tex:194 | "pushed to $0.99$" |
| F121 | ecrit_drop | 230× | theoretical ε_crit driven down at κ=0.99 | 10 | F_experiments.tex:194-195 | "driving the theoretical $\ecrit$ down $230\times$" |
| F122 | rho_Jz_bound | ρ(J_z)≤0.42 | trained ReLU pattern spectral radius bound | 10 | F_experiments.tex:195 | "the trained ReLU pattern keeps $\rho(J_z)\le0.42$" |
| F123 | resolvent_growth | 1.17→1.80 | resolvent growth across κ sweep | 10 | F_experiments.tex:195-196 | "the resolvent grows only $\mathbf{1.17{\to}1.80}$" |
| F124 | spectral_margin | 2–4× | spectral-radius margin to criticality (ρ=1) | 10 | F_experiments.tex:196-197 | "This is a $\mathbf{2}$--$\mathbf{4\times}$ spectral-radius margin to criticality ($\rho{=}1$)" |
| F125 | fixedpoint_budget | 50-step | matrix-free fixed-point budget | 10 | F_experiments.tex:197-198 | "reaches its 50-step fixed-point budget at $\kappa_{\max}{\ge}0.85$" |
| F126 | fixedpoint_kappa_threshold | κ_max≥0.85 | κ where 50-step budget is reached | 10 | F_experiments.tex:198 | "at $\kappa_{\max}{\ge}0.85$" |
| F127 | memory / dense_mem_cap | 24 GB | dense path exceeds beyond N=200 (fig:scalability) | n/a | F_experiments.tex:213-214 | "The dense path scales as $O((Nd)^3)$, exceeding $24$\,GB beyond $N{=}200$" |
| F128 | dense_scaling | O((Nd)^3) | dense path time scaling | n/a | F_experiments.tex:213 | "The dense path scales as $O((Nd)^3)$" |
| F129 | dense_mem_N | N=200 | dense path memory cap threshold | n/a | F_experiments.tex:214 | "exceeding $24$\,GB beyond $N{=}200$" |
| F130 | matrixfree_scaling | O(N log N) | matrix-free time scaling | n/a | F_experiments.tex:214 | "the matrix-free path is roughly $O(N\log N)$ in time and sub-linear in memory" |
| F131 | amazonphoto_N | 7,650 | Amazon Photo node count | n/a | F_experiments.tex:215 | "reaching Amazon Photo ($N{=}7{,}650$)" |
| F132 | runtime_matrixfree | 365 s | matrix-free runtime at Amazon Photo | n/a | F_experiments.tex:215 | "at $365$\,s and $5.5$\,GB" |
| F133 | memory_matrixfree | 5.5 GB | matrix-free memory at Amazon Photo | n/a | F_experiments.tex:215 | "$365$\,s and $5.5$\,GB" |
| F134 | sigma1_agreement | 0.03% | σ1 agreement dense vs matrix-free at N=200 | n/a | F_experiments.tex:222-223 | "$\sigma_1$ agrees within $0.03\%$ at $N{=}200$" |
| F135 | matrixfree_tau | τ=0.999 | per-edge agreement dense vs matrix-free at N=200 | n/a | F_experiments.tex:223 | "(per-edge $\tau{=}0.999$)" |
| F136 | neumann_residual | κ^200∈[10^-105,10^-48] | analytic truncation residual across suite | n/a | F_experiments.tex:223-224 | "the analytic truncation residual $\kappa^{200}\in[10^{-105},10^{-48}]$ across the suite" |
| F137 | pubmed_N | 19,717 | Pubmed node count; exceeds 24GB dense | n/a | F_experiments.tex:224-225 | "Pubmed ($N{=}19{,}717$) exceeds $24$\,GB on the dense path" |
| F138 | subgraph_sizes | N∈{30,50,100,200} | subgraph-size ablation on Cora | 10 | F_experiments.tex:229 | "Varying $N{\in}\{30,50,100,200\}$ on Cora (10 seeds)" |
| F139 | tightness_ratio_Nle100 | 1.01–1.02 | tightness for N≤100 | 10 | F_experiments.tex:229-230 | "tightness is stable at $1.01$--$1.02$ for $N{\le}100$" |
| F140 | tightness_ratio_N200 | 1.031 | tightness at N=200 | 10 | F_experiments.tex:230 | "degrades to $1.031$ at $N{=}200$" |
| F141 | subgraph_default_speedup | 66× | N=50 default speedup | 10 | F_experiments.tex:230-231 | "we use $N{=}50$ by default ($66\times$ faster)" |
| F142 | bfs_edge_coverage | ~1.8% | 50-node BFS edge coverage of citation graph | 10 | F_experiments.tex:231-232 | "A 50-node BFS covers only ${\sim}1.8\%$ of a citation graph's edges" |
| F143 | cora_subgraph_tau | τ=0.16 | Cora subgraph τ (low coverage) | 10 | F_experiments.tex:232 | "(Cora $\tau{=}0.16$)" |
| F144 | fullgraph_amp_citeseer | 9.82× | full-graph edge advantage over degree, Citeseer at k=10 | 10 | F_experiments.tex:233 | "amplifies to $\mathbf{9.82\times}$ (Citeseer)" |
| F145 | fullgraph_amp_cora | 3.25× | full-graph edge advantage over degree, Cora at k=10 | 10 | F_experiments.tex:234 | "$\mathbf{3.25\times}$ (Cora) at $k{=}10$" |
| F146 | subgraph_amp_baseline | ≈1.1× | edge advantage on subgraphs | 10 | F_experiments.tex:234 | "against ${\approx}1.1\times$ on subgraphs" |
| F147 | fullgraph_amp_k | k=10 | budget for amplification claim | 10 | F_experiments.tex:233-234 | "at $k{=}10$" |
| F148 | hidden_dims_sweep | d∈{16,32,64,128} | tightness stable across hidden dims | 10 | F_experiments.tex:235-236 | "Tightness is stable across $d{\in}\{16,32,64,128\}$" |
| F149 | c_cap_sweep | c∈{0.5,0.9} | spectral-norm ceiling sweep (accuracy-robustness frontier) | 10 | F_experiments.tex:236 | "the spectral-norm ceiling $c{\in}\{0.5,0.9\}$" |
| F150 | rv_at_c05 | r_v=0.147 | radius at c=0.5 | 10 | F_experiments.tex:236-237 | "($r_v{=}0.147$ at $72.1\%$ vs.\ $0.089$ at $80.6\%$)" |
| F151 | acc_at_c05 | 72.1% | accuracy at c=0.5 | 10 | F_experiments.tex:237 | "$r_v{=}0.147$ at $72.1\%$" |
| F152 | rv_at_c09 | r_v=0.089 | radius at c=0.9 | 10 | F_experiments.tex:237 | "vs.\ $0.089$ at $80.6\%$" |
| F153 | acc_at_c09 | 80.6% | accuracy at c=0.9 | 10 | F_experiments.tex:237 | "$0.089$ at $80.6\%$" |
| F154 | cora_fullgraph_N | N=2,708 | Cora full-graph node count (edge-protection expt) | 10 | F_experiments.tex:239 | "On the full graph (Cora, $N{=}2{,}708$, 10 seeds)" |
| F155 | edgeprotect_k5_reduction | 2.4±1.8% | σ1(S_c) damage cut at k=5 (full graph) | 10 | F_experiments.tex:240-241 | "cuts $\sigma_1(S_c)$ damage by $2.4{\pm}1.8\%$ at $k{=}5$" |
| F156 | edgeprotect_k10_reduction | 4.6±2.9% | σ1(S_c) damage cut at k=10 (full graph) | 10 | F_experiments.tex:241 | "$4.6{\pm}2.9\%$ at $k{=}10$" |
| F157 | edgeprotect_gain_vs_random | 12–46× | gain over random masking | 10 | F_experiments.tex:241-242 | "a $\mathbf{12}$--$\mathbf{46\times}$ gain over random masking" |
| F158 | subgraph_reduction | 42–61% | subgraph-scale reduction reported | 10 | F_experiments.tex:242 | "far below the $42$--$61\%$ a 50-node subgraph reports" |
| F159 | subgraph_k5_edge_pct | 10% | k=5 as % of subgraph edges | 10 | F_experiments.tex:242-243 | "(where $k{=}5$ is already $10\%$ of its edges)" |
| F160 | participation_ratio | 41–89 edges | delocalized leading attack mode participation ratio | 10 | F_experiments.tex:243-244 | "(participation ratio $41$--$89$ edges)" |
| F161 | subgraph_scale_edge_pct | 2–8% | edges to mask to reach subgraph-scale reduction | 10 | F_experiments.tex:244-245 | "needs $2$--$8\%$ of edges masked" |
| F162 | adaptive_recompute_erosion | 1–2 pp | adaptive recomputation erodes gain | 10 | F_experiments.tex:245 | "adaptive recomputation erodes the gain a further $1$--$2$\,pp" |
| F163 | independent_attacker_blunt | two orders of magnitude | GR-BCD blunted by σ1 penalty (fig:exp2_comovement) | 10 | F_experiments.tex:252-253 | "penalizing it blunts that attacker by two orders of magnitude" |
| F164 | compute_union_gap | ~700× | gap to union of separate tools (tab:compute) | n/a | F_experiments.tex:260-261 | "The ${\sim}700\times$ gap to the union of separate tools" |
| F165 | smoothing_forward_passes | 10^4 | randomized-smoothing forward passes dominating cost | n/a | F_experiments.tex:261-262 | "dominated by the randomized-smoothing certificate's $10^4$ forward passes" |
| F166 | n_architectures | 7 | GNN architectures in tab:explicit | 10 | F_experiments.tex:285-296 | "$S_c$ framework on 7 GNN architectures (Cora, 10 seeds)" |
| **tab:explicit (S_c on 7 GNN architectures, Cora, 10 seeds)** | | | | | F_experiments.tex:294-316 | caption ln296-299 |
| F167 | tightness_ratio_ignn | 1.01±.00 | IGNN (eq.) tightness | 10 | F_experiments.tex:306 | "IGNN (eq.) & $\ms{1.01}{.00}$ & $\ms{7.6}{0.5}\times$ & $\ms{+.32}{.04}$ & $\ms{77.5}{1.7}$" |
| F168 | atkadv_ignn | 7.6±0.5× | IGNN AtkAdv (AEGIS/random damage) | 10 | F_experiments.tex:306 | "IGNN ... $\ms{7.6}{0.5}\times$ ..." |
| F169 | tau_ignn | +.32±.04 | IGNN τ (unweighted v_ij vs single-edge removal) | 10 | F_experiments.tex:306 | "IGNN ... $\ms{+.32}{.04}$ ..." |
| F170 | arch_acc_ignn | 77.5±1.7 | IGNN full-graph test accuracy | 10 | F_experiments.tex:306 | "IGNN ... $\ms{77.5}{1.7}$" |
| F171 | tightness_ratio_gcn2 | 1.00±.00 | GCN-2 tightness | 10 | F_experiments.tex:308 | "GCN-2 & $\ms{1.00}{.00}$ & $\ms{3.6}{0.6}\times$ & $\ms{-.04}{.03}$ & $\ms{78.9}{0.9}$" |
| F172 | atkadv_gcn2 | 3.6±0.6× | GCN-2 AtkAdv | 10 | F_experiments.tex:308 | "GCN-2 ... $\ms{3.6}{0.6}\times$ ..." |
| F173 | tau_gcn2 | -.04±.03 | GCN-2 τ (near-uniform 2-hop shifts) | 10 | F_experiments.tex:308 | "GCN-2 ... $\ms{-.04}{.03}$ ..." |
| F174 | arch_acc_gcn2 | 78.9±0.9 | GCN-2 accuracy | 10 | F_experiments.tex:308 | "GCN-2 ... $\ms{78.9}{0.9}$" |
| F175 | tightness_ratio_gcn4 | 1.02±.00 | GCN-4 tightness | 10 | F_experiments.tex:309 | "GCN-4 & $\ms{1.02}{.00}$ & $\ms{3.4}{0.4}\times$ & $\ms{+.49}{.02}$ & $\ms{78.3}{1.8}$" |
| F176 | atkadv_gcn4 | 3.4±0.4× | GCN-4 AtkAdv | 10 | F_experiments.tex:309 | "GCN-4 ... $\ms{3.4}{0.4}\times$ ..." |
| F177 | tau_gcn4 | +.49±.02 | GCN-4 τ | 10 | F_experiments.tex:309 | "GCN-4 ... $\ms{+.49}{.02}$ ..." |
| F178 | arch_acc_gcn4 | 78.3±1.8 | GCN-4 accuracy | 10 | F_experiments.tex:309 | "GCN-4 ... $\ms{78.3}{1.8}$" |
| F179 | tightness_ratio_gin2 | 1.01±.00 | GIN-2 tightness | 10 | F_experiments.tex:310 | "GIN-2 & $\ms{1.01}{.00}$ & $\ms{2.6}{0.1}\times$ & $\ms{+.33}{.03}$ & $\ms{76.6}{1.9}$" |
| F180 | atkadv_gin2 | 2.6±0.1× | GIN-2 AtkAdv | 10 | F_experiments.tex:310 | "GIN-2 ... $\ms{2.6}{0.1}\times$ ..." |
| F181 | tau_gin2 | +.33±.03 | GIN-2 τ | 10 | F_experiments.tex:310 | "GIN-2 ... $\ms{+.33}{.03}$ ..." |
| F182 | arch_acc_gin2 | 76.6±1.9 | GIN-2 accuracy | 10 | F_experiments.tex:310 | "GIN-2 ... $\ms{76.6}{1.9}$" |
| F183 | tightness_ratio_gat | 1.01±.01 | GAT† tightness | 10 | F_experiments.tex:311 | "GAT$^\dagger$ & $\ms{1.01}{.01}$ & $\ms{2.1}{0.1}\times$ & $\ms{\mathbf{+.54}}{.06}$ & $\ms{77.8}{1.2}$" |
| F184 | atkadv_gat | 2.1±0.1× | GAT† AtkAdv | 10 | F_experiments.tex:311 | "GAT$^\dagger$ ... $\ms{2.1}{0.1}\times$ ..." |
| F185 | tau_gat | +.54±.06 | GAT† τ (bold; highest τ in table) | 10 | F_experiments.tex:311 | "GAT$^\dagger$ ... $\ms{\mathbf{+.54}}{.06}$ ..." |
| F186 | arch_acc_gat | 77.8±1.2 | GAT† accuracy | 10 | F_experiments.tex:311 | "GAT$^\dagger$ ... $\ms{77.8}{1.2}$" |
| F187 | tightness_ratio_sage2 | 1.00±.00 | SAGE-2 tightness | 10 | F_experiments.tex:312 | "SAGE-2 & $\ms{1.00}{.00}$ & $\ms{1.9}{0.2}\times$ & $\ms{+.22}{.10}$ & $\ms{77.5}{1.5}$" |
| F188 | atkadv_sage2 | 1.9±0.2× | SAGE-2 AtkAdv | 10 | F_experiments.tex:312 | "SAGE-2 ... $\ms{1.9}{0.2}\times$ ..." |
| F189 | tau_sage2 | +.22±.10 | SAGE-2 τ | 10 | F_experiments.tex:312 | "SAGE-2 ... $\ms{+.22}{.10}$ ..." |
| F190 | arch_acc_sage2 | 77.5±1.5 | SAGE-2 accuracy | 10 | F_experiments.tex:312 | "SAGE-2 ... $\ms{77.5}{1.5}$" |
| F191 | tightness_ratio_appnp | 1.02±.00 | APPNP tightness | 10 | F_experiments.tex:313 | "APPNP & $\ms{1.02}{.00}$ & $\ms{3.4}{0.3}\times$ & $\ms{+.35}{.01}$ & $\ms{\mathbf{82.2}}{0.6}$" |
| F192 | atkadv_appnp | 3.4±0.3× | APPNP AtkAdv | 10 | F_experiments.tex:313 | "APPNP ... $\ms{3.4}{0.3}\times$ ..." |
| F193 | tau_appnp | +.35±.01 | APPNP τ | 10 | F_experiments.tex:313 | "APPNP ... $\ms{+.35}{.01}$ ..." |
| F194 | arch_acc_appnp | 82.2±0.6 | APPNP accuracy (bold; highest acc in table) | 10 | F_experiments.tex:313 | "APPNP ... $\ms{\mathbf{82.2}}{0.6}$" |
| F195 | gat_tightness_prose | 0.99 | GAT† tightness (prose; note: table says 1.01) | 10 | F_experiments.tex:320 | "restoring differentiability and yielding tightness $0.99$" |
| F196 | gat_atkadv_prose | 4.4× | GAT† AtkAdv (prose; note: table says 2.1×) | 10 | F_experiments.tex:320-321 | "AtkAdv $4.4\times$, $\tau{=}{+}0.56$" |
| F197 | gat_tau_prose | +0.56 | GAT† τ edge-weighted (prose; table unweighted says +.54) | 10 | F_experiments.tex:321 | "AtkAdv $4.4\times$, $\tau{=}{+}0.56$" |
| F198 | robust_backbone_cap | σ1(W)≤0.9 | spectral cap for robust backbones | 10 | F_experiments.tex:322-323 | "Under the $\sigma_1(W){\le}0.9$ cap, RobustGCN-lite ... and GNNGuard-lite ... match or exceed IGNN" |
| F199 | edgeweighted_beat_baseline | Δτ≈+0.16 | edge-weighted A_ij·v_ij beats pure edge-weight baseline | 10 | F_experiments.tex:325-326 | "beats a pure edge-weight baseline by $\Delta\tau{\approx}{+}0.16$" |
| F200 | edgeweighted_fraud_dtau | +0.90 | Δτ on fraud graph (uniform weights uninformative) | 10 | F_experiments.tex:326-327 | "rising to $+0.90$ on the fraud graph" |
| F201 | gat_unweighted_transfer | +0.61 vs +0.56 | GAT-2† unweighted v_ij transfers better | 10 | F_experiments.tex:328-329 | "the unweighted $v_{ij}$ transfers better there ($+0.61$ vs.\ $+0.56$)" |
| F202 | transfer_pairwise_cond | 47–62% | edge pairs satisfying prop:transfer pairwise condition | 10 | F_experiments.tex:329-330 | "holds for $\mathbf{47}$--$\mathbf{62\%}$ of edge pairs" |
| F203 | gcn2_weighted_recovery | -0.28→+0.99 | GCN-2 weighted ranking recovers order (Citeseer) | 10 | F_experiments.tex:331-332 | "the weighted ranking recovers the order via the edge weight ($-0.28{\to}{+}0.99$, Citeseer)" |
| F204 | fraud_dtau_walkthrough | +0.90 | Δτ up to +0.90 over weight baseline (fraud walk-through) | 10 | F_experiments.tex:350 | "$v_{ij}$ adds the most rank information ($\Delta\tau$ up to $\mathbf{+0.90}$ over the weight baseline)" |
| F205 | transfer_cos_walkthrough | 0.99 | SVD direction transfers, zero shared gradients (fraud) | 10 | F_experiments.tex:351 | "transfers from a separately trained surrogate (zero shared gradients, $\cos{=}0.99$" |
| F206 | n_seeds_repro | 10 | reproducibility: 10 fixed seeds | 10 | F_experiments.tex:364 | "All results use 10 fixed random seeds (42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999)" |
| F207 | seed_list | 42,137,271,314,1729,2718,3141,5772,6561,9999 | the 10 preferred seeds (MATCHES hard-rule list) | 10 | F_experiments.tex:364 | "(42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999)" |
| F208 | ignn_W_dim | W∈R^{64×64} | IGNN weight matrix dimension | 10 | F_experiments.tex:367 | "$W\in\R^{64\times64}$ spectral-normalized" |
| F209 | learning_rate | 0.01 | Adam learning rate | 10 | F_experiments.tex:367-368 | "Adam (learning rate $0.01$)" |
| F210 | rsvd_k_p | k=p=10 | randomized SVD oversampling params | 10 | F_experiments.tex:371 | "The randomized SVD ... uses $k{=}p{=}10$, $n_{\mathrm{iter}}{=}2$" |
| F211 | rsvd_niter | n_iter=2 | randomized SVD power iterations | 10 | F_experiments.tex:371 | "$n_{\mathrm{iter}}{=}2$" |
| F212 | neumann_earlystop | <10^-6 ‖b‖ | Neumann series early-stop tolerance | 10 | F_experiments.tex:371-372 | "the Neumann series early-stops at $\norm{J_z^k b}<10^{-6}\norm{b}$" |
| F213 | hardware | RTX 4090 | single GPU for timings | n/a | F_experiments.tex:372 | "Timings are on a single RTX 4090." |
| F214 | aegis_alg_gated_steps | steps 3–4 | attack-direction reconstruction gated (alg:aegis) | n/a | F_experiments.tex:374-375 | "the attack-direction reconstruction (\cref{alg:aegis}, steps 3--4) is gated behind institutional-affiliation review" |

---

## A. INTERNAL INCONSISTENCIES within F

1. **GAT† numbers disagree between tab:explicit and prose (lines 311 vs 320-321).** The table (F183/F184/F185) reports tightness `1.01±.01`, AtkAdv `2.1±0.1×`, τ `+.54±.06`. The prose immediately below (F195/F196/F197) reports tightness `0.99`, AtkAdv `4.4×`, τ `+0.56`. Two are partly reconcilable by definition: the table τ is the **unweighted** v_ij ranking (+.54) while the prose τ is the **edge-weighted** A_ij·v_ij ranking (+0.56) — caption explicitly says table τ is unweighted, so this pair is consistent-by-design, NOT an error. BUT tightness (1.01 vs 0.99) and especially **AtkAdv (2.1× table vs 4.4× prose)** are the SAME quantity for the SAME model and differ materially. This is a genuine internal inconsistency to flag for the parent. (Possible that one row is "GAT†" and the prose is a differently-configured "GAT-2†", but the text presents them as the same model "our GAT† modulates attention...yielding...".)

2. **GAT naming drift.** Table row is `GAT$^\dagger$` (F183-186). Prose at line 328 refers to `GAT-2$^\dagger$` (F201) and line 320 to `GAT$^\dagger$`. Likely the same model; naming is inconsistent (GAT vs GAT-2). Minor, but the parent should confirm body uses one name.

3. **τ range for the radar (line 109, F038) = 0.16–0.99** vs **τ range vs GR-BCD in prose (line 67, F028) = 0.35–0.81.** These are different comparisons (radar = white-box gradient oracle proxy across all settings; F028 = rank agreement vs GR-BCD selection specifically), so not a contradiction, but the two "τ ranges" are easy to conflate — note the lower bound 0.16 also appears as the Cora subgraph τ (F143) and the GCN-2-style low-coverage value. Parent should ensure the body cites the correct range for the correct claim.

4. **Δτ "+0.16" (F199, line 326) vs the per-dataset Δτ column in tab:da_decomp (F081–F111).** Prose says edge-weighted beats pure edge-weight "by Δτ≈+0.16". The table's Δτ column (vs GR-BCD) ranges +0.07 to +0.39 (mean of the six = +0.207). +0.16 is plausibly a different aggregate (transfer/heatmap setting, fig:tau_heatmap) not the tab:da_decomp mean. Not a hard contradiction but the "≈+0.16" is not directly any single table cell — flag for provenance.

5. **Δτ "+0.90" appears three times** (F114 tab:da_decomp caption / brute-force; F200 line 326; F204 line 350) all attributed to Amazon Fraud over the weight baseline but in **two different comparison bases**: line 162 explicitly says +0.90 is vs **brute-force single-edge removal** (fig:tau_heatmap), whereas the tab:da_decomp Δτ column (vs GR-BCD) for Amazon Fraud is only **+0.27** (F111). The body must not present +0.90 as the GR-BCD increment. This is the most likely cross-file trap.

6. **Damage-match "~8%" (F027, line 64) vs the ratio table.** Prose says AEGIS matches attacker damage "to within ~8%". The smallest ratio in tab:baselines is Citeseer PR-BCD k=50 = 0.96 (a 4% shortfall) and GR-BCD k=50 = 0.97 (3%); WikiCS k=1 ratios are 1.33 (33% OVER). So "within 8%" describes the shortfalls but ignores the 33% overshoot at k=1 — internally the prose frames this as "and frequently exceeds it", so consistent, but "within ~8%" is a one-sided characterization of a table whose extreme cell is 1.33.

## B. SEED-COUNT AUDIT

- **Seed list MATCHES the hard rule exactly.** Line 364 (F207): `(42, 137, 271, 314, 1729, 2718, 3141, 5772, 6561, 9999)` — identical to required `[42,137,271,314,1729,2718,3141,5772,6561,9999]`. 
- **No result is stated at fewer than 10 seeds.** Every experimental subsection that names a seed count says "10 seeds" (lines 15, 33, 121, 192, 229, 239, 250, 272, 279, 296, 364). 
- **n/a-seed (pure compute/scaling) rows** — these are timing/memory/structural-cost numbers with no seed dependence, correctly NOT carrying a seed count: F016, F037, F039, F040, F127–F133, F164, F165, F213, F214 (smoothing sample counts, radar axes, AGNNCert T, scalability runtimes/memory, RTX 4090, gated steps). These are NOT seed-rule violations.
- **No seed-42-only ablation present** in this file. The memory flags a "log-penalty seed-42 ablation (supposedly dropped)" — it is indeed ABSENT here (good; see staleness C2).
- Conclusion: **F is clean on the 10-seed rule.** No fewer-seed red flags.

## C. STALENESS SIGNALS

1. **Amazon Fraud is PRESENT and load-bearing — NOT dropped.** The audit brief warned an "amzfg/Amazon-Fraud table may have been DROPPED." In F it is very much alive: a full row in tab:baselines (F072–F077), a full row in tab:da_decomp (F108–F113), the +0.90 fraud-transfer headline (F114/F200/F204), and an entire "Fraud-Detector Audit" section (lines 335–357). If the body or theory appendix dropped Amazon Fraud, F is the STALE one (or vice versa). **High-priority cross-check for the parent.** Note the dataset is cited as "Amazon Fraud~\cite{dou2020enhancing}" in repro (line 370), so it is a declared dataset.

2. **Log-penalty / seed-42 ablation: absent (consistent with "dropped").** No λ-on-log-scale seed-42 ablation table appears. If the body still references it, the reference is stale. Confirmed dropped here.

3. **Tables referenced but NOT defined in this file (defined elsewhere — verify they exist):**
   - `tab:attack_full` — referenced (lines 19, 27, 33, 53) and explicitly stated "promoted to the main text" (line 53). Its numbers (F002–F010) live in prose here but the TABLE is in the body. Parent must confirm body's tab:attack_full matches these prose values (esp. cos=0.99, 44±4%, 74–156×, 0–1.8% flips).
   - `tab:smoothing` — referenced (line 188) but defined elsewhere (likely body/another appendix). Smoothing speedups F116–F118 are prose-only here.
   - `tab:compute` — referenced (line 260) for the ~700× gap; defined elsewhere.
   - `fig:tau_heatmap` — referenced (lines 162, 298, 329, 349) but is a FIGURE elsewhere; the +0.90 brute-force increment (F114) is sourced to it.
   - `fig:positioning`, `fig:fraud_case` — referenced, defined elsewhere (header comment line 5 says fig_positioning is a "verified orphan figure" added here, but the `\label{fig:positioning}` is NOT in this file — only `\Cref{fig:positioning}` at line 105; possible orphan/missing-label risk).
   - **fig:positioning POSSIBLE ORPHAN:** header comment (lines 4-5) claims this file "Adds the two verified orphan figures (fig_attack_comparison, fig_positioning)" but only `fig:attack_comparison` is actually `\begin{figure}`+`\label` here (lines 30-39). `fig:positioning` is only *referenced* (line 105), never defined in this file. Either the comment is stale or the figure block was removed. **Flag.**

4. **No duplicated/leftover tables within F.** Each of the 4 in-file tables (tab:threat_model, tab:baselines, tab:da_decomp, tab:explicit) appears once. No obvious placeholder values (no `XX`, `TODO`, `??`, round-number stand-ins). The repeated `1.00`/`0.99` ratios in tab:baselines are legitimately near-1 (the damage-equivalence claim), not placeholders.

5. **Header-comment label list (lines 6-8) lists `app:fraud` and `app:repro`** — both present (lines 336, 361). Header also lists `app:smoothing`, `app:phase_scal`, `app:ablations`, `app:explicit` — all present. No stale label in the header.

6. **|E| values in tab:baselines (61/73/53/55/85/78) are SUBGRAPH edge counts** (N=50 BFS subgraph), NOT full-graph |E|. Full-graph N values appear separately (Cora 2,708; Pubmed 19,717; Amazon Photo 7,650). The body must not confuse subgraph |E| (~50–85) with full-graph edge counts (thousands). Easy stale-number trap.

7. **"Amazon" vs "Amazon Photo" vs "Amazon Fraud":** three distinct datasets. tab:baselines/tab:da_decomp list both "Amazon" (row, |E|=85) and "Amazon Fraud" (row, |E|=78). Repro (line 369-370) cites "Amazon Photo~\cite{shchur2018pitfalls}" and "Amazon Fraud~\cite{dou2020enhancing}". So "Amazon" in the tables = Amazon Photo. Parent should ensure the body's dataset names map consistently (Amazon = Amazon Photo).
