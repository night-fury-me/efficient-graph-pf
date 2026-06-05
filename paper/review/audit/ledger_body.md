# AEGIS Body Claims Ledger (main body of aaai_aegis.tex)

Scope: sections/{abstract,introduction,background,framework,theory,experiments,case_study,related_work,conclusion}.tex
Build order confirmed in aaai_aegis.tex lines 181-190 (case_study = sections/case_study.tex = FRAUD).
All file:line are 1-indexed against the current files.

| id | quantity_key | value | setting | n_seeds | location | exact_quote |
|----|--------------|-------|---------|---------|----------|-------------|
| B01 | scale_N | N=7,650 | single GPU, max graph size | - | abstract.tex:2 | "scales to $N{=}7{,}650$ nodes on one GPU" |
| B02 | conformal_smoothing_costratio | 10^4x lower cost | AEGIS-Conformal vs smoothing, matched Frobenius ball | - | abstract.tex:2 | "where smoothing abstains, at $10^4\times$ lower cost" |
| B03 | defense_anticorrelation | -0.65 | attack vs certified radius anticorrelation | 10 | abstract.tex:2 | "attack and certified radius anticorrelate, $-0.65$, 10 seeds" |
| B04 | n_datasets | 6 datasets | empirical eval | - | abstract.tex:2 | "across 6 datasets, 7 architectures, and 4 domains" |
| B05 | n_architectures | 7 architectures | empirical eval | - | abstract.tex:2 | "across 6 datasets, 7 architectures, and 4 domains" |
| B06 | n_domains | 4 domains | empirical eval | - | abstract.tex:2 | "across 6 datasets, 7 architectures, and 4 domains" |
| B07 | rank_agreement_edgeweighted_tau | tau=0.98 | single-edge-removal damage, edge-weighted | - | abstract.tex:2 | "predicts single-edge-removal damage ... (edge-weighted $\tau{=}0.98$" |
| B08 | tau_uplift_over_weight | +0.16 to +0.90 | sensitivity adds over weight-alone | - | abstract.tex:2 | "the sensitivity adds $\mathbf{+0.16}$ to $\mathbf{+0.90}$ over the weight" |
| B09 | per_query_damage_vs_pgd | 74-156x | one query vs 50-step PGD per-query equilibrium damage | - | abstract.tex:2 | "one query inflicts $\mathbf{74}$--$\mathbf{156\times}$ the per-query equilibrium damage of a 50-step PGD attack" |
| B10 | radar_axes | 7 axes / spans all seven | capability radar, AEGIS wins no axis but spans all 7 | - | introduction.tex:11 | "\AEGIS wins no axis outright but is the only method that spans all seven from one object" |
| B11 | positioning_claim | only AEGIS spans all seven | seven capability axes vs attacks/certifiers | - | introduction.tex:15 | "attacks and certifiers each cover only their own corner, whereas only \AEGIS spans all seven" |
| B12 | first_claim_one_pass | first/only one matrix-free pass | audits+certifies+defends | - | introduction.tex:15 | "No prior method audits, certifies, and defends in one matrix-free pass; \AEGIS ... does" |
| B13 | dense_match_pct | 0.03% | S_c matches dense sigma_1 at N=200 | - | introduction.tex:24 | "matching a dense $\sigma_1$ to $0.03\%$ at $N{=}200$" |
| B14 | scale_N | N=7,650 | scaling | - | introduction.tex:24 | "scaling to $N{=}7{,}650$" |
| B15 | conformal_alpha | coverage >= 1-alpha | AEGIS-Conformal over eps-ball | - | introduction.tex:24 | "with coverage $\geq 1{-}\alpha$ over the $\varepsilon$-ball" |
| B16 | epsilon_crit_formula | ecrit=(1-kappa)/||W||_2 | breaking point, contractive models | - | introduction.tex:24 | "the breaking point $\ecrit{=}(1{-}\kappa)/\norm{W}_2$ in the contraction factor $\kappa{=}\norm{J_z}_2$" |
| B17 | break_understatement | 2-9x | measured break exceeds certificate | 10 | introduction.tex:24 | "whose certificate the \emph{measured} break exceeds by $2$--$9\times$ ($10$ seeds)" |
| B18 | defense_anticorrelation | -0.65, 10/10 seeds | coupled defense, sigma_1 penalty | 10 | introduction.tex:24 | "the coupling is operational, not definitional ($-0.65$, $10/10$ seeds; \cref{sec:defense})" |
| B19 | n_runs | 420 runs | empirical evaluation total | - | introduction.tex:24 | "\textbf{(4) Empirical evaluation} (420 runs)" |
| B20 | kappa_bound_def | kappa <= ||Ahat||_2 ||W||_2 < 1 | IGNN spectral-norm contraction | - | background.tex:13 | "$\kappa \leq \norm{\Ahat}_2 \norm{W}_2 < 1$" |
| B21 | resolvent_gain | bounded by 1/(1-kappa) | IFT resolvent gain | - | background.tex:18 | "the resolvent $(I-J_z)^{-1}$ acts as a gain bounded by $1/(1-\kappa)$" |
| B22 | pseudospectral_index | eta = ||(I-Jz)^-1||_2 (1-rho) >= 1 | pseudospectral index def | - | background.tex:18 | "the pseudospectral index $\eta{=}\norm{(I-J_z)^{-1}}_2(1-\rho){\geq}1$" |
| B23 | threat_model | edge deletion + weight perturbation | symmetric, edge-only, continuous; no insertion | - | background.tex:25 | "\AEGIS targets edge deletion and weight perturbation; insertion attacks ... would enlarge the basis" |
| B24 | prop_attack_maximizer | delta A* = eps * reshape(v1, NxN), shift eps*sigma_1(S) | Prop (Maximally Sensitive 1st-Order Perturbation) | - | framework.tex:12 | "$\delta A^* = \varepsilon \cdot \mathrm{reshape}(v_1, N \times N)$ ... with shift $\varepsilon \cdot \sigma_1(S)$" |
| B25 | constrained_shift_ineq | eps*sigma_1(S_c) <= eps*sigma_1(S) | constrained vs unconstrained maximizer | - | framework.tex:12 | "with shift $\varepsilon\,\sigma_1(S_c){\leq}\varepsilon\,\sigma_1(S)$" |
| B26 | matrixfree_memory | O(Nd) vs O((Nd)^2) | matrix-free S_c application vs dense | - | framework.tex:34 | "applied in $O(Nd)$ memory rather than the $O((Nd)^2)$ of a dense $S_c$" |
| B27 | neumann_K | K in [20,50] for kappa<0.8 | Neumann series truncation | - | framework.tex:34 | "a geometric Neumann series ($K{\in}[20,50]$ for $\kappa{<}0.8$)" |
| B28 | sigma_gap | 43% above sigma_2; gap 0.39-0.50 across suite | spectral separation, Cora ego-graph | - | framework.tex:34 | "$\sigma_1$ a clear $43\%$ above $\sigma_2$ (gap $0.39$--$0.50$ across the suite)" |
| B29 | scale_ceiling_lift | N~300 -> N=7,650 | removing dense solve | - | framework.tex:34 | "lifts the $N{\approx}300$ ceiling to $N{=}7{,}650$" |
| B30 | runtime_query | 0.24 s, ~700x below union | all four artifacts, one query | 10 | framework.tex:34 | "returns all four artifacts in $0.24$\,s ... and ${\sim}700\times$ below their union" |
| B31 | sc_heatmap_params | N=50, d=16, |E|=61, sigma_1=41.2, 43% gap | Cora ego-graph fig | seed 0 | framework.tex:39 | "$N{=}50$, $d{=}16$, $|E|{=}61$ ... $\sigma_1{=}41.2$, $43\%$ gap to $\sigma_2$" |
| B32 | compute_runtime | ~700x below union | Cora, RTX 4090, 50-node subgraph | 10 | framework.tex:45 | "the cost of the single cheapest tool and ${\sim}700\times$ below the union of separate ones" |
| B33 | compute_tab_aegis | AEGIS one query: 0.24 s, queries=1 | tab:compute row | 10 | framework.tex:55 | "\AEGIS one query & $0.24$ & $1$ & ..." |
| B34 | compute_tab_grbcd | GR-BCD 125ep: 0.19 s, 0.8x | tab:compute row | 10 | framework.tex:56 | "GR-BCD ($125$\,ep) & $0.19$ & $0.8$ & ..." |
| B35 | compute_tab_prbcd | PR-BCD 125ep: 2.21 s, 9x | tab:compute row | 10 | framework.tex:57 | "PR-BCD ($125$\,ep) & $2.21$ & $9$ & ..." |
| B36 | compute_tab_smoothing | Smoothing 10^4: 169 s, 695x | tab:compute row | 10 | framework.tex:58 | "Smoothing ($10^4$) & $169$ & $695$ & ..." |
| B37 | compute_tab_union | Union: 169 s, 696x | tab:compute row | 10 | framework.tex:59 | "Union & $169$ & $696$ & ..." |
| B38 | assump_A3_kappa | kappa=0.14-0.59 across suite | (A3) post-training audit | - | theory.tex:14 | "$\kappa{=}0.14$--$0.59$ across our suite; \cref{tab:cross_domain}" |
| B39 | eglob_vs_ecrit | eglob=max(0,1/||W||_2-||Ahat||_2) <= ecrit=(1-kappa)/||W||_2 | global vs critical budget | - | theory.tex:18 | "$\eglob=\max(0,\tfrac{1}{\norm{W}_2}-\norm{\Ahat}_2)\ \le\ \ecrit=\frac{1-\kappa}{\norm{W}_2}$" |
| B40 | subcritical_bound | ||Delta z*||_F <= sigma_1(S) eps + O(eps^2) | Thm phase_transition (a) | - | theory.tex:25 | "$\norm{\Delta \zstar}_F \leq \sigma_1(S)\,\varepsilon + O(\varepsilon^2)$" |
| B41 | sigma1S_bound | sigma_1(S) <= ||J_A||_op/(1-kappa) | subcritical | - | theory.tex:28 | "$\sigma_1(S)\le\norm{J_A}_{\mathrm{op}}/(1-\kappa)$" |
| B42 | critical_divergence_rate | Omega(1/(ecrit-eps)) for normal Jz; eta<=2.47 ReLU | Thm phase_transition (b) | - | theory.tex:30 | "The rate is $\Omega(1/(\ecrit{-}\varepsilon))$ for normal $J_z'$ ... $\eta{\leq}2.47$ for ReLU" |
| B43 | spectral_radius_margin | 2-4x margin to rho=1 | supercritical; trained IGNNs | - | theory.tex:32 | "trained IGNNs sit well inside ($2$--$4\times$ spectral-radius margin to $\rho{=}1$" |
| B44 | ecrit_tightening | 7-14x; eta in [1.19,2.47] | data-dependent S_c tightens closed-form ecrit | - | theory.tex:37 | "the data-dependent $S_c$ tightens it $7$--$14\times$, the slack being $\eta\in[1.19,2.47]$" |
| B45 | nonnormality_gW | g_W=||W||_2/rho(W) >= 1 | W nonnormality | - | theory.tex:39 | "through $W$'s nonnormality $g_W{=}\norm{W}_2/\rho(W){\geq}1$" |
| B46 | bracket_inequality | ecrit <= eps_br <= (C/beta) ecrit | Thm cf2s, two-sided boundary | - | theory.tex:43-46 | "the all-active contraction boundary $\varepsilon_{\mathrm{br}}$ ... obeys ... $\ecrit \le \varepsilon_{\mathrm{br}} \le \tfrac{C}{\beta}\ecrit$" |
| B47 | bracket_constant_C | C = g_W (1+kappa)/(1-kappa) | Thm cf2s constant | - | theory.tex:46 | "$C = g_W\,\tfrac{1+\kappa}{1-\kappa}$" |
| B48 | bracket_equality_cond | equality iff g_W=1 and beta=1 | Thm cf2s collapse | - | theory.tex:49 | "the bracket collapses to equality iff $g_W{=}1$ and $\beta{=}1$" |
| B49 | bracket_slack | ~10-16x (proven); measured break 2-9x | suite | 10 | theory.tex:52 | "The bracket is loose (${\sim}10$--$16\times$) ... certificate under-states the \emph{measured} break by $\mathbf{2}$--$\mathbf{9\times}$ ($10$ seeds" |
| B50 | radius_formula | r_v = min_c m_v^(c)/||(W_yv-W_c) S_v||_2 | Prop per-node radius | - | theory.tex:58 | "$r_v = \min_{c\neq y_v} \frac{m_v^{(c)}}{\norm{(W_{y_v}{-}W_c)\,S_v}_2}$" |
| B51 | constrained_radius_ineq | r_v^(c) >= r_v | substituting S_{c,v} | - | theory.tex:61 | "substituting $S_{c,v}$ gives the tighter constrained radius $r_v^{(c)}{\geq}r_v$" |
| B52 | conformal_coverage_guarantee | Pr[y_v in C_eps(v)] >= 1-alpha | robust conformal certificate | - | theory.tex:70 | "$\Pr[y_v\in C_\varepsilon(v)]\;\geq\;1-\alpha$" |
| B53 | conformal_bound_form | L_1^c eps + C_v eps^2 replacing ~10^4-sample smoothing | curvature-corrected score-shift bound | - | theory.tex:68 | "a curvature-corrected bound $L_1^{c}\varepsilon{+}C_v\varepsilon^2$ replacing the ${\sim}10^4$-sample smoothing" |
| B54 | transfer_eq | d_k = w_k v_k + R_k, |R_k| <= L_J/(2(1-kappa)^2) w_k^2 | Prop continuous-to-discrete transfer | - | theory.tex:84 | "$d_k = w_k v_k + R_k, \qquad |R_k|\le \tfrac{L_J}{2(1-\kappa)^2}\,w_k^2$" |
| B55 | curvature_constant_LJ | L_J <= ||W||_2^2 ||z*|| | transfer curvature constant | - | theory.tex:87 | "with curvature constant $L_J\le\norm{W}_2^2\norm{\zstar}$" |
| B56 | explicit_unrolled_S | S_K = sum_l (prod_k Jz^k) J_A^l | Prop explicit K-layer GNNs | - | theory.tex:96 | "$S_K=...=\sum_{l=1}^{K}\Big(\prod_{k=l+1}^{K}J_z^{(k)}\Big)J_A^{(l)}$" |
| B57 | architectures_inherit | 6 of 7 inherit pipeline; only ecrit restricted | explicit extension scope | - | theory.tex:102 | "Six of our seven architectures thus inherit the full pipeline from $S_K$; only the closed-form $\ecrit$ stays restricted" |
| B58 | n_datasets | 6 datasets | experiments setup | 10 | experiments.tex:8 | "We evaluate \AEGIS on 6 datasets across 4 domains, 10 seeds each" |
| B59 | n_domains | 4 domains | experiments setup | 10 | experiments.tex:8 | "6 datasets across 4 domains, 10 seeds each" |
| B60 | dataset_list | Cora, Citeseer, Pubmed, Amazon Photo, WikiCS, Amazon Fraud | datasets | - | experiments.tex:8 | "Datasets: Cora, Citeseer, Pubmed ... Amazon Photo ... WikiCS ... Amazon Fraud" |
| B61 | ignn_config | W in R^{64x64}, Adam lr 0.01 | IGNN setup | - | experiments.tex:8 | "uses $W\in\R^{64\times64}$ spectral-normalised ... with Adam (lr $0.01$)" |
| B62 | attack_advantage_random | 3.2-4.1x | AtkAdv over random across datasets | 10 | experiments.tex:14 | "a $3.2$--$4.1\times$ attack advantage over random" |
| B63 | cross_domain_cora | Acc 77.5±1.7, kappa .33±.14, ecrit .66±.15, AtkAdv 3.6±.6, Cert .83±.10 | tab:cross_domain Cora | 10 | experiments.tex:26 | "Cora & $\ms{77.5}{1.7}$ & $\ms{.33}{.14}$ & $\ms{.66}{.15}$ & $\ms{3.6}{.6}$ & $\ms{.83}{.10}$" |
| B64 | cross_domain_citeseer | Acc 66.0±0.7, kappa .59±.02, ecrit .41±.02, AtkAdv 4.1±.5, Cert .94±.05 | tab:cross_domain Citeseer | 10 | experiments.tex:27 | "Citeseer & $\ms{66.0}{0.7}$ & $\ms{.59}{.02}$ & $\ms{.41}{.02}$ & $\ms{4.1}{.5}$ & $\ms{.94}{.05}$" |
| B65 | cross_domain_pubmed | Acc 78.9±0.7, kappa .41±.11, ecrit .59±.11, AtkAdv 3.2±.4, Cert .76±.18 | tab:cross_domain Pubmed | 10 | experiments.tex:28 | "Pubmed & $\ms{78.9}{0.7}$ & $\ms{.41}{.11}$ & $\ms{.59}{.11}$ & $\ms{3.2}{.4}$ & $\ms{.76}{.18}$" |
| B66 | cross_domain_amazon | Acc 94.8±0.3, kappa .14±.02, ecrit .86±.02, AtkAdv 3.8±.2, Cert .86±.08 | tab:cross_domain Amazon | 10 | experiments.tex:29 | "Amazon & $\ms{94.8}{0.3}$ & $\ms{.14}{.02}$ & $\ms{.86}{.02}$ & $\ms{3.8}{.2}$ & $\ms{.86}{.08}$" |
| B67 | cross_domain_wikics | Acc 77.9±0.4, kappa .34±.04, ecrit .66±.04, AtkAdv 3.8±.4, Cert .78±.04 | tab:cross_domain WikiCS | 10 | experiments.tex:30 | "WikiCS & $\ms{77.9}{0.4}$ & $\ms{.34}{.04}$ & $\ms{.66}{.04}$ & $\ms{3.8}{.4}$ & $\ms{.78}{.04}$" |
| B68 | surrogate_transfer_damage | 99% (cos=0.99) vs 44±4% for 512-query | one-query equilibrium damage transfer | 10 | experiments.tex:35 | "recovers $99\%$ of the one-query equilibrium damage ($\cos{=}0.99$), versus $44{\pm}4\%$ for a $512$-query black-box search" |
| B69 | query_count | 512-query black-box | comparison search | - | experiments.tex:35 | "versus $44{\pm}4\%$ for a $512$-query black-box search" |
| B70 | per_query_advantage | 74-156x | one query vs 50-step PGD at eps=0.10 | 10 | experiments.tex:35 | "matches or beats $50$-step PGD ... a $\mathbf{74}$--$\mathbf{156\times}$ per-query advantage" |
| B71 | flip_fraction_small | under 2% at eps=0.10 | prediction flips at this budget | 10 | experiments.tex:35 | "though prediction flips stay under $2\%$ at this budget" |
| B72 | fidelity_tau | tau=0.999 | finite differences reproduce S_c ranking | - | experiments.tex:35 | "finite differences reproduce the $S_c$ ranking ($\tau{=}0.999$)" |
| B73 | attack_full_cora | AEGIS 3.70±.79 ... random 0.97±.25 | tab:attack_full Cora, eps=0.10 | 10 | experiments.tex:47 | "Cora & $\ms{3.70}{.79}$ & $\ms{2.51}{.48}$ & $\ms{3.05}{.70}$ & $\ms{0.97}{.25}$" |
| B74 | attack_full_citeseer | 4.63±.71 / 2.97±.42 / 3.35±.57 / 1.01±.15 | tab:attack_full Citeseer | 10 | experiments.tex:48 | "Citeseer & $\ms{4.63}{.71}$ & $\ms{2.97}{.42}$ & $\ms{3.35}{.57}$ & $\ms{1.01}{.15}$" |
| B75 | attack_full_wikics | 2.10±.22 / 0.67±.11 / 1.93±.20 / 0.53±.09 | tab:attack_full WikiCS | 10 | experiments.tex:49 | "WikiCS & $\ms{2.10}{.22}$ & $\ms{0.67}{.11}$ & $\ms{1.93}{.20}$ & $\ms{0.53}{.09}$" |
| B76 | baseline_ranking_advantage | +6-148% AtkAdv at eps=0.01 | vs degree, edge-betweenness, spectral | - | experiments.tex:54 | "beats degree, edge-betweenness, and spectral rankings at $\varepsilon{=}0.01$ ($+6$--$148\%$ AtkAdv)" |
| B77 | mettack_advantage | 3-10x more damage; 149/150 wins, p<10^-43 | vs Mettack, early-warning regime | - | experiments.tex:54 | "$3$--$10\times$ more equilibrium damage than Mettack ... ($\mathbf{149/150}$ wins, $p{<}10^{-43}$" |
| B78 | first_order_match | within 1% at small budgets | first-order shifts | - | experiments.tex:54 | "First-order shifts match within $1\%$ at small budgets" |
| B79 | breach_validation | every breached node has eps > r_v | breach rates across datasets/budgets | - | experiments.tex:56 | "Every breached node satisfies $\varepsilon > r_v$ across all datasets and budgets" |
| B80 | flip_fraction_pubmed | under 8% except 27.4% Pubmed at eps=0.20 | flip fractions | 10 | experiments.tex:56 | "flip fractions ... stay under $8\%$ except a right-skewed $27.4\%$ on Pubmed at $\varepsilon{=}0.20$" |
| B81 | greedy_recovery | 54-67% of Cora damage at k=5-10, no labels | vs label-aware Greedy | 10 | experiments.tex:69 | "recovers $\mathbf{54}$--$\mathbf{67\%}$ of its Cora damage at $k{=}5$--$10$ with no label access" |
| B82 | transfer_n_runs | 6 datasets, 7 architectures (420 runs) | tau heatmap | 10 | experiments.tex:74 | "\cref{fig:tau_heatmap} reports Kendall $\tau$ over 6 datasets and 7 architectures (420 runs)" |
| B83 | transfer_cells_positive | all 42/42 cells positive, median tau=+0.98, p<10^-5 | continuous-to-discrete, edge-weighted | 10 | experiments.tex:74 | "all $\mathbf{42/42}$ cells positive, median $\tau{=}\mathbf{+0.98}$ ($p{<}10^{-5}$)" |
| B84 | tau_uplift_over_weight | +0.16 (median) to +0.90 | v_ij adds over weight-alone | 10 | experiments.tex:74 | "$v_{ij}$ adds the rest, $\mathbf{+0.16}$ (median) to $\mathbf{+0.90}$ over weight-alone ranking" |
| B85 | fullgraph_amazon_tau | tau=+0.996 at kappa~1.00, N=7,650 | Amazon Photo full-graph | 10 | experiments.tex:74 | "On full-graph Amazon Photo ($N{=}7{,}650$) it reaches $\tau{=}\mathbf{+0.996}$ even at $\kappa{\approx}1.00$" |
| B86 | architectures_list | GCN, SAGE, GIN, APPNP, GAT-dagger (+IGNN) | seven K-layer architectures | - | experiments.tex:74 | "for seven $K$-layer architectures (GCN, SAGE, GIN, APPNP, GAT$^\dagger$ ...)" |
| B87 | conformal_n | 10 seeds, four datasets | conformal certificate | 10 | experiments.tex:87 | "(\cref{tab:conformal}; 10 seeds, four datasets)" |
| B88 | conformal_gate | nominal 0.90 at eps=0.01; 0.94-1.00 at eps=0.05 | coverage gate under worst-case attack | 10 | experiments.tex:87 | "sits at the nominal $0.90$ at $\varepsilon{=}0.01$ and turns conservative ($0.94$--$1.00$) at $\varepsilon{=}0.05$" |
| B89 | conformal_setsize | ~1.0-3.5 labels | informative prediction sets | 10 | experiments.tex:87 | "with informative sets (${\sim}1.0$--$3.5$ labels)" |
| B90 | conformal_alpha | alpha=0.1, target 0.90 | tab:conformal; Cora,Citeseer,Pubmed,WikiCS | 10 | experiments.tex:90 | "10 seeds; Cora, Citeseer, Pubmed, WikiCS; $\alpha{=}0.1$, target $0.90$" |
| B91 | conformal_classes | classes 7/6/3/10; seed std <=0.07 | tab:conformal caption | 10 | experiments.tex:90 | "Classes 7/6/3/10; seed std ${\leq}0.07$; sets ${<}1$ reflect empty (abstaining) TPS sets" |
| B92 | conformal_cora_aps | 0.90/1.37 (gate 0.90 @.01); 0.89/1.06 (gate 0.98 @.05) | tab:conformal Cora APS | 10 | experiments.tex:96 | "Cora & APS & 0.90 / 1.37 & \textbf{0.90} & 0.89 / 1.06 & \textbf{0.98}" |
| B93 | conformal_cora_tps | 0.88/0.95 (gate 0.90); 0.89/0.99 (gate 0.98) | tab:conformal Cora TPS | 10 | experiments.tex:97 | "& TPS & 0.88 / 0.95 & \textbf{0.90} & 0.89 / 0.99 & \textbf{0.98}" |
| B94 | conformal_citeseer_aps | 0.92/1.50 (gate 0.92); 0.92/1.35 (gate 0.97) | tab:conformal Citeseer APS | 10 | experiments.tex:99 | "Citeseer & APS & 0.92 / 1.50 & \textbf{0.92} & 0.92 / 1.35 & \textbf{0.97}" |
| B95 | conformal_citeseer_tps | 0.91/1.20 (gate 0.92); 0.92/1.26 (gate 0.96) | tab:conformal Citeseer TPS | 10 | experiments.tex:100 | "& TPS & 0.91 / 1.20 & \textbf{0.92} & 0.92 / 1.26 & \textbf{0.96}" |
| B96 | conformal_pubmed_aps | 0.91/1.42 (gate 0.95); 0.92/1.52 (gate 1.00) | tab:conformal Pubmed APS | 10 | experiments.tex:102 | "Pubmed & APS & 0.91 / 1.42 & \textbf{0.95} & 0.92 / 1.52 & \textbf{1.00}" |
| B97 | conformal_pubmed_tps | 0.90/1.15 (gate 0.92); 0.90/1.15 (gate 0.98) | tab:conformal Pubmed TPS | 10 | experiments.tex:103 | "& TPS & 0.90 / 1.15 & \textbf{0.92} & 0.90 / 1.15 & \textbf{0.98}" |
| B98 | conformal_wikics_aps | 0.92/3.70 (gate 0.94); 0.93/3.62 (gate 0.95) | tab:conformal WikiCS APS | 10 | experiments.tex:105 | "WikiCS & APS & 0.92 / 3.70 & \textbf{0.94} & 0.93 / 3.62 & \textbf{0.95}" |
| B99 | conformal_wikics_tps | 0.93/3.52 (gate 0.94); 0.93/3.52 (gate 0.95) | tab:conformal WikiCS TPS | 10 | experiments.tex:106 | "& TPS & 0.93 / 3.52 & \textbf{0.94} & 0.93 / 3.52 & \textbf{0.95}" |
| B100 | smoothing_costratio | 10^3-10^4x cheaper | zero-sample S_c bound vs randomized smoothing | 10 | experiments.tex:111 | "the zero-sample $S_c$ bound runs $\mathbf{10^3}$--$\mathbf{10^4\times}$ cheaper" |
| B101 | smoothing_tab_params | Cora, n=200, 10 seeds, alpha=0.1, APS | tab:smoothing | 10 | experiments.tex:114 | "(Cora, $n{=}200$, 10 seeds, $\alpha{=}0.1$, APS)" |
| B102 | smoothing_aegis_001 | Cov 0.90, Set 1.36, Wall ~1 s | AEGIS-Conformal eps=0.01 | 10 | experiments.tex:120 | "\AEGIS-Conformal & -- & 0.01 & 0.90 & 1.36 & -- & \textbf{${\sim}1$\,s}" |
| B103 | smoothing_rs_frob_001 | Cov 1.00, Set 7.00, Cert 0.00, Wall 7,600 s | RandSmoothing frob eps=0.01 | 10 | experiments.tex:121 | "RandSmoothing & frob & 0.01 & 1.00 & 7.00 & 0.00 & 7{,}600\,s" |
| B104 | smoothing_rs_peredge_001 | Cov 0.95, Set 1.26, Cert 0.96, Wall 15,000 s | RandSmoothing per_edge eps=0.01 | 10 | experiments.tex:122 | "RandSmoothing & per\_edge& 0.01 & 0.95 & 1.26 & 0.96 & 15{,}000\,s" |
| B105 | smoothing_aegis_005 | Cov 0.90, Set 1.05, Wall ~1 s | AEGIS-Conformal eps=0.05 | 10 | experiments.tex:124 | "\AEGIS-Conformal & -- & 0.05 & 0.90 & 1.05 & -- & \textbf{${\sim}1$\,s}" |
| B106 | smoothing_rs_frob_005 | Cov 1.00, Set 7.00, Cert 0.00, Wall 10,900 s | RandSmoothing frob eps=0.05 | 10 | experiments.tex:125 | "RandSmoothing & frob & 0.05 & 1.00 & 7.00 & 0.00 & 10{,}900\,s" |
| B107 | smoothing_rs_peredge_005 | Cov 0.99, Set 2.39, Cert 0.77, Wall 36,300 s | RandSmoothing per_edge eps=0.05 | 10 | experiments.tex:126 | "RandSmoothing & per\_edge& 0.05 & 0.99 & 2.39 & 0.77 & 36{,}300\,s" |
| B108 | defense_sigma_drop | sigma_1 cut ~10x; cert lifted to 0.82±0.03 for ~4 acc pts | lambda=3e-4 | 10 | experiments.tex:135 | "at $\lambda{=}3{\times}10^{-4}$ this cuts $\sigma_1$ by ${\sim}10\times$ and lifts the certified fraction to $0.82{\pm}0.03$ for ${\sim}4$ accuracy points" |
| B109 | defense_comovement | -0.65±0.12, 10/10 seeds | one knob moves attack/certificate/attacker | 10 | experiments.tex:135 | "One knob moves attack, certificate, and that independent attacker together ($-0.65{\pm}0.12$, $\mathbf{10/10}$ seeds" |
| B110 | defense_tab_l0 | Acc 78.1±1.8, sigma_1 319±59, Cert .40±.08, Dmg 27.8±6.3 | tab:defense lambda=0 | 10 | experiments.tex:147 | "$0$ & $\ms{78.1}{1.8}$ & $\ms{319}{59}$ & $\ms{.40}{.08}$ & $\ms{27.8}{6.3}$" |
| B111 | defense_tab_l3e4 | Acc 73.9±0.8, sigma_1 32.6±1.7, Cert .82±.03, Dmg 3.1±0.2 | tab:defense lambda=3e-4 | 10 | experiments.tex:148 | "$3{\times}10^{-4}$ & $\ms{73.9}{0.8}$ & $\ms{32.6}{1.7}$ & $\ms{.82}{.03}$ & $\ms{3.1}{0.2}$" |
| B112 | defense_tab_l1e3 | Acc 69.0±0.7, sigma_1 10.7±0.6, Cert .89±.02, Dmg 1.1±0.1 | tab:defense lambda=1e-3 | 10 | experiments.tex:149 | "$1{\times}10^{-3}$ & $\ms{69.0}{0.7}$ & $\ms{10.7}{0.6}$ & $\ms{.89}{.02}$ & $\ms{1.1}{0.1}$" |
| B113 | defense_tab_l3e3 | Acc 61.9±0.6, sigma_1 3.9±0.2, Cert .92±.02, Dmg 0.4±0.0 | tab:defense lambda=3e-3 (Cert peak) | 10 | experiments.tex:150 | "$3{\times}10^{-3}$ & $\ms{61.9}{0.6}$ & $\ms{3.9}{0.2}$ & $\ms{.92}{.02}$ & $\ms{0.4}{0.0}$" |
| B114 | defense_tab_l1e2 | Acc 56.4±0.6, sigma_1 0.8±0.1, Cert .86±.04, Dmg 0.1±0.0 | tab:defense lambda=1e-2 | 10 | experiments.tex:151 | "$1{\times}10^{-2}$ & $\ms{56.4}{0.6}$ & $\ms{0.8}{0.1}$ & $\ms{.86}{.04}$ & $\ms{0.1}{0.0}$" |
| B115 | defense_cert_peak | Cert peaks at lambda=3e-3 | tab:defense caption | 10 | experiments.tex:139 | "Cert (certified fraction) peaks at $\lambda{=}3{\times}10^{-3}$ before margin collapse" |
| B116 | fraud_top5_overlap | top-5 by A_ij v_ij; 3 also in brute-force top-3 | fig fraud_case caption | - | case_study.tex:9 | "Green edges are the top-5 fault lines by $A_{ij}v_{ij}$; yellow dots mark the three also in the brute-force single-edge-removal top-3" |
| B117 | fraud_tau | tau=1.0 here, one query vs |E| reconvergences | Amazon Fraud cluster audit | - | case_study.tex:13 | "reproducing the brute-force single-edge-removal ranking ($\tau{=}1.0$ here) at the cost of one query rather than $|E|$ reconvergences" |
| B118 | fraud_dtau | Delta tau up to +0.90 on full fraud graph | non-circular audit | - | case_study.tex:13 | "both the ranking ($\Delta\tau$ up to $\mathbf{+0.90}$ on the full fraud graph; \cref{fig:tau_heatmap})" |
| B119 | one_query_vs_N | one query vs N (no labels, no retraining) | fraud detector audit | - | case_study.tex:13 | "\AEGIS audits one directly, with no labels and no retraining, reading from a \emph{single} $S_c$ query" |
| B120 | related_attacks_gap | structural attacks yield no analytic direction, no per-node radius | Nettack/Mettack/GR-BCD/PR-BCD | - | related_work.tex:6 | "optimise gradients to flip predictions but yield no analytic direction and no per-node radius" |
| B121 | related_conformal_gap | graph conformal certifies fixed graph; AEGIS covers eps-ball | vs zargarbashi2023conformal | - | related_work.tex:8 | "Graph conformal prediction certifies a \emph{fixed} graph, whereas \AEGIS-Conformal covers the adversarial $\varepsilon$-ball" |
| B122 | related_implicit_gap | prior addresses input sensitivity; AEGIS targets structural | DEQs/IGNN/influence | - | related_work.tex:10 | "address \emph{input} sensitivity $\partial \zstar/\partial x$; \AEGIS targets \emph{structural} sensitivity $\partial Z/\partial A$" |
| B123 | conclusion_unification | one matrix-free object unifies audit+cert+defense | conclusion | - | conclusion.tex:6 | "From one matrix-free object, $S_c$, \AEGIS unifies auditing, certification, and defense as readings of a single operator" |
| B124 | conclusion_ecrit_cost | ecrit track costs ~6% accuracy | limitations | - | conclusion.tex:8 | "the $\ecrit$ track costs ${\sim}6\%$ accuracy" |
| B125 | conclusion_physics_limit | contractive surrogate cannot model voltage collapse | limitations; complements PF screening | - | conclusion.tex:8 | "a contractive surrogate cannot model voltage collapse, so it complements rather than replaces power-flow contingency screening" |
| B126 | disclosure_window | 90-day coordinated-notification window | disclosure | - | conclusion.tex:10 | "We propose a 90-day coordinated-notification window" |
| B127 | accuracy_cost_constraint | ~6% Cora accuracy | IGNN constraint cost (also stated framework via 6% in setup) | 10 | experiments.tex:8 | "(implied) constraint cost; cf. conclusion ${\sim}6\%$" (see also abstract/intro 6% absent) |

## A. INTERNAL INCONSISTENCIES (within the main body)

1. **"42/42" cells vs "6 datasets x 7 architectures = 42" but eval is "6 datasets, 7 architectures".** Consistent arithmetic (6x7=42). NOT an inconsistency — flagged only to confirm 420 runs = 42 cells x 10 seeds is internally coherent. (No action.)

2. **Defense cuts sigma_1 "~10x" — check against tab:defense.** Body (experiments.tex:135) and intro/abstract say sigma_1 cut ~10x at lambda=3e-4. tab:defense: 319 (lambda=0) -> 32.6 (lambda=3e-4) = 9.8x. Consistent. The "~10x" defense claim matches the table.

3. **Certified-fraction at lambda=3e-4 = 0.82±0.03** appears identically in abstract-region prose, experiments.tex:135, and tab:defense row (B111). Consistent.

4. **"~6% accuracy" cost of the ecrit/IGNN constraint** is asserted in conclusion.tex:8 (B124) but the experiments.tex:8 setup no longer states a "~6% Cora accuracy" figure (older drafts did, per the stale index that read "the constraint costs ~6% Cora accuracy"). The current experiments setup line omits it; only conclusion carries the 6%. Mild internal gap: the number is asserted in the conclusion without being established in the experiments body (the supporting comparison must be in tab:cross_domain Cora 77.5% vs an unstated unconstrained baseline). Parent should confirm appendix carries the unconstrained baseline.

5. **Defense "~4 accuracy points" (experiments.tex:135) vs "~5% clean-accuracy cost" (stale index theory variant) vs "~6%".** Current body uses "~4 accuracy points" at lambda=3e-4 (78.1 -> 73.9 = 4.2 pts, consistent with tab:defense). The stale "~5%" appears only in indexed-but-removed theory text, NOT in current files. No live inconsistency, but note the family of accuracy-cost numbers (4 pts defense penalty vs 6% ecrit-constraint) describe DIFFERENT quantities and should not be conflated.

6. **No live abstract-vs-experiments numeric contradiction found.** Datasets (6), architectures (7), domains (4), runs (420), tau=0.98, +0.16-+0.90 uplift, 74-156x, -0.65, N=7,650, alpha, 0.82 cert — all agree across abstract / intro / experiments. This body is internally tight on headline numbers.

## B. STALENESS SIGNALS

1. **POWER-FLOW / N-1 residue while the case study is FRAUD.** The named case study file (case_study.tex) is now "Auditing a Fraud Detector" (Amazon Fraud, sec:fraud_case). But the body still contains power-grid contingency language left over from a dropped IEEE case study:
   - experiments.tex:74 — "ask whether the continuous ranking predicts brute-force **N-1** removal" ("N-1" is power-system contingency terminology applied to citation/fraud graphs).
   - conclusion.tex:8 — full sentence "a contractive surrogate cannot model voltage collapse, so it complements rather than replaces **power-flow contingency screening** [donon2019graph,nakiganda2023graph,varbella2024contingency]."
   These cite PF works (donon2019graph, varbella2024contingency) that no longer correspond to any case study in the body. Likely leftover from when there WAS an IEEE case14-118 power-flow case study. ACTION: parent should decide whether to (a) re-add the PF case study, or (b) scrub the N-1 / voltage-collapse / PF-citation residue.

2. **STALE INDEX ARTIFACT (NOT in current body) — flagged so parent does not chase it.** My first sandbox gather indexed an OLDER snapshot that contained: an IEEE case14-118 case_study (tau=+0.37-+0.62, P@10=0.66-0.81, LODF baselines, N=7,650 scalability), an intro variant saying "390 runs" / "9 datasets ... 330 runs" / "power-grid N-1 case study (sec:po...)" / cites tab:baselines, and a theory variant saying defense costs "~5% clean-accuracy". NONE of these strings exist in the current 9 body files (verified by direct grep). They are stale cached content. The CURRENT body uniformly says 6 datasets / 420 runs / fraud case / ~4 acc points. The "9 datasets" and "330/390 runs" are dead.

3. **Run-count lineage.** Body is now 420 runs everywhere. If the appendix still says 330 or 390 runs, or "9 datasets", that is a body↔appendix staleness; the other agent should check.

4. **AGNNCert positioning is "complementary"/scoped to app:baselines** (intro.tex:24, related_work.tex:8) — not a head-to-head win. The intro frames GR-BCD/PR-BCD as head-to-head but AGNNCert only as "complementary positioning." No overclaim, but note the asymmetry: do not let any summary claim AEGIS "beats AGNNCert."

5. **fraud audit tau=1.0 "here"** (case_study.tex:13, B117) is a single-cluster anecdote (the hedge word "here"), while the full-graph fraud Delta tau "up to +0.90" (B118) is the defensible aggregate. The tau=1.0 is figure-local, not a suite result — make sure no downstream text promotes "tau=1.0 on fraud" as a headline.

6. **"~6% accuracy" cost (conclusion B124)** has no establishing measurement in the experiments body (see Inconsistency #4) — looks like a number carried over from an earlier draft where the setup line stated it; verify the appendix establishes the unconstrained baseline, else it is a placeholder-grade claim.

## C. BODY -> APPENDIX PROMISES (each body claim -> the appendix/table label it cites)

- B13/B14 dense-match 0.03% @N=200 & scaling N=7,650 -> implicitly to \cref{app:phase_scal} (scaling) [intro.tex:24 cites no app inline; scaling figure fig:scalability is appendix].
- B17/B49 measured break exceeds certificate 2-9x (10 seeds) -> \cref{app:bracket} [theory.tex:52; intro.tex:24].
- B30/B32/B33-37 runtime 0.24s, ~700x, tab:compute -> \cref{tab:compute} (in framework body) ; smoothing 10^4 detail -> \cref{app:smoothing}.
- B38 kappa=0.14-0.59 -> \cref{tab:cross_domain} (body) [theory.tex:14].
- B42/B44 eta<=2.47, tightening 7-14x, eta in [1.19,2.47] -> \cref{obs:eta_bound,rem:eta_relu} (eta bound stated in theory; ReLU range "on our suite" -> app:ablations per stale text).
- B43 2-4x spectral-radius margin -> \cref{app:phase_scal} [theory.tex:32].
- B46-B48 bracket inequality, C=g_W(1+kappa)/(1-kappa), equality cond -> full statement/proof in \cref{app:bracket} [theory.tex:41-52].
- B52/B53 robust conformal coverage >=1-alpha, L_1^c eps + C_v eps^2 -> \cref{app:conformal} [theory.tex:68-70].
- B54/B55 transfer eq d_k=w_k v_k+R_k, L_J bound -> proof in \cref{app:transfer} [theory.tex:78-87].
- B56/B57 explicit S_K, 6-of-7 inherit -> proof in \cref{app:transfer}; per-arch detail \cref{app:explicit} [theory.tex:92-102].
- B62/B63-67 AtkAdv 3.2-4.1x, per-dataset table -> \cref{tab:cross_domain} (body table) + graph-level \cref{app:phase_scal}.
- B68/B70 surrogate transfer 99%/44±4% (512q), 74-156x -> \cref{tab:attack_full} (body) [experiments.tex:35].
- B76/B77 +6-148% AtkAdv, 3-10x vs Mettack, 149/150 wins p<10^-43 -> \cref{fig:greedy_topk} [experiments.tex:54].
- B78 first-order match within 1% -> \cref{fig:tightness_eps} [experiments.tex:54].
- B80 flip fractions <8% / 27.4% Pubmed -> \cref{fig:breach} [experiments.tex:56].
- B81 Greedy recovery 54-67% -> \cref{fig:greedy_topk} [experiments.tex:69].
- B82-85 transfer tau heatmap 42/42, median +0.98, +0.16-+0.90, +0.996 -> \cref{fig:tau_heatmap}; detail/ablations -> \cref{app:explicit,app:phase_scal,app:ablations} [experiments.tex:74].
- B87-99 conformal coverage table -> \cref{tab:conformal} (body) ; gate test detail -> \cref{app:conformal} [experiments.tex:87-90].
- B100-107 smoothing 10^3-10^4x cheaper, vs-smoothing table -> \cref{tab:smoothing} (body) + \cref{app:smoothing} [experiments.tex:111-126].
- B108/B109/B110-115 defense sigma_1 ~10x, 0.82 cert, -0.65 co-movement, tab:defense -> \cref{tab:defense} (body) ; independent-attacker/Lipschitz comparison -> \cref{app:ablations} ; co-movement fig -> \cref{fig:exp2_comovement} [experiments.tex:135].
- B116-119 fraud top-5/top-3 overlap, tau=1.0, one-query-vs-|E| -> \cref{fig:fraud_case} + full walk-through \cref{app:fraud} [case_study.tex].
- B118 fraud Delta tau up to +0.90 -> \cref{fig:tau_heatmap} [case_study.tex:13].
- B10/B11 capability radar (7 axes), per-method scoring -> \cref{fig:positioning} + axis definitions/scoring \cref{app:baselines} [intro.tex:11,15].
- B19 420 runs / four-quadrant / GR-BCD/PR-BCD head-to-head / AGNNCert -> \cref{app:baselines} [intro.tex:24].
- B124 "~6% accuracy" ecrit-constraint cost -> NO explicit appendix cref at point of claim (conclusion.tex:8); parent must locate the establishing baseline (likely app:experiments / tab:cross_domain).
- B125 voltage-collapse / PF limitation -> cites donon2019graph,nakiganda2023graph,varbella2024contingency (bib only; no case study backs it — see Staleness #1).

### Note on appendix-scoped labels referenced by the body (out of body scope; verify in appendix audit):
app:baselines, app:bracket, app:conformal, app:transfer, app:explicit, app:phase_scal, app:ablations, app:smoothing, app:fraud, app:experiments, app:repro, app:proof_phase, app:proof_radius. All resolve to a defined label set in the body's view (no body-internal broken \cref). tab:attack_full, tab:compute, tab:conformal, tab:cross_domain, tab:defense, tab:smoothing are defined IN the body files. No body \cref points to a nonexistent body label. (Older drafts referenced tab:baselines / fig:scalability / fig:ieee14_case — these do NOT appear in current body crefs.)
