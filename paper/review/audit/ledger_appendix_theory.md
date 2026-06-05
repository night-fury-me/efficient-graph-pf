# Claims Ledger — Appendix Theory (A–E)

Scope: `sections/appendix/{A_preliminaries,B_sensitivity,C_rankings,D_boundary,E_conformal}.tex`.
One row per quantitative or theorem-level claim. `location` is `file:line` where the file basename is the appendix letter (A=A_preliminaries, B=B_sensitivity, C=C_rankings, D=D_boundary, E=E_conformal).

## Claims table

| id | quantity_key | value | setting | location | exact_quote |
|----|--------------|-------|---------|----------|-------------|
| A01 | notation_Ahat | $D^{-1/2}(A{+}I)D^{-1/2}$ | symmetric normalized adjacency | A:37 | `$\Ahat$ & symmetric normalized adjacency $D^{-1/2}(A{+}I)D^{-1/2}$ & \cref{thm:phase_transition}` |
| A02 | notation_Jz | $\diag(\phi')(\Ahat\otimes W)$ | state Jacobian | A:41 | `$J_z$ & state Jacobian $\diag(\phi')(\Ahat\otimes W)$` |
| A03 | notation_JA | $\partial F/\partial\mathrm{vec}(A)$ | structural Jacobian | A:42 | `$J_A$ & structural Jacobian $\partial F/\partial\mathrm{vec}(A)$` |
| A04 | kappa_def | $\norm{J_z}_2<1$ | contraction factor | A:43 | `$\kappa$ & contraction factor $\norm{J_z}_2<1$ & (A3)` |
| A05 | notation_S | $(I{-}J_z)^{-1}J_A$ | structural sensitivity | A:46 | `$S$ & structural sensitivity $(I{-}J_z)^{-1}J_A$` |
| A06 | notation_Sc | $(I{-}J_z)^{-1}J_A P_c$ | edge-constrained sensitivity | A:47 | `$S_c$ & edge-constrained sensitivity $ (I{-}J_z)^{-1}J_A P_c$ & \cref{sec:framework}` |
| A07 | sigma1_def | leading singular value | operator spectral | A:50 | `$\sigma_1(\cdot)$ & leading singular value & \cref{prop:attack}` |
| A08 | notation_eglob | $\max(0,1/\norm{W}_2{-}\norm{\Ahat}_2)$ | mask-free safe radius | A:55 | `$\eglob$ & mask-free safe radius $\max(0,1/\norm{W}_2{-}\norm{\Ahat}_2)$` |
| A09 | ecrit_def | $(1{-}\kappa)/\norm{W}_2$ | norm certificate | A:56 | `$\ecrit$ & norm certificate $(1{-}\kappa)/\norm{W}_2$ & \cref{thm:phase_transition}` |
| A10 | espec_def | $1/\rho(W){-}\rho(\Ahat)$ | all-active spectral break | A:57 | `$\espec$ & all-active spectral break $1/\rho(W){-}\rho(\Ahat)$ & \cref{thm:cf2s}` |
| A11 | notation_ereach | measured nonlinear break budget | empirical | A:59 | `$\ereach$ & measured nonlinear break budget & \cref{rem:obs_o1}` |
| A12 | gW_def | $\norm{W}_2/\rho(W)\ge1$ | nonnormality of $W$ | A:60 | `$\gW$ & nonnormality of $W$, $\norm{W}_2/\rho(W)\ge1$ & \cref{thm:cf2s}` |
| A13 | notation_eta | Neumann-to-spectral slack (pseudospectral index) | nonnormality slack | A:61 | `$\eta$ & Neumann-to-spectral slack (pseudospectral index) & \cref{obs:eta_bound}` |
| A14 | beta_def | $\beta=\sigma_E$ = edge-supported mass of Perron mode $u_1$ | alignment | A:62 | `$\beta=\sigma_E$ & edge-supported mass of the Perron mode $u_1$ & \cref{thm:cf2s}` |
| A15 | bracket_constant_C | $\gW(1{+}\kappa)/(1{-}\kappa)$ | bracket constant | A:63 | `$C$ & bracket constant $\gW(1{+}\kappa)/(1{-}\kappa)$ & \cref{thm:cf2s}` |
| A16 | LJ_def | $\le\norm{W}_2^2\norm{\zstar}$ | equilibrium-map curvature | A:64 | `$L_J$ & equilibrium-map curvature, $\le\norm{W}_2^2\norm{\zstar}$ & \cref{prop:transfer}` |
| A17 | notation_rv | per-node first-order sensitivity radius | per-node certificate | A:69 | `$r_v$ & per-node first-order sensitivity radius & \cref{prop:radius}` |
| A18 | conformal_alpha | target miscoverage; calibration-set size | conformal | A:73 | `$\alpha,\ n_{\mathrm{cal}}$ & target miscoverage; calibration-set size & \cref{thm:robust-cov}` |
| A19 | assumption_A1 | $\phi$ 1-Lipschitz; $\norm{\diag(\phi')}_2\le1$ | (A1) Lipschitz activation | A:95-97 | `(A1)] \emph{Lipschitz activation.} $\phi$ is $1$-Lipschitz (ReLU or similar), so $\norm{\diag(\phi')}_2\le1$ pointwise` |
| A20 | assumption_A2 | $\norm{W}_2\le c$; does NOT imply (A3) | (A2) Bounded weight | A:98-100 | `(A2)] \emph{Bounded weight.} $\norm{W}_2\le c$ ... so (A2) does not imply (A3).` |
| A21 | kappa_band | $\kappa:=\norm{J_z}_2<1$; $\kappa\in[0.14,0.59]$ | (A3) trained contractivity, suite | A:101-102 | `$\kappa:=\norm{J_z}_2<1$, equivalently $\norm{\Ahat}_2\norm{W}_2<1$, verified post-training ($\kappa\in[0.14,0.59]$ across our suite).` |
| A22 | assumption_A4 | $\phi'\equiv1$, $J_z=\Ahat\otimes W$; only upper side uses it | (A4) all-active | A:103-105 | `(A4)] \emph{All-active operating point.} On the operating set every unit is active, $\phi'\equiv1$, so $J_z=\Ahat\otimes W$. Only the \emph{upper} side ... invokes (A4)` |
| A23 | certified_regime | $\varepsilon<\ecrit$ | resolvent well defined | A:107-108 | `Unless stated otherwise we work in the certified regime $\varepsilon<\ecrit$, where the perturbed operator is still a contraction and the resolvent $(I-J_z)^{-1}$ is well defined.` |
| A24 | S_def_eq | $S=(I-J_z)^{-1}J_A$ | eq:S-def | A:122-124 | `S=(I-J_z)^{-1}J_A, \label{eq:S-def}` |
| A25 | neumann_series | $S_c v=\sum_{j=0}^{\infty}J_z^{\,j}J_A P_c v$ | converges since $\kappa<1$ | A:144-148 | `S_c\,v=(I-J_z)^{-1}J_A P_c\,v=\sum_{j=0}^{\infty}J_z^{\,j}\,J_A P_c\,v ... which converges because $\kappa<1$` |
| A26 | neumann_tail | $\le\frac{\kappa^{K+1}}{1-\kappa}\norm{J_A P_c v}_2$ | truncation depth $K$ | A:150-155 | `\le\frac{\kappa^{\,K+1}}{1-\kappa}\,\norm{J_A P_c v}_2` ; `$K\in[20,50]$ for $\kappa<0.8$` |
| A27 | truncation_residual | $\kappa^{200}\in[10^{-105},10^{-48}]$; spectral gap $0.39$–$0.50$ | empirical, suite | A:158-160 | `gap $0.39$--$0.50$ ... empirical truncation residual $\kappa^{200}\in[10^{-105},10^{-48}]$ on the suite` |
| B01 | phase_transition_threshold | $\ecrit=(1-\kappa)/\norm{W}_2$; 3 regimes | thm:phase_transition restated | B:20-24 | `the critical budget $\ecrit=(1-\kappa)/\norm{W}_2$: subcritical ($\varepsilon<\ecrit$...), critical ($\varepsilon\to\ecrit$...), and supercritical ($\varepsilon\ge\ecrit$...)` |
| B02 | ift_expand | $\Delta\zstar=(I-J_z)^{-1}J_A\mathrm{vec}(\delta\Ahat)+O(\norm{\delta\Ahat}^2)$ | eq:ift-expand | B:33-36 | `\Delta\zstar=(I-J_z)^{-1}J_A\,\mathrm{vec}(\delta\Ahat)+O(\norm{\delta\Ahat}^2)` |
| B03 | neumann_bound | $\norm{(I-J_z)^{-1}}_2\le\frac{1}{1-\kappa}$ | eq:neumann-bound | B:41-44 | `\norm{(I-J_z)^{-1}}_2\le\sum_{j\ge0}\kappa^{\,j}=\frac{1}{1-\kappa}` |
| B04 | sigma1_S_bound | $\sigma_1(S)\le\norm{J_A}_2/(1-\kappa)$ | shift bound eq:shift_bound | B:45-47 | `$\norm{\Delta\zstar}_F\le\sigma_1(S)\,\varepsilon+O(\varepsilon^2)$ with $\sigma_1(S)\le\norm{J_A}_2/(1-\kappa)$` |
| B05 | Jzp_bound | $\norm{J_z'}_2\le(\norm{\Ahat}_2+\varepsilon)\norm{W}_2$ | eq:Jzp-bound, any mask | B:56-64 | `\norm{J_z'}_2=\norm{\diag(\phi')(\Ahat'\otimes W)}_2 \le\ \norm{\Ahat'}_2\norm{W}_2 \le\ (\norm{\Ahat}_2+\varepsilon)\norm{W}_2` |
| B06 | eglob_threshold | RHS$<1$ iff $\varepsilon<\eglob=1/\norm{W}_2-\norm{\Ahat}_2$ | rigorous safe radius | B:65-69 | `falls below $1$ exactly for $\varepsilon<\eglob=1/\norm{W}_2-\norm{\Ahat}_2$ ... This is the rigorous safe radius $\eglob$.` |
| B07 | eglob_value_normalized | $\eglob=1/c-1$ (since $\norm{\Ahat}_2=1$) | spectral-norm-capped model | B:71-73 | `a normalized adjacency with self-loops has $\norm{\Ahat}_2=1$, so $\eglob=1/c-1$ is small` |
| B08 | ecrit_ge_eglob | $\ecrit=(1-\kappa)/\norm{W}_2\ge\eglob$; reaches 1 at $\varepsilon=\ecrit$ | data-dependent budget | B:74-78 | `the larger, data-dependent budget $\ecrit=(1-\kappa)/\norm{W}_2\ge\eglob$ ... $\norm{J_z'}_2\le\kappa+\varepsilon\norm{W}_2$, which reaches $1$ at $\varepsilon=\ecrit$.` |
| B09 | budgets_coincide | $\ecrit=\eglob$ only in all-active (A4), $\kappa=\norm{\Ahat}_2\norm{W}_2$ | equality condition | B:84-85 | `The two budgets coincide, $\ecrit=\eglob$, only in the all-active case (A4), where $\kappa=\norm{\Ahat}_2\norm{W}_2$ and no region is crossed.` |
| B10 | break_above_ecrit | measured break lies above $\ecrit$; two scaling gaps open | rem:obs_o1 observation | B:89-91 | `the empirical regularity of \cref{rem:obs_o1} (the measured break lies above $\ecrit$), whose two scaling gaps remain open ... we record it as observation, not theorem.` |
| B11 | resolvent_lb | $\norm{(I-M)^{-1}}_2\ge\frac{1}{\min_i\abs{1-\lambda_i(M)}}$ | eq:resolvent-lb | B:95-98 | `\norm{(I-M)^{-1}}_2\ \ge\ \rho\!\big((I-M)^{-1}\big)=\frac{1}{\min_i\abs{1-\lambda_i(M)}}` |
| B12 | critical_divergence | gap $=\norm{W}_2(\ecrit-\varepsilon)$; $\Omega(1/(\ecrit-\varepsilon))$ | normal Perron case | B:99-103 | `$\min_i\abs{1-\lambda_i}=1-\norm{J_z'}_2=\norm{W}_2(\ecrit-\varepsilon)$ ... giving the $\Omega\!\big(1/(\ecrit-\varepsilon)\big)$ divergence.` |
| B13 | ecrit_one_sided | $\ecrit$ lower-bounds true divergence threshold; slack = nonnormality $\eta$ | one-sided | B:103-107 | `So $\ecrit$ lower-bounds the true divergence threshold, and the slack is governed by how far $J_z'$ is from normal, the nonnormality $\eta$ of \cref{obs:eta_bound}.` |
| B14 | eta_bound | $\eta\le\kappa(V_W)=\norm{V_W}_2\norm{V_W^{-1}}_2$ | obs:eta_bound, all-active | B:121-127 | `the nonnormality obeys $\eta\le\kappa(V_W)$, where $W=V_W D_W V_W^{-1}$ and $\kappa(V_W)=\norm{V_W}_2\norm{V_W^{-1}}_2$ is the eigenvector conditioning of $W$, independent of $\Ahat$.` |
| B15 | resolvent_diag_bound | $\norm{(I-M)^{-1}}_2\le\kappa(V)/(1-\rho(M))$; slack exactly $\kappa(V)$ | diagonalizable M | B:134-136 | `$\norm{(I-M)^{-1}}_2\le\kappa(V)/(1-\rho(M))$, and the slack relative to the spectral-radius rate $1/(1-\rho(M))$ is exactly $\kappa(V)$.` |
| B16 | eta_band | $\eta\in[1.19,2.47]$; tracks $\kappa(V_W)$ | rem:eta_relu, general ReLU, suite | B:139-144 | `the measured nonnormality stays moderate, $\eta\in[1.19,2.47]$ across the suite (\cref{app:ablations}), tracking $\kappa(V_W)$ closely.` |
| B17 | radius_def | $r_v=\min_{c\ne y_v}m_v^{(c)}/\norm{(W_{y_v}-W_c)S_v}_2$ | prop:radius restated | B:150-152 | `the certified radius $r_v=\min_{c\ne y_v}m_v^{(c)}/\norm{(W_{y_v}-W_c)S_v}_2$, below which the prediction cannot change.` |
| B18 | margin_move | $\abs{\Delta m_v^{(c)}}\le\norm{(W_{y_v}-W_c)S_v}_2\norm{\delta\Ahat}_F$ | eq:margin-move, Cauchy–Schwarz | B:159-162 | `\abs{\Delta m_v^{(c)}}=\abs{(W_{y_v}-W_c)\,S_v\,\mathrm{vec}(\delta\Ahat)}\le\norm{(W_{y_v}-W_c)S_v}_2\,\norm{\delta\Ahat}_F` |
| B19 | radius_restate | $r_v=\min_{c\ne y_v}\frac{m_v^{(c)}}{\norm{(W_{y_v}-W_c)S_v}_2}$ | eq:radius-restate | B:167-170 | `r_v=\min_{c\ne y_v}\frac{m_v^{(c)}}{\norm{(W_{y_v}-W_c)S_v}_2}` |
| B20 | constrained_radius | substituting $S_{c,v}$ for $S_v$ tightens denominator, larger radius | constrained | B:176 | `substituting the constrained $S_{c,v}$ for $S_v$ tightens the denominator and yields the larger constrained radius.` |
| C01 | transfer_damage | $d_k=w_k v_k+R_k$, $\abs{R_k}\le L_J w_k^2/(2(1-\kappa)^2)$ | prop:transfer restated | C:20-21 | `the true removal damage of edge $k$ factors as $d_k=w_k v_k+R_k$ with $\abs{R_k}\le L_J w_k^2/\big(2(1-\kappa)^2\big)$` |
| C02 | ranking_condition | $w_{k_1}<\frac{v_{k_1}-v_{k_2}}{L_J/(1-\kappa)^2}$ | eq:ranking pairwise order preservation | C:24-27 | `w_{k_1}<\frac{v_{k_1}-v_{k_2}}{L_J/(1-\kappa)^2}` |
| C03 | vk_def | $\norm{\Delta\zstar}\approx w_k v_k$, $v_k=\norm{[S_c]_{:,k}}_2$ | first-order removal damage | C:33-35 | `the damage is $\norm{\Delta\zstar}\approx w_k v_k$ with $v_k=\norm{[S_c]_{:,k}}_2$.` |
| C04 | bilinear_second_partials | $F_{zz}=0,\ F_{AA}=0$ | eq:bilinear, on ReLU region | C:42-46 | `F_{zz}=\frac{\partial^2 F}{\partial z^2}=0,\qquad F_{AA}=\frac{\partial^2 F}{\partial\mathrm{vec}(A)^2}=0` |
| C05 | hessian_full | $\frac{\partial^2\zstar}{\partial\mathrm{vec}(A)^2}=(I-J_z)^{-1}(F_{zA}+F_{Az})(I-J_z)^{-1}J_A$ | eq:hessian-full, complete | C:50-54 | `\frac{\partial^2\zstar}{\partial\mathrm{vec}(A)^2}=(I-J_z)^{-1}\big(F_{zA}+F_{Az}\big)(I-J_z)^{-1}J_A` |
| C06 | LJ_bound | $L_J\le\norm{W}_2^2\norm{\zstar}$, $\norm{\zstar}\le\norm{X_{proj}}/(1-\kappa)$ | eq:LJ-bound | C:59-62 | `L_J\le\norm{W}_2^2\,\norm{\zstar},\qquad \norm{\zstar}\le\frac{\norm{X_{\mathrm{proj}}}}{1-\kappa}` |
| C07 | hessian_norm_bound | $\norm{\partial^2\zstar/\partial\mathrm{vec}(A)^2}_2\le(1-\kappa)^{-2}L_J$ | derived | C:64 | `yields $\norm{\partial^2\zstar/\partial\mathrm{vec}(A)^2}_2\le(1-\kappa)^{-2}L_J$.` |
| C08 | Rk_bound | $\abs{R_k}\le\tfrac12(1-\kappa)^{-2}L_J w_k^2$ | eq:Rk-bound, complete remainder | C:68-71 | `\abs{R_k}\le\tfrac12\,(1-\kappa)^{-2}L_J\,w_k^2` |
| C09 | C_transfer_def | $C:=L_J/(2(1-\kappa)^2)$ | Step 3 ranking constant (NB reuse of symbol C) | C:77 | `With $d_k=w_k v_k+R_k$ from Steps 1--2 and $C:=L_J/\big(2(1-\kappa)^2\big)$` |
| C10 | rank_gap | $d_{k_1}-d_{k_2}\ge(w_{k_1}v_{k_1}-w_{k_2}v_{k_2})-C(w_{k_1}^2+w_{k_2}^2)$ | eq:rank-gap | C:78-81 | `d_{k_1}-d_{k_2}\ \ge\ \big(w_{k_1}v_{k_1}-w_{k_2}v_{k_2}\big)-C\big(w_{k_1}^2+w_{k_2}^2\big)` |
| C11 | tightness_ratio_kendall | $\abs{R_k}$ runs $2$–$10\times$ below $L_J w_k^2$; Kendall $\tau=+0.996$ | Amazon Photo full-graph | C:85-88 | `empirically $\abs{R_k}$ runs $2$--$10\times$ below $L_J w_k^2$, and the edge-weighted score reaches Kendall $\tau=+0.996$ against brute-force single-edge removal on full-graph Amazon Photo` |
| C12 | unrolled_SK | $S_K=\sum_{l=1}^{K}(\prod_{k=l+1}^{K}J_z^{(k)})J_A^{(l)}$ | eq:unrolled-restate, prop:explicit | C:105-110 | `S_K=\frac{\partial\,\mathrm{vec}(Z_K)}{\partial\,\mathrm{vec}(A)}=\sum_{l=1}^{K}\Big(\prod_{k=l+1}^{K}J_z^{(k)}\Big)J_A^{(l)}` |
| C13 | explicit_limit | $\sigma_1(S_K)\le\norm{J_A}_2\frac{1-\kappa^K}{1-\kappa}\to\frac{\norm{J_A}_2}{1-\kappa}$ | eq:explicit-limit, weight-tied | C:120-124 | `\sigma_1(S_K)\le\norm{J_A}_2\sum_{j=0}^{K-1}\kappa^{\,j}=\norm{J_A}_2\,\frac{1-\kappa^{K}}{1-\kappa}\xrightarrow[K\to\infty]{}\ \frac{\norm{J_A}_2}{1-\kappa}` |
| C14 | explicit_tightness | explicit models match implicit tightness $0.99$–$1.02$ | suite, app:explicit | C:130-131 | `Empirically the explicit models match the implicit tightness ($0.99$--$1.02$ across the suite, \cref{app:explicit}).` |
| D01 | ecrit_def_D | $\ecrit=\frac{1-\kappa}{\norm{W}_2}=\frac{1}{\norm{W}_2}-\norm{\Ahat}_2$ | bracket budgets | D:24-25 | `\ecrit&=\frac{1-\kappa}{\norm{W}_2}=\frac{1}{\norm{W}_2}-\norm{\Ahat}_2` |
| D02 | espec_def_D | $\espec=\frac{1}{\rho(W)}-\rho(\Ahat)$ | bracket budgets | D:26 | `\espec&=\frac{1}{\rho(W)}-\rho(\Ahat)` |
| D03 | additive_gap | $\espec-\ecrit=(\frac{1}{\rho(W)}-\frac{1}{\norm{W}_2})+(\norm{\Ahat}_2-\rho(\Ahat))\ge0$ | eq:additive-gap | D:32-34 | `\espec-\ecrit=\Big(\tfrac{1}{\rho(W)}-\tfrac{1}{\norm{W}_2}\Big)+\big(\norm{\Ahat}_2-\rho(\Ahat)\big)\ \ge\ 0` |
| D04 | gW_def_D | $\gW:=\norm{W}_2/\rho(W)\ge1$ | nonnormality, 2nd bracket vanishes (symmetric) | D:36-37 | `leaving only the term driven by $\gW:=\norm{W}_2/\rho(W)\ge1$.` |
| D05 | gW_band | $\gW\in[1.19,2.47]$; a-posteriori cert $\gW\le\kappa_2(V_W)$ | suite | D:41-42 | `we measure $\gW$ on each trained model ($\gW\in[1.19,2.47]$ on the suite, with the a-posteriori certificate $\gW\le\kappa_2(V_W)$, the eigenvector conditioning of $W$)` |
| D06 | beta_def_D | $\beta:=\langle u_1,Bu_1\rangle=\norm{g_E}_F^2/\sigma_E=\sigma_E$ | eq:beta-eq, alignment | D:54-58 | `\beta:=\langle u_1,Bu_1\rangle=\frac{\langle u_1,P_E(u_1u_1^\top)u_1\rangle}{\sigma_E}=\frac{\norm{g_E}_F^2}{\sigma_E}=\sigma_E` |
| D07 | sigma_E_def | $g_E:=P_E(u_1u_1^\top)$, $\sigma_E:=\norm{g_E}_F$, $B:=g_E/\sigma_E$ | edge-feasible direction | D:50-53 | `set $g_E:=P_E(u_1u_1^\top)$, $\sigma_E:=\norm{g_E}_F$, and the unit feasible direction $B:=g_E/\sigma_E$` |
| D08 | beta_positive | $\beta>0$ is observable not assumption (connected graph) | hypothesis status | D:59-63 | `So the alignment $\beta$ equals the edge-supported Frobenius mass $\sigma_E$ ... The hypothesis $\beta>0$ is therefore an observable, not an assumption.` |
| D09 | bracket_lower_side | $\ebreak\ge\eglob$; under (A4) $\eglob=\ecrit$ so $\ebreakall\ge\ecrit$ | thm:cf2s_full (i), any 1-Lip $\phi$ | D:73-77 | `the perturbed operator is an $\norm{\cdot}_2$-contraction ... $\ebreak\ge\eglob$. Under (A4), $\kappa=\norm{\Ahat}_2\norm{W}_2$ gives $\eglob=\ecrit$, so $\ebreak\ge\ecrit$ and in particular $\ebreakall\ge\ecrit$.` |
| D10 | bracket_upper_side | $\ebreakall\le\espec/\beta$; extremizer $(\espec/\beta)B$; $\beta=1\Rightarrow\ebreakall=\espec$ | thm:cf2s_full (ii), all-active (A4) | D:79-83 | `The feasible rank-one $\delta\Ahat^\star=(\espec/\beta)\,B\in\mathcal S_E$ ... drives the all-active spectral radius to $1$, so $\ebreakall\le\espec/\beta$. In the unconstrained-symmetric threat model ($P_E=I$, $\beta=1$) it is exact: $\ebreakall=\espec$` |
| D11 | bracket_constant_C | $C:=\gW\frac{1+\kappa}{1-\kappa}=\frac{\norm{W}_2}{\rho(W)}\cdot\frac{1+\norm{\Ahat}_2\norm{W}_2}{1-\norm{\Ahat}_2\norm{W}_2}$ | thm:cf2s_full (iii) | D:85-90 | `C:=\gW\,\frac{1+\kappa}{1-\kappa}=\frac{\norm{W}_2}{\rho(W)}\cdot\frac{1+\norm{\Ahat}_2\norm{W}_2}{1-\norm{\Ahat}_2\norm{W}_2}` |
| D12 | bracket_inequality | $\ecrit\le\ebreakall\le\frac{C}{\beta}\ecrit$ | eq:bracket_full (boxed) | D:91-95 | `\boxed{\ \ecrit\ \le\ \ebreakall\ \le\ \tfrac{C}{\beta}\,\ecrit\ }` |
| D13 | bracket_suite_value | $\beta\approx0.62$, $C/\beta\lesssim16$; $\beta$-free $\ecrit\le\ebreakall\le C\ecrit$ at $\beta=1$ | suite | D:96-98 | `on the suite $\beta\approx0.62$, giving $C/\beta\lesssim16$. The $\beta$-free form $\ecrit\le\ebreakall\le C\,\ecrit$ holds exactly for the unconstrained-symmetric model ($\beta=1$).` |
| D14 | equality_a | $\ecrit=\espec$ iff $\gW=1$ ($W$ normal), indep $\kappa,\beta$ | thm:cf2s_full (iv)(a) | D:100-101 | `(a) the endpoints coincide, $\ecrit=\espec$, iff $\gW=1$ ($W$ normal), independent of $\kappa,\beta$;` |
| D15 | equality_b | $\ecrit=\ebreakall=\espec$ iff $\gW=1$ and $\beta=1$ | thm:cf2s_full (iv)(b) | D:102 | `(b) all three coincide, $\ecrit=\ebreakall=\espec$, iff $\gW=1$ and $\beta=1$;` |
| D16 | equality_c | $C/\beta=1$ iff additionally $\kappa\to0^+$ | thm:cf2s_full (iv)(c) | D:103 | `(c) the slack constant is one, $C/\beta=1$, iff additionally $\kappa\to0^+$.` |
| D17 | weyl_inequality | $\rho(\Ahat')\le\rho(\Ahat)+\norm{\delta\Ahat}_2$, $\norm{\Ahat'}_2\le\norm{\Ahat}_2+\norm{\delta\Ahat}_2$ | proof setup | D:108-111 | `We use Weyl's inequality, $\rho(\Ahat')\le\rho(\Ahat)+\norm{\delta\Ahat}_2$ and $\norm{\Ahat'}_2\le\norm{\Ahat}_2+\norm{\delta\Ahat}_2$` |
| D18 | convexity_lower | $\rho(\Ahat+tB)\ge\rho(\Ahat)+\beta t$ for all $t\ge0$ | proof (ii), convex tangent | D:128-131 | `$\rho(\Ahat+tB)\ge\rho(\Ahat)+\beta t$ for all $t\ge0$ (this lower direction is what convexity buys; a concave version fails on every random instance).` |
| D19 | norm_A_in_ecrit | $\norm{\Ahat}_2=\frac{\kappa}{1-\kappa}\ecrit$ | proof (iii) identity | D:137-138 | `$\norm{\Ahat}_2=\kappa/\norm{W}_2=\kappa(\ecrit+\norm{\Ahat}_2)$ gives $\norm{\Ahat}_2=\tfrac{\kappa}{1-\kappa}\ecrit$` |
| D20 | espec_le_C_ecrit | $\espec\le\gW\frac{1}{1-\kappa}\ecrit\le\gW\frac{1+\kappa}{1-\kappa}\ecrit=C\ecrit$ | proof (iii) closing | D:138-139 | `\espec\le\gW\tfrac{1}{1-\kappa}\ecrit\le\gW\tfrac{1+\kappa}{1-\kappa}\ecrit=C\,\ecrit` |
| D21 | recovers_sharpens | "recovers and sharpens"; old result is lower side (i) | STALENESS/VERSIONING signal | D:148-149 | `This recovers and sharpens \cref{thm:phase_transition}: the old result is the lower side (i), part (ii) supplies the matching all-active upper side` |
| D22 | enclosure_suite | $C/\beta$ reaching $10$–$16\times$; random $\espec/\ecrit\in[1.02,18.2]$, median $2.25$ | suite, random instances | D:150-152 | `with enclosure constant $C/\beta$ reaching $10$--$16\times$ on the suite (random-instance $\espec/\ecrit\in[1.02,18.2]$, median $2.25$).` |
| D23 | all_active_spectral_law | $\rho(J_z')=\rho(\Ahat')\rho(W)$ (A4 only) | numerically exact identity | D:153-155 | `the all-active spectral law $\rho(J_z')=\rho(\Ahat')\rho(W)$ (A4 only), and the additive gap \eqref{eq:additive-gap}` |
| D24 | tightness_eps_operational | ratio $<1.16$ for $\varepsilon\le0.10$; citation reach $1.36$–$1.39$ at $\varepsilon=0.20$; product $\le1.07$ | fig:tightness_eps, 10 seeds, full graph | D:170-177 | `Within the operational regime ($\varepsilon\le0.10$) every curve stays below $1.16$; only at $\varepsilon=0.20$ do citation graphs (solid) reach $1.36$--$1.39$, while product graphs (dashed) stay within $1.07$.` |
| D25 | tightness_ranking_vs_shift | ranking tightness $1.00$–$1.02$ distinct from shift-prediction ratio | fig caption disambiguation | D:174-176 | `This shift-prediction ratio is distinct from the ranking tightness quoted elsewhere ($1.00$--$1.02$, \cref{app:explicit,app:ablations}), which is measured at the operational subgraph scale` |
| D26 | ereach_k05 | $\ereach/\espec=1.41\pm0.12$, $\ereach/\ecrit=2.17\pm0.18$ | rem:obs_o1, $\kappa_0=0.5$, 50-node ego, 10 seeds | D:186-189 | `at $\kappa_0=0.5$, $\ereach/\espec=1.41\pm0.12$ and $\ereach/\ecrit=2.17\pm0.18$` |
| D27 | ereach_k09 | $\ereach/\espec=1.51\pm0.18$, $\ereach/\ecrit=8.72\pm1.03$ | rem:obs_o1, $\kappa_0=0.9$, 10 seeds | D:188-189 | `at $\kappa_0=0.9$, $\ereach/\espec=1.51\pm0.18$ and $\ereach/\ecrit=8.72\pm1.03$` |
| D28 | ereach_model | $\ereach\approx\espec/a$, $a\approx0.6$, $1/a\approx1.5$–$1.7$; envelope $\ereach/\ecrit\lesssim C/a\in[2,10]$ | rem:obs_o1 CONJECTURE, mean-field | D:189-192 | `Empirically $\ereach\approx\espec/a$ with $a$ the ReLU active fraction: $a\approx0.6$ gives $1/a\approx1.5$--$1.7$ ... the observed envelope $\ereach/\ecrit\lesssim C/a$, spanning roughly $[2,10]$.` |
| D29 | obs_o1_gaps | conjecture; two open proof gaps; $\gamma\approx1.02$–$1.06$; $\ereach$ runs $5$–$11\times$ largest budget | rem:obs_o1 status, 10 seeds | D:194-205 | `\emph{This is a conjecture; two independent proof gaps remain open.} ... critical exponent $\gamma\approx1.02$--$1.06$ ... $\ereach$ runs $5$--$11\times$ the largest realistic budget ($\varepsilon\le0.2$)` |
| D30 | constants_kappa | $\kappa=\norm{J_z}_2$ trained contractivity ($<1$) | tab:constants row 1 | D:218 | `$\kappa=\norm{J_z}_2$ & trained contractivity ($<1$) & \cref{tab:cross_domain}` |
| D31 | constants_ecrit | $\ecrit=\frac{1-\kappa}{\norm{W}_2}$ norm certificate (safe radius) | tab:constants row 2 | D:219 | `$\ecrit=\tfrac{1-\kappa}{\norm{W}_2}$ & norm certificate (safe radius) & \cref{thm:phase_transition}` |
| D32 | constants_espec | $\espec$ all-active spectral break budget | tab:constants row 3 | D:220 | `$\espec$ & all-active spectral break budget & \cref{thm:cf2s}` |
| D33 | constants_op_margin | $\mathbf{2}$–$\mathbf{4\times}$ operating $\rho$-margin to $\rho=1$ | tab:constants row 4 | D:221 | `$\mathbf{2}$--$\mathbf{4\times}$ & operating $\rho$-margin to $\rho=1$ & \cref{app:phase_scal}` |
| D34 | constants_conservatism | $\mathbf{2}$–$\mathbf{9\times}$ empirical $\ereach/\ecrit$ (10 seeds) | tab:constants row 5 | D:222 | `$\mathbf{2}$--$\mathbf{9\times}$ & empirical $\ereach/\ecrit$ (10 seeds) & \cref{rem:obs_o1}` |
| D35 | constants_bracket | $\mathbf{10}$–$\mathbf{16\times}$ proven bracket constant $C/\beta$ | tab:constants row 6 | D:223 | `$\mathbf{10}$--$\mathbf{16\times}$ & proven bracket constant $C/\beta$ & \cref{thm:cf2s}` |
| D36 | constants_gamma | $\gamma\approx1.02$ resolvent critical exponent | tab:constants row 7 | D:224 | `$\gamma\approx1.02$ & resolvent critical exponent & \cref{rem:obs_o1}` |
| D37 | constants_gW | $\gW\in[1.19,2.47]$ nonnormality $\norm{W}_2/\rho(W)$ | tab:constants row 8 | D:225 | `$\gW\in[1.19,2.47]$ & nonnormality $\norm{W}_2/\rho(W)$ & \cref{thm:cf2s}` |
| E01 | conformal_eq | shift bound $L_1^{c}\varepsilon+C_v\varepsilon^2$; certified regime $\varepsilon<\ecrit$ | thm:robust-cov target, eq:conformal | E:12-20 | `a worst-case conformity-score-shift bound $L_1^{c}\varepsilon+C_v\varepsilon^2$ and the robust-coverage guarantee it implies. ... We use (A1)--(A3) and operate in the certified regime $\varepsilon<\ecrit$.` |
| E02 | ift_firstorder | $\Delta\zstar_v=S_{c,v}\boldsymbol\delta+R_v$, $\norm{\boldsymbol\delta}_2=\norm{\delta\Ahat}_F\le\varepsilon$ | eq:ift-firstorder | E:28-32 | `\Delta\zstar_v=S_{c,v}\,\boldsymbol\delta+R_v,\qquad \norm{\boldsymbol\delta}_2=\norm{\delta\Ahat}_F\le\varepsilon` |
| E03 | tps_score | $\score^{\mathrm{TPS}}_r(v)=\pi_r$ | def:conf-scores | E:42-43 | `\text{TPS:}\quad&\score^{\mathrm{TPS}}_r(v)=\pi_r` |
| E04 | aps_score | $\score^{\mathrm{APS}}_r(v)=1-(\rho_r+u_v\pi_r)$, $\rho_r:=\sum_{c:\pi_c>\pi_r}\pi_c$ | def:conf-scores, eq:aps | E:44-47 | `\score^{\mathrm{APS}}_r(v)=1-\big(\rho_r+u_v\,\pi_r\big) ... \rho_r:=\sum_{c:\,\pi_c>\pi_r}\pi_c` |
| E05 | tie_break | $u_v\sim\mathrm{Unif}[0,1]$ shared cal/test, cancels under exchangeability | def:conf-scores | E:48-49 | `$u_v\sim\mathrm{Unif}[0,1]$ a per-node tie-break drawn once and shared between calibration and test, so it cancels under exchangeability.` |
| E06 | L1_def | $L_1^{(c)}:=\norm{(W_r-W_c)S_{c,v}}_2$ | eq:L1def | E:51-53 | `L_1^{(c)}&:=\norm{(W_r-W_c)\,S_{c,v}}_2` |
| E07 | Cv_def | $C_v:=\norm{W_r-W_c}_2\cdot\frac{L_{J,v}}{2(1-\kappa)^2}$, $L_{J,v}\le\norm{W}_2^2\norm{\zstar}$ | eq:Cvdef | E:54-57 | `C_v&:=\norm{W_r-W_c}_2\cdot\frac{L_{J,v}}{2(1-\kappa)^2}` |
| E08 | aggregate_constants | $L_1^{c}=\max_{c\ne r}L_1^{(c)}$, $C_v$ with worst $\norm{W_r-W_c}_2$ | def:conf-scores aggregate | E:60-61 | `The aggregate node constant ... takes the competitor-worst values, $L_1^{c}=\max_{c\ne r}L_1^{(c)}$ and $C_v$ with the worst $\norm{W_r-W_c}_2$.` |
| E09 | no_quarter_slack | margin form avoids $1/4$ softmax-Lipschitz slack | rem:margin-not-grad | E:64-71 | `the margin form needs no bound on softmax curvature: it yields a sound, one-sided score drop without the $1/4$ softmax-Lipschitz slack.` |
| E10 | margin_shift | $\abs{g_c(\Ahat+\delta\Ahat)-g_c(\Ahat)}\le L_1^{(c)}\varepsilon+C_v\varepsilon^2$ | lem:score-shift eq:margin-shift | E:74-81 | `\big|g_c(\Ahat+\delta\Ahat)-g_c(\Ahat)\big|\le L_1^{(c)}\varepsilon+C_v\varepsilon^2` |
| E11 | score_drop | $\score_r(v;\Ahat)-\score_r(v;\Ahat+\delta\Ahat)\le\Delta_r(\varepsilon)$; headline $\le L_1^{c}\varepsilon+C_v\varepsilon^2$ | lem:score-shift eq:score-drop | E:84-90 | `\Delta_r(\varepsilon):=\Psi_r\!\big(\{L_1^{(c)}\varepsilon+C_v\varepsilon^2\}_{c\ne r}\big) ... the headline $\Delta_r(\varepsilon)\le L_1^{c}\varepsilon+C_v\varepsilon^2$.` |
| E12 | Rv_bound | $\norm{R_v}_2\le\tfrac12(1-\kappa)^{-2}L_{J,v}\varepsilon^2$ | lem:score-shift proof Step 2 | E:108-115 | `bounded in operator norm by $(1-\kappa)^{-2}L_{J,v}$ ... gives $\norm{R_v}_2\le\tfrac12(1-\kappa)^{-2}L_{J,v}\,\varepsilon^2$, hence $\abs{(W_r-W_c)R_v}\le C_v\varepsilon^2$` |
| E13 | conformal_gate_value | gate conservative $0.92$–$0.98$ at $\varepsilon=0.05$ | empirical, conformal | E:144-146 | `$L_1^{c}\varepsilon+C_v\varepsilon^2$ over-states the true drop for every $\varepsilon<\ecrit$, which is why the empirical gate is conservative ($0.92$--$0.98$ at $\varepsilon=0.05$) rather than tight.` |
| E14 | conf_kappa_subgraph | conformal subgraph $\kappa\approx0.68$, $(1-\kappa)^{-2}\approx9.8$ | rem:conf-caveats | E:153-156 | `On the conformal subgraph $\kappa\approx0.68$, so $(1-\kappa)^{-2}\approx9.8$ inflates $C_v$, yet the gate confirms the bound is not breached.` |
| E15 | conformal_exchangeability | (C1) clean true-label scores exchangeable (inductive/transductive perm-invariant) | thm:robust-cov (C1) | E:167-169 | `(C1) \emph{Exchangeability.} The clean true-label scores ... are exchangeable (an inductive split, or a transductive split with a permutation-invariant predictor;` |
| E16 | conformal_C2 | $\score_{y_w}(w;\Ahat+\delta\Ahat)\ge\score_{y_w}(w;\Ahat)-\Delta_{y_w}(\varepsilon)$ | thm:robust-cov (C2) | E:170-173 | `(C2) \emph{Score-shift bound.} ... $\score_{y_w}(w;\Ahat+\delta\Ahat)\ge\score_{y_w}(w;\Ahat)-\Delta_{y_w}(\varepsilon)$` |
| E17 | conformal_finite_sample | $\hat q_{rob}=\mathrm{Quantile}_{\lceil(n_{cal}+1)(1-\alpha)\rceil/n_{cal}}$ of lowered cal scores | eq:qrob, finite-sample correction | E:176-180 | `\hat q_{\mathrm{rob}}=\mathrm{Quantile}_{\lceil(n_{\mathrm{cal}}+1)(1-\alpha)\rceil/n_{\mathrm{cal}}}\Big(\big\{\score_{y_{v_i}}(v_i;\Ahat)-\Delta_{y_{v_i}}(\varepsilon)\big\}_{i\in\mathrm{cal}}\Big)` |
| E18 | conformal_coverage | $\Pr[y_v\in\Cset_\varepsilon(v;\Ahat+\delta\Ahat)]\ge1-\alpha$ for all feasible $\norm{\delta\Ahat}_F\le\varepsilon$ | thm:robust-cov eq:robcov | E:182-185 | `\Pr\!\big[y_v\in\Cset_\varepsilon(v;\Ahat+\delta\Ahat)\big]\ge1-\alpha \text{ for all feasible }\norm{\delta\Ahat}_F\le\varepsilon` |
| E19 | conformal_eps0_reduce | at $\varepsilon=0$, $\Delta\equiv0$, reduces to ordinary split-conformal $1-\alpha$ | proof Step 4 | E:215-216 | `At $\varepsilon=0$, $\Delta\equiv0$ and the statement reduces to the ordinary split-conformal $1-\alpha$ guarantee, as it must.` |
| E20 | conformal_clean_cov | clean coverage $\approx0.90$ | rem:exchange-honesty empirical | E:233-234 | `with the empirical coverage (clean coverage $\approx0.90$)` |
| E21 | conformal_gate_full | gate $0.90$ at $\varepsilon=0.01$; $0.92$–$0.98$ at $\varepsilon=0.05$; zero divergence across $4138$ gate nodes | rem:exchange-honesty | E:234-236 | `the worst-case gate ($0.90$ at $\varepsilon=0.01$, $0.92$--$0.98$ at $\varepsilon=0.05$, zero equilibrium divergence across all $4138$ gate nodes)` |
| E22 | conformal_divergence | zero equilibrium divergence across all 4138 gate nodes | rem:exchange-honesty | E:236 | `zero equilibrium divergence across all $4138$ gate nodes` |
| E23 | conformal_zero_mc | reduction to zargarbashi2023conformal faithful; computed analytically with zero Monte-Carlo smoothing | rem:exchange-honesty | E:236-239 | `\cref{lem:score-shift} is exactly such an envelope, computed analytically with zero Monte-Carlo smoothing.` |

---

## A. INTERNAL INCONSISTENCIES (within A–E)

1. **`tab:constants` bracket cell vs the boxed bracket inequality (numerical mismatch).**
   The boxed bracket is `ε_crit ≤ ε_break,all ≤ (C/β)·ε_crit` (D12, D:91-95). `tab:constants` row 6 (D35, D:223) labels the **proven bracket constant `C/β`** as **"10–16×"**. But the *constant* `C = g_W(1+κ)/(1−κ)` evaluated at the suite's own audited numbers does NOT reach 10–16 by itself: with `g_W∈[1.19,2.47]` (D05/D37) and `κ∈[0.14,0.59]` (A21), `C` ranges roughly `1.19·(1.14/0.86)≈1.58` up to `2.47·(1.59/0.41)≈9.6`; dividing by `β≈0.62` (D13) gives `C/β` up to `≈15.5`. So "10–16×" is only reachable at the *joint worst corner* (max g_W AND max κ AND the small β). The text's own headline "`C/β ≲ 16`" (D13) and "reaching 10–16×" (D22) are upper-corner values, yet the table presents "10–16×" as **the** bracket range without the "≲ / reaching" qualifier — a range-vs-worst-case framing inconsistency the parent should flag against any body claim that quotes a *typical* bracket factor.

2. **Symbol `C` overloaded with two different definitions inside the appendix.**
   - In A15 (A:63), D11 (D:85-90), D31, A:64-neighbourhood: `C` = **bracket constant** `g_W(1+κ)/(1−κ)`.
   - In C09 (C:77): `C := L_J/(2(1−κ)²)` — the **ranking/curvature constant**.
   These are unrelated quantities sharing one glyph. Within C_rankings the local `C` is self-consistent (used only in eq:rank-gap C10), but a cross-file reader who carries the `tab:notation` meaning of `C` (bracket constant) into Appendix C will misread eq:rank-gap. This is an internal naming collision, not a value error. (Note: E_conformal avoids the collision by writing the curvature constant as `L_J/(2(1−κ)²)` and naming the per-node version `C_v` instead — so the same object has three notations across the appendix: `C` in C, `L_J/(2(1−κ)²)` inline, and `C_v` scaled by the readout gap in E.)

3. **`espec/ecrit` median (2.25) vs `tab:constants` "10–16×" framing.**
   D22 (D:152) reports random-instance `espec/ecrit ∈ [1.02, 18.2]`, **median 2.25**. The bracket *constant* `C/β` is the worst-case enclosure, so a median ratio of 2.25 with the table advertising "10–16×" is internally consistent ONLY because the table cell is explicitly "worst-case enclosure constant" (caption D:209-212). But the proximity of three different "×" ranges in one table — op-margin 2–4× (D33), conservatism 2–9× (D34), bracket 10–16× (D35) — plus a 4th and 5th in prose (`ereach/ecrit ≲ C/a ∈ [2,10]` D28; `ereach` runs 5–11× largest budget D29) is a documented collision risk the caption tries to pre-empt ("distinct quantities, not estimates of one number"). Parent should verify the body never conflates these.

4. **`ε_reach/ε_crit` upper value: 8.72 (D27) vs table "2–9×" (D34) vs prose "[2,10]" (D28) vs "5–11×" (D29).**
   Four numbers for the same family of ratios appear:
   - D27: `ereach/ecrit = 8.72±1.03` at κ₀=0.9 (so up to ≈9.75 with the ±).
   - D34 (table): "2–9×" empirical `ereach/ecrit`.
   - D28: envelope `ereach/ecrit ≲ C/a` "spanning roughly [2,10]".
   - D29: "`ereach` runs 5–11× the largest realistic budget (ε≤0.2)" — NB this is `ereach`/budget, **not** `ereach/ecrit`.
   The table "2–9×" rounds the 8.72±1.03 down (9.75 → 9), while prose says "[2,10]". Mild rounding inconsistency (9 vs 10) between D34 and D28; the "5–11×" of D29 is a *different* denominator (the 0.2 budget, not ecrit) and must not be cross-matched to the others.

5. **`κ` operating band differs by sub-experiment — not contradictory but easy to mis-cross-check.**
   - Suite/standing assumption (A3): `κ∈[0.14,0.59]` (A21, A:102).
   - Conformal subgraph: `κ≈0.68` (E14, E:155).
   - obs:O1 / nonlinear-break experiment uses imposed operating points `κ₀=0.5` and `κ₀=0.9` (D26-27).
   `κ≈0.68` and `κ₀=0.9` both exceed the suite ceiling 0.59. This is internally consistent *if* the conformal subgraph and the O1 ego-subgraph are genuinely different objects from "the suite" (the text says so — conformal subgraph E:155, 50-node ego subgraph D:188), but it is a landmine: any body sentence that says "κ < 0.59 everywhere" or "κ ≤ 0.6" would contradict E14 and D27. Flag for parent.

6. **`η` band and `g_W` band are numerically identical `[1.19, 2.47]` (B16 vs D05/D37) — intended, verify it is not a copy-paste artifact.**
   `η∈[1.19,2.47]` (B16, B:143) and `g_W∈[1.19,2.47]` (D05, D:41; D37, D:225) share the exact same interval. The theory *predicts* a link (obs:eta_bound: `η ≤ κ(V_W)`; and `g_W ≤ κ₂(V_W)` D05), and B:143 says η "tracks κ(V_W) closely", so equal bands are plausible. But two distinct constants (Neumann-to-spectral slack η vs nonnormality ratio g_W = ‖W‖₂/ρ(W)) carrying byte-identical `[1.19,2.47]` is a classic stale-number signature. Parent should confirm both were measured, not one copied to the other.

7. **`espec` is defined as a budget but Appendix D never explicitly restricts it to be > 0 in the way ecrit is.**
   `ass:tight-v2` and D:17 require `ecrit > 0`. `espec = 1/ρ(W) − ρ(Â)` (D02). The additive-gap identity D03 gives `espec − ecrit ≥ 0`, hence `espec ≥ ecrit > 0` is implied, so this is consistent — but only via the gap lemma, never stated as a standing hypothesis. Minor; not an error.

---

## B. STALENESS / VERSIONING SIGNALS

1. **"recovers and sharpens" + "the old result is the lower side" (D21, D:148-149).** Exact quote: *"This recovers and sharpens \cref{thm:phase_transition}: the old result is the lower side (i), part (ii) supplies the matching all-active upper side, and (iv) separates the three equality conditions."* This is the explicit versioning marker the task flagged. It signals `thm:phase_transition` (body) is the **prior/lower-side-only** statement and the bracket (thm:cf2s / thm:cf2s_full) supersedes/extends it. **Parent must check the body's `thm:phase_transition` and any body capability table did not retain a one-sided claim that the appendix now calls "old".**

2. **`tab:notation` symbol `\espec` "First used" points to `\cref{thm:cf2s}` (A:57), but the appendix's full theorem is `thm:cf2s_full` (D:69).** Several notation rows and `tab:constants` cite `thm:cf2s` (A:57 espec, A:58 ebreak, A:60 gW, A:62 beta, A:63 C; D:220, D:223, D:225) while the actual full statement is labeled `thm:cf2s_full`. `thm:cf2s` is presumably the body's short version. Not wrong, but the parent should confirm body `thm:cf2s` and appendix `thm:cf2s_full` agree on C, β, and the bracket direction (the appendix is the authoritative long form).

3. **`obs:eta_bound` proves `η ≤ κ(V_W)` (eigenvector conditioning) while `g_W ≤ κ₂(V_W)` is asserted in D (D05).** Two different certificates (`η ≤ κ(V_W)` B14; `g_W ≤ κ₂(V_W)` D05) both invoke the eigenvector conditioning of W under two notations `κ(V_W)` (B) vs `κ₂(V_W)` (D). Likely the same object; the subscript-2 vs unsubscripted inconsistency is a leftover-notation signal. The matching numeric bands (item A6) suggest they may even be the same measured number wearing two names.

4. **`L_J` appears with three notations / scopes:** `L_J ≤ ‖W‖₂²‖z*‖` (A16, C06), per-node `L_{J,v} ≤ ‖W‖₂²‖z*‖` (E07, E:57), and the curvature constant alternately written `C` (C09) and `C_v` (E07). The global-vs-per-node distinction (`L_J` vs `L_{J,v}`) is real and intended, but `tab:notation` only lists `L_J` (A:64); `L_{J,v}` is introduced only in E without a notation-table entry — a dangling/under-documented symbol.

5. **`tab:notation` defines `\espec` as "all-active spectral break" and `\ebreak,\ebreakall` as "deployed / all-active contraction-break budget" (A:57-58), but the proof uses `ε_break,all` exclusively for the proven upper side; `ε_break` (deployed, ReLU) is bounded only from BELOW (`ε_break ≥ ε_glob`, D09) and never from above.** So the "deployed" upper budget promised by the symbol `ε_break` is never delivered as a two-sided result — it is the conjecture `ereach` (rem:obs_o1) instead. Symbol `ε_break` (as distinct from `ε_break,all`) is effectively a placeholder whose only theorem-grade use is the lower bound. Potential stale/aspirational symbol.

6. **`\ereach` "First used \cref{rem:obs_o1}" (A:59) is explicitly a CONJECTURE (D29, D:194), yet it sits in the main notation table beside proven budgets** with no marker that it is non-theorem. A reader pulling `ereach` from `tab:notation` gets no signal it is unproven until D. Versioning/honesty-scope risk.

7. **Possible stale glyph in `tab:notation`:** `$f=W\zstar_v{+}b$` row (A:67) writes node logits; E re-derives `f=Wz*_v+b` (E:23) consistently. No conflict, but `\score^{TPS}, \score^{APS}` "First used `\cref{def:conf-scores}`" (A:70) and `C_v` appears in `L_1^{(c)}, C_v` row (A:72) as "first-order and curvature score-shift constants" — `tab:notation` therefore DOES list `C_v` (A:72) but NOT `L_{J,v}`; confirms item B4's dangling `L_{J,v}`.

8. **`η` is named "Neumann-to-spectral slack (pseudospectral index)" in `tab:notation` (A:61) but the proofs only ever use it as the resolvent-norm/spectral-radius-rate slack bounded by `κ(V_W)` (B14, B15).** "Pseudospectral index" is never otherwise defined or used — a notation-table term with no operational appearance downstream (candidate leftover terminology).

---

## C. CONSTANT DICTIONARY (canonical definition + location)

| symbol | canonical definition | defined at | notes / aliases |
|--------|----------------------|------------|-----------------|
| `Â` (Ahat) | `D^{-1/2}(A+I)D^{-1/2}`, symmetric normalized adjacency; `‖Â‖₂=1` for self-loop normalization | A:37; B:72 | `ρ(Â)=‖Â‖₂` when symmetric (D:135) |
| `W` | tied layer weight; `‖W‖₂ ≤ c` (A2) | A:38; A:98 | numerator of g_W |
| `φ, diag(φ')` | 1-Lipschitz activation + derivative mask; `‖diag(φ')‖₂ ≤ 1` | A:39; A:95 | (A1) |
| `z*` | equilibrium fixed point `z*=F_θ(z*,Â)`; `‖z*‖ ≤ ‖X_proj‖/(1−κ)` | A:40; A:93; C:60 | |
| `J_z` | state Jacobian `diag(φ')(Â⊗W)`; all-active `J_z=Â⊗W` | A:41; A:117 | |
| `J_A` | structural Jacobian `∂F/∂vec(A) ∝ z*W^T`; `J_A ∝ ‖W‖₂‖z*‖` | A:42; C:57 | |
| **`κ`** | **contraction factor `κ := ‖J_z‖₂ < 1`**; all-active `κ = ‖Â‖₂‖W‖₂` | **A:43; A:101** | bands: suite `[0.14,0.59]` (A:102); conformal subgraph `≈0.68` (E:155); O1 imposes `κ₀∈{0.5,0.9}` (D:188) |
| `S` | structural sensitivity `(I−J_z)^{-1}J_A` | A:46; A:123 (eq:S-def) | |
| `S_c` | edge-constrained sensitivity `(I−J_z)^{-1}J_A P_c` | A:47; A:129 | `S_{c,v}` = block rows at node v (A:49) |
| `S_K` | unrolled K-layer sensitivity `Σ_{l=1}^K (Π_{k=l+1}^K J_z^{(k)}) J_A^{(l)}` | C:106 (eq:unrolled-restate) | →`‖J_A‖₂/(1−κ)` as K→∞ (C:122) |
| `P_c, P_E` | projections onto symmetric / edge-supported perturbations; `S_E={M=M^T: M_{ij}=0 if (i,j)∉E∪diag}` | A:48; D:50-51 | |
| `σ₁(·)` | leading singular value; `σ₁(S) ≤ ‖J_A‖₂/(1−κ)` | A:50; B:47 | |
| **`ε_glob`** | mask-free safe radius `max(0, 1/‖W‖₂ − ‖Â‖₂)`; normalized `=1/c−1` | **A:55; B:65** | rigorous for ANY mask; `ε_break ≥ ε_glob` (D:118) |
| **`ε_crit`** | norm certificate `(1−κ)/‖W‖₂ = 1/‖W‖₂ − ‖Â‖₂`; `ε_crit ≥ ε_glob`; `=ε_glob` only under A4 | **A:56; B:22; D:25** | LOWER side of bracket; one-sided (B:106) |
| **`ε_spec`** | all-active spectral break `1/ρ(W) − ρ(Â)` | **A:57; D:26** | exact for all-active unconstrained-symmetric break (D:82) |
| `ε_break, ε_break,all` | deployed / all-active contraction-break budget (the true threshold) | A:58 | `ε_break,all`: `ε_crit ≤ ε_break,all ≤ (C/β)ε_crit` (D:93); `ε_break` only lower-bounded |
| `ε_reach` | measured **nonlinear** break budget (CONJECTURE) `≈ ε_spec/a` | A:59; D:189 | a = ReLU active fraction ≈0.6; ratios in D26-29 |
| **`g_W`** | nonnormality of W: `‖W‖₂/ρ(W) ≥ 1` | **A:60; D:37** | band `[1.19,2.47]` (A-none; D:41, D:225); a-post cert `g_W ≤ κ₂(V_W)` (D:42) |
| `η` | Neumann-to-spectral slack (resolvent-norm vs spectral-radius-rate); `η ≤ κ(V_W)` all-active | A:61; B:122-136 | band `[1.19,2.47]` general-ReLU (B:143); "pseudospectral index" unused |
| **`β = σ_E`** | edge-supported Frobenius mass of Perron mode: `β = ⟨u₁,Bu₁⟩ = ‖g_E‖_F²/σ_E = σ_E ∈ (0,1]` | **A:62; D:55** | suite `β≈0.62` (D:96); `β=1` ⇔ unconstrained-symmetric |
| `σ_E` | `‖g_E‖_F`, `g_E = P_E(u₁u₁^T)`; equals β | D:52; D:56 | |
| `B` | unit feasible direction `g_E/σ_E` | D:53 | |
| `u₁` | unit top (Perron) eigenvector of Â; `u₁>0` entrywise on connected graph | D:51; D:62 | |
| **`C` (bracket)** | **bracket constant `g_W(1+κ)/(1−κ) = (‖W‖₂/ρ(W))·(1+‖Â‖₂‖W‖₂)/(1−‖Â‖₂‖W‖₂)`** | **A:63; D:88** | suite `C/β` reaches 10–16× (D:151,223) |
| **`C` (ranking)** | **`L_J/(2(1−κ)²)`** — DIFFERENT object, same glyph (collision) | **C:77** | only in eq:rank-gap (C:79) |
| `L_J` | equilibrium-map curvature `≤ ‖W‖₂²‖z*‖` | A:64; C:60 | governs Rk: `|R_k| ≤ L_J w_k²/(2(1−κ)²)` (C:21,69) |
| `L_{J,v}` | per-node curvature `≤ ‖W‖₂²‖z*‖` (NOT in tab:notation) | E:57 | feeds C_v |
| `L_1^{(c)}` | first-order score-shift constant `‖(W_r−W_c)S_{c,v}‖₂`; aggregate `L_1^c=max_c L_1^{(c)}` | A:72; E:52; E:61 | |
| `C_v` | curvature score-shift constant `‖W_r−W_c‖₂ · L_{J,v}/(2(1−κ)²)` | A:72; E:54 | per-node form of ranking-C scaled by readout gap |
| `f, m_v^{(c)}, r_v` | logits `Wz*_v+b`; margin `f_{y_v}−f_c`; radius `min_{c≠y_v} m_v^{(c)}/‖(W_{y_v}−W_c)S_v‖₂` | A:67-69; B:152,168 | |
| `w_k, v_k, d_k` | edge weight `[Â]_{ij}`; edge score `‖[S_c]_{:,k}‖₂`; true removal damage; `d_k=w_k v_k+R_k` | A:51-52; C:21,34 | |
| `α, n_cal` | target miscoverage; calibration-set size; `1−α` coverage | A:73; E:177,183 | quantile level `⌈(n_cal+1)(1−α)⌉/n_cal` |
| `γ` | resolvent critical exponent `≈1.02`–`1.06` | D:201,224 | empirical, O1 |
| `a` | ReLU active fraction `≈0.6` (CONJECTURE) | D:190 | `ε_reach ≈ ε_spec/a` |
| `κ(V_W) / κ₂(V_W)` | eigenvector conditioning of W = `‖V_W‖₂‖V_W^{-1}‖₂`; `W=V_W D_W V_W^{-1}` | B:124; D:42 | bounds both η and g_W (two notations, likely same object) |
