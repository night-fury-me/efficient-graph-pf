# SHOULD-FIX prep (ready to fold in during the integration pass)

Non-GPU editorial items prepared while the broadened-conformal (#1) + smoothing (#2)
10-seed runs finish on the 4090. Apply these in the single coordinated 7pp pass alongside
the 4-dataset Conformal table + the smoothing-comparison note.

## 1. ε-ball → edge counts (reviewer SHOULD-FIX: practical legibility)
Computed on Cora's normalized $\Ahat$ (N=2708, 5278 edges, $\norm{\Ahat}_F{=}24.9$, median
edge weight $\Ahat_{ij}{=}0.154{=}1/\sqrt{d_id_j}$). One symmetric edge deletion costs
$\norm{\delta\Ahat}_F{=}\sqrt2\,\Ahat_{ij}\approx 0.22$ (10–90 pct 0.10–0.41). So:
- $\varepsilon{=}0.01 \approx 0.05$ edge-equivalents (5% of one typical edge weight)
- $\varepsilon{=}0.05 \approx 0.23$ edge-equivalents (23% of one typical edge weight)

**Drop-in sentence (sec:conformal or its caption):** "In edge terms, on Cora
($\norm{\Ahat}_F{=}24.9$, median edge weight $0.154$, one deletion $\approx0.22$), the
certified budgets $\varepsilon{\in}\{0.01,0.05\}$ are $\approx0.05$/$0.23$ edge-equivalents
---continuous reweighting below a single edge, the threat \AEGIS targets, complementary to
discrete single-edge certifiers (\cref{app:baselines})." (Honest: small ball; reinforces
the continuous-threat positioning, not a hidden weakness.)

## 2. Constants-reconciliation table (defuses the "constant-zoo" critique)
Drop as a small `table` near \cref{thm:cf2s} or in the appendix.

| symbol | meaning | value | where |
|---|---|---|---|
| $\kappa{=}\norm{J_z}_2$ | trained contractivity | $0.14$–$0.59$ | Thm 1, tab:cross_domain |
| $\ecrit{=}(1{-}\kappa)/\norm{W}_2$ | norm certificate (certified-safe radius) | $0.41$–$0.86$/dataset | Thm 1 |
| $\varepsilon^\star{=}1/\rho(W){-}\rho(\Ahat)$ | all-active spectral break budget | $\approx \varepsilon_{\mathrm{reach}}/1.5$ | thm:cf2s |
| $2$–$4\times$ | **operating-point** $\rho$-margin to $\rho{=}1$ | how far the trained model sits *inside* the boundary | Thm 1c |
| $2$–$9\times$ | **empirical** break ratio $\varepsilon_{\mathrm{reach}}/\ecrit$ (10 seeds) | how far $\ecrit$ *under-states* the true nonlinear break | thm:cf2s / O1 |
| $10$–$16\times$ | **proven** bracket constant $C/\beta$ | worst-case enclosure of the all-active boundary | thm:cf2s |
| $\gamma{=}1.02$ | critical exponent | resolvent $\sim(1{-}\rho)^{-\gamma}$ | app:bracket |
| $\eta\in[1.19,2.47]$ | non-normality (pseudospectral) index | norm-vs-radius gap | obs:eta_bound |

The three "$\times$" rows are the trap; this table makes the operating-margin / empirical /
proven distinction legible in one place.

## 3. Mettack budget caveat (reviewer SHOULD-FIX: make the regime explicit inline)
In `experiments.tex` sec:cross_domain, the "$\mathbf{149/150}$ paired wins" already says
"in the early-warning regime" — make the budget explicit:
`in the early-warning regime ($\varepsilon{=}0.01$) ($\mathbf{149/150}$ paired wins, ...)`.

## 4. Smoothing-comparison note (skeleton — fill 10-seed numbers from #2)
Smoke (seed 42, Cora, n=200) — to be confirmed at 10 seeds:
- **AEGIS-Conformal is $\sim$10⁴× faster** wall-clock (0.5s vs 5,500s @ M=10⁴; zero-sample).
- On the **same $\varepsilon$-ball**, randomized smoothing is **vacuous** (matched
  $\sigma{=}\varepsilon/\sqrt{2|E|}$ tiny $\Rightarrow$ Cohen radius $\ll\varepsilon$ on a
  dense subgraph $\Rightarrow$ full 7-label set, cert_frac 0); it gives tight sets only on a
  larger per-coordinate ball.
- **Drop-in:** "Against randomized smoothing on the same $\varepsilon$-ball, \AEGIS-Conformal
  is non-vacuous where smoothing degenerates to the full set, at $\sim$10⁴$\times$ lower
  wall-clock (zero-sample vs $M$ reconverge passes; \cref{app:smoothing})." Put the full
  comparison table in the appendix.

**Status:** items 1–3 ready (data in hand); item 4 awaits the #2 10-seed run; the 4-dataset
Conformal table awaits #1. Page budget: the constants table (+~6 lines) + the notes will need
offsetting trims — handle in the one integration pass.
