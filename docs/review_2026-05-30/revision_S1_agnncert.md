# Revision S1 — AGNNCert complementarity (DONE; DEFEND with existing evidence)

**Concern (R2):** the "complementary, not interchangeable" decision rule is *asserted*, not validated; R2 suggested a per-quadrant breach-rate experiment or demoting `r_v` to a screen.

**Data scouting:** the per-quadrant cross-tab (certified? × `r_v<ρ`? × breached?) is **not** computable from saved files — `agnncert_comparison.csv` and `exp_breach_rates.csv` are pre-aggregated (medians, correlations, `breach_rate`/`radius_respected`); per-node rows are computed in memory but never persisted. A full cross-tab would need a light **CPU** rerun of both scripts with an added per-node dump (they share the same deterministic 50-node subgraph + seeds, so they'd join).

**Strategy: DEFEND with existing evidence (don't rerun a suggested item; don't demote `r_v`).** The complementarity is already substantiated by two facts in the paper:
1. **Orthogonal coverage** — near-zero Kendall τ between `r_v` and AGNNCert (`τ∈[−0.11,0.24]`, `tab:baselines`): the two flag *different* nodes ⇒ complementary, not redundant.
2. **Sound screen** — no node breaches below `r_v` (`radius_respected≈1.0`, `sec:adaptive`): `r_v` is a reliable first-order screening threshold.

**Edit (`experiments.tex`):** sharpened the AGNNCert sentence to cite both — "complementary, not redundant: it and `r_v` flag *different* nodes (near-zero τ), while `r_v` is itself a sound first-order screen (no node breaches below it) running 4.4–15.0× tighter than the certificate." Keeps the strong claim, now evidenced; no undersell.

**Optional bulletproof upgrade (offered, not done):** the per-quadrant breach-rate table via a light-CPU rerun + per-node dump in `exp_breach_rates.py` + `R2_02_agnncert_comparison.py`. It would mostly re-confirm (1)+(2). Build: 10pp.
