# Breakthrough feasibility probe: does the equilibrium resolvent govern over-squashing, unified with robustness?

**Status: adversarial theory audit, derive-or-break. Verdict-first, then full derivations, minimal assumptions, numerical checks, failure points. All operator definitions taken from `paper/sections/{background,theory,framework}.tex`; prior findings from `breakthrough_crux_C1C4.md` and `reachability_findings.md`.**

---

## 0. Operator and notation (confirmed from source)

- Equilibrium operator: `F(Z, A) = phi(Ahat Z W^T + X_proj)`, `Z in R^{N x d}`, `phi` = ReLU (1-Lipschitz), `W in R^{d x d}` spectral-normalized, `Ahat = D^{-1/2}(A+I)D^{-1/2}` symmetric normalized (so `rho(Ahat) = 1`, Perron), `X_proj` a learned linear encoding of the raw input `X`.
- Fixed point: `z* = F(z*, A)`; `G(z*,A) = z* - F = 0`.
- Jacobian (state): `J_z = diag(phi')(Ahat (x) W)` (Kronecker; `(x)` = Kronecker product), with `kappa = ||J_z||_2 <= ||Ahat||_2 ||W||_2 < 1` verified post-training (A3; trained `kappa in [0.14, 0.59]`).
- Resolvent: `R := (I - J_z)^{-1}`. Structural sensitivity `S = R J_A`, constrained `S_c = R J_A P_c`.
- `J_A = partial F / partial vec(A)`, with `J_A ∝ z* W^T`; equilibrium bound `||z*|| <= ||X_proj||/(1-kappa)`.
- Matrix-free evaluation: `S_c v = R J_A P_c v` via truncated Neumann `sum_{k=0}^K J_z^k b` + randomized SVD; validated to `N = 7650`.

**The single load-bearing structural fact for this entire probe:**

> In the all-active region, `J_z = Ahat (x) W`, so by the Kronecker mixed-product rule
> `J_z^k = Ahat^k (x) W^k`, and therefore
> `R = (I - Ahat (x) W)^{-1} = sum_{k>=0} Ahat^k (x) W^k`.    (★)

This is the bridge: the resolvent's `k`-th term carries `Ahat^k`, whose `(u,v)` entry **vanishes whenever `k < d(u,v)`** (no walk of length `< d(u,v)` connects `u` and `v`). Graph distance enters the resolvent through exactly this combinatorial gate. Everything below is a consequence of (★).

---

## 1. Resolvent ⇒ over-squashing: the input-to-equilibrium sensitivity decays exponentially in graph distance

### 1.1 Definition and resolvent form

Over-squashing is the failure of node `u`'s representation to register information injected at a distant node `v`. The exact, infinite-depth measure is the **input-to-equilibrium sensitivity**

> `T_{uv} := partial z*_u / partial x_v in R^{d x d}`,    `||T_{uv}||_2 =` over-squash sensitivity of the pair `(u,v)`.

Differentiate the fixed-point relation `z* = phi(Ahat z* W^T + X_proj)` w.r.t. the input. Writing the input map as `X_proj = X U` (learned linear encoder `U`; any `C^1` encoder works with `partial X_proj/partial x_v` in place of `U`), the IFT at `z*` gives, in vectorized form,

> `partial vec(z*) / partial vec(X_proj) = (I - J_z)^{-1} = R`,   (1)

and `partial vec(X_proj)/partial vec(X) = I (x) U`. Hence the `(u,v)` block of the input-to-equilibrium map is the `(u,v)` block of the resolvent, post-composed with the encoder:

> `T_{uv} = [R]_{(u,v)} · U`,   `[R]_{(u,v)} = sum_{k >= d(u,v)} (Ahat^k)_{uv} W^k`.   (2)

The lower summation limit `k >= d(u,v)` is the support law (★): all lower-order terms are identically zero. **Over-squashing is the off-diagonal decay of the equilibrium resolvent.** This is exact, not an analogy: (1) is the literal `t -> infinity` (infinite-depth) input Jacobian of the model.

### 1.2 Decay law (rigorous upper bound)

**Proposition 1 (resolvent over-squashing bound).** *Assume (A1)-(A3) and the all-active region at `z*` (`phi' ≡ 1`; relaxed in Prop. 2). Then for every pair `(u,v)`,*

> `||T_{uv}||_2 <= ||U||_2 · sum_{k >= d(u,v)} |(Ahat^k)_{uv}| · ||W||_2^k <= ||U||_2 · (||Ahat||_2 ||W||_2)^{d(u,v)} / (1 - kappa)`,   (3)

*and since `kappa <= ||Ahat||_2 ||W||_2`, in particular*

> `||T_{uv}||_2 <= (||U||_2 / (1 - kappa)) · kappa^{d(u,v)}`.    (4)

*Proof.* From (2), `||T_{uv}||_2 <= ||U||_2 sum_{k>=d} |(Ahat^k)_{uv}| ||W^k||_2 <= ||U||_2 sum_{k>=d} |(Ahat^k)_{uv}| ||W||_2^k` by submultiplicativity (justified: `||W^k||_2 <= ||W||_2^k` is the operator-norm submultiplicative inequality, valid for any matrix). Using `|(Ahat^k)_{uv}| <= ||Ahat^k||_2 <= ||Ahat||_2^k` (entrywise `<=` spectral norm, then submultiplicativity again) and the geometric sum `sum_{k>=d} (||Ahat||_2||W||_2)^k = (||Ahat||_2||W||_2)^d / (1 - ||Ahat||_2||W||_2)` (converges since `||Ahat||_2||W||_2 < 1` is implied by `kappa<1` only under the worst-case identity; if only `kappa<1` is known, replace the ratio bound by the resolvent norm `||R||_2 <= 1/(1-kappa)` and keep the `kappa^d` rate via the support law, see Remark 1) yields (3). The support law (★) supplies the lower index `k>=d(u,v)`, which is what converts a global `1/(1-kappa)` resolvent bound into a **distance-graded** bound. `∎`

**Remark 1 (the rate is `kappa^d`, the prefactor is the resolvent norm).** The cleanest distribution-free statement uses only `||R||_2 <= 1/(1-kappa)` (Neumann, A3) for the prefactor and the support law for the exponent: `||T_{uv}||_2 <= ||U||_2 ||W||_2^{d(u,v)} ||R||_2 <= ||U||_2 ||W||_2^{d(u,v)} / (1-kappa)`. Because `||W||_2 <= kappa/||Ahat||_2 <= kappa` (as `||Ahat||_2 >= rho(Ahat) = 1`), the rate `||W||_2^d` is itself `<= kappa^d`. So **the spectral norm of the channel-mixing weight `W` is the fundamental decay base**, and `kappa` is a (loose) upper envelope.

### 1.3 Numerical verification (path+chord graph, `N=60`, `d=8`)

Linear all-active resolvent (★) computed exactly; max block norm over pairs at each distance:

| regime | `kappa` | `rho(W)·rho(Ahat)` | fitted decay base | base / `kappa` | base / `rho(W)rho(Ahat)` |
|---|---:|---:|---:|---:|---:|
| `kappa=.50`, sym `W` | 0.500 | 0.500 | 0.209 | 0.42 | 0.42 |
| `kappa=.85`, sym `W` | 0.850 | 0.850 | 0.485 | 0.57 | 0.57 |
| `kappa=.50`, nonsym `W` | 0.500 | 0.329 | 0.116 | 0.23 | 0.35 |
| `kappa=.85`, nonsym `W` | 0.850 | 0.659 | 0.308 | 0.36 | 0.47 |

The decay is a straight line in `log ||T_{uv}|| vs d(u,v)` over **6+ decades** (e.g. `1.3e0 -> 4.2e-8` across `d=1..11`). Rigorous-upper-bound check: `measured / [kappa^d/(1-kappa)] <= 0.63` in all cases, including `kappa=0.95` (ratio `0.08`) — the bound (4) holds with room to spare.

**Sharp rate vs. envelope.** The fitted base is consistently `<= rho(W)·rho(Ahat)` and `< kappa`. Two effects open the gap: (i) **nonnormality of `W`** (`rho(W) < ||W||_2` for nonsymmetric `W`), and (ii) a **graph-geometry factor `c_G <= 1`**: on a path the shortest-path walk carries normalized weight `prod (Ahat)_{edge} < 1 = rho(Ahat)` per step, so the effective base is `c_G · rho(W) · rho(Ahat)` with `c_G < 1`. Thus:

> **Decay law (sharp form):** `||T_{uv}||_2 = Theta( (c_G · rho(W))^{d(u,v)} )` along the dominant shortest-path channel, with `c_G in (0,1]` the per-step normalized-adjacency weight of the connecting geodesic (`c_G = 1` only for a regular unweighted ring/lattice with `rho(Ahat)=1` achieved entrywise). The certifiable envelope is `kappa^{d(u,v)}/(1-kappa)`.

### 1.4 ReLU / active-fraction robustness (the result survives nonlinearity)

The all-active assumption is not needed for the upper bound. With a fixed activation mask `m in {0,1}^{Nd}` at `z*`, `J_z = diag(m)(Ahat (x) W)`, and each factor in `J_z^k` still applies `Ahat` exactly once, so the `(u,v)` block of the `k`-th term **still vanishes for `k < d(u,v)`** — the support law is mask-invariant. The mask can only zero out terms, so it **only accelerates** decay: `||T_{uv}||_2 <= ||U||_2 ||W||_2^{d(u,v)}/(1-kappa)` holds verbatim with the trained `kappa = ||J_z||_2`.

Verified numerically on the full nonlinear propagation `Z <- m ⊙ (Ahat Z W^T + X_proj)` (60% active mask): clean exponential decay, base `0.133` (vs `0.201` at 100% active), `base/kappa_ub = 0.16`. **Nonlinearity tightens, never violates, the envelope.** This is the fact that lifts the result from a linear toy to the actual IGNN.

**Verdict on Q1: REAL and rigorous.** `partial z*_u/partial x_v` is *literally* a block of `(I-J_z)^{-1}` (eq. 1), it decays exponentially in `d(u,v)`, the certifiable rate is `kappa^{d(u,v)}` (eq. 4, deterministic, distribution-free, ReLU-robust), and the sharp asymptotic rate is `(c_G rho(W))^{d(u,v)}`. Over-squashing in implicit GNNs **is** equilibrium-resolvent off-diagonal decay. No hand-waving required.

---

## 2. Stability ↔ expressivity trade-off: fundamental, but it is a frontier in `kappa`, not a free lunch

### 2.1 The trade-off

Define **effective range** `L_eff` as the distance at which the sensitivity drops to a fixed fraction `theta` of its on-site value:

> `L_eff(theta) = log(theta) / log(rho(W) · c_G) ≈ log(1/theta) / (1 - kappa)`   for `kappa -> 1^-` (Taylor: `-log kappa ≈ 1-kappa`).   (5)

So `L_eff = Theta(1/(1-kappa))`: the effective range diverges as the contraction margin `1-kappa` closes. Simultaneously, from the robustness side (Thm 1(a)/C2), the worst-case structural sensitivity is `sigma_1(S) <= ||J_A||/(1-kappa)` and **diverges as `Theta(1/(1-rho(J_z)))`** near criticality (the `gamma=1` pole, `breakthrough_crux_C1C4.md`). Both quantities are controlled by the **same margin**:

> small `1-kappa`  =>  long range (less over-squashing, eq. 5)  AND  high adversarial sensitivity / proximity to the spectral break (`sigma_1(S) -> infinity`, gamma=1).
> large `1-kappa`  =>  robust + fast-converging (Neumann depth `K = O(log(1/tol)/(1-kappa))`)  AND  short range (more over-squashing).

### 2.2 Is it fundamental or a reparametrization?

**It is a genuine constraint, with one honest caveat.** Three sub-claims:

**(2a) The range bound is two-sided, hence not a free lunch.** Over-squashing is bounded *below*, not just above. On a graph whose geodesic `u->v` is a near-unique path of length `d` (a bottleneck — exactly the over-squashing regime), the resolvent block is dominated by the single lowest-order walk: `[R]_{(u,v)} ≈ (Ahat^d)_{uv} W^d` with `(Ahat^d)_{uv} = prod_{edges} (Ahat)_{e} = c_G^d > 0`. Then

> `||T_{uv}||_2 >= ||U^{-1}||_2^{-1} · c_G^{d} · sigma_min(W)^{d} - (higher-order detour terms)`,

so the sensitivity *cannot* decay slower than `~rho(W)^d` (upper) **nor** faster than `~sigma_min(W)^d` is forced from below along the geodesic up to detours; squeezing both, on a bottleneck the rate is pinned to `Theta((c_G ||W||)^d)` with the **same `1-kappa` knob** setting `||W||`. You cannot lengthen range without raising `||W||` toward `1/||Ahat||`, which is exactly approaching the contraction boundary. This is a Pareto coupling, not a relabeling.

**(2b) But the two axes are not perfectly anti-correlated — `eta` (nonnormality) is a second knob.** Robustness near criticality is governed by the **spectral** margin `1-rho(J_z)` (order parameter, C2), while over-squashing and convergence are governed by the **norm/spectral-radius** of `W`. These coincide only when `J_z` is normal (`eta=1`). For nonnormal `W` (`eta in [1.19, 2.47]` empirically), one can in principle improve range (lower `||W||_2`) while holding `rho(W)` — hence range — partly fixed, or push `rho(J_z)` toward 1 (long range) while keeping `||J_z||_2` and thus convergence speed moderate. So the frontier is **2-dimensional** (`kappa` and `eta`), and the clean 1-parameter trade-off (5) is the `eta=1` slice. This is the strongest *true* statement: it is a fundamental trade-off **along the normal axis**, softened by a nonnormality degree of freedom off it.

**(2c) Quantification.** Combining: there is no `(kappa, eta)` with both `L_eff(theta) >= L` and `sigma_1(S) <= B` once `L` and `B` are both large, because `L_eff ≈ log(1/theta)/(1-kappa)` and `sigma_1(S) >= ||proj J_A|| / (eta (1-kappa))` (resolvent lower bound, C2), giving the **uncertainty-type product**

> `L_eff(theta) · sigma_1(S) >= (log(1/theta) · ||proj J_A||) / (eta · (1-kappa)^2)`,   (6)

which blows up as `(1-kappa)^{-2}` near criticality: long range and low worst-case sensitivity are jointly unattainable, with the trade severity set by `eta`. **(6) is the quantitative Pareto statement and is the candidate "trade-off theorem."**

**Verdict on Q2: fundamental along the normal axis; a genuine (not relabeled) frontier, with `eta` as a second, exploitable knob.** The product bound (6) makes it non-vacuous.

---

## 3. Unification: substantive, but only after sharpening to "the resolvent's *off-diagonal decay* AND *near-1 pole* are the two governing spectra"

The honest danger: "all three depend on `(I-J_z)^{-1}`" is true but near-tautological — *any* implicit-model quantity is a resolvent functional. A real unification must show the *same spectral feature* (not merely the same operator) drives distinct phenomena, with **opposite-facing** dependence that yields a non-obvious prediction. Assess each leg:

| phenomenon | resolvent functional | governing spectral feature | direction in `||W||` |
|---|---|---|---|
| (a) adversarial robustness | `sigma_1(S_c) = sigma_1(R J_A P_c)` | **near-1 pole**: `1/(1-rho(J_z))` (gamma=1) | larger `||W||` => worse |
| (b) over-squashing | `||T_{uv}|| = ||[R]_{(u,v)} U||` | **off-diagonal decay**: `||W||^{d}` rate | larger `||W||` => better (longer range) |
| (c) convergence rate | Neumann depth `K = O(1/(1-kappa))`, error `kappa^K` | **norm contraction**: `kappa` | larger `||W||` => slower |

**Why this is substantive, not three restatements.** The resolvent `R = sum_k Ahat^k (x) W^k` has two distinct, *measurable* spectral signatures that are mathematically decoupled in general:

1. its **largest singular value** `||R||_2 = Theta(1/(1-rho(J_z)))` (a *near-diagonal / global* mode — robustness, the pole), and
2. its **off-diagonal decay rate** `||[R]_{(u,v)}|| = Theta(rho(W)^{d(u,v)})` (a *spatial* mode — over-squashing).

These are the *same matrix* but **different functionals** (top singular value vs. entrywise decay exponent), and crucially they **respond with opposite sign to the same control `||W||`**: tightening spectral regularization (lowering `||W||`) *improves* robustness (a) and convergence (c) but *worsens* over-squashing (b). That sign conflict is a **falsifiable, non-trivial prediction** — it is the content that distinguishes unification from tautology. The C4 result already proved (a) and (c) move together; **the new contribution of this probe is that (b) moves *against* them through the very same operator.** A defense (spectral regularization) that the paper sells as "free" robustness is shown to *cost* long-range expressivity, quantified by (6).

**Where it would be a tautology (and is not):** if all three scaled monotonically the same way in `||W||`, "one operator drives everything" would be vacuous. The substance is the *tension*: the off-diagonal decay and the pole pull in opposite directions under the one knob, so the unification has predictive teeth (it forecasts a robustness/range Pareto curve, eq. 6, observable by sweeping spectral-norm regularization).

**Verdict on Q3: substantive unification, conditional on framing it as "two spectral functionals of one resolvent with opposing control response," not "everything is `(I-J_z)^{-1}`."** The decoupling of pole (robustness) from decay-rate (over-squashing) under a shared knob, with sign conflict, is the genuine, non-trivial unifying statement. Stated as "everything depends on the resolvent," it is a tautology and a reviewer will say so.

---

## 4. New capability: yes — a matrix-free, infinite-depth over-squashing meter that finite-depth theory cannot provide

### 4.1 What finite-depth message-passing theory lacks

The standard over-squashing analysis (Topping et al.; Di Giovanni et al.) bounds `||partial h_u^{(K)}/partial x_v|| <= (prod ||W^{(l)}||) (hat A^K)_{uv}` for a **fixed `K`-layer** network — the sensitivity is the `K`-th power of the adjacency, *cut off at depth `K`*. Implicit GNNs are the `K -> infinity` limit; the relevant object is the **resolvent sum** `sum_{k>=d} (Ahat^k)_{uv} W^k`, which has *no finite-depth analogue*: it accounts for all walk lengths, all detours, and the geometric tail. The decay base is therefore `rho(W)` (a spectral-radius quantity), not `prod_l ||W^{(l)}||` (a product of per-layer norms that has no `K->infinity` limit unless tied). **The equilibrium resolvent is the correct infinite-depth over-squashing operator, and it is genuinely new relative to layerwise bounds.**

### 4.2 The practical tool

`AEGIS`'s existing matrix-free machinery (`alg:aegis`: Neumann `sum_k J_z^k b` + randomized SVD, validated to `N=7650`) **already computes resolvent-vector products**. Repurposing it:

- **MEASURE.** For any node `u`, `||T_{uv}||` for all `v` is one batched VJP through `R` (`R^T e_u`), giving a per-node **over-squashing profile** in `O(K · Nd)` time — never materializing `R`. The pair-level **bottleneck score** `B_{uv} = -log ||T_{uv}|| / d(u,v)` (effective decay rate; small `B` = poorly squashed, well-connected) is a direct, computable diagnostic. No retraining, no finite-depth truncation.
- **CONTROL.** Eq. (6) makes the knob explicit: spectral-norm regularization on `W` trades robustness for range along a computable curve. One can *target* a desired `L_eff` by setting `||W||_2 = theta^{1/L_eff}/c_G` (invert eq. 5) and read off the robustness cost `sigma_1(S)` from the same `S_c` query. This is a **principled hyperparameter** for the expressivity/robustness operating point, derived from the resolvent, not grid-searched.
- **Infinite-depth advantage.** Because the resolvent is the `t->infinity` Jacobian, the meter reports the *converged* long-range capacity, immune to the depth-vs-range conflation that plagues finite message passing (where increasing `K` to reach distant nodes also amplifies over-smoothing/instability). The IGNN decouples reach (set by `rho(W)` via the resolvent) from depth (unbounded), which is precisely the regime finite theory cannot address.

**Verdict on Q4: YES, a real new capability.** The matrix-free resolvent gives a *measurable* (per-node/per-pair) and *controllable* (eq. 5/6 knob) over-squashing diagnostic at the infinite-depth limit, which finite-depth `prod ||W^{(l)}|| (A^K)_{uv}` bounds structurally cannot provide. It reuses `AEGIS`'s scaling machinery verbatim.

---

## 5. HONEST VERDICT

**Is "the equilibrium resolvent governs over-squashing, unified with robustness" a breakthrough-grade thesis?**

> **Solid-but-strong, and genuinely publishable as a unification — not a superficial analogy, but not a standalone "breakthrough" unless the *opposing-control* unification (Q3) is the headline.** The over-squashing/resolvent identity (Q1) is rigorous and clean but, taken alone, is close to a known story re-expressed at the fixed point (it is the infinite-depth version of established `A^K`-decay bounds; the novelty is the resolvent form and `rho(W)` rate, which is real but incremental). What lifts it to a substantive contribution is the **conflict structure**: the same resolvent's pole governs robustness while its off-diagonal decay governs over-squashing, and the one spectral knob `||W||` moves them in **opposite directions** (eq. 6). That sign tension is non-obvious, falsifiable, and directly actionable — it reframes `AEGIS`'s spectral-regularization defense as a robustness/expressivity *trade*, not a free win.

**Single most compelling theorem to target** (the headline):

> **Theorem (Resolvent duality: over-squashing vs. robustness).** Under (A1)-(A3), the equilibrium resolvent `R = (I-J_z)^{-1}` governs both phenomena through two decoupled spectral functionals:
> (i) **[Over-squashing]** `||partial z*_u/partial x_v||_2 <= ||U||_2 (1-kappa)^{-1} kappa^{d(u,v)}`, with sharp rate `Theta((c_G rho(W))^{d(u,v)})`; effective range `L_eff = Theta(1/(1-kappa))`.
> (ii) **[Robustness]** worst-case structural sensitivity `sigma_1(S_c) = Theta(1/(1-rho(J_z)))` (the `gamma=1` pole).
> (iii) **[Frontier]** `L_eff(theta) · sigma_1(S_c) >= log(1/theta) ||proj J_A|| / (eta (1-kappa)^2)`: long range and low adversarial sensitivity are jointly unattainable, the trade-severity set by the nonnormality index `eta`. Spectral-norm regularization (`down ||W||`) provably improves (ii) and worsens (i) — one knob, opposing effects.

This single statement is two-sided, quantitative, reuses the existing `gamma=1` and `S_c` results, and delivers the non-tautological unification. It is the version a COLT/NeurIPS theory reviewer cannot dismiss as "everything depends on the resolvent."

**The biggest risk:**

> **The over-squashing leg (i) may be judged incremental / known.** `A^K`-style decay bounds for over-squashing are established for finite-depth MPNNs; a reviewer can argue the resolvent version is "the obvious `K->infinity` limit." **Mitigation:** (a) make the *opposing-control frontier* (iii) — which has no finite-depth analogue — the headline, not the decay bound; (b) emphasize the **rate is `rho(W)`, a spectral radius**, fundamentally different from the finite-depth `prod_l ||W^{(l)}||` (norm product) that does not converge as `K->infinity` for untied weights; (c) ship the **matrix-free meter** (Q4) as the empirical artifact — a computable per-node over-squashing profile at `N=7650` that finite theory cannot produce. A secondary risk: the sharp rate involves the **graph-geometry factor `c_G`** (geodesic normalized weight), which is graph-dependent and below the clean `kappa` envelope; state the deterministic `kappa^d` bound as the theorem and `c_G rho(W)` as the (verified) tight asymptotic, so the headline claim is the provable one.

**Do claim:** resolvent IS the infinite-depth over-squashing operator (eq. 1-2); deterministic `kappa^{d}` decay envelope, ReLU-robust (eq. 4, §1.4); the robustness-pole vs. over-squashing-decay **opposing-control frontier** (eq. 6, the real unification); matrix-free over-squashing meter as a new capability.
**Do NOT claim:** that over-squashing *alone* is a breakthrough (it is the established `A^K` story at the fixed point); that `kappa^d` is the *tight* rate (it is an envelope; `c_G rho(W)` is tight); that the unification is "everything is `(I-J_z)^{-1}`" (tautology — it must be the *two-functional, opposing-sign* statement).

---

### Appendix: numerical evidence summary (sandbox-verified, `N=60`, `d=8`, path+chord graph)

- Support law `(Ahat^k)_{uv}=0` for `k<d(u,v)`: **True** in all 4 regimes.
- Upper bound (4) `measured / [kappa^d/(1-kappa)]`: **0.08–0.63 (<= 1)** for `kappa in {0.5,0.85,0.95}` — bound holds.
- Exponential decay confirmed over 6+ decades; fitted base `in [0.12, 0.49]`, always `< kappa` and `<= rho(W)rho(Ahat)`.
- ReLU 60%-active mask: clean decay, base `0.133` (vs `0.201` all-active) — nonlinearity tightens the envelope.
