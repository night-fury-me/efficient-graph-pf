# AEGIS — Project Progress Tracker

**Paper:** *Matrix-Free Diagnostics for the Adversarial Fault Lines of Graph Neural
Networks* (ICDM, strict 10 pp).
**Last updated:** 2026-06-01.
**How to read this:** each work item records **What / Why / How / Result / Status /
Files**. Status legend: ✅ done & validated · 🔬 validated in pilots (not yet in
the paper) · 🔄 in progress · 📝 drafted (text only) · ❌ not started · ⏬ tested then
downgraded/dropped.

> **One-line state of the project:** the targeted improvements are *implemented and
> validated* — including **AEGIS-Conformal** (the runner-up, now **10-seed validated on
> Cora & Citeseer**, gate holds at the nominal 1−α with 0 divergence). The **paper is the
> bottleneck**: the `.tex` still reflects the
> pre-improvement framing, so **paper integration (#6) is the remaining work.**

> **HARD RULE — 10 seeds.** Every result presented in the paper MUST use the 10
> preferred seeds `[42,137,271,314,1729,2718,3141,5772,6561,9999]`; no fewer-seed
> result goes in. **Audited 2026-06-01:** all experiments comply — campaign, coupling,
> MonDEQ, Universal at 10; **Conformal at 10 on both Cora & Citeseer**; amzfg
> deferred (non-load-bearing, killed to unthrottle Conformal). The log-penalty ablation
> (seed 42 only) is **dropped** — the paper presents only the raw 10-seed frontier.

---

## 1. Goal & strategy

**Goal.** Overcome the standing ICDM critique that AEGIS's *scope is too narrow* /
"it's a diagnostic, not a guarantee," and ship a defensible 10-page paper.

**Strategy (this campaign).**
1. **Literature search** across 14 fields → candidate breakthrough extensions
   (`breakthrough_extensions.md`).
2. Pick the **"Core kill-the-critique" package**: Certify + Universal + Stackelberg
   + a spine reframe.
3. Translate the ICDM weaknesses into **6 concrete recommendations** (#1–#6).
4. **Per-experiment protocol:** implement → critique-inspect for bugs → smoke-test →
   validate (independent recompute / dense ground-truth) → save findings md → next.
   One experiment at a time; local 4090 first, then cluster for multi-seed.

---

## 2. Status dashboard (at a glance)

| Item | What | Status | In paper? |
|---|---|---|---|
| Foundation: c=0.9 recipe | hard ‖W‖≤0.9 cap → 80% Cora, genuine κ<1 | ✅ | ✅ |
| Four-quadrant headline | SVD ≥ {PGD,Cls,Shift} | ✅ verified 10/10 seeds | ✅ |
| **#1 σ₁-regularizer defense** | working defense (penalize σ₁(S_c)) | ✅ multi-seed | 🔬 not yet |
| **#2 Certify vs AGNNCert** | integrity fix + complementary reframe | ✅ | ✅ (this session) |
| **#3 MonDEQ breadth** | AEGIS on a monotone graph DEQ | ✅ probe @ **10 preferred seeds** (FEASIBLE 10/10, ρ<1, σ₁ 0.0000%) | 🔬 not yet |
| **#4 influence framing** | delta paragraph | 📝 drafted | ❌ |
| **#5 attack–defense duality** | coupling proposition | ✅ validated | 🔬 not yet |
| **#6 10pp integration plan** | the spine + cut list | 📝 drafted | — |
| P1 **AEGIS-Certify** (TOP PICK) | sound matrix-free certified radius ρ_v | 🔬 piloted | ❌ not yet |
| P4 **AEGIS-Universal** | operator-agnostic (RL Bellman) | ✅ @ **10 preferred seeds** (YES 10/10, τ=0.988±0.002) | ❌ not yet |
| R2.1 **AEGIS-Stackelberg** | certified defender | ⏬ downgraded | ❌ (dropped) |
| P2 **AEGIS-Conformal** (RUNNER-UP) | robust split-CP on the S_c bound | 🔬 gate HOLDS @ **10 seeds, both Cora & Citeseer** (nominal 1−α: 0.90/0.98 @ε=.01/.05; 0 diverged); **local 4090** n=200 | ❌ |
| P3 Fisher · P5 Tight · R2.2 Escape · R2.3 Equivariant · R2.4 Resistance | exploratory | ❌ | ❌ |

---

## 3. The 6 recommendations (detailed)

### #1 — σ₁(S_c)-regularized training (the *working* defense) ✅
- **Why.** The paper's only "defense" was the c=0.9 spectral cap, which constrains
  ‖W‖ (hence κ) but **not S_c directly**. Reviewers want an actual defense.
- **How.** A differentiable, matrix-free `σ₁(S_c)` penalty (truncated Neumann +
  power iteration, `create_graph=True`) added to CE: `loss = CE + λ·σ₁̂(S_c)`. Swept
  λ on Cora; **raw vs log** penalty compared; multi-seed sweep on the cluster.
- **Result (multi-seed, 10 seeds, Cora).** Operating point **λ=0.0003:** acc
  **0.739±0.008** (−4.2), σ₁ **32.6±1.7** (10× lower), cert_frac **0.40→0.823±0.033**,
  attack damage 9× lower, flips 10.8→2.3. σ₁ & attack damage strictly monotone ↓;
  cert_frac peaks 0.923 at λ=0.003 (margin-collapse beyond). **Raw penalty is the
  headline (10-seed `regsweep`).** The log-penalty comparison
  (`regularizer_log_vs_raw.md`, seed 42) is **dropped from the paper** per the 10-seed
  rule — kept only as a review methodology note; the paper presents the raw 10-seed
  frontier alone. **Bug-audited:** matrix-free σ₁ matches a dense-SVD ground truth to 0.00%;
  the σ₁ drop is genuine S_c reshaping (‖J_z‖₂ stays ~0.9), not ‖W‖→0.
- **Files.** `scripts/exp_aegis_regularized_training.py`,
  `regularized_defense_findings.md`, `regularizer_multiseed_findings.md`,
  `regularizer_log_vs_raw.md`, `regularizer_bug_audit.md`.

### #2 — Certify vs AGNNCert → integrity fix + complementary reframe ✅
- **Why.** Position the certificate against the published AGNNCert; address
  "diagnostic, not a guarantee."
- **How.** Critique-first. **Scoped AGNNCert** (Li & Wang, USENIX Security 2025,
  arXiv:2502.00765): it is **hash-partition majority voting**, certifies **discrete
  edge edits (L0)** — *incommensurable* with AEGIS's continuous Frobenius radius, and
  it can't white-box-certify an equilibrium model. A naive head-to-head is
  self-defeating (AEGIS certifies ~0% on the discrete axis at full scale).
- **Result.** **No head-to-head.** Found and **fixed three integrity defects in the
  live draft:** (a) AGNNCert was mislabeled "IBP" in 4 places → corrected to
  voting/discrete; (b) the bib had wrong author/venue (was "Li Yuning, ICLR 2025");
  (c) `tab:baselines` reported a **fabricated `1.414`** (a home-grown IBP proxy)
  under the AGNNCert name → row + proxy-footnote deleted, reframed as
  complementary/incommensurable. Net length-negative (reclaimed ~5 lines).
- **Files.** `agnncert_scoping.md`; edits in `experiments.tex`, `related_work.tex`,
  `introduction.tex`, `aegis.bib`. (Verified: `1.414` gone, no dangling refs.)

### #3 — MonDEQ breadth (AEGIS on a monotone graph DEQ) ✅ probe
- **Why.** Rebut "your theory is for one specific IGNN."
- **How.** Critique-first → feasibility probe. Built a genuinely different model
  (Winston–Kolter monotone parameterization, forward–backward solver, **no norm
  cap**) and fed it unmodified to `ScalableSensitivity`.
- **Result. FEASIBLE-WITH-CAVEAT (the caveat is the contribution).** Re-run at the
  **10 preferred seeds** `[42,137,271,314,1729,2718,3141,5772,6561,9999]` (one
  `--seed` per run, per-seed logs `results/mondeq_s<seed>.log`, no clobbering):
  **FEASIBLE on 10/10** — ρ(J_z^FB)=**0.929±0.017** (<1 on **10/10**, max 0.963),
  dense-vs-matrix-free σ₁ = **0.0000% on 10/10** (exact every seed), mono_m=0.154±0.002
  (>0 on 10/10), v₁ beats random **3.46±0.43×**. No seed had ρ≥1 or σ₁ degradation.
  AEGIS's true applicability condition is **ρ(J_z)<1** — *strictly weaker* than the
  norm cap (‖J_z^plain‖₂=2.04±0.28>1 on all 10, which the IGNN cap would reject),
  and **not implied by monotonicity** (a genuinely monotone counterexample with
  ρ≈2.95 diverges). The probe at the 10 preferred seeds is sufficient as a
  supplementary breadth subsection.
- **Files.** `scripts/exp_mondeq_probe.py`, `mondeq_probe_findings.md`,
  `results/mondeq_s<seed>.log` (10 preferred seeds).

### #4 — Relation to influence functions 📝
- **Why.** Pre-empt "incremental over influence functions."
- **How/Result.** A delta paragraph (structure vs training-set; equilibrium
  resolvent vs training Hessian; matrix-free at N=7,650; emits attack + defense +
  certificate). **Drafted, not yet in the .tex.**
- **Files.** `rec_4_5_6_framing.md` (§#4).

### #5 — Attack–defense coupling/duality proposition ✅
- **Why.** Turn the "where you can attack = what you can certify" slogan into a
  checked claim (depth).
- **How.** Critique-first → refined design (margin-confound control, metric
  robustness L1/L2/max, permutation null, 10 seeds).
- **Result. VALIDATED.** Margin-controlled partial `a_v↔ρ_v | margin` =
  **−0.646±0.117**, negative 10/10 seeds, p≈1e-160; **independently recomputed**
  (seed 42 = −0.666, exact). Honest nuance: raw ρ_v is margin-dominated
  (ρ_v↔margin≈+0.79), so the coupling is a v₁-direction effect, sharp once margin is
  partialled out. Part (b): resolvent norm 1.97→16.49 and σ₁ blow up as κ→1 while
  cert_frac→0 (acc flat → not an accuracy artifact).
- **Files.** `scripts/exp_coupling_validation.py`, `coupling_validation_findings.md`.

### #6 — 10-page integration plan 📝 (this is the pending paper work)
- The spine, the section-by-section plan, and the cut list (PF gone → space to
  Certify; defense section shrinks to the regularizer + honest delocalization;
  Stackelberg → one paragraph/supplementary). **Drafted; execution = remaining.**
- **Files.** `rec_4_5_6_framing.md` (§#6).

---

## 4. Lit-search proposals (`breakthrough_extensions.md`) — status

- **P1 AEGIS-Certify (TOP PICK).** 🔬 Coded + tested, **not in the paper.** Sound
  matrix-free certified radius `ρ_v` = positive root of `m_v − √2·σ₁(S_{c,v})·‖Δw‖·r
  − C_v·r² = 0` (T3 curvature). Pilot: **0 breaches** (sound); non-vacuous in the
  **contractive regime** (dense Cora 96% @ ε=0.05) but **thin at full scale** (full
  Cora 41% @ ε=0.05, 2.2% @ ε=0.10) and **~0% against a single discrete edge edit**.
  The paper currently still *concedes* (`rem:certificates`, `conclusion.tex` (i))
  that r_v is not a sound certificate — the sound ρ_v theorem is the headline that
  needs writing in. Files: `exp_certify_{pilot,tighten,soundness}.py`,
  `certify_pilot_findings.md`.
- **P2 AEGIS-Conformal (RUNNER-UP).** 🔬 **10-seed validated on Cora — gate HOLDS at the
  nominal 1−α** (2026-06-01). Robust split-CP with the analytic S_c certify bound (no
  smoothing) → a sound, distribution-free, finite-sample structural coverage certificate;
  model-agnostic (any GNN via S_K). Over the **10 preferred seeds** (n=200, n_cal=n_test=
  100, α=0.10): gate (coverage under the worst-case AEGIS attack) = **0.900±0.062 / 0.895
  @ ε=0.01** (right at target) and **0.983 @ ε=0.05** (conservative), with **0.000
  divergence across all 4138 gate nodes** — not a breach. Sets ~0.95–1.37 (APS proper;
  TPS abstains/empty) — **non-vacuous at the very ε where the deterministic Certify is
  thin**, so Conformal is the more *practically useful* guarantee. Ran on the **local 4090
  (24 GB)** — the dense `S_c` path OOMs 8 GB cluster cards (matrix-free = future work).
  Citeseer also **10/10** (clean cov 0.91–0.92 — above nominal; gate 0.918–0.968, 0
  diverged). File: `conformal_findings.md`.
- **P3 AEGIS-Fisher.** ❌ Not started (flagged vacuity-risky in the doc).
- **P4 AEGIS-Universal.** ✅ Run at the **10 preferred seeds**
  `[42,137,271,314,1729,2718,3141,5772,6561,9999]` (canonical S=60, succ=6, γ=0.9;
  `RL_SEED` per run, per-seed CSV snapshots `results/universal_rl_s<seed>.csv`,
  no clobbering). **Operator-agnostic VERDICT = YES on 10/10**: edge-weighted
  Kendall τ = **0.988±0.002** (>0.8 on 10/10), S_value-vs-FD max rel err =
  **3.5e-6±0.8e-6** (<1e-4 on 10/10), SVD-vs-random margin 5.8±0.35×, ρ(J_z)=γ=0.9.
  (Earlier doc reported seeds 0–3; now the preferred 10.) **Not in the paper.**
  Extended by #3 (MonDEQ). File: `universal_findings.md`,
  `results/universal_rl_s<seed>.csv`.
- **P5 AEGIS-Tight.** ❌ Not done (the T3 curvature tightening in
  `exp_certify_tighten.py` is a simpler, separate device; P5's non-Euclidean /
  mixed-monotone reachability is unbuilt).
- **R2.1 AEGIS-Stackelberg.** ⏬ Tested then **downgraded** (best_r=1; a rank-1
  portfolio doesn't beat the single direction). Dropped from the paper (0 mentions).
  Cluster `stack` jobs also OOM. File: `stackelberg_findings.md`.
- **R2.2 Escape · R2.3 Equivariant · R2.4 Resistance.** ❌ Not started. (R2.5 Value =
  the Universal RL result.)

---

## 5. The cluster campaign (re-run at c=0.9)

- **Outcome:** completed 2026-06-01 07:07 — **291 done / 31 failed of 322**;
  scheduler exited cleanly. Headline experiments (four-quadrant, certify, transfer,
  cross-domain incl. Amazon ego-subgraph, regularizer multi-seed) all succeeded.
- **Failures triaged:** 11 stale `mettack`/`scallg` (superseded by `_rr` requeues,
  which succeeded); 20 genuine = `amzfg` (Amazon full-graph; missing-output-dir bug,
  now fixed + re-running locally) + `stack` (Stackelberg; OOM on 8 GB cards + not in
  the paper → **dropped**). **No failure blocks the paper.**
- **Orchestration note:** the scheduler runs on **cip3c0** (not cip7f0, the SSH
  gateway). NFS log ≠ liveness → scan all hosts before any restart (a false "dead"
  reading once spawned a duplicate scheduler). See
  `[[feedback_remote_launchpad_orchestration]]`.

---

## 6. Honest caveats / scoping (do not over-claim)

- **Certify** is a *sound but contractive-regime, small-ε continuous* certificate;
  it does **not** certify discrete edge edits at full scale. Ship as a
  contractive-regime guarantee.
- **PF physics claim is unsound** (separate prior finding): the PF N-1 headline was
  scored vs a surrogate, not real AC; rescope to model-auditing, not ground-truth
  physics. See `[[project_aegis_pf_physics_unsound]]`. (PF is being **cut** from the
  10pp main text per the #6 plan.)
- **Page budget** is a strict 10 pp at the ceiling — every length-adding edit must be
  offset. See `[[project_aegis_page_budget]]`.

---

## 7. What's remaining (prioritized)

1. **Paper integration (#6) — THE bottleneck.** Fold the validated results into the
   10 pp `.tex`: the **sound ρ_v Certify theorem** (replacing the `rem:certificates`
   concession), **Universal**, the **regularizer defense + multi-seed table**, the
   **coupling proposition**, the **MonDEQ breadth subsection**, and the **#4**
   influence paragraph — honoring the page budget.
2. **AEGIS-Conformal (P2 runner-up)** — 🔬 **10-seed validated on BOTH Cora & Citeseer,
   gate holds at nominal 1−α, 0 divergence** (table in `conformal_findings.md`). Local
   4090, n=200; run complete 2026-06-01. **Remaining:** fold the Cora+Citeseer table into
   the paper as the distribution-free runner-up certificate (part of #6).
3. **`amzfg` re-run** — **deferred** (killed 2026-06-01 to unthrottle Conformal on the
   shared 4090). Non-load-bearing; re-run only if the supplementary full-graph τ number
   is wanted.
4. (Optional) MonDEQ multi-seed if the breadth subsection needs more than the probe.

---

## 8. Artifacts index

| Kind | Path |
|---|---|
| Lit search (targets) | `paper/review/breakthrough_extensions.md` |
| Regularizer | `scripts/exp_aegis_regularized_training.py` · `regularized_defense_findings.md` · `regularizer_multiseed_findings.md` · `regularizer_log_vs_raw.md` · `regularizer_bug_audit.md` |
| Certify | `scripts/exp_certify_{pilot,tighten,soundness_fullgraph}.py` · `certify_pilot_findings.md` |
| Universal | `universal_findings.md` |
| Coupling (#5) | `scripts/exp_coupling_validation.py` · `coupling_validation_findings.md` |
| MonDEQ (#3) | `scripts/exp_mondeq_probe.py` · `mondeq_probe_findings.md` |
| AGNNCert (#2) | `agnncert_scoping.md` (+ paper edits) |
| Stackelberg | `stackelberg_findings.md` |
| Framing #4/#5/#6 | `rec_4_5_6_framing.md` |
| Foundation | `ignn_accuracy_findings.md` · `stepA_c09_adoption.md` · `stepB_b1b4_findings.md` |
| Paper source | `paper/aegis.tex` · `paper/sections/*.tex` · `paper/aegis.bib` |
| Cluster | `scripts/cluster_scheduler.py` · `scripts/run_job.sh` · (hub) `/proj/ciptmp/up89uvox/aegis/results/cluster/` |
