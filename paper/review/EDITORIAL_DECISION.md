# Editorial Decision — AEGIS: Matrix-Free Diagnostics for the Adversarial Fault Lines of GNNs

**Venue (simulated):** ICDM (regular track, 10 pp)
**Decision:** **MAJOR REVISION**
**Panel:** R1 Theory (ml-theory-reviewer) · R2 Empirical methodology · R3 Domain/positioning · R4 Devil's Advocate · EIC synthesis
**Per-reviewer reports:** `review/R1_theory.md`, `review/R2_methodology.md`, `review/R3_domain.md`, `review/R4_devils_advocate.md`

---

## 1. Summary judgment

AEGIS unifies three GNN-vulnerability diagnostics — the SVD-optimal structural perturbation direction, per-edge sensitivity rankings, and per-node first-order radii — under one matrix-free object, the constrained sensitivity matrix `S_c`, and scales it to N=7,650 on one GPU. The panel agrees this is a *genuine* (if niche) contribution: a clean operator-level unification of IFT/influence-function sensitivity re-pointed at structural edge perturbations, with a credible label-free power-flow N-1 study as the cross-domain differentiator. The writing is honest — **every unfavorable number is actually reported in the paper** (R4 verified this explicitly); the problems below are about *framing precision, one theoretical soundness gap, and 2–3 missing control experiments*, not data integrity.

No reviewer argued Accept; no reviewer argued Reject. The Devil's Advocate raised CRITICAL issues (IRON RULE → cannot Accept), but all are rebuttable with bounded work. **Major Revision** is the consensus call.

### Scorecard (0–100)

| Dimension | Score | One-line basis |
|---|---|---|
| Originality / novelty | 62 | Real unification, but recombination of known IFT machinery (Koh–Liang, Gould, Lorraine); R3 "borderline, niche." |
| Theoretical rigor | 54 | Proofs mostly careful, but the flagship `ε_crit` safety-boundary is unsound *as stated* under measured κ; several claims over-scoped. |
| Empirical rigor | 60 | Broad (390 runs), honest, well-hedged; undercut by a topology confound, 50-node-subgraph reliance, and a strawman black-box. |
| Significance | 58 | Niche robustness/sensitivity tool; power-flow is the hook but is topology-confounded as currently shown. |
| Clarity | 65 | Well-organized and cross-referenced; abstract is over-dense, several headline sentences overstate. |
| Reproducibility | 55 | Seeds/HW/hparams reported; no code artifact, several protocols (stratified-N-1, BFS seeding, splits) underspecified. |

---

## 2. Consensus findings (raised independently by ≥2 reviewers)

- **C-1 — Power-flow result is topology-confounded; no centrality null.** [R4 CRITICAL · R2 MAJOR] `S_c` is derived from a *learned* surrogate, then "recovers" N-1 rankings — but the paper itself concedes the signal "reflects topology-driven flow concentration" and that **binary adjacency beats admittance-weighting (P@10 0.81 vs 0.27)**. The only baselines are LODF/PTDF/PI; no degree / betweenness / current-flow-betweenness ranker is run. Until a graph-centrality null is beaten, the flagship cross-domain validation is not distinguishable from "trivial topology centrality."

- **C-2 — `ε_crit` is not a sufficient safety boundary as written (κ reconciliation).** [R1 CRITICAL · R4 C2 corroborates] `ε_crit=(1−κ)/‖W‖₂` is sufficient (`‖J_z'‖₂<1` for ε<ε_crit) **only in the all-active case** where κ=‖Â‖₂‖W‖₂. The paper instead plugs in the *measured* partial-ReLU κ=0.14–0.59. EIC independently verified from Table 1 that the implied ‖W‖₂≈1.0; with ‖Â‖₂=1 exactly (renormalized adjacency), the honest worst-case budget `1/‖W‖₂−1 ≈ 0`. **The reported per-dataset ε_crit (0.41–0.86) and the "2–4× margin" are artifacts of the optimistic measured-κ substitution.** Reframing path exists (below) — the *eigenvalue* story (ρ(J_z)≤0.42) is the real, defensible content; the *norm-based* certificate is near-vacuous in worst case and must be labeled as such.

- **C-3 — Theory does less work than the framing implies.** [R1 MAJOR · R4 M2/C2] (i) Prop 3(b) "ranking preservation" is a *pairwise-sufficient* condition that holds for only 47–62% of pairs; the headline τ=+0.99 / +0.996 is **empirical, not implied**. (ii) The phase-transition regimes (b)/(c) are never reached by trained models (huge margin), so the theorem is operationally a *consistency check*, not a working safety tool. State this plainly.

- **C-4 — Headline four-quadrant + 42% defense live on 50-node BFS subgraphs the paper shows are 16%-faithful.** [R2 MAJOR · R4 M1] Subgraph↔full-graph τ=0.16; yet the 42±8% defense, four-quadrant attack table, and breach analysis are subgraph-only. Re-run on the full graph via the existing matrix-free path, or justify.

- **C-5 — 512-query "black-box" is an unspecified/random strawman.** [R2 MAJOR · R4] The non-circularity argument leans on "44% for 512-query black-box." Replace with a real query attacker (NES / SimBA / bandits) at matched budget.

- **C-6 — Several abstract/intro sentences overstate.** [R2 · R3 · R4] (a) "one query recovers the direction 50-step PGD finds" misattributes the *surrogate-transfer* cos=0.99 result to PGD. (b) "No single method returns all three" over-claims (collective certs surface influential edge sets; surrogate gradients are a discrete direction) — soften to "no prior *single matrix-free pass*." (c) AGNNCert near-zero per-seed τ∈[−0.11,0.24] is reframed as "complementary"; state the low correlation first.

---

## 3. Arbitration of tensions

- **R1 ("ε_crit unsound, CRITICAL") vs R4 ("theorem correct but inert").** Not a contradiction. The *proof's core* (resolvent blows up as an eigenvalue→1; ε_crit *lower-bounds* that threshold) is correct and already hedged in part (b). The defect is the *presentation*: Table 1's per-dataset ε_crit and the "safety boundary with 2–4× margin" headline use measured κ, which is optimistic. Resolution: keep the eigenvalue result; relabel the norm-based ε_crit as a conservative (worst-case ≈0) certificate and move the "2–4× margin" claim onto ρ(J_z), where it is genuine.
- **Overall severity.** R3 "borderline-accept," R1 "sound-with-fixes + 1 CRITICAL," R2 "gaps closable," R4 "major revision, not fraud." → **Major Revision**, gated on the P0 items.

---

## 4. Revision Roadmap (prioritized; R&R-ready)

### P0 — Gating (must fix for acceptance)

1. **Restate `ε_crit` honestly.** [C-2 / R1] Either (a) report the worst-case sufficient budget using the all-active bound `1/‖W‖₂−‖Â‖₂` (note ‖Â‖₂=1), independent of measured κ; or (b) explicitly scope the measured-κ ε_crit as a *local, activation-pattern-stable* threshold, not a global certificate. Move the "2–4×" margin claim onto the spectral radius ρ(J_z). Re-verify Table 1 and the phase-transition narrative against the corrected quantity. Fix the shared root cause everywhere (conflation of `1−‖Â‖₂‖W‖₂` with `1−κ`), including the L_J denominator in Prop 3 (use `1−κ`).
2. **Add a graph-centrality null to the power-flow study.** [C-1 / R4,R2] Rank N-1 severity by degree, betweenness, and current-flow betweenness on the IEEE topologies. Either beat them (→ strengthens the paper) or concede that `S_c` ≈ topology centrality on this task and rescope the contribution.
3. **Fix factual/attribution errors.** [R2] (a) `case_study.tex` τ range "+0.37 to +0.62" contradicts Table `tab:ieee` (case57=+0.67) and the section's own "0.62–0.67"; correct to **+0.37 to +0.67**. (b) Rewrite the abstract "PGD direction" sentence to name the zero-gradient surrogate transfer.

### P1 — Major (expected by reviewers)

4. **Re-run the four-quadrant attack + 42% defense on the full graph** (matrix-free path already exists), not the 50-node subgraph. [C-4]
5. **Replace the 512-query black-box with NES/SimBA** at matched budget; re-report the non-circularity number. [C-5]
6. **Tighten theory scope.** [R1] Prop 1: state the maximizer over the symmetric/edge-supported subspace with magnitude `ε·σ₁(S_c)` (not `σ₁(S)`); Prop 3(b): label "pairwise-sufficient" and separate the empirical τ; Obs 1 + the Ω(1/(ε_crit−ε)) rate: title/scope to the all-active (φ′≡1) case, demote general-ReLU η∈[1.19,2.47] to clearly-empirical.
7. **Soften positioning over-claims** and add the nearest-neighbor literature. [R3] Reframe "returns all three"; lead the AGNNCert contrast with the low correlation. **Cite (already-orphaned-in-bib or standard):** oversquashing-via-curvature edge sensitivity — Topping et al. ICLR'22, Di Giovanni et al.; Topology Attack (Xu et al. IJCAI'19); node-injection insertion family (NIPA/AFGSM) as the concrete out-of-scope class; GNN stability (Gama et al.; Kenlay et al.); GCORN (Abbahaddou et al. ICLR'24).
8. **Power-flow out-of-distribution check.** [R2] The surrogate is trained on uniform 70–130% load only; N-1 is evaluated in-distribution. Add an out-of-envelope (redispatch/seasonal) operating-point test, or rescope "screening" → "in-distribution rank triage."
9. **Release a diagnostic-path code artifact** (anonymized) and specify the stratified-N-1, BFS-center/seed, and dataset-split protocols. [R2]

### P2 — Minor / polish

- Statistics: report effect sizes + CIs alongside p<10⁻⁴³ (large-n artifact); standardize SD vs 95%-CI usage (n=10); note multiple-comparison stance. [R2]
- η is double-defined (background vs theory); κ overloaded with cond(V_W) in Obs 1; "certified" used for a first-order threshold in tables. [R1]
- Pubmed 27.4% breach is the one consequential flip case — engage it, don't label-and-drop as "outlier." [R4]
- GAT† vs "any GNN" claim: standard GAT/GATv2/binary-mask are out of scope — state the boundary near the abstract claim, not only in §experiments. [R4]
- Bib hygiene: 36/92 entries uncited — prune or wire in. [R3]

---

## 5. What the paper already survives (do not over-correct)

- **Tautology / gradient-artifact charge** — defused by the separately-trained surrogate transfer (zero shared gradients, cos=0.99). The residual concern is *significance*, not circularity. [R4]
- **Scalability claims** — dense↔matrix-free σ₁ agreement (0.03%), 43–50% singular gap, N=7,650 on one GPU are concrete and real. [R4]
- **`r_v` "unsound"** — never claimed as a certificate (Rem. 3 explicit); "no breach below r_v" shown empirically. Keep as-is. [R1,R4]
- **Theorem correctness** — the resolvent/eigenvalue core is right and carefully hedged; the issue is the ε_crit *certificate framing* (P0-1), not the math. [R1,R4]

---

*Decision prepared by EIC synthesis over five independent reviews. Every roadmap item traces to a specific reviewer report; the C-2 soundness gap and the C-3 numeric contradiction were additionally re-verified by the editor.*
