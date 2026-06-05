# Editorial Decision — AEGIS (AAAI-2026 anonymous submission)

**Paper:** *AEGIS: A Matrix-Free Operator to Audit, Certify, and Defend Graph Neural Networks*
**Panel:** EIC + Methodology (R1) + Theory/proof-audit (R2, via ml-theory-reviewer) + Cross-disciplinary (R3) + Devil's Advocate (R4)
**Date:** 2026-06-05 (fresh panel; independent of the prior R1–R4 round)
**Reviews on file:** `EIC.md`, `R1_methodology.md`, `R2_theory.md`, `R3_perspective.md`, `R4_devils_advocate.md`

---

## DECISION: MAJOR REVISION

Unanimous across all five reviewers (5/5 Major Revision, confidence 4/5 each). The Devil's Advocate raised a CRITICAL issue, so per panel rules an Accept is precluded regardless; but the recommendation is independently unanimous.

### Score summary (0–10)

| Dimension | EIC | R1 Meth | R2 Theory | R3 Persp | R4 DA | Mean |
|---|---|---|---|---|---|---|
| Novelty | 6 | 7 | 4 | 7 | 4 | **5.6** |
| Soundness | 7 | 4 | 6 | 5 | 6 | **5.6** |
| Clarity | 8 | 8 | 7 | 7 | 8 | **7.6** |
| Significance | 5 | 6 | 6 | 6 | 4 | **5.4** |
| Reproducibility | 8 | 7 | 7 | 7 | 7 | **7.2** |

---

## The throughline (editor's synthesis)

The paper is **well-written** (Clarity 7.6) and the central object — a single matrix-free operator $S_c$ read out three ways — is a **clean, genuinely useful organizing idea** with disciplined 10-seed empirics and strong reproducibility. Every reviewer affirmed this.

The problem is singular and consistent: **nearly every headline claim is over-calibrated relative to what is delivered.** The paper's own body and appendix already contain the honest caveats; they have simply not been lifted to the abstract/intro, and the marquee numbers are reported on axes that flatter the method.

- The attack "matches/beats 50-step PGD (74–156×)" — but only on *equilibrium/hidden-state shift* $\|\Delta z^\star\|$, the quantity the SVD direction maximizes by construction; **prediction flips stay <2%** at the same budget (R1-C1, R4-M1).
- The certificate is "non-vacuous where smoothing abstains, at $10^4\times$ lower cost" — but **AEGIS-Conformal also abstains on the matched Frobenius ball** (Tab. smoothing); it certifies only on a larger, easier per-coordinate ball (R4-C1, R1-M5, R3-M1, EIC-M3).
- The "unification" and "scales to $N{=}7{,}650$" hold mainly **outside the contractive regime the theory actually certifies** ($\tau{=}0.996$ at $\kappa{\approx}1.00$, where the truncated Neumann series has no convergence guarantee) (EIC-C1/C2, R3-C1, R4-M2, R2-M3).
- The displayed conformal guarantee Eq. (conformal) is printed **unconditionally**, but only the **exchangeability-conditional** version is proved, and the white-box threat model can break exchangeability (R2-CRITICAL-1).

**Implication for the authors:** the remedy is mostly **claim re-calibration + four targeted experiments**, not new theory. This is a fixable Major Revision, not a Reject. The risk if unaddressed is that a hostile AAAI reviewer reaches the same matched-ball / decision-flip observations and recommends rejection for over-claiming.

---

## Cross-reviewer consensus matrix

Severity per reviewer (C = Critical, M = Major, · = raised as minor/in passing, blank = not raised). "N" = consensus count.

| Issue | EIC | R1 | R2 | R3 | R4 | N | Verdict |
|---|---|---|---|---|---|---|---|
| **A. Attack metric not decision-relevant** (flips <2%; $\|\Delta z^\star\|$ flatters) | (sig) | **C** |  |  | **M** | 3 | **CRITICAL** |
| **B. Smoothing/conformal comparison not apples-to-apples** (AEGIS also abstains on matched Frobenius ball; "$10^4\times$" = 0-sample vs $10^4$-sample) | M | M |  | M | **C** | 4 | **CRITICAL** |
| **C. Conformal coverage printed unconditional; only exchangeability-conditional proved; threat model breaks it** | M |  | **C** |  | · | 3 | **CRITICAL** |
| **D. "Unification"/breadth overclaimed**; theory restricted to niche contractive IGNN, decoupled from empirical flagship; pillars trail their baselines | **C×2** |  | M |  | **M×2** | 3 | **CRITICAL** |
| **E. $N{=}7{,}650$ scaling computed at $\kappa{\approx}1.00$** where Neumann has no convergence guarantee | · | (M3) |  | **C** |  | 2 | **CRITICAL** |
| F. Two-sided bracket too loose / $\beta{=}\Theta(1/\sqrt N)$ vacuity; "constant-factor characterisation" oversold | M |  | M |  |  | 2 | MAJOR |
| G. Theory novelty thin over Neumann/resolvent theory; undelimited | (C2) |  | M |  | (CA) | 2 | MAJOR |
| H. $\tau{=}0.98$ carried by edge weight; vs *independent* GR-BCD only ~0.5; no CIs; Prop. transfer proves pairwise, not global $\tau$ | | M | M |  | · | 3 | MAJOR |
| I. External validity: 50-node BFS ego-subgraph; own ablation shows it doesn't generalize (subgraph→full $\tau{=}0.16$) |  | M |  | (·) | · | 2 | MAJOR |
| J. Statistics: $p{<}10^{-43}$ / $-0.65$ — no stated test, no multiplicity control, n inadequate, ± = seed-std only |  | M |  |  |  | 1 | MAJOR |
| K. "One query vs 125 epochs" compute framing overstated (GR-BCD 0.19s < AEGIS 0.24s on attack axis) |  | M |  |  |  | 1 | MAJOR |
| L. Pseudospectral $\eta$ machinery decorative; numbers measured not derived; changes no conclusion |  |  | (·) | M |  | 1 | MAJOR |
| M. Deployment evidence thin: fraud n=1 ($\tau{=}1.0$, no variance); grid story rescoped to nil but intro still opens on it |  |  |  | M | · | 2 | MAJOR |
| N. $\varepsilon_{\mathrm{crit}}$ track trails best explicit arch by ~5 acc. points (only closed-form-certified class is also accuracy-dominated) | M |  |  |  |  | 1 | MAJOR |
| O. 90-day disclosure window asserted, not justified; the "harmless" diagnostic path is itself a $\tau{=}0.98$ attack prioritizer |  |  |  | M |  | 1 | MAJOR |
| P. Defense's only edge over generic $\|W\|$ cap buried in appendix, no number in Tab. defense |  |  |  |  | M | 1 | MAJOR |

**Five CRITICAL threads (A–E).** A, B, D are the *same disease* (claims out-calibrated vs delivery). C and E are specific soundness/numerical gaps.

---

## Arbitration of divergences

1. **Soundness 7 (EIC) vs 4 (R1).** Not a contradiction: EIC explicitly defers fine methodology and scored organizational/theoretical soundness; R1 is the empirical-soundness specialist and found the metric + comparison defects. **Because the paper's headline claims are empirical, empirical-evaluation soundness is the binding dimension** — the decision weights R1/R4 here. Theory soundness (R2 = 6) is moderate but gated by the conformal overclaim (C).

2. **Novelty 4 (R2, R4) vs 6–7 (EIC, R1, R3).** The *perturbation theory* is thin (R2 authoritative: $1/(1-\kappa)$ is the standard Neumann bound). The *engineering-level* novelty — one operator feeding attack+radius+penalty, the conformal-over-$\varepsilon$-ball construction vs Zargarbashi's fixed-graph conformal, and the operational $\sigma_1(S_c)$ defense coupling — is moderate. **Net: novelty is real but concentrated in the unification + conformal-over-ball, not in new theory.** Authors should relocate the novelty claim there and stop selling Thm. 1 as a new phenomenon.

3. **Significance** (4–6): genuine consensus that this is the weakest dimension, driven by D (the transferable contribution, once the contractive theory is stripped, is an incremental edge ranking).

**Points of unanimous agreement (weight heavily):** the operator idea is clean and the writing is strong (all reviewers); claims are mis-calibrated (all five, via A/B/D); the certificate comparison needs a matched-ball restatement (4/5).

---

## Revision Roadmap (prioritized; directly consumable by academic-paper revision mode)

Tags — **[REFRAME]** wording/claim calibration, **[EXP]** new/extended experiment, **[PROOF]** statement/proof change, **[STATS]** statistical reporting, **[WRITE]** exposition. Effort: S/M/L.

### P0 — CRITICAL (must close before resubmission; map to threads A–E)

- **P0-1 [EXP+REFRAME, M] Decision-level attack comparison (thread A).** Add a flip-rate / accuracy-drop vs $\varepsilon$ curve at matched budget, AEGIS vs Cls-PGD vs Shift-PGD vs Mettack vs GR-BCD, same axis, with CIs. If flips remain ~0 in the cited regime, **retitle the attack as a label-free first-order *equilibrium-shift diagnostic*** and qualify the "74–156× / beats PGD" numbers inline in abstract+intro as per-query equilibrium damage. Promote **Shift-PGD to a first-class matched baseline** and report AEGIS/Shift-PGD as the honest solver-efficiency claim (R1-C1, R1-C2, R4-M1).
- **P0-2 [EXP+REFRAME, M] Matched-ball certification (thread B).** Report a quantity on the **same Frobenius ball** where AEGIS certifies a non-trivial fraction and smoothing does not — or delete the "non-vacuous where smoothing abstains" clause. State the $10^3$–$10^4\times$ cost ratio **only where both deliver a comparable guarantee**; label it zero-sample-bound vs $10^4$-sample-sampler. Move the appendix concession ("favors smoothing twice") into the abstract (R4-C1, R1-M5, R3-M1, EIC-M3).
- **P0-3 [PROOF+REFRAME, S] Fix the conformal guarantee statement (thread C).** Lift the **(C1) exchangeability** condition onto the displayed Eq. (conformal) and the abstract; **separate "score-shift envelope (proved, adversary-uniform)" from "coverage (assumed under exchangeability + empirical gate)."** Name the deployment protocol that enforces exchangeability under attack; report where the gate turns conservative/vacuous (R2-CRITICAL-1, EIC-M3).
- **P0-4 [EXP+REFRAME, M] Full-graph numerics at $\kappa{\approx}1$ (thread E).** Report the actual $N{=}7{,}650$ protocol (Neumann terms used, realized residual, whether tolerance was met). Switch the $\kappa{\approx}1$ case to a **Krylov/GMRES resolvent solve** (converges by eigenvalue clustering, not $\kappa<1$) with a reported residual; add a $\kappa$-vs-truncation-error curve; tabulate per-dataset $\kappa$. Mark $\kappa{\approx}1.00$ results as empirical-only (R3-C1).
- **P0-5 [REFRAME, M] Re-scope "unification" and decouple proven-vs-observed (thread D).** Retitle/reword to: *"a unified sensitivity operator that **audits broadly** and **certifies/defends provably in the contractive regime**."* In abstract + main text, separate **proven (contractive IGNN)** from **observed (general architectures)**. Either (a) demonstrate the certificate + $\sigma_1$ defense coupling on **≥1 deployed explicit architecture**, or (b) foreground the two genuinely transferable novel pieces (conformal-over-ball; $\sigma_1$ defense) as the contribution and present the ranking as supporting (EIC-C1, EIC-C2, R4-M2, R4-M3, R2-M).

### P1 — MAJOR (expected for a credible revision)

- **P1-1 [PROOF, M] Bracket (Thm. cf2s) (F).** Report $\beta(N)$ on full graphs; the "$\lesssim 16\times$" is a 50-node artifact ($\beta{=}\Theta(1/\sqrt N)\Rightarrow C/\beta{=}\Theta(\sqrt N)$). Either restate the extremiser via the $P_E$-restricted map (so $\beta$ disappears) or scope "constant-factor / non-vacuous" to the subgraph regime. Reserve the word **"certificate"** for AEGIS-Conformal; call the bracket a qualitative phase-transition/coupling result (R2-M3, EIC-M2).
- **P1-2 [WRITE, S] Delimit theory novelty (G).** State explicitly which steps of Thm. 1 are standard (Neumann gain, Horn-Johnson/Trefethen) vs the contribution ($\varepsilon_{\mathrm{glob}}$, $\eta{\le}\kappa(V_W)$, the $S_c$ unification). Retitle Thm. 1 away from "phase transition / vulnerability characterization" implying a new phenomenon (R2-M2).
- **P1-3 [STATS, M] $\tau$ headline (H).** Lead with **$\Delta\tau$ (sensitivity's marginal over edge weight) + bootstrap CIs**, not the raw $\tau{=}0.98$ (which the weight largely carries). Report $\tau$ vs the **independent** GR-BCD ($\approx0.35$–$0.81$) with equal prominence. State the weight's dominance in the abstract. Label $\tau$ empirical and move it out of the Prop. transfer proof environment (R1-M6, R2-M4).
- **P1-4 [EXP, M] External validity (I).** Foreground full-graph results for every claim meant to generalize; tabulate subgraph-vs-full $\tau$ for all datasets; show whether the defense effect survives at scale (own ablation: 42–61%→2.4–4.6%) (R1-M3).
- **P1-5 [STATS, S] Significance testing (J).** Name the test and its independence assumptions; test at dataset level (n=6) or use cluster-robust inference; report effect sizes with bootstrap CIs; apply Holm/BH across the $6\times7$ grid; drop the headline $p{<}10^{-43}$ (R1-M7).
- **P1-6 [REFRAME, S] Compute framing (K).** Separate axes: attack/ranking is compute-comparable to GR-BCD (0.24s vs 0.19s) — the real win is *label-free + certificate-bearing*; confine "$\sim700\times$ / $10^4\times$" to certificate-vs-smoothing (R1-M4).
- **P1-7 [WRITE/EXP, M] Pseudospectral apparatus (L).** Label the all-active $\eta{\le}\kappa(V_W)$ as the only derived result; relabel $[1.19,2.47]$ and "7–14×" as empirical; either tie the pseudospectral framing to a bound it actually controls or trim it (R3-M2).
- **P1-8 [EXP, M] Deployment evidence (M).** Add 5–10 fraud clusters with a $\tau$ distribution (and a failure case) or demote the single $\tau{=}1.0$ hub to an illustration. Reframe the power grid as general motivation, not a delivered capability; soften the intro's grid opening to match the conclusion's concession (R3-M3).
- **P1-9 [WRITE/EXP, S] Closed-form regime is accuracy-dominated (N).** Plot the accuracy/certifiability frontier and name the deployment (regulated/audited) where ~5 accuracy points buys a closed-form budget; else soften "deploy safely" (EIC-M1).
- **P1-10 [WRITE, S] Disclosure (O).** Justify the 90 days, name the notification target, define the ethics gate concretely, and **quantify attack damage from the released ranking alone vs the gated $v_1$ reconstruction** — if the gap is small, acknowledge the gating is largely cosmetic (R3-M4).
- **P1-11 [EXP, S] Defense vs $\|W\|$ cap (P).** Promote the matched $\sigma_1$-penalty-vs-Lipschitz-cap frontier into the main paper with a numbered, strictly-better Pareto point in/near Tab. defense (R4-M3).

### P2 — MINOR (polish; in individual reports)
GAT$^\dagger$ counts toward "7 architectures" but is modified; $-0.65$ coupling is partly mechanical (acknowledge); exchangeability caveat absent from abstract; clarify whether the "non-vacuous certificate" claim refers to deterministic $r_v$ or AEGIS-Conformal (R3 flags an attribution slip); plus the per-report MINOR lists in `R1`–`R4`/`EIC`.

---

## Devil's Advocate "So what?" verdict (flagged for the authors)
> For a practitioner who already owns an attacker + a smoothing certifier + a robust-training knob, AEGIS currently delivers **convenience, not capability** — a unified, scalable, model-intrinsic *diagnostic operator*, but one that ties or trails every dedicated baseline at its own task and flips ~0 predictions. The revision must either (i) show a task where the *union* of off-the-shelf tools cannot match the single operator, or (ii) reposition the paper honestly as a unified diagnostic + a novel conformal-over-$\varepsilon$-ball certificate, and let that narrower-but-true claim carry it.

## What is genuinely strong (defend, don't dilute)
Clean organizing idea; the matrix-free $O(|E|)$ construction; exemplary scope honesty already present in the body; the non-circular audit evidence (separately-trained surrogate recovers 99% of damage; finite-difference $\tau{=}0.999$); the operational attack↔defense coupling; disciplined 10-seed protocol and strong reproducibility.
