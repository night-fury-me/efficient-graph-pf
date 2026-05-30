# Reviewer 4 — Perspective (Power Systems)

## Summary

The paper applies $S_c$ to AC power flow on IEEE case14/30/57/118 (and case300 as scalability stress-test) via a ContractiveGCN-PF model trained on 2,000 PandaPower-generated load samples per case. Reported metrics: Kendall $\tau = +0.37$ to $+0.62$ on case14–118 vs brute-force N-1 contingency; P@10 = 0.66–0.81; LODF baseline P@10 $\le 0.20$ on case118. The paper consistently labels this as proof-of-concept and disclaims operator-grade use.

I read this as a power-systems audience reader. The framing has improved substantially over the original (case300 explicitly flagged as un-converged; abstract and conclusion both say "proof-of-concept"). Three substantive concerns remain.

## Domain validity of the isomorphism

The case study leans on a *correspondence*: the IFT-resolvent $(I-J_z)^{-1}$ at the GNN's fixed point is the "linearised analogue" of the post-contingency steady-state operator; $v_{ij}$ tracks N-1 severity the way PTDF/LODF linearise line trips for DC-PF.

**Concern P.1 (must address).** A correspondence is not a derivation. PTDF is derived from DC-PF as $\partial P_{ij} / \partial P_k = H_{ij,k}$ where $H$ is the inverse susceptance matrix; LODF is a one-line consequence of PTDF. AEGIS's $v_{ij}$ is derived from a GNN's IFT. The claim "$S_c$ is the linearised analogue of the post-contingency operator" requires either (a) a formal derivation showing that the GNN's $J_A$ corresponds to $\partial f / \partial \theta$ in PF — i.e., the GNN has actually learned PF physics — or (b) honest re-labeling as "empirical correlation; the structural form of the IFT mirrors the form of the line-outage distribution factor but the coefficients are not physically interpretable."

The Table 1 row case300 with $\theta$ RMSE = 22.6° provides a clean empirical proof of (b): the GNN has *not* learned AC-PF physics, yet the framework still flags $\tau$ values. This is consistent with the empirical-correlation reading. Adopt (b) explicitly.

## Operating-condition envelope

Training data: 2,000 samples per case at 70–130 % uniform load scaling.

**Concern P.2 (must address).** This is an extremely narrow envelope. Real contingency screening must hold across (a) seasonal peaks / valleys, (b) generator-dispatch variations including merit-order shifts, (c) wind / solar net-load ramps and curtailment, (d) post-contingency redispatch states. The current "Limitations of setup" paragraph in §VII acknowledges this. Strengthen the Table 1 caption to say: *"all rows conditional on uniform load scaling within 70–130 % of nominal; rankings derived here may not generalise to stressed conditions where contingencies are most dangerous."* The case14–118 numbers should not be quoted (in the abstract or conclusion) without this qualifier.

A more ambitious next step — out of scope for this paper but worth flagging in §Future Work — is evaluation on Grid2Op or the GO-Competition stressed-grid scenarios. These provide the directional load shifts, generator outages, and renewable ramps that uniform scaling cannot.

## Binary adjacency vs admittance-weighted edges

The paper reports that binary adjacency *outperforms* a naive admittance-weighted variant. This is a striking result — and a warning.

**Concern P.3 (must address).** In a DC-PF model the relevant Laplacian is $B \theta = P$ where $B$ is the susceptance matrix; $B_{ij}$ is the per-line susceptance. PTDF coefficients are exact functions of $B^{-1}$. A binary-adjacency GNN cannot recover line-specific impedance; it can only learn an averaged proxy. The fact that binary adjacency *wins* against naive admittance weighting suggests two things: (a) the naive admittance weight (whatever scaling was used) was poorly normalised relative to the GNN's input range, and (b) the GNN is implicitly memorising case-specific topology rather than learning admittance-weighted message passing. Both are signs that $v_{ij}$ on these grids is closer to a *graph-topology heuristic* than a *physics-informed sensitivity*.

Recommended: (i) report the exact admittance normalisation used and (ii) add a third row, "log-admittance edge weights with input normalisation," before concluding that binary adjacency is preferable. The current claim invites the wrong takeaway — that admittance information is unhelpful.

## LODF comparison

LODF P@10 $\le 0.20$ on case118 vs AEGIS P@10 = 0.81 is reported in Table 1.

**Concern P.4 (must address).** LODF is computed on DC-PF; the AC contingency-severity ground truth used by AEGIS is computed via PandaPower's Newton–Raphson AC solver. Make explicit which contingency-severity metric defines the ground truth — line-flow deviation, voltage-violation count, generation-imbalance? If the ground truth is AC-derived, LODF (a DC linearisation) is at a built-in disadvantage and the comparison is unfair to LODF. State the metric explicitly in the table caption and, if AC-derived, add a DC-PTDF-vs-AEGIS comparison restricted to DC ground truth so the framework's advantage over the right baseline is auditable.

Additionally: PTDF (not just LODF) is the standard sensitivity baseline; LODF is the line-outage variant. Report PTDF–edge correlations as well as LODF for a complete comparison.

## Operational utility

The paper correctly states $\tau = +0.37$ to $+0.62$ is insufficient for direct operator-grade screening. With current numbers, AEGIS is a *candidate-list reducer* that runs upstream of full N-1 simulation, not a replacement.

**Concern P.5 (should address).** Add a one-paragraph operational utility framing in §Case Study: "AEGIS-as-prefilter would reduce the candidate set from $|E|$ contingencies to top-$K$, then full Newton–Raphson is run on the top-$K$. P@10 = 0.81 on case118 means 8 of the 10 most-severe contingencies are caught in AEGIS's top-10; an operator using $K = 20$ would catch 16+ of the top-10 with no false-positive cost beyond the doubled simulation count." This converts a soft proof-of-concept into a credibly-positioned screening contribution.

## Decision lean

**Major Revision.** The case study is honestly disclosed as proof-of-concept. The technical concerns (correspondence vs derivation, operating envelope, binary-vs-admittance, AC-vs-DC ground truth) are addressable in revision. Recommended scope: re-frame as a *graph-machine-learning paper with a power-systems proof of concept*, not as a power-systems contribution.
