# Reviewer 3 (Domain Expert) — AEGIS: Matrix-Free Diagnostics for the Adversarial Fault Lines of Graph Neural Networks

**Scope of this review:** literature positioning, novelty, related-work completeness, significance. I did not verify proofs or audit statistics (owned by R1/R2).

**Files read:** `sections/related_work.tex`, `sections/introduction.tex`, `sections/background.tex`, `sections/case_study.tex`, `sections/conclusion.tex`, `aegis.bib` (92 entries; histogram peaks 2019–2021, current through 2026; recency is good).

---

## 1. NOVELTY & DELTA

### What the contribution actually is
The core object `S_c` is implicit-function-theorem (IFT) sensitivity `∂z*/∂p = (I−J_z)^{-1} ∂F/∂p` — the Koh–Liang influence-function / Lorraine / Gould (deep declarative / hyperparameter-IFT) machinery — **re-pointed from loss/hyperparameter perturbations to a structural edge-weight perturbation**, via a projection `P_c` onto the edge subspace, made matrix-free (randomized SVD over the resolvent), and read three ways: leading singular direction (attack direction), column norms (per-edge ranking), per-node block norms (radius `r_v`).

### Strongest case FOR novelty
- The **specialization is non-obvious and correct in spirit**: prior IFT-on-graphs work differentiates through *features* or *loss*; differentiating the *operator itself* through both `J_z = diag(φ')(Â⊗W)` and `J_A = ∂F/∂vec(A)` is a genuinely different sensitivity object, and the background section (eq. for `J_z`, `J_A`) makes this distinction crisply.
- The **unification under one matrix-free pass is the real delta**: three diagnostics that the community currently obtains from three disjoint pipelines (attacks, explainers, certifiers) fall out of one `S_c` construction. That is a *framing/operator* contribution, not just an algorithm.
- Matrix-free scaling to N=7,650 with 0.03% agreement to dense σ₁ is a concrete systems contribution on top of the conceptual one.

### Strongest case AGAINST novelty
- Every *ingredient* is known: IFT sensitivity (Koh–Liang, Gould, Lorraine), resolvent/Neumann amplification (standard for DEQs — Bai, El Ghaoui, Revay), pseudospectra (Trefethen), randomized SVD (Halko). The novelty is **recombination + projection `P_c`**, which a skeptic will call incremental.
- The "**three diagnostics, no prior method returns all three**" claim is a *unification* claim, and unification claims are weak currency at ICDM unless the joint object enables something the parts cannot. The paper asserts joint provenance ("one query") but the *scientific payoff of jointness* (beyond convenience/runtime) is thin — direction, ranking, and radius are read off independently; nothing requires them to be computed together except efficiency.

### Is "no prior method returns all three" defensible?
**Partially, and it needs softening (MAJOR, see §2).** Two-of-three are arguably covered by single existing lines:
- **Edge ranking + a notion of direction**: gradient/surrogate attacks (Mettack, PR-BCD) already produce an edge-perturbation *direction* (the surrogate gradient) and an implicit ranking. The paper's distinction is "analytic, continuous, first-order SVD-optimal" vs. "discrete surrogate" — real but narrower than "they give none."
- **Radius + ranking**: certified-robustness work yields per-node radii, and several certificate papers *do* surface which edges/region drive the certificate (e.g., the sparsity-aware and collective certificates reason explicitly about influential perturbation sets). So "certifiers give no edge ranking" is *over-stated* for some members of that family.
- **GNNExplainer** already gives per-edge importance (the paper concedes this is "the closest analogue") — so the ranking leg is independently solved; AEGIS's claim there is *semantic* (vulnerability vs. fidelity), which is correct but should be stated as a re-purposing, not a new capability.

**Where ICDM lands:** I expect **borderline-positive on novelty**. The operator/unification framing plus the matrix-free realization clears the bar for a *methods* paper, but the "no prior method returns all three" sentence is the most attackable claim in the paper and will draw fire unless reframed as "no prior *single object/pass* returns all three" (which is true and defensible) rather than implying the *capabilities* are individually new.

---

## 2. RELATED-WORK COMPLETENESS

The bib is broad. The serious problem is **orphaned high-relevance entries**: several of the most topically-adjacent papers are present in `aegis.bib` but **cited nowhere in any section** (verified: 86 `\cite` commands, these keys appear in 0 source files outside the .bib). For an *edge-sensitivity* paper these are exactly the citations a domain reviewer looks for first.

### CRITICAL / MAJOR missing or under-used

**[MAJOR] Oversquashing-via-curvature line is in the bib but UNCITED.**
- Location: `related_work.tex` (Implicit/IFT paragraph) and `background.tex` (after the `J_z`/`J_A` decomposition).
- Gap: `topping2022understanding` (Topping et al., "Understanding over-squashing and bottlenecks on graphs via curvature," ICLR 2022) and `digiovanni2023oversquashing` (Di Giovanni et al. survey) are the **most theoretically relevant prior work to "which edges are most sensitive"** — they relate *edge-level structural sensitivity* (Jacobian sensitivity `∂h_v/∂x_u`) directly to graph curvature/bottlenecks. AEGIS's per-edge `v_{ij}` is conceptually a cousin of their sensitivity bounds. The theory section even uses "curvature" (for the second-order Taylor remainder) without connecting to *graph* curvature, which a reader will find conspicuous.
- Fix: add one-to-two sentences in Related Work distinguishing AEGIS's *adversarial/equilibrium* edge sensitivity from the *message-passing/oversquashing* edge sensitivity of `\cite{topping2022understanding,digiovanni2023oversquashing}` (and ideally Di Giovanni et al. "On over-squashing... width, depth and topology," ICML 2023, if added). This converts an orphan into a positioning asset.

**[MAJOR] Topology/structural-attack baseline `xu2019topology` is in the bib but UNCITED.**
- Location: `related_work.tex` "Structural attacks" sentence (currently Nettack/Mettack/GR-BCD/PR-BCD + `wu2019adversarial`).
- Gap: Xu et al. "Topology attack and defense for GNNs: An optimization perspective" (IJCAI 2019) is a standard structural-attack reference and is already sitting unused. Omitting it from the very sentence it belongs in is the kind of gap a structural-attacks reviewer flags immediately.
- Fix: add `\cite{xu2019topology}` to the structural-attacks enumeration (one-word change).

**[MAJOR] GNN-stability / spectral-sensitivity line under-cited (orphans present).**
- Location: `background.tex` (pseudospectral index `η`) or `related_work.tex`.
- Gap: `gama2020stability` (Gama et al., "Stability properties of GNNs"), `kenlay2021stability` (Kenlay et al., stability via spectral graph perturbation), `liu2021stability` (Liu et al., stability & generalization of GNNs) are all in the bib, all uncited. These are precisely the "sensitivity of GNN outputs to graph perturbations" literature; an edge-sensitivity paper that does not engage them looks unaware of its nearest non-adversarial neighbors.
- Fix: one sentence acknowledging the stability-theory line and contrasting *worst-case adversarial* edge sensitivity (AEGIS) vs. their *perturbation-stability bounds*.

**[MAJOR] GCORN missing entirely (a directly-adjacent 2024 norm-based robustness result).**
- Location: `related_work.tex` "Certified robustness and defense" paragraph.
- Gap: Abbahaddou, Ennadir, Lutzeyer, Vazirgiannis, Boström, "Bounding the Expected Robustness of GNNs Subject to Node Feature Attacks" (GCORN), **ICLR 2024** — connects robustness to **weight-matrix orthonormality/norms**, which is conceptually close to AEGIS's `‖W‖₂`-driven critical budget `ε_crit=(1−κ)/‖W‖₂`. Not in the bib at all. (Confirmed real via OpenReview/ICLR 2024.) Caveat: GCORN is *feature*-attack-scoped, so it is a contrast, not a competitor — but a reviewer will expect it cited given the norm/spectral framing overlap.
- Fix: add `\cite{abbahaddou2024bounding}` to the certification paragraph with a half-sentence noting it bounds robustness via weight orthonormality for *feature* attacks, complementary to AEGIS's structural `ε_crit`.

**[MAJOR] Node-injection attacks omitted — and this intersects the paper's OWN scoping claim.**
- Location: `background.tex` and `conclusion.tex` (both say "insertion attacks (Nettack-class) are deliberately scoped out").
- Gap: The paper frames insertion as "Nettack-class," but the canonical *insertion* threat model in the community is **node/graph-injection attacks** — NIPA (Sun et al., "Node injection attacks on graphs via reinforcement learning," 2020) and AFGSM (Wang et al., approximate fast gradient injection). These are the works a reviewer will cite back when the authors say "insertion is future work." Conflating insertion with Nettack (which is primarily edge/feature flips) is a small mischaracterization.
- Fix: in the scoping sentence, cite the node-injection line (`\cite{sun2020nipa}` / AFGSM) as the concrete insertion-attack family AEGIS does not yet cover, rather than labeling insertion "Nettack-class." Confirmed both exist; add only if you can fix the exact bib entry.

**[MINOR→MAJOR] Influence-functions-on-graphs specifically.**
- Location: `related_work.tex` (Implicit/IFT paragraph).
- Gap: The lineage cited (Koh–Liang, Pruthi TracIn `pruthi2020estimating` [orphan], Yeh representer `yeh2018representer` [orphan]) is generic deep-learning influence. There is graph-specific influence work the community associates with this exact move (e.g., influence/Jacobian-based GNN attribution). I am **not** confident enough in a single canonical "influence functions for GNN node classification" citation to name one without fabricating — I flag this as a gap to fill rather than prescribe a specific key. At minimum, the *already-present* orphans `pruthi2020estimating` and `yeh2018representer` should either be cited (they belong in this paragraph) or removed.

### Orphan hygiene (MINOR but a reviewer will notice)
36 of 92 bib entries are uncited. Several are clearly relevant and *should* be wired in (above). Others (`chamberlain2021grand`, `thorpe2022grand`, `chen2020simple`, `pei2020geomgcn`, `entezari2020all`, `geisler2020reliable`, `scholten2022randomized`, `scholten2023hierarchical`, `wang2021certified`, `li2025xgnncert`, `jia2020certified`, `ma2020towards`, `chang2020restricted`, `tang2020transferring`, `luo2020parameterized`, `luo2021learning`, `bojchevski2019adversarial`) are defensible to drop or to fold into the survey sentences. A large orphan set signals an under-developed related-work pass.

### Adequately covered (credit where due)
Structural attacks (Nettack/Mettack/PR-BCD), smoothing/IBP certificates (Bojchevski, Schuchardt collective, AGNNCert), sober-look evaluation discipline (Mujkanović, Gosch), implicit/equilibrium nets (DEQ, IGNN, monotone, JFB), and power-flow + contingency (Donon, Bolz, Nakiganda, Varbella, N-2 Yang, PTDF/LODF, PandaPower) are all present and mostly cited. Power-systems coverage is genuinely strong for an ML venue.

---

## 3. POSITIONING HONESTY

- **"Structural attacks ... provide no analytic maximally sensitive first-order direction and no per-node radius."** Fair on *radius*; slightly strawmanned on *direction* — surrogate gradients ARE a first-order direction, just discrete/non-SVD-optimal. Soften to "no *continuous, SVD-optimal closed-form* direction." (MAJOR-adjacent; ties to §1.)
- **"Certifiers ... do not identify vulnerability-driving edges."** Over-broad: collective/sparsity-aware certificates reason about influential perturbation sets. Soften to "do not return a *ranked per-edge sensitivity score*." (MAJOR, §2.)
- **GNNExplainer contrast** is handled honestly (explicitly named "closest analogue," fidelity-vs-vulnerability distinction is correct).
- **Threat-model scoping** (edge deletion / weight perturbation; insertion excluded; standard GAT excluded; binary-mask excluded): this is a **convenient narrowing**, and the paper is admirably explicit about it (background + conclusion limitations). It is a *reasonable* contribution boundary for a first paper because the IFT machinery genuinely requires a continuous, differentiable edge basis. BUT it materially limits impact: the two most practically common GNN threats — **node injection** and **attention-based (GAT) models** — are both outside scope. That is honest, not deceptive, yet a reviewer is entitled to weigh it against significance (see §4). The "extends once the basis is enlarged to Ē ⊇ E" promissory note is plausible but unproven.

Overall positioning is **mostly fair and unusually candid about limitations**; the two soften-the-verb fixes above are the only places it tips into strawman.

---

## 4. SIGNIFICANCE

**Who uses this:** practitioners deploying *equilibrium/implicit* GNNs (a small but growing slice) who want pre-deployment vulnerability triage; secondarily, the explicit-GNN extension widens this to any continuous edge-weight-modulated MPNN. The audience that benefits most is **robustness/ML-systems**, not classical data mining.

**Power-flow application:** This is the strongest cross-domain signal in the paper and **more than a thin add-on** — recovering brute-force N-1 critical-line severity (P@10 0.66–0.81, τ up to +0.62) *without the admittance matrix* is a concrete, checkable, domain-meaningful result, and beating standalone PTDF / matching LODF is a real baseline comparison. It is genuinely compelling as validation that `S_c` rankings track a physical ground truth. Caveats (uniform-load-only training, rank-triage not severity estimation) are stated honestly. This case study is what lifts the paper above "yet another sensitivity operator."

**ICDM fit:** This is the central tension. AEGIS is fundamentally a **robustness/sensitivity-analysis methods paper with a power-systems validation** — its natural homes are a robustness/ML venue (e.g., the certification/attack community) or a power-ML venue. For ICDM specifically, the **data-mining hook is the per-edge *mining* of vulnerability structure** (ranking which edges/subgraphs are fault lines), which is legitimately a graph-mining flavor, and the multi-domain empirical breadth (10 datasets, 5 domains, 390 runs) plays well at ICDM. I judge it **significant enough for ICDM but not a natural-fit slam dunk**: an ICDM PC may ask why this is not at a robustness venue. The authors should lean the framing toward *diagnostic graph mining of adversarial structure* (which they partly do via "fault lines") to consolidate venue fit.

**Bottom line on significance:** Real but **niche** contribution; the power-flow study is the differentiator that makes it worth accepting. Without that case study it would read as an incremental sensitivity-operator paper.

---

## SEVERITY-TAGGED ISSUE LIST

**CRITICAL:** none. (No directly-competing method is uncited in a way that *invalidates* the contribution; the closest competitor, GNNExplainer, is cited and honestly distinguished. The novelty claim is defensible once reframed — see below.)

**MAJOR**
1. `related_work.tex` / `background.tex` — Oversquashing-via-curvature edge-sensitivity line (`topping2022understanding`, `digiovanni2023oversquashing`) is **in the bib but cited nowhere**; it is the nearest theoretical neighbor to an edge-sensitivity paper. Fix: cite both and add one contrast sentence (equilibrium/adversarial vs. message-passing edge sensitivity).
2. `related_work.tex` — Structural-attack baseline `xu2019topology` (Topology Attack, IJCAI 2019) is in the bib but uncited. Fix: add to the structural-attacks enumeration.
3. `background.tex` / `related_work.tex` — GNN-stability/spectral-sensitivity line (`gama2020stability`, `kenlay2021stability`, `liu2021stability`) orphaned. Fix: one sentence contrasting worst-case adversarial sensitivity with perturbation-stability bounds.
4. `related_work.tex` (certification para) — Missing **GCORN** (Abbahaddou et al., ICLR 2024, "Bounding the Expected Robustness of GNNs..."), conceptually close via weight-norm/orthonormality robustness. Fix: add `\cite{abbahaddou2024bounding}` with a half-sentence (feature-attack-scoped, complementary to structural `ε_crit`).
5. `intro.tex` / `related_work.tex` — "No single method returns all three / certifiers identify no vulnerability-driving edges / attacks give no direction" is over-claimed; collective certificates surface influential edge sets and surrogate gradients are a (discrete) direction. Fix: reframe to "no prior *single object/matrix-free pass*" and "no *ranked per-edge sensitivity score* / no *continuous SVD-optimal* direction." This protects the headline claim.
6. `background.tex` / `conclusion.tex` — Insertion scoped out as "Nettack-class," but the canonical insertion family is **node-injection** (NIPA, AFGSM), uncited. Fix: cite the node-injection line as the concrete out-of-scope insertion family.

**MINOR**
- `related_work.tex` (IFT para) — Generic influence orphans `pruthi2020estimating` (TracIn) and `yeh2018representer` (representer points) belong in this paragraph or should be removed; a graph-specific influence-function citation would strengthen the lineage (flagged as gap; not prescribing a specific key to avoid fabrication).
- Bib hygiene — 36/92 entries uncited; prune or wire in. A large orphan set reads as an unfinished related-work pass to a domain reviewer.
- `case_study.tex` — Consider citing the node-injection/insertion absence as an explicit power-flow limitation too (line *insertion* = new tie-line, physically meaningful), reinforcing scope honesty.
- `related_work.tex` — `li2025xgnncert` (robust *explainable* GNN certificates, 2025) is in the bib, uncited, and is relevant to the explainer-vs-AEGIS contrast; consider one citation.

---

## VERDICT (positioning/novelty axis)

Genuine *operator/unification + matrix-free realization* contribution, validated by a credible cross-domain power-flow study; **borderline-accept on novelty/positioning** for ICDM, conditional on (a) reframing the over-claimed "all three / certifiers give no edges / attacks give no direction" statements, and (b) wiring in the orphaned and missing nearest-neighbor literature (oversquashing-curvature, topology attack, GNN stability, GCORN, node-injection). No issue is fatal; the listed MAJORs are the bar a domain reviewer would require cleared before acceptance.
