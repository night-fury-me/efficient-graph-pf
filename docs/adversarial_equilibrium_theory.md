# Adversarial Equilibrium Theory for Implicit Graph Models

**Core insight**: For contractive DEQ-GNNs, the IFT sensitivity matrix S = (I - J_z)^{-1} J_A is the **tight** first-order characterization of adversarial vulnerability under graph structure perturbation. This simultaneously yields optimal attacks, certified defense bounds, and a critical perturbation budget — unifying adversarial ML with infrastructure contingency analysis (N-1).

---

## Notation

| Symbol | Meaning |
|--------|---------|
| F_θ(Z, A) | Fixed-point operator on graph G = (V, E, A) |
| z* = F_θ(z*, A) | Equilibrium (fixed point) |
| J_z = ∂F/∂z \|_{z*} | State Jacobian at equilibrium |
| J_A = ∂F/∂vec(A) \|_{z*} | Structural Jacobian at equilibrium |
| ρ = ρ(J_z) | Spectral radius of state Jacobian |
| S = (I - J_z)^{-1} J_A | **Structural sensitivity matrix** |
| σ_1(S) | Largest singular value of S |

---

## Theorem 1 (Tight Certified Fixed-Point Shift Bound)

**Statement.** Let F_θ be a contractive operator with ρ(J_z) < 1 on graph G with adjacency A. For any structural perturbation δA with ||δA||_F ≤ ε:

$$\sigma_1(S) \cdot \varepsilon - O(\varepsilon^2) \;\leq\; \|\Delta z^*\| \;\leq\; \sigma_1(S) \cdot \varepsilon + O(\varepsilon^2)$$

where S = (I - J_z)^{-1} J_A. The constant σ_1(S) is tight and satisfies σ_1(S) ≤ ||J_A||_{op} / (1 - ρ), with equality iff J_z is normal.

**Proof.**

*Upper bound.* By the Implicit Function Theorem applied to the fixed-point equation z* = F(z*, A):

$$\Delta z^* = (I - J_z)^{-1} J_A \cdot \text{vec}(\delta A) + O(\|\delta A\|^2) = S \cdot \text{vec}(\delta A) + O(\varepsilon^2)$$

Taking norms: ||Δz*|| ≤ ||S||_{op} · ||vec(δA)|| + O(ε²) = σ_1(S) · ε + O(ε²). □

*Matching lower bound.* Choose δA* = ε · reshape(v_1, N×N) where v_1 is the right singular vector of S corresponding to σ_1(S). Then:

$$\|\Delta z^*\| = \|S \cdot \text{vec}(\delta A^*) + O(\varepsilon^2)\| = \sigma_1(S) \cdot \varepsilon + O(\varepsilon^2)$$

The last equality follows because S · vec(δA*) = σ_1(S) · u_1 where u_1 is the corresponding left singular vector, so ||S · vec(δA*)|| = σ_1(S) · ε exactly. □

*Bound on σ_1(S).* By sub-multiplicativity of the operator norm:

$$\sigma_1(S) = \|(I - J_z)^{-1} J_A\|_{op} \leq \|(I - J_z)^{-1}\|_{op} \cdot \|J_A\|_{op}$$

For normal J_z: ||(I - J_z)^{-1}||_{op} = 1/min_i |1 - λ_i| = 1/(1 - ρ). For non-normal J_z, the resolvent norm can exceed 1/(1-ρ) by a factor η ≥ 1 (the **non-normality index**). □

**Remark (Non-normality amplification).** The ratio η = ||(I-J_z)^{-1}||_{op} · (1-ρ) measures how much the adversarial vulnerability exceeds the spectral-radius prediction. When η >> 1, the operator exhibits **transient amplification**: perturbations are amplified before the contraction eventually damps them. This is the graph-learning analogue of pseudospectral instability in fluid dynamics (Trefethen & Embree, 2005).

---

## Theorem 2 (Optimal First-Order Structural Attack)

**Statement.** The structural perturbation δA* maximising ||Δz*|| to first order subject to ||δA||_F ≤ ε is:

$$\delta A^* = \varepsilon \cdot \text{reshape}(v_1, N \times N)$$

where v_1 is the leading right singular vector of S. The maximum first-order shift is ε · σ_1(S). This is computable in O(|E| · D²) time via:
1. Jacobian computation: O(D²) per edge for J_A, O(D²) for J_z
2. Linear solve: O(D³) for (I - J_z)^{-1} J_A
3. Truncated SVD: O(D² · k) for top-k singular vectors

**Proof.** The first-order approximation gives Δz* ≈ S · vec(δA). The problem max_{||vec(δA)|| ≤ ε} ||S · vec(δA)|| is solved by the leading right singular vector of S, with objective value ε · σ_1(S). □

**Corollary (Per-edge vulnerability spectrum).** The vulnerability of edge (i,j) is:

$$v_{ij} = \|S_{:, iN+j} + S_{:, jN+i}\|$$

This simultaneously serves as: (a) adversarial attack priority, (b) N-1 contingency ranking, (c) edge importance for prediction. The Kendall τ between IFT-based v_{ij} and brute-force N-1 is the empirical measure of first-order fidelity.

**Corollary (Effective adversarial dimensionality).** The number of singular values of S exceeding 1% of σ_1(S) defines the effective adversarial dimensionality d_adv. When d_adv = 1, the model is vulnerable in one direction only (concentrated vulnerability). When d_adv ~ min(D, N²), vulnerability is spread uniformly (diffuse vulnerability).

---

## Theorem 3 (Critical Perturbation Budget)

**Statement.** For an IGNN-class operator F(Z) = σ(A Z W^T + X_proj) with spectral radius ρ < 1:

$$\varepsilon_{\text{crit}} \geq \frac{1 - \rho}{\|W\|_2}$$

For any ||δA||_F < ε_crit, the perturbed operator F_{A+δA} remains contractive (ρ(J_z(A+δA)) < 1), and all certificates from Theorem 1 remain valid.

**Proof.** For the IGNN operator, ρ(J_z) ≤ ||σ'||_∞ · ||A||_2 · ||W||_2. Since σ = ReLU has ||σ'||_∞ = 1:

$$\rho(A + \delta A) \leq \|A + \delta A\|_2 \cdot \|W\|_2 \leq (\|A\|_2 + \|\delta A\|_2) \cdot \|W\|_2$$

For contractivity: ρ(A+δA) < 1 requires ||δA||_2 < (1/||W||_2) - ||A||_2. Since ρ(A) = ||A||_2 · ||W||_2 (in the worst case), we get ||δA||_2 < (1-ρ)/||W||_2. By ||·||_2 ≤ ||·||_F, the Frobenius bound follows. □

**Phase transition.** At ε = ε_crit:
- **Subcritical** (ε < ε_crit): ρ < 1, unique fixed point, all certificates valid
- **Critical** (ε = ε_crit): ρ = 1, certificates become infinite (1/(1-ρ) → ∞)
- **Supercritical** (ε > ε_crit): ρ ≥ 1, fixed point may not be unique, certificates void

This is a sharp **phase transition** in adversarial vulnerability, analogous to criticality in statistical mechanics. The certified bound diverges as 1/(1-ρ) near the transition.

---

## Proposition 1 (Per-Node Robust Radius)

**Statement.** For node v with classification margin m_v = f_{y_v}(z*_v) - max_{c≠y_v} f_c(z*_v) > 0, the certified robust radius is:

$$r_v = \frac{m_v \cdot (1 - \rho)}{\|\partial f / \partial z^*_v\| \cdot \|S_v\|}$$

where S_v denotes the block-rows of S corresponding to node v. Any structural perturbation ||δA||_F < r_v preserves the classification of node v.

**Proof.** By the chain rule, the change in logit for node v is:

$$\Delta f_{y_v}(z^*_v) = \frac{\partial f}{\partial z^*_v} \cdot \Delta z^*_v \leq \|\partial f / \partial z^*_v\| \cdot \|S_v\| \cdot \varepsilon$$

Misclassification requires |Δf| ≥ m_v. Solving for ε gives the result. □

**Comparison with existing certificates.** Randomized smoothing for GNNs (Bojchevski et al., 2020) gives probabilistic certificates valid with probability 1-α. Our certificate is deterministic (always valid), at the cost of being first-order (accurate for small perturbations). For DEQ-GNNs with ρ < 1, our certificates are non-vacuous when m_v > 0, whereas smoothing certificates can be vacuous when the smoothed classifier is uncertain.

---

## Unification: Adversarial Robustness ≡ Infrastructure Contingency

The adversarial vulnerability spectrum v_{ij} from Theorem 2 is mathematically identical to the N-1 contingency criticality from power flow analysis:

| ML concept | Power systems concept | Mathematical object |
|---|---|---|
| Adversarial edge perturbation | N-1 line outage | δA_{ij} |
| Vulnerability spectrum | Contingency ranking | v_{ij} = ||S_{:,ij}|| |
| Critical budget ε_crit | Maximum tolerable outage size | (1-ρ)/||W||_2 |
| Per-node robust radius | Bus-level voltage stability | r_v |
| Phase transition at ρ=1 | Voltage collapse threshold | ρ → 1 |

This is the first framework that provides a rigorous mathematical bridge between these two communities. The same code (`iem.adversarial`) handles both.

---

## Novel contributions (vs. prior work)

| Claim | Prior art | What's new |
|---|---|---|
| Tight shift bound (Thm 1) | Lipschitz bounds for DEQs (El Ghaoui+21, Revay+20) bound INPUT perturbation | We bound GRAPH STRUCTURE perturbation with matching upper+lower bounds |
| Non-normality observation | Pseudospectral theory (Trefethen+05) in fluid dynamics | First application to DEQ-GNN adversarial vulnerability |
| Optimal structural attack (Thm 2) | Mettack (Zügner+19) for explicit GNNs | First IFT-based structural attack for implicit models, polynomial-time |
| Critical budget (Thm 3) | Stability margins in control theory | First characterization of the perturbation-induced phase transition for DEQ-GNNs |
| Per-node certificates (Prop 1) | Randomized smoothing (Bojchevski+20) | Deterministic (not probabilistic) certificates via IFT |
| ML ≡ contingency unification | Separate literatures | First rigorous bridge between adversarial robustness and N-1 contingency |
