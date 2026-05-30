# Revision C2 — Re-scope Theorem 1 (DONE, honest-not-undersold)

**Concern (R1 — methodology):** W1 (CRITICAL) — A1 advertises "ReLU or any 1-Lipschitz," but the *sharp* regime-(b) rate `Ω(1/(ε_crit−ε))` and the η-slack bound are proved only for the all-active case (φ'≡1; Obs 1 titled "all-active case", Rem 1 concedes ReLU η∈[1.19,2.47] empirical). W2 (Major) — the `L_J` finiteness uses `‖z*‖≤‖X_proj‖/(1−‖Â‖₂‖W‖₂)`, whose denominator R1 worried "can be ≤0 for a contractive model."

## Strategic principle applied: **scope the over-claim, don't dilute the claim**
The theorem genuinely *characterises three regimes* for the contractive IGNN — that part is honest and strong. Only the **sharp rate / η-bound in regime (b)** is the all-active part. So we keep "three-regime characterisation" and scope *only* the rate. (First draft over-hedged abstract to "safety boundary" — reverted, because that needlessly weakened an accurate, strong descriptor.)

## Edits
- **Abstract:** kept "closed-form **three-regime characterisation** with critical budget ε_crit", added "**(rate sharp in the all-active case)**", kept "**2–4× margin**". Honest scope, full strength.
- **Theorem regime (b):** η-slack now explicit — "bounded by η (`obs:eta_bound`; **η≤2.47 for general ReLU**, `rem:eta_relu`)" — states the *tight empirical bound* (a strength), not a bare "empirical" hedge.
- **Prop. transfer proof (W2) — DEFENDED, not weakened:** the `L_J` denominator is positive because `‖Â‖₂‖W‖₂<1` is the IGNN **well-posedness contraction** (standing existence condition from §background), and it *implies* A3 since `κ≤‖Â‖₂‖W‖₂`. Added "which implies (A3) since κ≤‖Â‖₂‖W‖₂."

## Rigor finding: R1's suggested W2 fix was itself wrong
R1 suggested "bound via `1/(1−κ)` using A3 only." That is an **invalid tightening**: the ‖z*‖ geometric series is governed by the *global* Lipschitz constant `‖Â‖₂‖W‖₂`, not the *local* Jacobian norm κ (κ≤‖Â‖₂‖W‖₂, so 1/(1−κ)≥1/(1−‖Â‖₂‖W‖₂) — using κ would under-bound). The correct, honest fix is the well-posedness framing above. (Verified: z*=φ(Âz*Wᵀ+X_proj), ‖φ‖ 1-Lipschitz with φ(0)=0 ⟹ ‖z*‖(1−‖Â‖₂‖W‖₂)≤‖X_proj‖.)

## Length management (10pp held)
Offsetting trims (no claim touched): ε_crit "in practice / 50-node" aside; Neumann-vs-ρ aside; `|R_k|` "vacuous→tight" wording. Build: 10 pages, 0 overfull, 0 errors.

## Not changed
- Intro contribution (2) already reads "a **provable** structural-perturbation **safety threshold** … 2–4× margin" — honest and strong; left as-is.
- Theorem assumptions A1–A3, the unconditional resolvent bound, Props 1/2/4 — all correct per R1; untouched.
