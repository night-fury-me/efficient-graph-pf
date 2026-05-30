# AEGIS Framework Architecture — Diagram Specification

**Style reference:** Layered horizontal pipeline (similar to the cryptographic watermarking diagram), with color-coded layers, processing blocks connected by arrows, input icons on the left, output icons on the right, and an extension/application strip at the bottom.

---

## Layout: 4 horizontal layers + input/output strips

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUTS (left strip)                                                        │
│  ┌──────────┐  ┌──────────┐                                                │
│  │ Trained   │  │ Target   │                                                │
│  │ GNN f_θ   │  │ Graph G  │                                                │
│  │ [icon:    │  │ [icon:   │                                                │
│  │ neural    │  │ graph    │                                                │
│  │ network]  │  │ nodes]   │                                                │
│  └─────┬─────┘  └─────┬───┘                                                │
│        └──────┬────────┘                                                    │
│               ▼                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — GRAPH PREPROCESSING (light blue, #D6EAF8)                       │
│                                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │  BFS Subgraph │ ──► │  Forward     │ ──► │  Fixed-Point │                │
│  │  Extraction   │     │  Pass        │     │  Convergence │                │
│  │               │     │              │     │              │                │
│  │  Center: max  │     │  Z = f_θ(X,A)│     │  Z* : F(Z*,A)│              │
│  │  degree node  │     │              │     │  = Z*        │                │
│  │  N_sub ≤ 200  │     │              │     │  (IGNN only) │                │
│  └──────────────┘     └──────────────┘     └──────┬───────┘                │
│                                                    │                        │
│                           Outputs: Z*, A_sub, ctx  │                        │
├────────────────────────────────────────────────────┼────────────────────────┤
│  INTERFACE: Auto-select dense (N≤200) or matrix-free (N>200)               │
│             ┌──────────────────────────────────────┐                        │
│             │     N ≤ 200? ──► Dense Path          │                        │
│             │     N > 200? ──► Matrix-Free Path    │                        │
│             └──────────────────────────────────────┘                        │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — SENSITIVITY COMPUTATION (light green, #D5F5E3)                  │
│                                                                             │
│  ┌─ Dense Path (N ≤ 200) ──────────────────────────────────────────────┐   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────────┐                  │   │
│  │  │  J_z via  │ ─► │  J_A via │ ─► │  Linear Solve│                  │   │
│  │  │  Autograd │    │  Autograd│    │  S=(I-J_z)⁻¹ │                  │   │
│  │  │           │    │          │    │  J_A          │                  │   │
│  │  │  O(D²)    │    │  O(D×N²)│    │  O(D³)       │                  │   │
│  │  └──────────┘    └──────────┘    └──────┬───────┘                  │   │
│  └─────────────────────────────────────────┼──────────────────────────┘   │
│                                             │                              │
│  ┌─ Matrix-Free Path (N > 200) ────────────┼──────────────────────────┐   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────┴───────┐                  │   │
│  │  │  ρ via    │ ─► │  JVPs via│ ─► │  Neumann     │                  │   │
│  │  │  Power    │    │  torch.  │    │  Series      │                  │   │
│  │  │  Iteration│    │  func.jvp│    │  Σ J_z^k b   │                  │   │
│  │  │           │    │          │    │  O(K·D)      │                  │   │
│  │  └──────────┘    └──────────┘    └──────┬───────┘                  │   │
│  └─────────────────────────────────────────┼──────────────────────────┘   │
│                                             │                              │
│                              S (unconstrained sensitivity)                  │
├─────────────────────────────────────────────┼──────────────────────────────┤
│  LAYER 3 — CONSTRAINED PROJECTION (light orange, #FDEBD0)                 │
│                                             ▼                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              S ──► P_c ──► S_c                                   │      │
│  │                                                                   │      │
│  │  Projection: N² dims ──► |E| dims                                │      │
│  │  [S_c]_{:,k} = S_{:,iN+j} + S_{:,jN+i}  for each edge (i,j)   │      │
│  │  Enforces: symmetric + edge-only                                  │      │
│  │                                                                   │      │
│  │  "The central technical contribution"                             │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                              │                                             │
│                              ▼ S_c (constrained sensitivity matrix)        │
├──────────────────────────────┼──────────────────────────────────────────────┤
│  LAYER 4 — VULNERABILITY ANALYSIS (light purple, #E8DAEF)                  │
│                              │                                             │
│          ┌───────────────────┼───────────────────┐                         │
│          ▼                   ▼                   ▼                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                   │
│  │  Randomized   │   │  Column Norms│   │  Per-Node    │                   │
│  │  SVD of S_c   │   │  of S_c      │   │  Radius      │                   │
│  │               │   │              │   │              │                   │
│  │  (U,σ,V) =   │   │  v_k =       │   │  r_v =       │                   │
│  │  rSVD(S_c)    │   │  ‖[S_c]_{:,k}│   │  m_v /       │                   │
│  │               │   │  ‖_2         │   │  (‖∇f‖·‖S_v‖)│                   │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘                   │
│         │                  │                   │                            │
├─────────┼──────────────────┼───────────────────┼────────────────────────────┤
│  OUTPUTS (right strip)     │                   │                            │
│         ▼                  ▼                   ▼                            │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐                   │
│  │ 🎯 Optimal   │   │ 📊 Vulnerability│  │ 🛡️ Sensitivity│                  │
│  │ Attack δA*   │   │ Spectrum v_ij │   │ Radii r_v    │                   │
│  │              │   │               │   │              │                   │
│  │ SVD-optimal  │   │ Per-edge      │   │ Per-node     │                   │
│  │ perturbation │   │ rankings      │   │ tolerance    │                   │
│  │ direction    │   │               │   │ budgets      │                   │
│  └──────────────┘   └──────────────┘   └──────────────┘                   │
│                                                                             │
│  + For IGNN only: ε_crit (critical budget) + convergence diagnostics       │
├─────────────────────────────────────────────────────────────────────────────┤
│  APPLICATION EXTENSIONS (bottom strip, light gray, #F2F3F4)                │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐ │
│  │  Defense      │   │  Power Flow  │   │  Explicit    │   │  Cross-    │ │
│  │  Design       │   │  N-1         │   │  GNN         │   │  Domain    │ │
│  │               │   │  Contingency │   │  Extension   │   │  Transfer  │ │
│  │  Mask top-k   │   │              │   │              │   │            │ │
│  │  vulnerable   │   │  v_ij ≈ line │   │  S_K via     │   │  7 archs   │ │
│  │  edges from   │   │  trip        │   │  unrolled    │   │  × 9       │ │
│  │  perturbation │   │  severity    │   │  Jacobian    │   │  datasets  │ │
│  │  space        │   │  (τ=0.37–    │   │  (Obs. 1)    │   │  (τ>0 in   │ │
│  │  (Sec. V-F)   │   │  0.72)       │   │              │   │  29/33)    │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Notes

### Color Palette
| Layer | Color | Hex | Meaning |
|-------|-------|-----|---------|
| Input strip | White/light gray | #FAFAFA | Neutral |
| Layer 1: Graph Preprocessing | Light blue | #D6EAF8 | Data preparation |
| Interface: Path Selection | White with dashed border | — | Decision point |
| Layer 2: Sensitivity Computation | Light green | #D5F5E3 | Core computation |
| Layer 3: Constrained Projection | Light orange | #FDEBD0 | Key contribution (highlight) |
| Layer 4: Vulnerability Analysis | Light purple | #E8DAEF | Analysis extraction |
| Output strip | White | #FFFFFF | Results |
| Application Extensions | Light gray | #F2F3F4 | Downstream uses |

### Visual Elements
- **Processing blocks**: Rounded rectangles with drop shadow, each containing a title (bold), equation (italic/math), and complexity annotation (small, gray)
- **Arrows**: Solid arrows between blocks within a layer; thicker arrows between layers. Label intermediate quantities on arrows (e.g., "Z*, A_sub" between Layer 1→2, "S" between Layer 2→3, "S_c" between Layer 3→4)
- **Layer headers**: Full-width colored banner with layer name in bold and a short descriptor in regular weight (e.g., "LAYER 2 — SENSITIVITY COMPUTATION — core IFT-based analysis")
- **Dense vs Matrix-Free**: Show as two parallel tracks within Layer 2, with a switch/router icon at the top choosing between them based on N
- **Icons**: Use simple icons for inputs (neural network icon for GNN, graph/network icon for Graph G) and outputs (crosshair for attack, bar chart for spectrum, shield for radii)
- **Highlight box**: Layer 3 (Constrained Projection) should have a slightly thicker border or glow effect — it's the paper's central contribution
- **IGNN-only badge**: Small badge/tag on ε_crit and convergence diagnostics indicating "IGNN only"

### Annotations on Arrows (data flow)
```
Input ──► Layer 1:  "Trained GNN + Graph G"
Layer 1 ──► Layer 2:  "Z*, A_sub, ctx"
Layer 2 ──► Layer 3:  "S ∈ R^{Nd × N²} (unconstrained)"
Layer 3 ──► Layer 4:  "S_c ∈ R^{Nd × |E|} (constrained)"
Layer 4 ──► Output:   "δA*, v_ij, r_v"
```

### Key Equations Per Block
| Block | Equation |
|-------|----------|
| BFS Subgraph | A_sub = A[idx][:, idx], N_sub ≤ 200 |
| Forward Pass | Z = f_θ(X, A) |
| Fixed-Point | Z* : F(Z*, A) = Z* |
| J_z (dense) | J_z = ∂F/∂vec(Z) via autograd |
| J_A (dense) | J_A = ∂F/∂vec(A) via autograd |
| Linear Solve | S = (I − J_z)⁻¹ J_A |
| ρ Estimation | ρ(J_z) via power iteration |
| Neumann Series | (I − J_z)⁻¹ b ≈ Σ_{k=0}^{K} J_z^k b |
| Constrained Projection | [S_c]_{:,k} = S_{:,iN+j} + S_{:,jN+i} |
| rSVD | (U, σ, V) = rSVD(S_c, k=6) |
| Column Norms | v_k = ‖[S_c]_{:,k}‖₂ |
| Per-Node Radius | r_v = m_v / (‖∇f‖₂ · ‖S_v‖₂) |
| Critical Budget | ε_crit = (1 − κ) / ‖W‖₂  [IGNN only] |

### Dimensions / Complexity Per Block
| Block | Complexity |
|-------|-----------|
| BFS Subgraph | O(N_sub) |
| Forward Pass | O(N_sub · d) |
| Fixed-Point | O(T · N_sub · d), T ≤ 50 iterations |
| J_z (dense) | O((Nd)²) |
| J_A (dense) | O(Nd · N²) |
| Linear Solve (dense) | O((Nd)³) — **bottleneck for dense path** |
| ρ Estimation | O(30 · Nd) |
| JVPs | O(Nd) per product |
| Neumann Series | O(K · Nd) per solve, K ≈ 20–100 |
| rSVD | O(k · n_iter · K · Nd) |
| Constrained Projection | O(|E|) mapping |
| Column Norms | O(|E| · Nd) or via rSVD |
| Per-Node Radius | O(N · d) |

### Figure Size
- Target: full-width figure (\columnwidth for single-column, or \textwidth for two-column span)
- Aspect ratio: approximately 3:2 (landscape)
- Font: serif, 11pt (matching paper body)
- Format: PDF output

### Suggested Tools
- **matplotlib + matplotlib.patches**: For programmatic generation matching the paper's plotting style
- **draw.io / diagrams.net**: For manual refinement
- **Figma / Illustrator**: For publication-quality polish
