# Revision S5 — disclosure-protocol asymmetry (DONE; DEFEND, prose-only)

**Concern (R3):** the protocol releases the per-edge `v_ij` target list unconditionally while gating the SVD-reconstruction code — R3 called this "backwards" (a ranked target list seems the more actionable artifact).

**Strategy: DEFEND (don't weaken a responsible-by-design feature).** Demoting/gating the diagnostic release would undersell; instead sharpen *why* the asymmetry is principled.

**Edit (`conclusion.tex`, disclosure (iii)):**
- *Before:* "…released unconditionally, since it cannot directly synthesise a perturbation."
- *After:* "…released unconditionally: a score **ranks** which edges are sensitive but is not itself an attack; the damaging perturbation δA\* (`prop:attack`) comes only from the gated reconstruction (steps 3–4)."
- The argument: the score is a *ranking* (a diagnostic), not the perturbation. The actual attack δA\* = ε·reshape(v₁) requires the SVD reconstruction (the gated artifact); a ranking alone supports only the weaker per-edge attacks the SVD direction already dominates. So the gating is *calibrated*, not inverted.

**Offset (to hold 10pp):** generalized disclosure (i) recipients ("the affected benchmark and attacker-tool maintainers") — the specific OGB/GR-BCD/Mettack/AGNNCert names were a detail.

**Result:** 10 pages, 0 overfull. The disclosure stays a strength; no undersell.
