# Conclusion — minimal-edit, length-neutral (consistent framing)

Aligns the conclusion with the new abstract (`12`) and intro (`13`): lands the **coupling** thesis
explicitly, adds the **exchangeability** caveat, and softens "safety boundary" → "characterisation" to
match the Option-B theory (`09`). ~140 words vs the current ~137 — length-neutral. Keeps the physics line
verbatim (your intro now forward-references it).

---
## Updated conclusion (drop-in for `sections/conclusion.tex`)

```latex
From one matrix-free object, $S_c$, \AEGIS unifies auditing, certification, and defense as readings of a
single operator: the $\sigma_1(S_c)$ that names the worst perturbation bounds its safe radius and,
penalized, defends against it. $S_c$ scales the audit to $N{=}7{,}650$ and extends to any continuous
edge-weight GNN, with a closed-form safety characterisation (\cref{thm:phase_transition}) for contractive
implicit models.

\textbf{Limitations.} The radii $r_v$ are first-order; \AEGIS-Conformal assumes exchangeability and runs
densely at $N{=}200$; standard GAT, binary-mask architectures, and insertion attacks lie outside scope
(\cref{sec:explicit_extension,sec:background}); and the $\ecrit$ track costs ${\sim}6\%$ accuracy. \AEGIS
audits vulnerability, not physics: a contractive surrogate cannot model voltage collapse, so it
complements rather than replaces power-flow contingency screening~\cite{donon2019graph,nakiganda2023graph,varbella2024contingency}.

\textbf{Disclosure.} We propose a 90-day coordinated-notification window: the diagnostic path ($r_v$,
$v_{ij}$) releases unconditionally, while attack-direction reconstruction (\cref{alg:aegis}) is gated
behind ethics review, since it alone yields the damaging $\delta A^*$ (\cref{prop:attack}).
```

---
## What changed (and why), and what's kept

| Part | Change | Why |
|---|---|---|
| Opening | "unifies … coupled by construction" → spells out the coupling: "the $\sigma_1(S_c)$ that names the worst perturbation bounds its safe radius and, penalized, defends against it" | Lands the **coupling thesis** explicitly — the repositioning payload, echoing abstract/intro. |
| Opening | added "$S_c$ scales the audit to $N{=}7{,}650$" | Conveys the **scale** half of "coupling + cost," consistent with abstract/intro. |
| Opening | "closed-form safety **boundary**" → "safety **characterisation**" | Matches Option-B theory (`09`): the boundary is now `ε_glob` (global) + `ε_crit` (linearized), a characterisation, not a single clean threshold. |
| Limitations | "+ \AEGIS-Conformal **assumes exchangeability** and runs densely at N=200" | Consistency with abstract/intro's "sound under exchangeability"; the load-bearing (C1) caveat (`rem:exchange-honesty`). |
| Limitations | merged the GAT / binary-mask / insertion scope items into one clause | Pure tightening — funds the coupling + exchangeability additions so the paragraph stays length-neutral. No content lost. |
| **Kept verbatim** | the physics line ("audits vulnerability, not physics … cannot model voltage collapse … complements rather than replaces power-flow screening") | This is the **payoff of the intro's forward-reference** — the two now read as one consistent scoping. |
| **Kept verbatim** | the entire Disclosure paragraph | Policy statement, unchanged. |

---
## One flagged tension (NOT changed here — your call)

The Disclosure says the diagnostic path "releases unconditionally … since [attack-direction reconstruction]
**alone** yields the damaging $\delta A^*$." But the paper now shows the diagnostic ranking `A_ij·v_ij`
*itself* reproduces brute-force removal damage (the fraud case, `sec:fraud_case`) — i.e. the "diagnostic"
has real offensive utility, so "alone" slightly over-claims. This is the R3-5 / P2 roadmap item. It's a
*disclosure-policy* question, not a framing one, so I left it untouched. If you want, the honest minimal
fix is to drop "alone" and note the ranking is dual-use but defender-oriented — say the word.
