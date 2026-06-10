# TLS-QEC
Documentation for simulations regarding TLS inspired QEC

# TLS-QEC

**Topic:** Two-Level Systems and Quantum Error Correction  
**Institution:** KIT, Karlsruhe  
**Supervisors:** Mathieu, Nico (Ioan)  
**Started:** June 2026

---

## Research Goal

Investigate how long-lived TLS defects coupled to superconducting qubits introduce time-correlated errors that challenge quantum error correction. Starting from the full quantum evolution of a qubit+TLS system, then scaling to a repetition code to test whether TLS-induced bunching causes correlated syndrome patterns.

---

## First Tasks (from Mathieu)

- [ ] **Task 1 — Full quantum evolution of qubit + TLS**  
  Write the full Lindblad master equation for the coupled qubit+TLS system. Start from the Jaynes-Cummings Hamiltonian, add dissipation channels (γ_q, γ_t, dephasing), solve numerically with QuTiP.

- [ ] **Task 2 — Derive the Solomon equations**  
  Show that the Solomon equations emerge from the full quantum evolution when TLS coherences are traced out (Markov approximation on the TLS). This is the limit where off-diagonal elements of ρ_TLS → 0.

- [ ] **Task 3 — Compare full quantum vs Solomon near the assumption limit**  
  Sweep the parameters where the Solomon approximation is expected to break down (strong coupling g, long TLS coherence time T2_TLS). Quantify the deviation between the two models. Check against existing code from Martin and Nico.

---

## Background

The group's recent paper showed that long-lived TLSs cause **bunching** in qubit quantum jumps — after a qubit decay, the TLS re-excites it, causing correlated errors. They modeled this with Solomon equations (coupled rate equations for qubit+TLS populations). The thesis extends this to multi-qubit QEC codes.

---

## Repository Structure

```
TLS-QEC/
├── logs/                        # daily research logs
│   ├── template.md
│   └── YYYY-MM-DD.md
├── notes/
│   ├── papers/                  # paper summaries
│   └── derivations/             # worked derivations
├── simulations/
│   ├── qubit_tls/
│   │   ├── closed_system.py     # unitary evolution U(t)
│   │   ├── lindblad.py          # open system master equation
│   │   ├── solomon.py           # Solomon equations (rate eqs)
│   │   └── comparison.py        # full quantum vs Solomon
│   └── utils/
│       └── plotting.py
├── figures/
└── thesis/                      # LaTeX source
```

---

## Key References

### Textbooks
| Book | Why |
|------|-----|
| Haroche & Raimond — *Exploring the Quantum* (Oxford, 2006) | Gold standard for qubit+TLS (JC model), dressed states, full unitary evolution |
| Walls & Milburn — *Quantum Optics* (Springer) | Master equation, open system treatment alongside coherent dynamics |
| Breuer & Petruccione — *The Theory of Open Quantum Systems* | Complete Lindblad / density matrix formalism |
| Gardiner & Zoller — *Quantum Noise* | Input-output theory, field-coupled TLS |
| Gerry & Knight — *Introductory Quantum Optics* | Gentler intro to JC model mathematics |

### Key Papers
| Paper | Focus |
|-------|-------|
| Blais, Grimsmo, Girvin, Wallraff — Rev. Mod. Phys. 93, 025005 (2021) | Circuit QED bible — JC Hamiltonian, dispersive limit, dissipation |
| Ashhab, Johansson, Nori — arXiv:cond-mat/0604475 | Decoherence and Rabi oscillations in qubit coupled to quantum TLS |
| arXiv:1004.4664 | Driven qubit+TLS: anomalous Rabi oscillations, two-photon transitions |
| de Graaf et al. (Chalmers, 2020) | TLS defects causing qubit parameter fluctuations and relaxation |
| PRX Quantum 4, 020356 (2023) | TLS spectral dynamics destabilizing qubit lifetimes |
| IOP Quantum Sci. Technol. (2023) | Simulating qubit+TLS bath with Lindblad, 150 TLS model |
| Group paper (Fechant, Gosling et al.) | Solomon equations, TLS bunching, time-correlated errors |

### Blais et al. RMP 2021 — Reading Roadmap
Sections to read in order:
1. §II.D — transmon basics
2. §III.A — derive the qubit+TLS Hamiltonian
3. §III.B — exact spectrum / dressed states
4. §III.C + Appendix B — dispersive limit (Schrieffer-Wolff + Bogoliubov)
5. §IV.B–C — open system, Lindblad, T1/T2
6. §VI.A–B — full map of dynamical regimes

---

## Commit Convention

| Prefix | Use |
|--------|-----|
| `feat(sim):` | new simulation or code |
| `fix(sim):` | bug fix |
| `log:` | daily research log |
| `notes:` | paper summaries, derivations |
| `fig:` | new or updated figure |
| `thesis:` | LaTeX writing |
| `chore:` | repo maintenance |

---

## Daily Workflow

```bash
git pull
# ... work ...
git add logs/YYYY-MM-DD.md simulations/...
git commit -m "log: YYYY-MM-DD [one line summary]"
git push
```
