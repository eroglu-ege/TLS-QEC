"""
07_clean_comparison.py
=======================
Clean side-by-side comparison: P_q(t) and P_t(t), Lindblad vs Solomon,
at a few hand-picked g values spanning weak to strong coupling.

No fitting, no colormap — just direct population curves so you can
see the Solomon breakdown with your own eyes.

Initial state: qubit ground (P_q=0), TLS excited (P_t=1).

CONFIGURE HERE — pick g values and N_TH.

Run from: TLS-QEC/analysis/
    python 07_clean_comparison.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

FIXED = dict(
    wq      = 1.0,
    wt      = 1.0,     # resonant
    gamma_q = 0.01,
    gamma_t = 0.005,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

# Hand-picked g values spanning weak -> strong coupling relative to gamma_t=0.005
G_VALUES = [0.001, 0.01, 0.1]   # g/gamma_t = 0.2, 2, 20

T_END   = 1500
N_STEPS = 1000
# ──────────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"


def run():
    n = len(G_VALUES)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), sharex=True)

    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))  # |ge>: qubit ground, TLS excited
    rho0 = qt.ket2dm(psi0)

    print(f"N_TH = {N_TH:.4f}   gamma_q={FIXED['gamma_q']}   gamma_t={FIXED['gamma_t']}\n")

    for col, g in enumerate(G_VALUES):
        ratio = g / FIXED['gamma_t']
        print(f"Running g={g:.4f}  (g/gamma_t = {ratio:.2f}) ...")

        res_L = lindblad_evolve(**FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
                                rho0=rho0)
        res_S = solomon_evolve(**FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
                               P_q0=0.0, P_t0=1.0)

        t = res_L['t']

        # ── Top row: qubit population ───────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(t, res_L['P_e_q'], lw=2.2, color='C0', label='Lindblad')
        ax.plot(t, res_S['P_e_q'], lw=2.2, color='C1', ls='--', label='Solomon')
        ax.axhline(res_L['P_e_q_ss'], color='gray', ls=':', lw=1,
                  label=f'$P_{{ss}}$={res_L["P_e_q_ss"]:.3f}')
        ax.set_title(rf'$g={g:.3f}$  ($g/\gamma_t$={ratio:.2g})', fontsize=13)
        ax.set_ylabel('Qubit $P_e$', fontsize=12)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=9)

        # ── Bottom row: TLS population ──────────────────────────────────────────
        ax2 = axes[1, col]
        ax2.plot(t, res_L['P_e_t'], lw=2.2, color='C0', label='Lindblad')
        ax2.plot(t, res_S['P_e_t'], lw=2.2, color='C1', ls='--', label='Solomon')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('TLS $P_e$', fontsize=12)
        ax2.set_ylim(-0.02, 1.05)
        ax2.legend(fontsize=9)

        max_diff = np.max(np.abs(res_L['P_e_q'] -
                                 np.interp(t, res_S['t'], res_S['P_e_q'])))
        print(f"  max|P_L - P_S| (qubit) = {max_diff:.4f}")

    fig.suptitle(
        rf'Lindblad vs Solomon: weak $\to$ strong coupling  '
        rf'($\Delta=0$, $n_{{th}}$={N_TH:.1f}, qubit ground / TLS excited)',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(f'../figures/single_trajectory/clean_comparison_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → figures/single_trajectory/clean_comparison_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
