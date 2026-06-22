"""
07_clean_comparison.py
=======================
Clean side-by-side: P_q(t) and P_t(t), Lindblad vs Solomon,
at hand-picked g values spanning weak to strong coupling.

Initial state: qubit ground, TLS excited.

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
from utils.physics import make_params

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.005,
    T2_q    = None,
    T2_t    = None,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)
FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}

# Hand-picked g values: weak, intermediate, strong coupling
G_VALUES = [0.001, 0.005, 0.01, 0.1]   # g/gamma_t = 0.2, 1, 2, 20

T_END   = 1500
N_STEPS = 1000
TAG     = f"nth{N_TH:.4f}_T2q{PARAMS['T2_q']:.0f}_T2t{PARAMS['T2_t']:.0f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    n = len(G_VALUES)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), sharex=True)

    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))  # |ge>
    rho0 = qt.ket2dm(psi0)

    print(f"N_TH={N_TH:.4f}  T2_q={PARAMS['T2_q']:.0f}  T2_t={PARAMS['T2_t']:.0f}\n")

    for col, g in enumerate(G_VALUES):
        ratio = g / FIXED['gamma_t']
        print(f"Running g={g:.4f}  (g/gamma_t={ratio:.2f})...")

        res_L = lindblad_evolve(**FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
                                rho0=rho0)
        res_S = solomon_evolve(**FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
                               P_q0=0.0, P_t0=1.0)

        t = res_L['t']

        ax = axes[0, col]
        ax.plot(t, res_L['P_e_q'], lw=2.2, color='C0', label='Lindblad')
        ax.plot(t, res_S['P_e_q'], lw=2.2, color='C1', ls='--', label='Solomon')
        ax.axhline(res_L['P_e_q_ss'], color='gray', ls=':', lw=1,
                  label=f"$P_{{ss}}$={res_L['P_e_q_ss']:.3f}")
        ax.set_title(rf'$g={g:.3f}$  ($g/\gamma_t$={ratio:.1f})', fontsize=12)
        ax.set_ylabel('Qubit $P_e$', fontsize=11)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=8)

        ax2 = axes[1, col]
        ax2.plot(t, res_L['P_e_t'], lw=2.2, color='C0', label='Lindblad')
        ax2.plot(t, res_S['P_e_t'], lw=2.2, color='C1', ls='--', label='Solomon')
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('TLS $P_e$', fontsize=11)
        ax2.set_ylim(-0.02, 1.05)
        ax2.legend(fontsize=8)

        max_d = np.max(np.abs(res_L['P_e_q'] -
                              np.interp(t, res_S['t'], res_S['P_e_q'])))
        print(f"  max|P_L - P_S| = {max_d:.4f}")

    fig.suptitle(
        rf'Lindblad vs Solomon: weak $\to$ strong coupling  '
        rf'($\Delta=0$, $n_{{th}}$={N_TH:.1f}, '
        rf'$T_2^q$={PARAMS["T2_q"]:.0f}, qubit ground / TLS excited)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(f'../figures/single_trajectory/clean_comparison_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → figures/single_trajectory/clean_comparison_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
