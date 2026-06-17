"""
06_coupling_sweep_tevo.py
=========================
Sweep coupling strength g and plot P_q(t) as a colormap.

Initial conditions: qubit ground (P_q=0), TLS excited (P_t=1).
This is the reverse of the standard setup — shows how the TLS
dumps its excitation into the qubit over time.

Physical question: how does coupling strength control the
speed and completeness of TLS->qubit energy transfer?

The colormap shows:
    x-axis: time
    y-axis: coupling strength g
    color:  qubit population P_q(t) — blue=0, red=1

Run from: TLS-QEC/analysis/
    python 06_coupling_sweep_tevo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.physics import n_thermal
import qutip as qt


# ─── Parameters ───────────────────────────────────────────────────────────────

FIXED = dict(
    wq      = 1.0,
    wt      = 1.0,     # resonant: Delta = 0
    gamma_q = 0.01,
    gamma_t = 0.005,
    n_th_q  = 0.0,     # set > 0 for thermal, e.g. n_thermal(5.0, 0.020)
    n_th_t  = 0.0,
)

# Sweep g from weak to strong coupling
G_VALUES = np.linspace(0.005, 0.3, 40)

T_END   = 400
N_STEPS = 600


def run():
    print(f"Sweeping g from {G_VALUES[0]:.3f} to {G_VALUES[-1]:.3f}")
    print(f"Initial state: qubit ground (P_q=0), TLS excited (P_t=1)")
    print(f"Fixed: gamma_q={FIXED['gamma_q']}, gamma_t={FIXED['gamma_t']}\n")

    # Storage
    P_q_grid_L = np.zeros((len(G_VALUES), N_STEPS))
    P_q_grid_S = np.zeros((len(G_VALUES), N_STEPS))
    t_array    = None

    for i, g in enumerate(tqdm(G_VALUES, desc="g sweep")):

        # Initial state: qubit ground |g>, TLS excited |e> → |ge>
        psi0  = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))  # |ge>
        rho0  = qt.ket2dm(psi0)

        res_L = lindblad_evolve(
            **FIXED, g=g,
            t_end=T_END, n_steps=N_STEPS,
            rho0=rho0,
        )
        res_S = solomon_evolve(
            **{k: v for k, v in FIXED.items()
               if k not in ('n_th_q', 'n_th_t')},
            g=g,
            n_th_q=FIXED['n_th_q'], n_th_t=FIXED['n_th_t'],
            t_end=T_END, n_steps=N_STEPS,
            P_q0=0.0,   # qubit starts in ground
            P_t0=1.0,   # TLS starts excited
        )

        P_q_grid_L[i, :] = res_L['P_e_q']
        P_q_grid_S[i, :] = res_S['P_e_q']

        if t_array is None:
            t_array = res_L['t']

    # ── Save data ─────────────────────────────────────────────────────────────
    np.savez('../data/sweeps/coupling_tevo.npz',
             P_q_L=P_q_grid_L,
             P_q_S=P_q_grid_S,
             g_values=G_VALUES,
             t=t_array)
    print("Data saved → data/sweeps/coupling_tevo.npz")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid, title in zip(
        axes,
        [P_q_grid_L, P_q_grid_S],
        ['Lindblad (full quantum)', 'Solomon (rate equations)']
    ):
        im = ax.pcolormesh(
            t_array, G_VALUES, grid,
            cmap='RdYlBu_r',
            vmin=0, vmax=1,
            shading='auto',
        )
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Qubit $P_e$', fontsize=11)

        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Coupling $g$', fontsize=12)
        ax.set_title(title, fontsize=12)

        # Mark g = gamma_q and g = gamma_t lines
        ax.axhline(FIXED['gamma_q'], color='white', ls='--', lw=1.5,
                   label=f"$g=\\gamma_q$={FIXED['gamma_q']}")
        ax.axhline(FIXED['gamma_t'], color='cyan',  ls='--', lw=1.5,
                   label=f"$g=\\gamma_t$={FIXED['gamma_t']}")
        ax.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        r'Qubit $P_e(t)$ vs coupling $g$  '
        f'(initial state: qubit ground, TLS excited)\n'
        rf'$\Delta=0$, $\gamma_q={FIXED["gamma_q"]}$, '
        rf'$\gamma_t={FIXED["gamma_t"]}$',
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig('../figures/sweeps/coupling_tevo.png', dpi=150, bbox_inches='tight')
    print("Figure saved → figures/sweeps/coupling_tevo.png")
    plt.show()


if __name__ == "__main__":
    run()
