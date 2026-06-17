"""
06_coupling_sweep_tevo.py
=========================
Sweep coupling g and plot P_q(t) as a colormap.

Initial state: qubit ground (P_q=0), TLS excited (P_t=1).

CONFIGURE HERE — change N_TH to run at finite temperature.

Run from: TLS-QEC/analysis/
    python 06_coupling_sweep_tevo.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import qutip as qt

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.physics import n_thermal

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.0    # thermal photon number (0.0 = T=0)
               # e.g. N_TH = n_thermal(5.0, 0.100) for 5GHz at 100mK

FIXED = dict(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.005,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

G_VALUES = np.linspace(0.005, 0.3, 40)
T_END    = 400
N_STEPS  = 600
# ──────────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"


def run():
    print(f"Sweeping g: {G_VALUES[0]:.3f} to {G_VALUES[-1]:.3f}")
    print(f"Initial state: qubit ground, TLS excited")
    print(f"N_TH = {N_TH:.4f}\n")

    P_q_grid_L = np.zeros((len(G_VALUES), N_STEPS))
    P_q_grid_S = np.zeros((len(G_VALUES), N_STEPS))
    t_array    = None

    for i, g in enumerate(tqdm(G_VALUES, desc="g sweep")):
        psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))  # |ge>
        rho0 = qt.ket2dm(psi0)

        res_L = lindblad_evolve(
            **FIXED, g=g,
            t_end=T_END, n_steps=N_STEPS,
            rho0=rho0,
        )
        res_S = solomon_evolve(
            **FIXED, g=g,
            t_end=T_END, n_steps=N_STEPS,
            P_q0=0.0,
            P_t0=1.0,
        )

        P_q_grid_L[i, :] = res_L['P_e_q']
        P_q_grid_S[i, :] = res_S['P_e_q']
        if t_array is None:
            t_array = res_L['t']

    np.savez(f'../data/sweeps/coupling_tevo_{TAG}.npz',
             P_q_L=P_q_grid_L, P_q_S=P_q_grid_S,
             g_values=G_VALUES, t=t_array)
    print(f"Data saved → data/sweeps/coupling_tevo_{TAG}.npz")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid, title in zip(
        axes,
        [P_q_grid_L, P_q_grid_S],
        ['Lindblad (full quantum)', 'Solomon (rate equations)']
    ):
        im = ax.pcolormesh(t_array, G_VALUES, grid,
                           cmap='RdYlBu_r', vmin=0, vmax=1, shading='auto')
        fig.colorbar(im, ax=ax).set_label('Qubit $P_e$', fontsize=11)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Coupling $g$', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.axhline(FIXED['gamma_q'], color='white', ls='--', lw=1.5,
                   label=f"$g=\\gamma_q$={FIXED['gamma_q']}")
        ax.axhline(FIXED['gamma_t'], color='cyan', ls='--', lw=1.5,
                   label=f"$g=\\gamma_t$={FIXED['gamma_t']}")
        ax.legend(fontsize=9, loc='upper right')

    fig.suptitle(
        rf'Qubit $P_e(t)$ vs coupling $g$  '
        f'(qubit ground, TLS excited, $n_{{th}}$={N_TH:.4f})',
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/coupling_tevo_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/coupling_tevo_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
