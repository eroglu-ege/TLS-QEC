"""
06b_coupling_tevo_zoom.py
==========================
Zoomed-in version of 06: colormap of P_q(t) vs g, restricted to
g/gamma_t in [0.1, 1.1] — the region where the breakdown actually
begins (per the fine_threshold_scan results: onset at g/gt~0.07,
sign reversal starting ~0.45).

Same design as 06 (time in units of 1/gamma_t, threshold line marked)
but with much higher g-resolution since the range is narrow, and
N_STEPS tuned to avoid aliasing at the top of this range.

Run from: TLS-QEC/analysis/
    python 06b_coupling_tevo_zoom.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import qutip as qt

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

# Build consistent params from physical timescales.
# T2=None means no pure dephasing (T2 = 2*T1, Lindblad limit).
# Set T2_q < 2*T1_q to add realistic pure dephasing.
PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,    # T1_q = 100
    gamma_t = 0.005,   # T1_t = 200
    T2_q    = None,    # None => T2 = 2*T1 (no pure dephasing)
    T2_t    = None,    # None => T2 = 2*T1 (no pure dephasing)
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)
FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}
# ──────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"


def run():
    # Sampling check at the top of the range
    g_max      = G_VALUES[-1]
    T_rabi_min = np.pi / g_max
    dt         = T_END / N_STEPS
    print(f"Zoom range: g/gamma_t = [{G_OVER_GAMMA_T[0]:.3f}, {G_OVER_GAMMA_T[-1]:.3f}]")
    print(f"  {N_G} points, N_TH={N_TH}")
    print(f"Sampling check: T_rabi_min={T_rabi_min:.2f}, dt={dt:.3f}, "
          f"samples/period={T_rabi_min/dt:.1f}  (want >10)\n")

    P_q_grid_L = np.zeros((N_G, N_STEPS))
    P_q_grid_S = np.zeros((N_G, N_STEPS))
    t_array    = None

    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))  # |ge>
    rho0 = qt.ket2dm(psi0)

    for i, g in enumerate(tqdm(G_VALUES, desc="Zoom sweep")):
        res_L = lindblad_evolve(
            **FIXED, g=g, t_end=T_END, n_steps=N_STEPS, rho0=rho0,
        )
        res_S = solomon_evolve(
            **FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
            P_q0=0.0, P_t0=1.0,
        )
        P_q_grid_L[i, :] = res_L['P_e_q']
        P_q_grid_S[i, :] = res_S['P_e_q']
        if t_array is None:
            t_array = res_L['t']

    t_scaled = t_array * FIXED['gamma_t']

    np.savez(f'../data/sweeps/coupling_tevo_zoom_{TAG}.npz',
             P_q_L=P_q_grid_L, P_q_S=P_q_grid_S,
             g_over_gamma_t=G_OVER_GAMMA_T, g_values=G_VALUES,
             t=t_array, t_scaled=t_scaled, gamma_t=FIXED['gamma_t'])
    print(f"Data saved → data/sweeps/coupling_tevo_zoom_{TAG}.npz")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    for col, (grid, title) in enumerate(zip(
        [P_q_grid_L, P_q_grid_S],
        ['Lindblad (full quantum)', 'Solomon (rate equations)']
    )):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(t_scaled, G_OVER_GAMMA_T, grid,
                           cmap='RdYlBu_r', vmin=0, vmax=None,
                           shading='gouraud')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Qubit $P_e$', fontsize=11)

        ax.set_xlabel(r'Time  [$1/\gamma_t$]', fontsize=12)
        ax.set_ylabel(r'$g/\gamma_t$', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.axhline(1.0, color='black', lw=2, ls='-',
                  label=r'$g=\gamma_t$')
        ax.legend(fontsize=9, loc='upper right',
                 facecolor='white', framealpha=0.85)

    # ── Line cuts at g/gamma_t = 0.1, 0.5, 0.86 (max overshoot), 1.1 ───────────
    cut_values = [0.1, 0.5, 0.86, 1.1]
    colors     = ['C0', 'C1', 'C2', 'C3']

    ax_cut = fig.add_subplot(gs[1, :])
    for gv, c in zip(cut_values, colors):
        idx = np.argmin(np.abs(G_OVER_GAMMA_T - gv))
        actual_g = G_OVER_GAMMA_T[idx]
        ax_cut.plot(t_scaled, P_q_grid_L[idx], color=c, lw=2,
                   label=f'Lindblad $g/\\gamma_t$={actual_g:.2f}')
        ax_cut.plot(t_scaled, P_q_grid_S[idx], color=c, lw=2, ls='--',
                   label=f'Solomon $g/\\gamma_t$={actual_g:.2f}')

    ax_cut.set_xlabel(r'Time  [$1/\gamma_t$]', fontsize=12)
    ax_cut.set_ylabel('Qubit $P_e$', fontsize=12)
    ax_cut.set_title('Line cuts across the zoomed window', fontsize=12)
    ax_cut.legend(fontsize=8, ncol=4, loc='upper right')

    fig.suptitle(
        rf'Zoomed threshold colormap: $g/\gamma_t \in [0.1, 1.1]$  '
        rf'($n_{{th}}$={N_TH}, $\gamma_q$={FIXED["gamma_q"]}, '
        rf'$\gamma_t$={FIXED["gamma_t"]})',
        fontsize=14
    )
    plt.savefig(f'../figures/sweeps/coupling_tevo_zoom_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/coupling_tevo_zoom_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
