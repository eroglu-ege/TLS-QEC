"""
06_coupling_sweep_tevo.py
=========================
Sweep coupling g (log-spaced, spanning weak to strong coupling) and
plot P_q(t) as a colormap, with the Solomon breakdown threshold
g = gamma_t made maximally visible.

Initial state: qubit ground (P_q=0), TLS excited (P_t=1).

KEY DESIGN CHOICES to make the threshold visible:
  1. g-axis is LOG-SPACED, covering orders of magnitude:
     from g << gamma_t (Solomon valid) to g >> gamma_t (Solomon invalid)
  2. time axis is in units of 1/gamma_t (physical decoherence time),
     NOT rescaled per-row — this keeps the x-axis meaningful while
     the qualitative change (smooth -> oscillatory) appears naturally
     as you scan up the g-axis
  3. A solid horizontal line marks g = gamma_t exactly
  4. Side panel: line cuts at g/gamma_t = 0.1, 1, 10 for direct comparison

Run from: TLS-QEC/analysis/
    python 06_coupling_sweep_tevo.py
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

FIXED = dict(
    wq      = 1.0,
    wt      = 1.0,     # resonant
    gamma_q = 0.01,
    gamma_t = 0.005,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

# Log-spaced g, deep into BOTH regimes:
#   g/gamma_t = 0.001  -> deep Solomon-valid regime
#   g/gamma_t = 1000   -> deep Rabi (Solomon-invalid) regime
N_G = 60
G_OVER_GAMMA_T = np.logspace(-3, 1.7, N_G)
G_VALUES       = G_OVER_GAMMA_T * FIXED['gamma_t']

# Time axis in units of 1/gamma_t — physical decoherence timescale,
# long enough to resolve slow Rabi cycles at the smallest g
T_END_UNITS = 50          # in units of 1/gamma_t
T_END       = T_END_UNITS / FIXED['gamma_t']
N_STEPS     = 800
# ──────────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"


def run():
    print(f"Sweeping g/gamma_t: {G_OVER_GAMMA_T[0]:.4f} to {G_OVER_GAMMA_T[-1]:.1f}")
    print(f"  ({N_G} log-spaced points)")
    print(f"t_end = {T_END:.1f} = {T_END_UNITS}/gamma_t")
    print(f"N_TH = {N_TH:.4f}\n")

    P_q_grid_L = np.zeros((N_G, N_STEPS))
    P_q_grid_S = np.zeros((N_G, N_STEPS))
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

    # Time in units of 1/gamma_t for the plot
    t_scaled = t_array * FIXED['gamma_t']

    np.savez(f'../data/sweeps/coupling_tevo_log_{TAG}.npz',
             P_q_L=P_q_grid_L, P_q_S=P_q_grid_S,
             g_over_gamma_t=G_OVER_GAMMA_T, g_values=G_VALUES,
             t=t_array, t_scaled=t_scaled,
             gamma_t=FIXED['gamma_t'])
    print(f"Data saved → data/sweeps/coupling_tevo_log_{TAG}.npz")

    # ── Main figure: two colormaps + three line cuts ───────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    for col, (grid, title) in enumerate(zip(
        [P_q_grid_L, P_q_grid_S],
        ['Lindblad (full quantum)', 'Solomon (rate equations)']
    )):
        ax = fig.add_subplot(gs[0, col])
        im = ax.pcolormesh(t_scaled, G_OVER_GAMMA_T, grid,
                           cmap='RdYlBu_r', vmin=0, vmax=1,
                           shading='gouraud')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Qubit $P_e$', fontsize=11)

        ax.set_yscale('log')
        ax.set_xlabel(r'Time  [$1/\gamma_t$]', fontsize=12)
        ax.set_ylabel(r'$g/\gamma_t$', fontsize=12)
        ax.set_title(title, fontsize=12)

        # THE threshold line — impossible to miss
        ax.axhline(1.0, color='black', lw=2.5, ls='-', alpha=0.9,
                   label=r'$g=\gamma_t$ (Solomon breakdown)')
        ax.legend(fontsize=9, loc='upper right',
                  facecolor='white', framealpha=0.85)

    # ── Bottom row: line cuts at g/gamma_t = 0.1, 1, 10 ─────────────────────────
    cut_values = [0.1, 1.0, 10.0]
    colors     = ['C0', 'C1', 'C2']

    ax_cut = fig.add_subplot(gs[1, :])
    for gv, c in zip(cut_values, colors):
        idx = np.argmin(np.abs(G_OVER_GAMMA_T - gv))
        actual_g = G_OVER_GAMMA_T[idx]
        ax_cut.plot(t_scaled, P_q_grid_L[idx], color=c, lw=2,
                   label=f'Lindblad $g/\\gamma_t$={actual_g:.2g}')
        ax_cut.plot(t_scaled, P_q_grid_S[idx], color=c, lw=2, ls='--',
                   label=f'Solomon $g/\\gamma_t$={actual_g:.2g}')

    ax_cut.set_xlabel(r'Time  [$1/\gamma_t$]', fontsize=12)
    ax_cut.set_ylabel('Qubit $P_e$', fontsize=12)
    ax_cut.set_title('Line cuts: below, at, and above threshold $g=\\gamma_t$',
                     fontsize=12)
    ax_cut.legend(fontsize=8, ncol=3, loc='upper right')
    ax_cut.set_ylim(-0.02, 1.05)

    fig.suptitle(
        rf'Solomon breakdown threshold, made visible  '
        rf'($n_{{th}}$={N_TH:.1f}, $\gamma_q$={FIXED["gamma_q"]}, '
        rf'$\gamma_t$={FIXED["gamma_t"]})',
        fontsize=14
    )

    plt.savefig(f'../figures/sweeps/coupling_tevo_log_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/coupling_tevo_log_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
