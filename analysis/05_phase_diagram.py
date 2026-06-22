"""
05_phase_diagram.py
===================
2D phase diagram of Solomon validity in (g/Delta, g/gamma_t) space.

CONFIGURE HERE — change N_TH to run at finite temperature.
Results saved with n_th label so runs don't overwrite each other.

WARNING: N_X * N_Y solver calls. Default 20x20 = ~15 minutes.

Run from: TLS-QEC/analysis/
    python 05_phase_diagram.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
from itertools import product

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.metrics import ISE, coherence_metrics
from utils.io import save, save_sweep, save_phase_diagram, load_phase_diagram
from utils.physics import make_params, n_thermal_phase_diagram, load_phase_diagram

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

TAG       = f"nth{N_TH:.4f}"
DATA_PATH = f'../data/phase_diagram/phase_diagram_{TAG}.h5'

X_RANGE = np.logspace(-1.3, 0.3, N_X)
Y_RANGE = np.logspace(-2,   1.3, N_Y)


def run_grid():
    print(f"Running {N_X}x{N_Y} grid  (N_TH={N_TH:.4f})")

    ISE_grid       = np.zeros((N_X, N_Y))
    coherence_grid = np.zeros((N_X, N_Y))

    pbar = tqdm(total=N_X*N_Y, desc="Phase diagram")

    for i, j in product(range(N_X), range(N_Y)):
        g       = G_FIXED
        delta   = g / X_RANGE[i]
        gamma_t = g / Y_RANGE[j]
        wt      = FIXED['wq'] - delta

        try:
            res_L = lindblad_evolve(
                wq=FIXED['wq'], wt=wt, g=g,
                gamma_q=FIXED['gamma_q'], gamma_t=gamma_t,
                n_th_q=N_TH, n_th_t=N_TH,
                t_end=T_END, n_steps=N_STEPS,
            )
            res_S = solomon_evolve(
                wq=FIXED['wq'], wt=wt, g=g,
                gamma_q=FIXED['gamma_q'], gamma_t=gamma_t,
                n_th_q=N_TH, n_th_t=N_TH,
                t_end=T_END, n_steps=N_STEPS,
            )

            P_S = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
            ISE_grid[i, j]       = ISE(res_L['t'], res_L['P_e_q'], P_S)
            coherence_grid[i, j] = coherence_metrics(
                res_L['t'], res_L['coherence'])['max']
        except Exception as e:
            ISE_grid[i, j]       = np.nan
            coherence_grid[i, j] = np.nan

        pbar.update(1)

    pbar.close()

    grid_result = {
        'x_values':       X_RANGE,
        'y_values':       Y_RANGE,
        'ISE_grid':       ISE_grid,
        'coherence_grid': coherence_grid,
        'x_label':        'g/Delta',
        'y_label':        'g/gamma_t',
    }
    save_phase_diagram(grid_result, DATA_PATH)
    return grid_result


def plot_phase_diagram(grid_result):
    X        = grid_result['x_values']
    Y        = grid_result['y_values']
    ISE_grid = grid_result['ISE_grid']

    log_ISE = np.log10(np.where(ISE_grid > 0, ISE_grid, 1e-16))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    im = ax.pcolormesh(X, Y, log_ISE.T, cmap='RdYlBu_r',
                       norm=mcolors.Normalize(
                           vmin=np.nanpercentile(log_ISE, 5),
                           vmax=np.nanpercentile(log_ISE, 95)))
    fig.colorbar(im, ax=ax).set_label('log$_{10}$(ISE)', fontsize=11)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$g/\\Delta$', fontsize=12)
    ax.set_ylabel('$g/\\gamma_t$', fontsize=12)
    ax.set_title('Solomon validity (blue=valid, red=broken)', fontsize=12)
    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.axhline(1, color='gray', ls=':', lw=1)

    ax2 = axes[1]
    coh = grid_result['coherence_grid']
    im2 = ax2.pcolormesh(X, Y,
                         np.log10(np.where(coh > 0, coh, 1e-16)).T,
                         cmap='Purples')
    fig.colorbar(im2, ax=ax2).set_label(
        r'log$_{10}$(max $|\rho_{eg,ge}|$)', fontsize=11)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('$g/\\Delta$', fontsize=12)
    ax2.set_ylabel('$g/\\gamma_t$', fontsize=12)
    ax2.set_title('Peak coherence', fontsize=12)

    fig.suptitle(
        rf'Phase diagram  ($g={G_FIXED}$, $n_{{th}}$={N_TH:.4f})',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../figures/phase_diagram/phase_diagram_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/phase_diagram/phase_diagram_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        print(f"Loading existing data from {DATA_PATH}")
        grid_result = load_phase_diagram(DATA_PATH)
    else:
        grid_result = run_grid()
    plot_phase_diagram(grid_result)
