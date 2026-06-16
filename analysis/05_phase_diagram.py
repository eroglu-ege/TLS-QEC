"""
05_phase_diagram.py
===================
Step 5: 2D phase diagram of Solomon validity in (g/Delta, g/gamma_t) space.

This is the thesis figure. It maps the entire parameter space and shows:
  - Where Solomon is valid (blue, low ISE)
  - Where Solomon breaks down (red, high ISE)
  - The breakdown boundary

The two axes correspond to the two independent approximation conditions:
  - g/Delta << 1 : dispersive condition
  - g/gamma_t << 1: Markov condition on TLS coherences

The diagram shows whether both conditions are needed simultaneously
or whether satisfying one is sufficient.

WARNING: This runs N_X * N_Y Lindblad+Solomon solves. With default
settings (20x20 grid) this takes ~10-20 minutes. Reduce grid size
for a quick test.

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
from utils.io import save_phase_diagram, load_phase_diagram


# ─── Grid settings ────────────────────────────────────────────────────────────

FIXED = dict(
    wq      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.01,   # gamma_t = gamma_q
)

# x-axis: g/Delta (near-resonant to dispersive)
# We keep g fixed and vary Delta
G_FIXED  = 0.05
N_X      = 20
N_Y      = 20

X_RANGE  = np.logspace(-1.3, 0.3, N_X)   # g/Delta
Y_RANGE  = np.logspace(-2,   1.3, N_Y)   # g/gamma_t

T_END    = 600
N_STEPS  = 600

DATA_PATH = '../data/phase_diagram/phase_diagram.h5'


def run_grid():
    print(f"Running {N_X}x{N_Y} = {N_X*N_Y} parameter points")
    print(f"  x-axis: g/Delta  from {X_RANGE[0]:.2f} to {X_RANGE[-1]:.2f}")
    print(f"  y-axis: g/gamma_t from {Y_RANGE[0]:.2f} to {Y_RANGE[-1]:.2f}")
    print(f"  Estimated time: {N_X*N_Y*0.5:.0f}–{N_X*N_Y*1.5:.0f} seconds\n")

    ISE_grid       = np.zeros((N_X, N_Y))
    coherence_grid = np.zeros((N_X, N_Y))

    total = N_X * N_Y
    pbar  = tqdm(total=total, desc="Phase diagram")

    for i, j in product(range(N_X), range(N_Y)):
        g_over_delta = X_RANGE[i]
        g_over_gamma = Y_RANGE[j]

        g       = G_FIXED
        delta   = g / g_over_delta
        gamma_t = g / g_over_gamma

        wt = FIXED['wq'] - delta

        try:
            res_L = lindblad_evolve(
                wq=FIXED['wq'], wt=wt, g=g,
                gamma_q=FIXED['gamma_q'], gamma_t=gamma_t,
                t_end=T_END, n_steps=N_STEPS,
            )
            res_S = solomon_evolve(
                wq=FIXED['wq'], wt=wt, g=g,
                gamma_q=FIXED['gamma_q'], gamma_t=gamma_t,
                t_end=T_END, n_steps=N_STEPS,
            )

            P_S_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
            ISE_grid[i, j] = ISE(res_L['t'], res_L['P_e_q'], P_S_interp)
            coherence_grid[i, j] = coherence_metrics(
                res_L['t'], res_L['coherence'])['max']

        except Exception as e:
            print(f"\n  Error at ({i},{j}): {e}")
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


def plot_phase_diagram(grid_result: dict):
    X   = grid_result['x_values']
    Y   = grid_result['y_values']
    ISE_grid = grid_result['ISE_grid']

    # Normalize: log10(ISE / ISE_min)
    ISE_safe = np.where(ISE_grid > 0, ISE_grid, 1e-16)
    log_ISE  = np.log10(ISE_safe)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: ISE heatmap ────────────────────────────────────────────────────
    ax = axes[0]
    im = ax.pcolormesh(X, Y, log_ISE.T,
                       cmap='RdYlBu_r',
                       norm=mcolors.Normalize(
                           vmin=np.nanpercentile(log_ISE, 5),
                           vmax=np.nanpercentile(log_ISE, 95)
                       ))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log$_{10}$(ISE)', fontsize=11)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$g/\\Delta$',     fontsize=12)
    ax.set_ylabel('$g/\\gamma_t$',   fontsize=12)
    ax.set_title('Solomon validity map\n(blue=valid, red=broken)',
                 fontsize=12)

    # Diagonal g/Delta = g/gamma_t reference
    xy = np.logspace(np.log10(min(X.min(), Y.min())),
                     np.log10(min(X.max(), Y.max())), 50)
    ax.plot(xy, xy, 'k--', lw=1.5, label='$g/\\Delta = g/\\gamma_t$')
    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.axhline(1, color='gray', ls=':', lw=1)
    ax.legend(fontsize=9)

    # ── Right: coherence heatmap ─────────────────────────────────────────────
    ax2 = axes[1]
    coh_grid = grid_result['coherence_grid']
    im2 = ax2.pcolormesh(X, Y, np.log10(
                         np.where(coh_grid > 0, coh_grid, 1e-16)).T,
                         cmap='Purples')
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label(r'log$_{10}$(max $|\rho_{eg,ge}|$)', fontsize=11)
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('$g/\\Delta$',   fontsize=12)
    ax2.set_ylabel('$g/\\gamma_t$', fontsize=12)
    ax2.set_title('Peak coherence\n(large = Solomon unjustified)', fontsize=12)
    ax2.axvline(1, color='white', ls=':', lw=1)
    ax2.axhline(1, color='white', ls=':', lw=1)

    fig.suptitle(
        rf'Phase diagram: Solomon approximation validity  '
        rf'($g={G_FIXED}$, $\gamma_q={FIXED["gamma_q"]}$)',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig('../figures/phase_diagram/phase_diagram.png',
                dpi=150, bbox_inches='tight')
    print("Figure saved → figures/phase_diagram/phase_diagram.png")
    plt.show()


if __name__ == "__main__":
    # Check if data already exists — skip computation if so
    if os.path.exists(DATA_PATH):
        print(f"Loading existing data from {DATA_PATH}")
        grid_result = load_phase_diagram(DATA_PATH)
    else:
        grid_result = run_grid()

    plot_phase_diagram(grid_result)
