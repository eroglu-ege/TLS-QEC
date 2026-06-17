"""
03_sweep_g_gamma.py
===================
Sweep g/gamma_t at resonance (Delta=0).

CONFIGURE HERE — change N_TH to run at finite temperature.

Run from: TLS-QEC/analysis/
    python 03_sweep_g_gamma.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.metrics import ISE, max_diff, coherence_metrics, fit_biexp
from utils.io import save_sweep

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.0    # thermal photon number (0.0 = T=0)

FIXED = dict(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.01,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

G_OVER_GAMMA = np.logspace(-2, 1.3, 40)
G_VALUES     = G_OVER_GAMMA * FIXED['gamma_t']
T_END        = 800
N_STEPS      = 800
# ──────────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"


def run_sweep():
    print(f"Sweeping g/gamma_t  (resonant, Delta=0, N_TH={N_TH:.4f})")

    ise_arr    = []
    maxd_arr   = []
    coh_arr    = []
    g1L_arr    = []
    g2L_arr    = []
    g1S_arr    = []
    g2S_arr    = []

    for g in tqdm(G_VALUES, desc="Sweep"):
        params = {**FIXED, 'g': g}

        res_L = lindblad_evolve(**params, t_end=T_END, n_steps=N_STEPS)
        res_S = solomon_evolve(**params,  t_end=T_END, n_steps=N_STEPS)

        P_S_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])

        ise_arr.append(ISE(res_L['t'], res_L['P_e_q'], P_S_interp))
        maxd_arr.append(max_diff(res_L['t'], res_L['P_e_q'], P_S_interp)['value'])
        coh_arr.append(coherence_metrics(res_L['t'], res_L['coherence'])['max'])

        fL = fit_biexp(res_L['t'], res_L['P_e_q'])
        fS = fit_biexp(res_S['t'], res_S['P_e_q'])
        g1L_arr.append(fL['Gamma1'] if fL['success'] else np.nan)
        g2L_arr.append(fL['Gamma2'] if fL['success'] else np.nan)
        g1S_arr.append(fS['Gamma1'] if fS['success'] else np.nan)
        g2S_arr.append(fS['Gamma2'] if fS['success'] else np.nan)

    sweep = {
        'sweep_values':  G_OVER_GAMMA,
        'ISE':           np.array(ise_arr),
        'max_diff':      np.array(maxd_arr),
        'coherence_max': np.array(coh_arr),
        'gamma1_L':      np.array(g1L_arr),
        'gamma2_L':      np.array(g2L_arr),
        'gamma1_S':      np.array(g1S_arr),
        'gamma2_S':      np.array(g2S_arr),
        'sweep_param':   'g/gamma_t',
        'fixed_params':  {k: v for k, v in FIXED.items()
                          if not isinstance(v, float) or k != 'g'},
    }

    save_sweep(sweep, f'../data/sweeps/sweep_g_gamma_{TAG}.h5')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)
    x   = G_OVER_GAMMA

    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(x, sweep['ISE'], 'o-', lw=2, color='black')
    ax1.axvline(1, color='red', ls='--', lw=1, label='$g=\\gamma_t$')
    ax1.set_xlabel('$g/\\gamma_t$'); ax1.set_ylabel('ISE')
    ax1.set_title('Integrated Squared Error'); ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.semilogx(x, sweep['max_diff'], 'o-', lw=2, color='darkblue')
    ax2.axvline(1, color='red', ls='--', lw=1)
    ax2.set_xlabel('$g/\\gamma_t$'); ax2.set_ylabel('max $|P_L - P_S|$')
    ax2.set_title('Peak Difference')
    ax2.grid(True, which='both', alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    ax3.semilogx(x, sweep['coherence_max'], 'o-', lw=2, color='purple')
    ax3.axvline(1, color='red', ls='--', lw=1)
    ax3.set_xlabel('$g/\\gamma_t$'); ax3.set_ylabel(r'max $|\rho_{eg,ge}|$')
    ax3.set_title('Peak Coherence')
    ax3.grid(True, which='both', alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    ax4.loglog(x, sweep['gamma1_L'], 'o-', lw=2, label='$\\Gamma_1$ Lindblad')
    ax4.loglog(x, sweep['gamma1_S'], 's--', lw=2, label='$\\Gamma_1$ Solomon')
    ax4.axvline(1, color='red', ls='--', lw=1)
    ax4.set_xlabel('$g/\\gamma_t$'); ax4.set_ylabel('Fitted rate')
    ax4.set_title('Decay Rates'); ax4.legend(fontsize=8)
    ax4.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        rf'Solomon breakdown vs $g/\gamma_t$  '
        rf'($\Delta=0$, $n_{{th}}$={N_TH:.4f})',
        fontsize=13
    )
    plt.savefig(f'../figures/sweeps/sweep_g_gamma_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → figures/sweeps/sweep_g_gamma_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run_sweep()
