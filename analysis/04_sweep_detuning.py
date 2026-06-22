"""
04_sweep_detuning.py
====================
Sweep g/Delta at fixed gamma_t and g.

CONFIGURE HERE — change N_TH, T2_q, T2_t as needed.

Run from: TLS-QEC/analysis/
    python 04_sweep_detuning.py
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
from utils.physics import make_params

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.005,
    T2_q    = None,    # None => T2 = 2*T1
    T2_t    = None,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)
FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}

G_FIXED      = 0.05
G_OVER_DELTA = np.logspace(-1.3, 0.3, 40)
DELTA_VALUES = G_FIXED / G_OVER_DELTA
T_END        = 800
N_STEPS      = 800
TAG          = f"nth{N_TH:.4f}_T2q{PARAMS['T2_q']:.0f}_T2t{PARAMS['T2_t']:.0f}"
# ──────────────────────────────────────────────────────────────────────────────


def run_sweep():
    print(f"Sweeping g/Delta  (N_TH={N_TH:.4f}, g={G_FIXED})")
    print(f"  T1_q={PARAMS['T1_q']:.0f}  T2_q={PARAMS['T2_q']:.0f}  "
          f"T1_t={PARAMS['T1_t']:.0f}  T2_t={PARAMS['T2_t']:.0f}\n")

    ise_arr  = []
    maxd_arr = []
    coh_arr  = []
    g1L_arr  = []
    g2L_arr  = []
    g1S_arr  = []
    g2S_arr  = []

    for delta in tqdm(DELTA_VALUES, desc="Sweep"):
        wt     = FIXED['wq'] - delta
        params = {**FIXED, 'wt': wt, 'g': G_FIXED}

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
        'sweep_values':  G_OVER_DELTA,
        'ISE':           np.array(ise_arr),
        'max_diff':      np.array(maxd_arr),
        'coherence_max': np.array(coh_arr),
        'gamma1_L':      np.array(g1L_arr),
        'gamma2_L':      np.array(g2L_arr),
        'gamma1_S':      np.array(g1S_arr),
        'gamma2_S':      np.array(g2S_arr),
        'sweep_param':   'g/Delta',
        'fixed_params':  FIXED,
    }

    save_sweep(sweep, f'../data/sweeps/sweep_detuning_{TAG}.h5')
    print(f"Saved → data/sweeps/sweep_detuning_{TAG}.h5")

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)
    x   = G_OVER_DELTA

    ax1 = fig.add_subplot(gs[0])
    ax1.loglog(x, sweep['ISE'], 'o-', lw=2, color='black')
    ax1.axvline(1, color='red', ls='--', lw=1, label='$g=\\Delta$')
    ax1.set_xlabel('$g/\\Delta$'); ax1.set_ylabel('ISE')
    ax1.set_title('Integrated Squared Error'); ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.semilogx(x, sweep['max_diff'], 'o-', lw=2, color='darkblue')
    ax2.axvline(1, color='red', ls='--', lw=1)
    ax2.set_xlabel('$g/\\Delta$'); ax2.set_ylabel('max $|P_L - P_S|$')
    ax2.set_title('Peak Difference')
    ax2.grid(True, which='both', alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    ax3.semilogx(x, sweep['coherence_max'], 'o-', lw=2, color='purple')
    ax3.axvline(1, color='red', ls='--', lw=1)
    ax3.set_xlabel('$g/\\Delta$'); ax3.set_ylabel(r'max $|\rho_{eg,ge}|$')
    ax3.set_title('Peak Coherence')
    ax3.grid(True, which='both', alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    ax4.loglog(x, sweep['gamma1_L'], 'o-', lw=2, label='$\\Gamma_1$ Lindblad')
    ax4.loglog(x, sweep['gamma1_S'], 's--', lw=2, label='$\\Gamma_1$ Solomon')
    ax4.axvline(1, color='red', ls='--', lw=1)
    ax4.set_xlabel('$g/\\Delta$'); ax4.set_ylabel('Fitted rate')
    ax4.set_title('Decay Rates'); ax4.legend(fontsize=8)
    ax4.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        rf'Solomon breakdown vs $g/\Delta$  '
        rf'($g={G_FIXED}$, $n_{{th}}$={N_TH:.3f}, '
        rf'$T_2^q$={PARAMS["T2_q"]:.0f}, $T_2^t$={PARAMS["T2_t"]:.0f})',
        fontsize=13
    )
    plt.savefig(f'../figures/sweeps/sweep_detuning_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/sweep_detuning_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run_sweep()
