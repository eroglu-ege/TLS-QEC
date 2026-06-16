"""
04_sweep_detuning.py
====================
Step 4: Sweep g/Delta (coupling-to-detuning ratio) at fixed gamma_t.

The dispersive approximation requires g << Delta. As g/Delta grows,
the qubit and TLS hybridize more strongly and coherences matter more.

At g/Delta → 0: dispersive regime, Solomon is excellent
At g/Delta ~ 1: near-resonance, Solomon fails

Note: Delta=0 is singular for this sweep (can't divide by zero),
so we sweep from g/Delta = 0.05 to g/Delta = 2.0.

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


# ─── Fixed parameters ─────────────────────────────────────────────────────────

FIXED = dict(
    wq      = 1.0,
    g       = 0.05,    # fixed coupling
    gamma_q = 0.01,
    gamma_t = 0.01,
)

# Sweep Delta = wq - wt: from Delta=g/0.05 (large detuning) to Delta=g/2 (near resonant)
G_OVER_DELTA = np.logspace(-1.3, 0.3, 40)   # 40 points
DELTA_VALUES = FIXED['g'] / G_OVER_DELTA     # Delta = g / (g/Delta)

T_END   = 800
N_STEPS = 800


def run_sweep():
    print("Sweeping g/Delta  (fixed g, varying detuning)")
    print(f"  g={FIXED['g']},  gamma_q={FIXED['gamma_q']},  "
          f"gamma_t={FIXED['gamma_t']}")
    print(f"  {len(DELTA_VALUES)} points from g/Delta = "
          f"{G_OVER_DELTA[0]:.2f} to {G_OVER_DELTA[-1]:.2f}")
    print()

    ise_arr  = []
    maxd_arr = []
    coh_arr  = []
    g1L_arr  = []
    g2L_arr  = []
    g1S_arr  = []
    g2S_arr  = []

    for delta in tqdm(DELTA_VALUES, desc="Sweep"):
        wt = FIXED['wq'] - delta
        params = {**FIXED, 'wt': wt}

        res_L = lindblad_evolve(**params, t_end=T_END, n_steps=N_STEPS)
        res_S = solomon_evolve( **params, t_end=T_END, n_steps=N_STEPS)

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

    save_sweep(sweep, '../data/sweeps/sweep_detuning.h5')

    # ── Figure ────────────────────────────────────────────────────────────────
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
    ax3.axvline(1, color='red', ls='--', lw=1, label='$g=\\Delta$')
    ax3.set_xlabel('$g/\\Delta$'); ax3.set_ylabel(r'max $|\rho_{eg,ge}|$')
    ax3.set_title('Peak Coherence'); ax3.legend()
    ax3.grid(True, which='both', alpha=0.3)

    ax4 = fig.add_subplot(gs[3])
    ax4.loglog(x, sweep['gamma1_L'], 'o-',  lw=2, label='$\\Gamma_1$ Lindblad')
    ax4.loglog(x, sweep['gamma1_S'], 's--', lw=2, label='$\\Gamma_1$ Solomon')
    ax4.loglog(x, sweep['gamma2_L'], 'o-',  lw=2, label='$\\Gamma_2$ Lindblad',
               color='C1')
    ax4.loglog(x, sweep['gamma2_S'], 's--', lw=2, label='$\\Gamma_2$ Solomon',
               color='C3')
    ax4.axvline(1, color='red', ls='--', lw=1)
    ax4.set_xlabel('$g/\\Delta$'); ax4.set_ylabel('Fitted rate')
    ax4.set_title('Decay Rates'); ax4.legend(fontsize=8)
    ax4.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        rf'Solomon breakdown vs $g/\Delta$  '
        rf'($g={FIXED["g"]:.3f}$, $\gamma_q=\gamma_t={FIXED["gamma_q"]:.3f}$)',
        fontsize=13
    )
    plt.savefig('../figures/sweeps/sweep_detuning.png',
                dpi=150, bbox_inches='tight')
    print("\nFigure saved → figures/sweeps/sweep_detuning.png")
    plt.show()


if __name__ == "__main__":
    run_sweep()
