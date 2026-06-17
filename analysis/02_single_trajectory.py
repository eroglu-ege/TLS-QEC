"""
02_single_trajectory.py
=======================
Full comparison of Lindblad vs Solomon at a single parameter point.

CONFIGURE HERE — change N_TH to run at finite temperature.
Results are saved with n_th label so old runs are not overwritten.

Run from: TLS-QEC/analysis/
    python 02_single_trajectory.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.metrics import full_comparison, print_summary
from utils.physics import n_thermal
from utils.io import save

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH   = 0.0     # thermal photon number (0.0 = T=0)
                  # use n_thermal(freq_GHz, temp_K) for physical units
                  # e.g. N_TH = n_thermal(5.0, 0.100) for 5GHz at 100mK

PARAMS = dict(
    wq      = 1.0,
    wt      = 1.0,   # resonant: Delta=0
    g       = 0.1,
    gamma_q = 0.01,
    gamma_t = 0.005,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

T_END   = 600
N_STEPS = 1000
# ──────────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"   # label for file names


def run():
    print(f"Running with N_TH = {N_TH:.4f}")
    print(f"  n_th_q = {PARAMS['n_th_q']:.6f}")
    print(f"  n_th_t = {PARAMS['n_th_t']:.6f}\n")

    print("Running Lindblad solver...")
    res_L = lindblad_evolve(
        **PARAMS, t_end=T_END, n_steps=N_STEPS,
        save_path=f'../data/single_trajectory/lindblad_{TAG}.h5'
    )

    print("Running Solomon solver...")
    res_S = solomon_evolve(
        **PARAMS, t_end=T_END, n_steps=N_STEPS,
        save_path=f'../data/single_trajectory/solomon_{TAG}.h5'
    )

    m = full_comparison(
        res_L['t'], res_L['P_e_q'], res_L['coherence'],
        res_S['t'], res_S['P_e_q'],
    )
    print_summary(m, PARAMS)

    metrics_out = {k: v for k, v in m.items() if not isinstance(v, dict)}
    metrics_out['params'] = PARAMS
    save(metrics_out, f'../data/single_trajectory/metrics_{TAG}.h5')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(res_L['t'], res_L['P_e_q'], lw=2,       label='Lindblad')
    ax1.plot(res_S['t'], res_S['P_e_q'], lw=2, ls='--', label='Solomon')
    if N_TH > 0:
        ax1.axhline(res_L['P_e_q_ss'], color='gray', ls=':', lw=1,
                    label=f'$P_{{ss}}$={res_L["P_e_q_ss"]:.4f}')
    ax1.set_xlabel('Time'); ax1.set_ylabel('Population')
    ax1.set_title('Qubit $P_e$'); ax1.legend(); ax1.set_ylim(-0.02, 1.05)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(res_L['t'], res_L['P_e_t'], lw=2,       color='C1', label='Lindblad')
    ax2.plot(res_S['t'], res_S['P_e_t'], lw=2, ls='--', color='C3', label='Solomon')
    ax2.set_xlabel('Time')
    ax2.set_title('TLS $P_e$'); ax2.legend(); ax2.set_ylim(-0.02, 1.05)

    ax3 = fig.add_subplot(gs[2])
    coh = np.where(res_L['coherence'] > 0, res_L['coherence'], 1e-16)
    ax3.semilogy(res_L['t'], coh, color='purple', lw=2)
    ax3.set_xlabel('Time'); ax3.set_ylabel(r'$|\rho_{eg,ge}|$')
    ax3.set_title('Coherence (Lindblad)')

    ax4 = fig.add_subplot(gs[3])
    P_S_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
    diff = np.abs(res_L['P_e_q'] - P_S_interp)
    ax4.semilogy(res_L['t'], np.where(diff > 0, diff, 1e-16), color='black', lw=2)
    ax4.set_xlabel('Time'); ax4.set_ylabel(r'$|P_L - P_S|$')
    ax4.set_title(f'Discrepancy  ISE={m["ISE"]:.3e}')

    p = PARAMS
    fig.suptitle(
        rf"$\Delta$={p['wq']-p['wt']:.2f}, $g$={p['g']:.3f}, "
        rf"$\gamma_q$={p['gamma_q']:.4f}, $\gamma_t$={p['gamma_t']:.4f}, "
        rf"$n_{{th}}$={N_TH:.4f}",
        fontsize=12
    )

    plt.savefig(f'../figures/single_trajectory/lindblad_vs_solomon_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → figures/single_trajectory/lindblad_vs_solomon_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
