"""
02_single_trajectory.py
=======================
Step 2: Full comparison of Lindblad vs Solomon at a single parameter point.

This is the core qualitative figure:
  - Left:   qubit populations (both models)
  - Middle: TLS populations (both models)
  - Right:  off-diagonal coherence |rho_{eg,ge}| (Lindblad only)
            + population difference |P_L - P_S|

Also prints all metrics from metrics.py.

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
from utils.io import save


# ─── Parameters ───────────────────────────────────────────────────────────────
# Working in units where gamma_q = 1
# Tune these to explore different regimes

PARAMS = dict(
    wq      = 1.0,
    wt      = 1.0,     # resonant: Delta = 0
    g       = 0.1,     # g/gamma_q = 10  (strong coupling)
    gamma_q = 0.01,
    gamma_t = 0.005,   # TLS lives longer than qubit
)

T_END   = 600
N_STEPS = 1000


def run():
    print("Running Lindblad solver...")
    res_L = lindblad_evolve(
        **PARAMS, t_end=T_END, n_steps=N_STEPS,
        save_path='../data/single_trajectory/lindblad.h5'
    )

    print("Running Solomon solver...")
    res_S = solomon_evolve(
        **PARAMS, t_end=T_END, n_steps=N_STEPS,
        save_path='../data/single_trajectory/solomon.h5'
    )

    # ── Compute metrics ───────────────────────────────────────────────────────
    m = full_comparison(
        res_L['t'], res_L['P_e_q'], res_L['coherence'],
        res_S['t'], res_S['P_e_q'],
    )
    print_summary(m, PARAMS)

    # Save metrics
    metrics_out = {k: v for k, v in m.items()
                   if not isinstance(v, dict)}
    metrics_out['params'] = PARAMS
    save(metrics_out, '../data/single_trajectory/metrics.h5')

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    # Qubit population
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(res_L['t'], res_L['P_e_q'], lw=2,       label='Lindblad')
    ax1.plot(res_S['t'], res_S['P_e_q'], lw=2, ls='--', label='Solomon')
    ax1.set_xlabel('Time'); ax1.set_ylabel('Population')
    ax1.set_title('Qubit $P_e$'); ax1.legend(); ax1.set_ylim(-0.02, 1.05)

    # TLS population
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(res_L['t'], res_L['P_e_t'], lw=2,       label='Lindblad', color='C1')
    ax2.plot(res_S['t'], res_S['P_e_t'], lw=2, ls='--', label='Solomon',  color='C3')
    ax2.set_xlabel('Time')
    ax2.set_title('TLS $P_e$'); ax2.legend(); ax2.set_ylim(-0.02, 1.05)

    # Coherence
    ax3 = fig.add_subplot(gs[2])
    coh_safe = np.where(res_L['coherence'] > 0, res_L['coherence'], 1e-16)
    ax3.semilogy(res_L['t'], coh_safe, color='purple', lw=2)
    ax3.set_xlabel('Time'); ax3.set_ylabel(r'$|\rho_{eg,ge}|$')
    ax3.set_title('Coherence (Lindblad)')
    ax3.axhline(1e-3, color='gray', ls=':', lw=1, label='1e-3')
    ax3.legend()

    # Population difference
    ax4 = fig.add_subplot(gs[3])
    P_S_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
    diff = np.abs(res_L['P_e_q'] - P_S_interp)
    ax4.semilogy(res_L['t'], np.where(diff > 0, diff, 1e-16),
                 color='black', lw=2)
    ax4.set_xlabel('Time')
    ax4.set_ylabel(r'$|P_L - P_S|$')
    ax4.set_title(f'Discrepancy  ISE={m["ISE"]:.3e}')

    p = PARAMS
    delta = p['wq'] - p['wt']
    fig.suptitle(
        rf"$\Delta$={delta:.2f}, $g$={p['g']:.3f}, "
        rf"$\gamma_q$={p['gamma_q']:.4f}, $\gamma_t$={p['gamma_t']:.4f}  "
        rf"[$g/\gamma_t$={p['g']/p['gamma_t']:.1f}]",
        fontsize=12
    )

    plt.savefig('../figures/single_trajectory/lindblad_vs_solomon.png',
                dpi=150, bbox_inches='tight')
    print("\nFigure saved → figures/single_trajectory/lindblad_vs_solomon.png")
    plt.show()


if __name__ == "__main__":
    run()
