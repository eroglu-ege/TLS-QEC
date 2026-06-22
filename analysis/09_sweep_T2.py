"""
09_sweep_T2.py
==============
Sweep T2/T1 at multiple fixed g values in the threshold region
g/gamma_t in [0.2, 2.0] — where Solomon first breaks down.

From fine_threshold_scan results:
    g/gamma_t = 0.216 : D first exceeds 0.05 (significance threshold)
    g/gamma_t = 0.860 : maximum Solomon overshoot (sign reversal)
    g/gamma_t = 1.950 : L_max = S_max crossing (sign change back)

QUESTION: As T2 decreases (more dephasing), does Solomon recover
in this threshold region? Expected: yes — shorter T2 kills coherences
faster, pushing Lindblad toward Solomon's incoherent picture.

For each fixed g, sweep T2/T1 from 0.1 to 2.0 and track D_max.
The T2* at which D drops below 0.05 is the "recovery threshold".

Run from: TLS-QEC/analysis/
    python 09_sweep_T2.py
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
from utils.metrics import ISE, coherence_metrics
from utils.metrics_extra import trace_distance_trajectory
from utils.physics import make_params

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH    = 0.1
GAMMA_Q = 0.01    # T1_q = 100
GAMMA_T = 0.005   # T1_t = 200

# Fixed g values in the threshold region [0.2, 2.0] in g/gamma_t units
# Chosen to hit: below threshold, sign reversal, crossing, above crossing
G_OVER_GAMMA_T_FIXED = [0.1, 0.216, 0.5, 0.86, 1.0, 1.95, 3.0]
G_FIXED_VALUES = [r * GAMMA_T for r in G_OVER_GAMMA_T_FIXED]

# T2/T1 sweep axis
N_T2    = 30
T2_RATIO = np.linspace(0.1, 2.0, N_T2)   # T2/T1, from heavy dephasing to max

T_END   = 1200
N_STEPS = 600

TAG = f"nth{N_TH:.4f}_threshold_region"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    print(f"T2 sweep in threshold region")
    print(f"  g/gamma_t values: {G_OVER_GAMMA_T_FIXED}")
    print(f"  T2/T1 range: [{T2_RATIO[0]:.2f}, {T2_RATIO[-1]:.2f}]  ({N_T2} points)")
    print(f"  N_TH={N_TH}\n")

    # Results: shape (n_g, n_T2)
    n_g = len(G_FIXED_VALUES)
    D_grid   = np.zeros((n_g, N_T2))
    ISE_grid = np.zeros((n_g, N_T2))
    coh_grid = np.zeros((n_g, N_T2))

    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))  # |eg>: qubit excited
    rho0 = qt.ket2dm(psi0)

    for i, (g, ratio_label) in enumerate(zip(G_FIXED_VALUES, G_OVER_GAMMA_T_FIXED)):
        print(f"  g/gamma_t = {ratio_label:.3f}  (g={g:.5f})")
        for j, T2r in enumerate(tqdm(T2_RATIO, desc=f"    T2 sweep", leave=False)):
            T2_q = T2r / GAMMA_Q
            T2_t = T2r / GAMMA_T

            try:
                p = make_params(wq=1.0, wt=1.0,
                               gamma_q=GAMMA_Q, gamma_t=GAMMA_T,
                               T2_q=T2_q, T2_t=T2_t,
                               n_th_q=N_TH, n_th_t=N_TH)
            except ValueError:
                p = make_params(wq=1.0, wt=1.0,
                               gamma_q=GAMMA_Q, gamma_t=GAMMA_T,
                               n_th_q=N_TH, n_th_t=N_TH)

            fixed = {k: p[k] for k in
                     ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
                      "n_th_q","n_th_t"]}

            res_L = lindblad_evolve(**fixed, g=g, t_end=T_END, n_steps=N_STEPS,
                                    rho0=rho0)
            res_S = solomon_evolve(**fixed, g=g, t_end=T_END, n_steps=N_STEPS,
                                   P_q0=1.0, P_t0=0.0)

            P_S_q = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
            P_S_t = np.interp(res_L['t'], res_S['t'], res_S['P_e_t'])

            ISE_grid[i, j] = ISE(res_L['t'], res_L['P_e_q'], P_S_q)
            coh_grid[i, j] = coherence_metrics(res_L['t'], res_L['coherence'])['max']
            D = trace_distance_trajectory(res_L['states'], P_S_q, P_S_t)
            D_grid[i, j] = D.max()

    np.savez(f'../data/sweeps/sweep_T2_{TAG}.npz',
             T2_ratio=T2_RATIO, g_values=G_FIXED_VALUES,
             g_over_gamma_t=G_OVER_GAMMA_T_FIXED,
             D_grid=D_grid, ISE_grid=ISE_grid, coh_grid=coh_grid)
    print(f"\nData saved → data/sweeps/sweep_T2_{TAG}.npz")

    # Print recovery thresholds
    print("\n=== SOLOMON RECOVERY THRESHOLDS (D < 0.05) ===")
    for i, ratio in enumerate(G_OVER_GAMMA_T_FIXED):
        below = T2_RATIO[D_grid[i] < 0.05]
        if len(below) > 0:
            print(f"  g/γt={ratio:.3f}: Solomon valid (D<0.05) for T2/T1 < {below[-1]:.3f}")
        else:
            print(f"  g/γt={ratio:.3f}: Solomon NEVER valid in scanned T2 range")

    # ── Figure: D_max vs T2/T1 for each g value ─────────────────────────────
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_g))

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    for i, ratio in enumerate(G_OVER_GAMMA_T_FIXED):
        ax1.plot(T2_RATIO, D_grid[i], 'o-', lw=2, ms=4,
                color=colors[i], label=f'$g/\\gamma_t$={ratio:.2f}')
    ax1.axhline(0.05, color='green', ls='--', lw=1.5, label='$D=0.05$')
    ax1.set_xlabel('$T_2/T_1$', fontsize=12)
    ax1.set_ylabel('Max trace distance $D$', fontsize=12)
    ax1.set_title('Rigorous validity vs $T_2$', fontsize=12)
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    for i, ratio in enumerate(G_OVER_GAMMA_T_FIXED):
        ax2.semilogy(T2_RATIO, ISE_grid[i], 'o-', lw=2, ms=4,
                    color=colors[i], label=f'$g/\\gamma_t$={ratio:.2f}')
    ax2.set_xlabel('$T_2/T_1$', fontsize=12)
    ax2.set_ylabel('ISE', fontsize=12)
    ax2.set_title('Total disagreement vs $T_2$', fontsize=12)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, which='both', alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    for i, ratio in enumerate(G_OVER_GAMMA_T_FIXED):
        ax3.plot(T2_RATIO, coh_grid[i], 'o-', lw=2, ms=4,
                color=colors[i], label=f'$g/\\gamma_t$={ratio:.2f}')
    ax3.set_xlabel('$T_2/T_1$', fontsize=12)
    ax3.set_ylabel('Peak coherence', fontsize=12)
    ax3.set_title('Coherence vs $T_2$', fontsize=12)
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        rf'Solomon recovery as $T_2$ decreases — threshold region $g/\gamma_t \in [0.1, 3.0]$'
        rf'  ($n_{{th}}$={N_TH})',
        fontsize=13
    )
    plt.savefig(f'../figures/sweeps/sweep_T2_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/sweep_T2_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
