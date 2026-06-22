"""
09_sweep_T2.py
==============
Sweep T2/T1 ratio and measure Solomon breakdown metrics.

PHYSICS MOTIVATION:
    T2 controls how fast quantum coherences die:
        1/T2 = 1/(2*T1) + 1/T_phi   (where T_phi = 1/gamma_phi)

    Since coherences are the CAUSE of Solomon's failure (r=0.996),
    shorter T2 should make Solomon more valid — coherences die before
    they can build up. This sweep tests that prediction directly.

    Expected behavior:
        T2 -> 2*T1 (no dephasing)  : Solomon fails most (max ISE)
        T2 -> 0    (heavy dephasing): Solomon becomes exact (ISE -> 0)

    The crossing point T2* where D < 0.05 tells experimentalists
    what coherence time regime makes Solomon trustworthy.

SWEEP AXIS: T2_q/T1_q from 0.1 to 2.0
    (T2 = 2*T1 is the no-dephasing Lindblad limit, maximum T2)
    (T2 = 0.1*T1 means very strong pure dephasing)

Note: we sweep T2_q and set T2_t = T2_q for simplicity.
      A 2D sweep over (T2_q, T2_t) is a natural extension.

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
G       = 0.1      # fixed coupling (g/gamma_t = 20, well into Rabi regime
                   # so Solomon is clearly wrong at T2=2*T1 — gives
                   # maximum dynamic range for the sweep)

# Base relaxation rates
GAMMA_Q = 0.01     # T1_q = 100
GAMMA_T = 0.005    # T1_t = 200

# Sweep T2_q/T1_q from 0.1 to 2.0
# (2.0 = no dephasing limit; 0.1 = very short T2)
N_POINTS   = 40
T2_RATIO   = np.linspace(0.1, 2.0, N_POINTS)   # T2/T1 for qubit
T2_Q_VALUES = T2_RATIO * (1.0 / GAMMA_Q)        # absolute T2_q values

T_END   = 600
N_STEPS = 800

TAG = f"nth{N_TH:.4f}_g{G:.4f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    print(f"T2 sweep: T2_q/T1_q from {T2_RATIO[0]:.2f} to {T2_RATIO[-1]:.2f}")
    print(f"  g={G:.3f}  gamma_q={GAMMA_Q}  gamma_t={GAMMA_T}  N_TH={N_TH}")
    print(f"  g/gamma_t = {G/GAMMA_T:.1f}  (deep in Rabi regime at T2=2*T1)\n")

    ise_arr   = np.zeros(N_POINTS)
    coh_arr   = np.zeros(N_POINTS)
    D_arr     = np.zeros(N_POINTS)
    gamma_phi_arr = np.zeros(N_POINTS)

    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))  # |eg>: qubit excited
    rho0 = qt.ket2dm(psi0)

    for i, T2_q in enumerate(tqdm(T2_Q_VALUES, desc="T2 sweep")):
        # Set T2_t = T2_q * (T1_t/T1_q) to keep T2/T1 ratio the same for TLS
        T2_t = T2_RATIO[i] * (1.0 / GAMMA_T)

        try:
            p = make_params(
                wq=1.0, wt=1.0,
                gamma_q=GAMMA_Q, gamma_t=GAMMA_T,
                T2_q=T2_q, T2_t=T2_t,
                n_th_q=N_TH, n_th_t=N_TH,
            )
        except ValueError:
            # T2 > 2*T1 not physically possible — cap at 2*T1
            p = make_params(wq=1.0, wt=1.0,
                           gamma_q=GAMMA_Q, gamma_t=GAMMA_T,
                           n_th_q=N_TH, n_th_t=N_TH)

        gamma_phi_arr[i] = p['gamma_phi_q']

        fixed = {k: p[k] for k in
                 ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
                  "n_th_q","n_th_t"]}

        res_L = lindblad_evolve(**fixed, g=G, t_end=T_END, n_steps=N_STEPS,
                                rho0=rho0)
        res_S = solomon_evolve(**fixed, g=G, t_end=T_END, n_steps=N_STEPS,
                               P_q0=1.0, P_t0=0.0)

        P_S_q = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
        P_S_t = np.interp(res_L['t'], res_S['t'], res_S['P_e_t'])

        ise_arr[i] = ISE(res_L['t'], res_L['P_e_q'], P_S_q)
        coh_arr[i] = coherence_metrics(res_L['t'], res_L['coherence'])['max']
        D = trace_distance_trajectory(res_L['states'], P_S_q, P_S_t)
        D_arr[i] = D.max()

    np.savez(f'../data/sweeps/sweep_T2_{TAG}.npz',
             T2_ratio=T2_RATIO, T2_q_values=T2_Q_VALUES,
             gamma_phi=gamma_phi_arr,
             ISE=ise_arr, coherence_max=coh_arr, D_max=D_arr)
    print(f"\nData saved → data/sweeps/sweep_T2_{TAG}.npz")

    # Find where Solomon becomes valid (D < 0.05)
    valid_idx = np.where(D_arr < 0.05)[0]
    if len(valid_idx) > 0:
        print(f"Solomon valid (D<0.05) for T2/T1 < {T2_RATIO[valid_idx[-1]]:.3f}")
        print(f"  corresponding gamma_phi_q = {gamma_phi_arr[valid_idx[-1]]:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.semilogy(T2_RATIO, ise_arr, 'o-', lw=2, color='black')
    ax1.axvline(1.0, color='blue', ls='--', lw=1.5, label='$T_2=T_1$')
    ax1.axvline(2.0, color='gray', ls=':', lw=1.5, label='$T_2=2T_1$ (max)')
    ax1.set_xlabel('$T_2^q/T_1^q$', fontsize=12)
    ax1.set_ylabel('ISE', fontsize=12)
    ax1.set_title('Total disagreement', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', alpha=0.3)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(T2_RATIO, coh_arr, 'o-', lw=2, color='purple',
             label='Peak coherence')
    ax2.axhline(0.0, color='gray', ls=':', lw=1)
    ax2.axvline(1.0, color='blue', ls='--', lw=1.5)
    ax2.set_xlabel('$T_2^q/T_1^q$', fontsize=12)
    ax2.set_ylabel(r'max $|\rho_{eg,ge}|$', fontsize=12)
    ax2.set_title('Peak coherence vs $T_2$', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(T2_RATIO, D_arr, 'o-', lw=2, color='darkred',
             label='Trace distance')
    ax3.axhline(0.05, color='green', ls='--', lw=1.5,
               label='$D=0.05$ (significance)')
    ax3.axvline(1.0, color='blue', ls='--', lw=1.5, label='$T_2=T_1$')
    # Shade the Solomon-valid region
    valid = T2_RATIO[D_arr < 0.05]
    if len(valid) > 0:
        ax3.axvspan(T2_RATIO[0], valid[-1], alpha=0.1, color='green',
                   label='Solomon valid')
    ax3.set_xlabel('$T_2^q/T_1^q$', fontsize=12)
    ax3.set_ylabel('Max trace distance $D$', fontsize=12)
    ax3.set_title('Rigorous validity boundary', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        rf'Solomon validity vs $T_2$  '
        rf'($g/\gamma_t$={G/GAMMA_T:.0f}, $n_{{th}}$={N_TH:.1f}, '
        rf'$\Delta=0$)',
        fontsize=14
    )
    plt.savefig(f'../figures/sweeps/sweep_T2_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/sweep_T2_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
