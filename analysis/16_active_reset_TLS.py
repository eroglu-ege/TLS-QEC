"""
16_active_reset_TLS.py
======================
Active reset protocol: TLS is repeatedly re-initialized to P_t=1
every fixed time interval dt, while the qubit starts at P_q=0.

DIFFERENCE FROM INITIALIZATION:
  Initialization : set state once at t=0, evolve freely
  Active reset   : every dt, forcibly return TLS to P_t=1,
                   leave qubit untouched, evolve for another dt

PHYSICAL MOTIVATION:
  Models a TLS that is continuously re-excited by its environment
  (substrate phonons, charge noise, cosmic rays, parasitic drives).
  Question: how does the qubit accumulate population over N reset cycles?

PROTOCOL:
  Cycle 0: P_q=0, P_t=1  →  evolve dt  →  record P_q(dt), P_t(dt)
  Cycle 1: P_q unchanged, P_t=1  →  evolve dt  →  record
  Cycle 2: P_q unchanged, P_t=1  →  evolve dt  →  record
  ...
  Cycle N: same

FIGURES:
  1. P_TLS(t) and P_q(t) on same axes vs continuous time
     (shows the sawtooth TLS reset and gradual qubit buildup)
  2. P_TLS and P_q sampled at end of each cycle vs cycle number n
  3. P_TLS(end of cycle) vs n — does TLS drain faster each cycle?

Run from: TLS-QEC/analysis/
    python 16_active_reset_TLS.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qutip as qt

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.physics import make_params

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,       # resonant: Delta=0
    gamma_q = 0.01,      # T1_q = 100
    gamma_t = 0.005,     # T1_t = 200
    T2_q    = 50.0,      # T2_q = T1/2 = 50  -> gamma_phi_q = 0.030
    T2_t    = 100.0,     # T2_t = T1/2 = 100 -> gamma_phi_t = 0.010
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

# g/gamma_t = 0.7 -> g = 0.0035
# g/gamma_t = 0.8 -> g = 0.0040
# g/gamma_t = 0.5 -> g = 0.0025  (below threshold, for comparison)
G_VALUES = [0.0025, 0.0035, 0.0040]

# Reset protocol parameters
N_CYCLES = 20       # number of active reset cycles
DT       = 100      # time between resets = 1*T1_q = 2*T1_t
N_STEPS  = 200      # time steps per cycle

TAG = f"nth{N_TH:.3f}_T2q{PARAMS['T2_q']:.0f}_T2t{PARAMS['T2_t']:.0f}_dt{DT}"
# ──────────────────────────────────────────────────────────────────────────────

FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}


def run_active_reset(g, model='lindblad'):
    """
    Run the active reset protocol for N_CYCLES cycles.

    Returns:
        t_full     : continuous time array (length N_CYCLES * N_STEPS)
        Pq_full    : qubit population over full time
        Pt_full    : TLS population over full time
        Pq_end     : qubit population at end of each cycle (length N_CYCLES)
        Pt_end     : TLS population at end of each cycle (length N_CYCLES)
        Pt_start   : TLS population just before reset (= Pt_end, for clarity)
    """
    # Initial conditions
    P_q = 0.0   # qubit starts in ground
    P_t = 1.0   # TLS starts excited

    t_full   = []
    Pq_full  = []
    Pt_full  = []
    Pq_end   = []
    Pt_end   = []

    for cycle in range(N_CYCLES):
        t_offset = cycle * DT

        if model == 'lindblad':
            # Build initial density matrix from current populations
            # Assume no coherences between cycles (active reset destroys them)
            # rho = P_q*P_t|ee><ee| + P_q*(1-P_t)|eg><eg|
            #     + (1-P_q)*P_t|ge><ge| + (1-P_q)*(1-P_t)|gg><gg|
            rho0 = (P_q * P_t * qt.ket2dm(qt.tensor(qt.basis(2,0),qt.basis(2,0)))
                  + P_q*(1-P_t) * qt.ket2dm(qt.tensor(qt.basis(2,0),qt.basis(2,1)))
                  + (1-P_q)*P_t * qt.ket2dm(qt.tensor(qt.basis(2,1),qt.basis(2,0)))
                  + (1-P_q)*(1-P_t)*qt.ket2dm(qt.tensor(qt.basis(2,1),qt.basis(2,1))))

            res = lindblad_evolve(**FIXED, g=g,
                                  t_end=DT, n_steps=N_STEPS,
                                  rho0=rho0)
        else:  # solomon
            res = solomon_evolve(**FIXED, g=g,
                                 t_end=DT, n_steps=N_STEPS,
                                 P_q0=P_q, P_t0=P_t)

        t_full.extend(res['t'] + t_offset)
        Pq_full.extend(res['P_e_q'])
        Pt_full.extend(res['P_e_t'])

        # Record end-of-cycle values
        P_q_new = res['P_e_q'][-1]
        P_t_new = res['P_e_t'][-1]
        Pq_end.append(P_q_new)
        Pt_end.append(P_t_new)

        # Active reset: force TLS back to 1, leave qubit unchanged
        P_q = P_q_new   # qubit accumulates
        P_t = 1.0       # TLS forcibly reset to excited

    return {
        't':       np.array(t_full),
        'Pq':      np.array(Pq_full),
        'Pt':      np.array(Pt_full),
        'Pq_end':  np.array(Pq_end),
        'Pt_end':  np.array(Pt_end),
        'cycles':  np.arange(1, N_CYCLES+1),
    }


def run():
    os.makedirs('../data/sweeps', exist_ok=True)
    os.makedirs('../figures/sweeps', exist_ok=True)

    P_ss = N_TH / (2*N_TH + 1)
    print(f"Active reset protocol: {N_CYCLES} cycles, dt={DT}")
    print(f"N_TH={N_TH}  P_ss={P_ss:.4f}  (thermal steady state)")
    print(f"g/gamma_t values: {[round(g/PARAMS['gamma_t'],2) for g in G_VALUES]}\n")

    colors = ['C0', 'C1', 'C2']

    for g, color in zip(G_VALUES, colors):
        ratio = g / PARAMS['gamma_t']
        print(f"Running g/gamma_t={ratio:.2f} (g={g:.4f})...")

        res_L = run_active_reset(g, model='lindblad')
        res_S = run_active_reset(g, model='solomon')

        # ── Figure 1: full time traces ────────────────────────────────────────
        fig1, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(res_L['t'], res_L['Pq'], lw=2, color='C0',
                    label='Qubit (Lindblad)')
        axes[0].plot(res_L['t'], res_L['Pt'], lw=2, color='C1',
                    label='TLS (Lindblad)')
        axes[0].plot(res_S['t'], res_S['Pq'], lw=1.5, ls='--', color='C0',
                    alpha=0.7, label='Qubit (Solomon)')
        axes[0].plot(res_S['t'], res_S['Pt'], lw=1.5, ls='--', color='C1',
                    alpha=0.7, label='TLS (Solomon)')
        # Mark reset events
        for n in range(N_CYCLES):
            axes[0].axvline(n*DT, color='gray', ls=':', lw=0.8, alpha=0.5)
        axes[0].axhline(P_ss, color='black', ls=':', lw=1,
                       label=f'$P_{{ss}}$={P_ss:.3f}')
        axes[0].set_ylabel('Population', fontsize=12)
        axes[0].set_title(
            rf'$P_{{TLS}}$ and $P_q$ over {N_CYCLES} active reset cycles  '
            rf'($g/\gamma_t$={ratio:.2f}, $n_{{th}}$={N_TH})',
            fontsize=12)
        axes[0].legend(fontsize=9, ncol=2)
        axes[0].set_ylim(-0.02, 1.05)
        axes[0].grid(True, alpha=0.3)

        # Difference between Lindblad and Solomon
        axes[1].plot(res_L['t'], res_L['Pq'] - res_S['Pq'],
                    lw=2, color='C0', label='Qubit $P_L - P_S$')
        axes[1].plot(res_L['t'], res_L['Pt'] - res_S['Pt'],
                    lw=2, color='C1', label='TLS $P_L - P_S$')
        for n in range(N_CYCLES):
            axes[1].axvline(n*DT, color='gray', ls=':', lw=0.8, alpha=0.5)
        axes[1].axhline(0, color='black', ls='-', lw=0.5)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('$P_L - P_S$', fontsize=12)
        axes[1].set_title('Lindblad $-$ Solomon (Solomon error accumulation)',
                          fontsize=12)
        axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'../figures/sweeps/active_reset_traces_g{ratio:.2f}_{TAG}.png',
                    dpi=150, bbox_inches='tight')

        # ── Figure 2: per-cycle quantities vs n ───────────────────────────────
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        n = res_L['cycles']

        axes2[0].plot(n, res_L['Pq_end'], 'o-', lw=2, color='C0',
                     ms=6, label='Qubit (Lindblad)')
        axes2[0].plot(n, res_S['Pq_end'], 's--', lw=2, color='C0',
                     ms=6, alpha=0.7, label='Qubit (Solomon)')
        axes2[0].plot(n, res_L['Pt_end'], 'o-', lw=2, color='C1',
                     ms=6, label='TLS (Lindblad)')
        axes2[0].plot(n, res_S['Pt_end'], 's--', lw=2, color='C1',
                     ms=6, alpha=0.7, label='TLS (Solomon)')
        axes2[0].axhline(P_ss, color='black', ls=':', lw=1,
                        label=f'$P_{{ss}}$={P_ss:.3f}')
        axes2[0].set_xlabel('Reset cycle $n$', fontsize=12)
        axes2[0].set_ylabel('Population at end of cycle', fontsize=12)
        axes2[0].set_title('$P_q$ and $P_{{TLS}}$ at end of each cycle',
                           fontsize=12)
        axes2[0].legend(fontsize=9); axes2[0].grid(True, alpha=0.3)

        # TLS end-of-cycle population: how much does TLS drain each cycle?
        axes2[1].plot(n, res_L['Pt_end'], 'o-', lw=2, color='darkgreen',
                     ms=6, label='Lindblad')
        axes2[1].plot(n, res_S['Pt_end'], 's--', lw=2, color='olive',
                     ms=6, label='Solomon')
        axes2[1].axhline(P_ss, color='black', ls=':', lw=1,
                        label=f'Thermal $P_{{ss}}$')
        axes2[1].set_xlabel('Reset cycle $n$', fontsize=12)
        axes2[1].set_ylabel('$P_{{TLS}}$ at end of cycle', fontsize=12)
        axes2[1].set_title(
            '$P_{{TLS}}$ vs reset count $n$\n'
            '(how much TLS drains per cycle)',
            fontsize=12)
        axes2[1].legend(fontsize=9); axes2[1].grid(True, alpha=0.3)

        fig2.suptitle(
            rf'Active reset: per-cycle populations  '
            rf'($g/\gamma_t$={ratio:.2f}, $\Delta t$={DT}, $n_{{th}}$={N_TH})',
            fontsize=13)
        plt.tight_layout()
        plt.savefig(f'../figures/sweeps/active_reset_percycle_g{ratio:.2f}_{TAG}.png',
                    dpi=150, bbox_inches='tight')

        # Save data
        np.savez(f'../data/sweeps/active_reset_g{ratio:.2f}_{TAG}.npz',
                 t=res_L['t'],
                 Pq_L=res_L['Pq'], Pt_L=res_L['Pt'],
                 Pq_S=res_S['Pq'], Pt_S=res_S['Pt'],
                 Pq_end_L=res_L['Pq_end'], Pt_end_L=res_L['Pt_end'],
                 Pq_end_S=res_S['Pq_end'], Pt_end_S=res_S['Pt_end'],
                 cycles=res_L['cycles'],
                 g=g, g_over_gamma_t=ratio, DT=DT, N_cycles=N_CYCLES)

        print(f"  Final P_q after {N_CYCLES} cycles: "
              f"L={res_L['Pq_end'][-1]:.4f}  S={res_S['Pq_end'][-1]:.4f}")
        print(f"  TLS at end of last cycle: "
              f"L={res_L['Pt_end'][-1]:.4f}  S={res_S['Pt_end'][-1]:.4f}")

    print(f"\nData → data/sweeps/active_reset_*")
    print(f"Figures → figures/sweeps/active_reset_*")
    plt.show()


if __name__ == "__main__":
    run()
