"""
17_active_reset_ensemble.py
============================
Active reset protocol with randomized initial qubit population.

PROTOCOL:
  - TLS always starts at P_t = 1 (fully excited)
  - Qubit starts at P_q^(0) drawn uniformly from [0, 1]
  - N_SHOTS independent realizations, each with different P_q^(0)
  - Each shot runs the full active reset protocol (N_CYCLES cycles of dt)
  - TLS forcibly reset to 1 at start of each cycle

PHYSICAL MOTIVATION:
  Models the situation where the qubit state before a QEC round is
  unknown — it could be anywhere between ground and excited due to
  previous operations, incomplete initialization, or measurement
  backaction. The TLS is assumed to be freshly excited at the start
  of each round (e.g. pumped by substrate noise).

FIGURES:
  1. P_q(t) ensemble: all N_SHOTS trajectories as thin lines +
     mean and ±1 std band. TLS mean overlaid.
  2. P_q at end of each cycle vs n: trajectories + mean
  3. P_TLS at end of each cycle vs n: how TLS drain depends on P_q^(0)
  4. Final P_q (after N_CYCLES) vs initial P_q^(0): does memory persist?

Run from: TLS-QEC/analysis/
    python 17_active_reset_ensemble.py
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
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.005,
    T2_q    = 50.0,    # T2 = T1/2
    T2_t    = 100.0,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

G          = 0.0040    # g/gamma_t = 0.8 — threshold region
N_SHOTS    = 30        # number of independent random initializations
N_CYCLES   = 20        # active reset cycles per shot
DT         = 100       # time per cycle
N_STEPS    = 200       # time steps per cycle
SEED       = 42        # reproducibility

TAG = (f"g{G:.4f}_nth{N_TH:.3f}_"
       f"T2q{PARAMS['T2_q']:.0f}_"
       f"shots{N_SHOTS}_dt{DT}")
# ──────────────────────────────────────────────────────────────────────────────

FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}
P_SS = N_TH / (2*N_TH + 1)


def run_one_shot(P_q0, model='lindblad'):
    """Run active reset protocol for one initial P_q0, TLS always starts at 1."""
    P_q = P_q0
    P_t = 1.0

    t_full  = []
    Pq_full = []
    Pt_full = []
    Pq_end  = []
    Pt_end  = []

    for cycle in range(N_CYCLES):
        t_offset = cycle * DT

        if model == 'lindblad':
            # Diagonal density matrix — no coherences after reset
            rho0 = (P_q*P_t         * qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,0)))
                  + P_q*(1-P_t)     * qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,1)))
                  + (1-P_q)*P_t     * qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,0)))
                  + (1-P_q)*(1-P_t) * qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,1))))

            res = lindblad_evolve(**FIXED, g=G, t_end=DT,
                                  n_steps=N_STEPS, rho0=rho0)
        else:
            res = solomon_evolve(**FIXED, g=G, t_end=DT,
                                 n_steps=N_STEPS, P_q0=P_q, P_t0=P_t)

        t_full.extend(res['t'] + t_offset)
        Pq_full.extend(res['P_e_q'])
        Pt_full.extend(res['P_e_t'])
        Pq_end.append(res['P_e_q'][-1])
        Pt_end.append(res['P_e_t'][-1])

        P_q = res['P_e_q'][-1]   # qubit carries over
        P_t = 1.0                 # TLS reset

    return {
        't':      np.array(t_full),
        'Pq':     np.array(Pq_full),
        'Pt':     np.array(Pt_full),
        'Pq_end': np.array(Pq_end),
        'Pt_end': np.array(Pt_end),
    }


def run():
    os.makedirs('../data/sweeps', exist_ok=True)
    os.makedirs('../figures/sweeps', exist_ok=True)

    rng    = np.random.default_rng(SEED)
    Pq0_values = rng.uniform(0.0, 1.0, N_SHOTS)

    print(f"Ensemble active reset: {N_SHOTS} shots, g/gamma_t={G/PARAMS['gamma_t']:.2f}")
    print(f"  T2_q={PARAMS['T2_q']:.0f}  T2_t={PARAMS['T2_t']:.0f}")
    print(f"  N_TH={N_TH}  P_ss={P_SS:.4f}")
    print(f"  P_q^(0) uniform in [0,1], TLS always starts at 1\n")

    # Run all shots for both models
    all_Pq_L  = []   # shape (N_SHOTS, N_CYCLES*N_STEPS)
    all_Pt_L  = []
    all_end_Pq_L = []  # shape (N_SHOTS, N_CYCLES)
    all_end_Pt_L = []
    all_end_Pq_S = []
    all_end_Pt_S = []
    final_Pq_L   = []
    final_Pq_S   = []

    t_axis = None

    for i, Pq0 in enumerate(Pq0_values):
        print(f"  Shot {i+1:2d}/{N_SHOTS}  P_q^(0)={Pq0:.3f}")
        res_L = run_one_shot(Pq0, model='lindblad')
        res_S = run_one_shot(Pq0, model='solomon')

        all_Pq_L.append(res_L['Pq'])
        all_Pt_L.append(res_L['Pt'])
        all_end_Pq_L.append(res_L['Pq_end'])
        all_end_Pt_L.append(res_L['Pt_end'])
        all_end_Pq_S.append(res_S['Pq_end'])
        all_end_Pt_S.append(res_S['Pt_end'])
        final_Pq_L.append(res_L['Pq_end'][-1])
        final_Pq_S.append(res_S['Pq_end'][-1])

        if t_axis is None:
            t_axis = res_L['t']

    all_Pq_L     = np.array(all_Pq_L)      # (N_SHOTS, T_total)
    all_Pt_L     = np.array(all_Pt_L)
    all_end_Pq_L = np.array(all_end_Pq_L)  # (N_SHOTS, N_CYCLES)
    all_end_Pt_L = np.array(all_end_Pt_L)
    all_end_Pq_S = np.array(all_end_Pq_S)
    all_end_Pt_S = np.array(all_end_Pt_S)
    final_Pq_L   = np.array(final_Pq_L)
    final_Pq_S   = np.array(final_Pq_S)
    cycles       = np.arange(1, N_CYCLES+1)

    # Sort by initial P_q for clean coloring
    sort_idx = np.argsort(Pq0_values)

    # Save
    np.savez(f'../data/sweeps/active_reset_ensemble_{TAG}.npz',
             Pq0_values=Pq0_values, t=t_axis, cycles=cycles,
             all_Pq_L=all_Pq_L, all_Pt_L=all_Pt_L,
             all_end_Pq_L=all_end_Pq_L, all_end_Pt_L=all_end_Pt_L,
             all_end_Pq_S=all_end_Pq_S, all_end_Pt_S=all_end_Pt_S,
             final_Pq_L=final_Pq_L, final_Pq_S=final_Pq_S,
             g=G, g_over_gamma_t=G/PARAMS['gamma_t'],
             DT=DT, N_cycles=N_CYCLES, N_shots=N_SHOTS)
    print(f"\nData → data/sweeps/active_reset_ensemble_{TAG}.npz")

    # ── Color map: color each trajectory by its P_q^(0) ──────────────────────
    cmap   = plt.cm.coolwarm
    norm   = plt.Normalize(0, 1)

    # ── Figure 1: full time traces ────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    # Top: P_q trajectories
    for i in sort_idx:
        c = cmap(norm(Pq0_values[i]))
        axes1[0].plot(t_axis, all_Pq_L[i], color=c, lw=0.8, alpha=0.5)

    # Mean and std band
    Pq_mean = all_Pq_L.mean(axis=0)
    Pq_std  = all_Pq_L.std(axis=0)
    axes1[0].fill_between(t_axis, Pq_mean-Pq_std, Pq_mean+Pq_std,
                          color='black', alpha=0.15, label='±1 std')
    axes1[0].plot(t_axis, Pq_mean, 'k-', lw=2.5, label='Mean $P_q$')
    # TLS mean
    Pt_mean = all_Pt_L.mean(axis=0)
    axes1[0].plot(t_axis, Pt_mean, 'r-', lw=2, alpha=0.8, label='Mean $P_{TLS}$')
    axes1[0].axhline(P_SS, color='gray', ls=':', lw=1,
                    label=f'$P_{{ss}}$={P_SS:.3f}')
    for n in range(N_CYCLES):
        axes1[0].axvline(n*DT, color='gray', ls=':', lw=0.6, alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=axes1[0], label='Initial $P_q^{(0)}$')
    axes1[0].set_ylabel('Population', fontsize=12)
    axes1[0].set_title(
        rf'{N_SHOTS} shots: $P_q^{{(0)}} \sim \mathcal{{U}}[0,1]$, '
        rf'$P_{{TLS}}^{{(0)}}=1$, active reset every $\Delta t={DT}$  '
        rf'($g/\gamma_t$={G/PARAMS["gamma_t"]:.2f})',
        fontsize=12)
    axes1[0].legend(fontsize=9, loc='upper right')
    axes1[0].set_ylim(-0.02, 1.05)
    axes1[0].grid(True, alpha=0.3)

    # Bottom: std over time — does spread grow or collapse?
    axes1[1].plot(t_axis, Pq_std, 'k-', lw=2, label='Std of $P_q$')
    axes1[1].axhline(Pq_std[0], color='blue', ls='--', lw=1,
                    label=f'Initial std={Pq_std[0]:.3f}')
    for n in range(N_CYCLES):
        axes1[1].axvline(n*DT, color='gray', ls=':', lw=0.6, alpha=0.4)
    axes1[1].set_xlabel('Time', fontsize=12)
    axes1[1].set_ylabel('Std($P_q$)', fontsize=12)
    axes1[1].set_title('Spread of ensemble — does initial P_q^(0) matter at long times?',
                       fontsize=12)
    axes1[1].legend(fontsize=9); axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/ensemble_traces_{TAG}.png',
                dpi=150, bbox_inches='tight')

    # ── Figure 2: per-cycle — P_q and P_TLS vs n ─────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    # P_q end-of-cycle
    ax = axes2[0]
    for i in sort_idx:
        c = cmap(norm(Pq0_values[i]))
        ax.plot(cycles, all_end_Pq_L[i], color=c, lw=1.2, alpha=0.6)
    Pq_end_mean = all_end_Pq_L.mean(axis=0)
    Pq_end_std  = all_end_Pq_L.std(axis=0)
    ax.fill_between(cycles, Pq_end_mean-Pq_end_std,
                    Pq_end_mean+Pq_end_std, color='black', alpha=0.15)
    ax.plot(cycles, Pq_end_mean, 'k-', lw=2.5, label='Mean')
    ax.axhline(P_SS, color='gray', ls=':', lw=1, label=f'$P_{{ss}}$')
    plt.colorbar(sm, ax=ax, label='$P_q^{(0)}$')
    ax.set_xlabel('Reset cycle $n$', fontsize=12)
    ax.set_ylabel('$P_q$ at end of cycle', fontsize=12)
    ax.set_title('Qubit population vs reset count', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # P_TLS end-of-cycle
    ax = axes2[1]
    for i in sort_idx:
        c = cmap(norm(Pq0_values[i]))
        ax.plot(cycles, all_end_Pt_L[i], color=c, lw=1.2, alpha=0.6)
    Pt_end_mean = all_end_Pt_L.mean(axis=0)
    ax.plot(cycles, Pt_end_mean, 'k-', lw=2.5, label='Mean $P_{TLS}$ end')
    plt.colorbar(sm, ax=ax, label='$P_q^{(0)}$')
    ax.set_xlabel('Reset cycle $n$', fontsize=12)
    ax.set_ylabel('$P_{TLS}$ at end of cycle', fontsize=12)
    ax.set_title('TLS drain per cycle vs reset count\n(backpressure from qubit?)',
                 fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Final P_q vs initial P_q^(0) — memory test
    ax = axes2[2]
    ax.scatter(Pq0_values, final_Pq_L, c=Pq0_values, cmap=cmap,
               s=80, zorder=5, label='Lindblad')
    ax.scatter(Pq0_values, final_Pq_S, c=Pq0_values, cmap=cmap,
               s=40, marker='s', alpha=0.6, label='Solomon')
    ax.axhline(P_SS, color='gray', ls=':', lw=1.5, label=f'$P_{{ss}}$={P_SS:.3f}')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.3, label='No change (y=x)')
    ax.set_xlabel('Initial $P_q^{(0)}$', fontsize=12)
    ax.set_ylabel(f'Final $P_q$ after {N_CYCLES} cycles', fontsize=12)
    ax.set_title('Memory erasure test:\ninitial condition vs final state',
                 fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig2.suptitle(
        rf'Active reset ensemble: $P_q^{{(0)}} \sim \mathcal{{U}}[0,1]$, '
        rf'$g/\gamma_t$={G/PARAMS["gamma_t"]:.2f}, '
        rf'$n_{{th}}$={N_TH}, $T_2^q$={PARAMS["T2_q"]:.0f}',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/ensemble_percycle_{TAG}.png',
                dpi=150, bbox_inches='tight')

    print(f"Figures → figures/sweeps/ensemble_*_{TAG}.png")

    # Print memory erasure stats
    print(f"\n=== MEMORY ERASURE AFTER {N_CYCLES} CYCLES ===")
    print(f"  Initial P_q std:  {Pq0_values.std():.4f}")
    print(f"  Final   P_q std (L): {final_Pq_L.std():.4f}")
    print(f"  Final   P_q std (S): {final_Pq_S.std():.4f}")
    print(f"  Final   P_q mean (L): {final_Pq_L.mean():.4f}  (P_ss={P_SS:.4f})")
    corr_L = np.corrcoef(Pq0_values, final_Pq_L)[0,1]
    corr_S = np.corrcoef(Pq0_values, final_Pq_S)[0,1]
    print(f"  Correlation final vs initial (L): {corr_L:.4f}")
    print(f"  Correlation final vs initial (S): {corr_S:.4f}")
    print(f"  (0 = memory fully erased, 1 = perfect memory)")

    plt.show()


if __name__ == "__main__":
    run()
