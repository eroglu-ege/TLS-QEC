"""
18_free_evolution_ensemble.py
==============================
Free evolution ensemble: TLS starts at 1, qubit randomized in [0,1],
NO active reset. System evolves freely for the full duration.

Compare with 17_active_reset_ensemble.py to isolate the effect of
the active reset protocol.

PROTOCOL:
  - TLS starts at P_t = 1 (fully excited)
  - Qubit starts at P_q^(0) drawn uniformly from [0, 1]
  - N_SHOTS independent realizations
  - No resets — system evolves freely for N_CYCLES * DT total time
  - Same total time as the active reset protocol for direct comparison

FIGURES: identical structure to 17 for direct visual comparison
  1. Full time traces: all trajectories + mean + std band
  2. Per-cycle snapshots (sampled at same times as reset protocol)
  3. Memory erasure: final P_q vs initial P_q^(0)

Run from: TLS-QEC/analysis/
    python 18_free_evolution_ensemble.py
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
    T2_q    = 50.0,
    T2_t    = 100.0,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

G        = 0.0040     # g/gamma_t = 0.8
N_SHOTS  = 30
N_CYCLES = 20         # kept same as script 17 for comparison
DT       = 100        # same time interval for snapshot sampling
N_STEPS  = N_CYCLES * 200   # total steps for one long evolution

T_TOTAL  = N_CYCLES * DT    # same total time as reset protocol
SEED     = 42               # same seed → same Pq0 values as script 17

TAG = (f"g{G:.4f}_nth{N_TH:.3f}_"
       f"T2q{PARAMS['T2_q']:.0f}_"
       f"shots{N_SHOTS}_free")
# ──────────────────────────────────────────────────────────────────────────────

FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}
P_SS = N_TH / (2*N_TH + 1)
SNAPSHOT_TIMES = np.arange(1, N_CYCLES+1) * DT   # sample at same times as reset


def run_one_shot(Pq0, model='lindblad'):
    """Free evolution from (Pq0, Pt=1) for T_TOTAL time."""
    P_t = 1.0

    if model == 'lindblad':
        rho0 = (Pq0*P_t         * qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,0)))
              + Pq0*(1-P_t)     * qt.ket2dm(qt.tensor(qt.basis(2,0), qt.basis(2,1)))
              + (1-Pq0)*P_t     * qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,0)))
              + (1-Pq0)*(1-P_t) * qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(2,1))))
        res = lindblad_evolve(**FIXED, g=G, t_end=T_TOTAL,
                              n_steps=N_STEPS, rho0=rho0)
    else:
        res = solomon_evolve(**FIXED, g=G, t_end=T_TOTAL,
                             n_steps=N_STEPS, P_q0=Pq0, P_t0=P_t)

    # Sample at snapshot times (equivalent cycle endpoints)
    t = res['t']
    Pq_snap = np.array([res['P_e_q'][np.argmin(np.abs(t - ts))]
                        for ts in SNAPSHOT_TIMES])
    Pt_snap = np.array([res['P_e_t'][np.argmin(np.abs(t - ts))]
                        for ts in SNAPSHOT_TIMES])

    return {
        't':      t,
        'Pq':     res['P_e_q'],
        'Pt':     res['P_e_t'],
        'Pq_snap': Pq_snap,
        'Pt_snap': Pt_snap,
    }


def run():
    os.makedirs('../data/sweeps', exist_ok=True)
    os.makedirs('../figures/sweeps', exist_ok=True)

    rng = np.random.default_rng(SEED)
    Pq0_values = rng.uniform(0.0, 1.0, N_SHOTS)

    print(f"Free evolution ensemble: {N_SHOTS} shots, g/gamma_t={G/PARAMS['gamma_t']:.2f}")
    print(f"  T_total={T_TOTAL}  (same as {N_CYCLES} cycles x dt={DT})")
    print(f"  T2_q={PARAMS['T2_q']:.0f}  T2_t={PARAMS['T2_t']:.0f}")
    print(f"  N_TH={N_TH}  P_ss={P_SS:.4f}\n")

    all_Pq_L    = []
    all_Pt_L    = []
    all_Pq_snap_L = []
    all_Pt_snap_L = []
    all_Pq_snap_S = []
    all_Pt_snap_S = []
    final_Pq_L  = []
    final_Pq_S  = []
    t_axis      = None

    for i, Pq0 in enumerate(Pq0_values):
        print(f"  Shot {i+1:2d}/{N_SHOTS}  P_q^(0)={Pq0:.3f}")
        res_L = run_one_shot(Pq0, model='lindblad')
        res_S = run_one_shot(Pq0, model='solomon')

        all_Pq_L.append(res_L['Pq'])
        all_Pt_L.append(res_L['Pt'])
        all_Pq_snap_L.append(res_L['Pq_snap'])
        all_Pt_snap_L.append(res_L['Pt_snap'])
        all_Pq_snap_S.append(res_S['Pq_snap'])
        all_Pt_snap_S.append(res_S['Pt_snap'])
        final_Pq_L.append(res_L['Pq'][-1])
        final_Pq_S.append(res_S['Pq'][-1])

        if t_axis is None:
            t_axis = res_L['t']

    all_Pq_L     = np.array(all_Pq_L)
    all_Pt_L     = np.array(all_Pt_L)
    all_Pq_snap_L = np.array(all_Pq_snap_L)
    all_Pt_snap_L = np.array(all_Pt_snap_L)
    all_Pq_snap_S = np.array(all_Pq_snap_S)
    all_Pt_snap_S = np.array(all_Pt_snap_S)
    final_Pq_L   = np.array(final_Pq_L)
    final_Pq_S   = np.array(final_Pq_S)
    cycles       = np.arange(1, N_CYCLES+1)
    sort_idx     = np.argsort(Pq0_values)

    np.savez(f'../data/sweeps/free_evolution_ensemble_{TAG}.npz',
             Pq0_values=Pq0_values, t=t_axis, cycles=cycles,
             all_Pq_L=all_Pq_L, all_Pt_L=all_Pt_L,
             all_Pq_snap_L=all_Pq_snap_L, all_Pt_snap_L=all_Pt_snap_L,
             all_Pq_snap_S=all_Pq_snap_S, all_Pt_snap_S=all_Pt_snap_S,
             final_Pq_L=final_Pq_L, final_Pq_S=final_Pq_S,
             g=G, g_over_gamma_t=G/PARAMS['gamma_t'],
             T_total=T_TOTAL, N_shots=N_SHOTS)
    print(f"\nData → data/sweeps/free_evolution_ensemble_{TAG}.npz")

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(0, 1)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # ── Figure 1: full time traces ────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    for i in sort_idx:
        c = cmap(norm(Pq0_values[i]))
        axes1[0].plot(t_axis, all_Pq_L[i], color=c, lw=0.8, alpha=0.5)

    Pq_mean = all_Pq_L.mean(axis=0)
    Pq_std  = all_Pq_L.std(axis=0)
    Pt_mean = all_Pt_L.mean(axis=0)

    axes1[0].fill_between(t_axis, Pq_mean-Pq_std, Pq_mean+Pq_std,
                          color='black', alpha=0.15, label='±1 std')
    axes1[0].plot(t_axis, Pq_mean, 'k-', lw=2.5, label='Mean $P_q$')
    axes1[0].plot(t_axis, Pt_mean, 'r-', lw=2, alpha=0.8, label='Mean $P_{TLS}$')
    axes1[0].axhline(P_SS, color='gray', ls=':', lw=1,
                    label=f'$P_{{ss}}$={P_SS:.3f}')
    # Mark snapshot times (same as reset times for comparison)
    for ts in SNAPSHOT_TIMES:
        axes1[0].axvline(ts, color='gray', ls=':', lw=0.6, alpha=0.4)

    plt.colorbar(sm, ax=axes1[0], label='Initial $P_q^{(0)}$')
    axes1[0].set_ylabel('Population', fontsize=12)
    axes1[0].set_title(
        rf'FREE EVOLUTION: {N_SHOTS} shots, $P_{{TLS}}^{{(0)}}=1$, '
        rf'$P_q^{{(0)}}\sim\mathcal{{U}}[0,1]$, NO reset  '
        rf'($g/\gamma_t$={G/PARAMS["gamma_t"]:.2f})',
        fontsize=12)
    axes1[0].legend(fontsize=9, loc='upper right')
    axes1[0].set_ylim(-0.02, 1.05)
    axes1[0].grid(True, alpha=0.3)

    axes1[1].plot(t_axis, Pq_std, 'k-', lw=2, label='Std of $P_q$')
    axes1[1].axhline(Pq_std[0], color='blue', ls='--', lw=1,
                    label=f'Initial std={Pq_std[0]:.3f}')
    for ts in SNAPSHOT_TIMES:
        axes1[1].axvline(ts, color='gray', ls=':', lw=0.6, alpha=0.4)
    axes1[1].set_xlabel('Time', fontsize=12)
    axes1[1].set_ylabel('Std($P_q$)', fontsize=12)
    axes1[1].set_title('Spread collapse — free evolution', fontsize=12)
    axes1[1].legend(fontsize=9); axes1[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/free_evolution_traces_{TAG}.png',
                dpi=150, bbox_inches='tight')

    # ── Figure 2: snapshots + memory test ─────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes2[0]
    for i in sort_idx:
        c = cmap(norm(Pq0_values[i]))
        ax.plot(cycles, all_Pq_snap_L[i], color=c, lw=1.2, alpha=0.6)
    Pq_snap_mean = all_Pq_snap_L.mean(axis=0)
    Pq_snap_std  = all_Pq_snap_L.std(axis=0)
    ax.fill_between(cycles, Pq_snap_mean-Pq_snap_std,
                    Pq_snap_mean+Pq_snap_std, color='black', alpha=0.15)
    ax.plot(cycles, Pq_snap_mean, 'k-', lw=2.5, label='Mean')
    ax.axhline(P_SS, color='gray', ls=':', lw=1)
    plt.colorbar(sm, ax=ax, label='$P_q^{(0)}$')
    ax.set_xlabel('Snapshot $n$ (= cycle number in reset protocol)', fontsize=11)
    ax.set_ylabel('$P_q$ at snapshot', fontsize=12)
    ax.set_title('Qubit population (free evolution snapshots)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes2[1]
    for i in sort_idx:
        c = cmap(norm(Pq0_values[i]))
        ax.plot(cycles, all_Pt_snap_L[i], color=c, lw=1.2, alpha=0.6)
    Pt_snap_mean = all_Pt_snap_L.mean(axis=0)
    ax.plot(cycles, Pt_snap_mean, 'k-', lw=2.5, label='Mean $P_{TLS}$')
    plt.colorbar(sm, ax=ax, label='$P_q^{(0)}$')
    ax.set_xlabel('Snapshot $n$', fontsize=12)
    ax.set_ylabel('$P_{TLS}$ at snapshot', fontsize=12)
    ax.set_title('TLS drain — free evolution\n(no backpressure from resets)', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes2[2]
    ax.scatter(Pq0_values, final_Pq_L, c=Pq0_values, cmap=cmap,
               s=80, zorder=5, label='Lindblad')
    ax.scatter(Pq0_values, final_Pq_S, c=Pq0_values, cmap=cmap,
               s=40, marker='s', alpha=0.6, label='Solomon')
    ax.axhline(P_SS, color='gray', ls=':', lw=1.5, label=f'$P_{{ss}}$={P_SS:.3f}')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.3, label='y=x (no change)')
    ax.set_xlabel('Initial $P_q^{(0)}$', fontsize=12)
    ax.set_ylabel(f'Final $P_q$ after $T={T_TOTAL}$', fontsize=12)
    ax.set_title('Memory erasure — free evolution', fontsize=12)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    fig2.suptitle(
        rf'Free evolution ensemble: $g/\gamma_t$={G/PARAMS["gamma_t"]:.2f}, '
        rf'$n_{{th}}$={N_TH}, $T_2^q$={PARAMS["T2_q"]:.0f}',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/free_evolution_percycle_{TAG}.png',
                dpi=150, bbox_inches='tight')

    print(f"Figures → figures/sweeps/free_evolution_*_{TAG}.png")

    print(f"\n=== MEMORY ERASURE (FREE EVOLUTION) ===")
    print(f"  Initial P_q std:       {Pq0_values.std():.4f}")
    print(f"  Final P_q std (L):     {final_Pq_L.std():.4f}")
    print(f"  Final P_q mean (L):    {final_Pq_L.mean():.4f}  (P_ss={P_SS:.4f})")
    corr_L = np.corrcoef(Pq0_values, final_Pq_L)[0,1]
    corr_S = np.corrcoef(Pq0_values, final_Pq_S)[0,1]
    print(f"  Correlation (L): {corr_L:.4f}")
    print(f"  Correlation (S): {corr_S:.4f}")
    print(f"  (compare with active reset script 17)")

    plt.show()


if __name__ == "__main__":
    run()
