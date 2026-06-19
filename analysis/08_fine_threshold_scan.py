"""
08_fine_threshold_scan.py
==========================
High-resolution scan zoomed tightly around the threshold g = gamma_t.

PHYSICS: g = gamma_t marks where two competing timescales cross:
  - Rabi period       T_rabi = pi/g           (coherent exchange time)
  - TLS decoherence   1/gamma_t               (how long coherence survives)

  g << gamma_t: TLS dephases before exchange completes -> looks incoherent
                -> Solomon (rate equations) is exact
  g >> gamma_t: many coherent Rabi cycles survive before dephasing
                -> genuine quantum oscillation -> Solomon fails
  g ~  gamma_t: crossover, both timescales comparable

This script uses DENSE sampling concentrated near g/gamma_t = 1,
with extra resolution in [0.5, 2.0] specifically.

Run from: TLS-QEC/analysis/
    python 08_fine_threshold_scan.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import qutip as qt

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.metrics import ISE, coherence_metrics

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

FIXED = dict(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.005,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)

# Two-tier sampling: coarse background [0.05, 10] + dense zoom [0.5, 2.0]
G_OVER_GAMMA_T_COARSE = np.geomspace(0.05, 10, 30)
G_OVER_GAMMA_T_ZOOM   = np.linspace(0.5, 2.0, 60)
G_OVER_GAMMA_T = np.unique(np.concatenate(
    [G_OVER_GAMMA_T_COARSE, G_OVER_GAMMA_T_ZOOM]))
G_OVER_GAMMA_T.sort()

G_VALUES = G_OVER_GAMMA_T * FIXED['gamma_t']
N_G      = len(G_VALUES)

T_END   = 4000
N_STEPS = 1200
# ──────────────────────────────────────────────────────────────────────────────

TAG = f"nth{N_TH:.4f}"


def run():
    print(f"Fine threshold scan: {N_G} points")
    print(f"  background: {len(G_OVER_GAMMA_T_COARSE)} pts, log-spaced [0.05, 10]")
    print(f"  zoom:       {len(G_OVER_GAMMA_T_ZOOM)} pts, linear [0.5, 2.0]")
    print(f"  N_TH={N_TH}\n")

    T_rabi_min = np.pi / G_VALUES[-1]
    dt = T_END / N_STEPS
    print(f"Sampling check: smallest T_rabi={T_rabi_min:.2f}, dt={dt:.3f}, "
          f"samples/period={T_rabi_min/dt:.1f}  (want >10)\n")

    L_max   = np.zeros(N_G)
    S_max   = np.zeros(N_G)
    osc_amp = np.zeros(N_G)
    ise_arr = np.zeros(N_G)
    coh_max = np.zeros(N_G)

    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    rho0 = qt.ket2dm(psi0)

    for i, g in enumerate(tqdm(G_VALUES, desc="Fine scan")):
        res_L = lindblad_evolve(**FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
                                rho0=rho0)
        res_S = solomon_evolve(**FIXED, g=g, t_end=T_END, n_steps=N_STEPS,
                               P_q0=0.0, P_t0=1.0)

        L_max[i]   = res_L['P_e_q'].max()
        S_max[i]   = res_S['P_e_q'].max()
        osc_amp[i] = res_L['P_e_q'].max() - res_L['P_e_q'].min()

        P_S_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
        ise_arr[i] = ISE(res_L['t'], res_L['P_e_q'], P_S_interp)
        coh_max[i] = coherence_metrics(res_L['t'], res_L['coherence'])['max']

    np.savez(f'../data/sweeps/fine_threshold_{TAG}.npz',
             g_values=G_VALUES, g_over_gamma_t=G_OVER_GAMMA_T,
             L_max=L_max, S_max=S_max,
             osc_amp=osc_amp, ISE=ise_arr, coherence_max=coh_max)
    print(f"\nData saved → data/sweeps/fine_threshold_{TAG}.npz")

    # ── Figure ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(G_OVER_GAMMA_T, L_max, 'o-', lw=1.5, ms=3, label='Lindblad max $P_q$')
    ax.plot(G_OVER_GAMMA_T, S_max, 's-', lw=1.5, ms=3, label='Solomon max $P_q$')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\gamma_t$')
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Peak qubit population')
    ax.set_title('Peak population vs coupling'); ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(G_OVER_GAMMA_T, ise_arr, 'o-', lw=1.5, ms=3, color='black')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('ISE')
    ax.set_title('Integrated Squared Error')

    ax = axes[1, 0]
    ax.plot(G_OVER_GAMMA_T, coh_max, 'o-', lw=1.5, ms=3, color='purple')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Peak coherence')
    ax.set_title('Coherence buildup')

    ax = axes[1, 1]
    ax.plot(G_OVER_GAMMA_T, osc_amp, 'o-', lw=1.5, ms=3, color='darkgreen')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Oscillation amplitude')
    ax.set_title('Rabi oscillation onset (Lindblad)')

    fig.suptitle(
        rf'Fine threshold scan around $g=\gamma_t$  '
        rf'(zoom: 60 pts in [0.5,2.0]), $n_{{th}}$={N_TH}',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/fine_threshold_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/fine_threshold_{TAG}.png")

    # ── Zoomed-in second figure: just the [0.5, 2.0] region ────────────────────
    zoom_mask = (G_OVER_GAMMA_T >= 0.5) & (G_OVER_GAMMA_T <= 2.0)
    fig2, ax = plt.subplots(figsize=(8, 5))
    ax.plot(G_OVER_GAMMA_T[zoom_mask], L_max[zoom_mask], 'o-', lw=2,
           label='Lindblad max $P_q$')
    ax.plot(G_OVER_GAMMA_T[zoom_mask], S_max[zoom_mask], 's-', lw=2,
           label='Solomon max $P_q$')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\gamma_t$')
    ax.set_xlabel('$g/\\gamma_t$', fontsize=12)
    ax.set_ylabel('Peak qubit population', fontsize=12)
    ax.set_title('Zoom: threshold region [0.5, 2.0]', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/fine_threshold_zoom_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Zoom figure saved → figures/sweeps/fine_threshold_zoom_{TAG}.png")

    diff_pct = (L_max - S_max) / S_max * 100
    above5 = G_OVER_GAMMA_T[diff_pct > 5]
    if len(above5) > 0:
        print(f"\n5% divergence threshold: g/gamma_t = {above5[0]:.4f}")
    sign_change = np.where(np.diff(np.sign(L_max - S_max)))[0]
    if len(sign_change) > 0:
        print(f"L_max = S_max crossing at g/gamma_t ≈ "
              f"{G_OVER_GAMMA_T[sign_change[0]]:.4f}")

    plt.show()


if __name__ == "__main__":
    run()
