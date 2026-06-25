"""
12_JC_sweep_g_kappa.py
======================
Sweep g/kappa across the Solomon breakdown boundary [0.1, 20].

Tracks:
    n_ss        : steady-state photon number
    tot_photons : cumulative photons emitted
    peak_Pe     : max qubit population
    peak_coh    : max qubit-resonator coherence

This directly mirrors sweep 03 (g/gamma_t) from the TLS model,
with kappa playing the role of gamma_t.

Run from: TLS-QEC/analysis/
    python 12_JC_sweep_g_kappa.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from qubit_tls.jaynes_cummings import evolve, purcell_rate, n_ss_analytic
from utils.physics import n_thermal

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
WR    = 1.0
GAMMA = 0.01
KAPPA = 0.01
N_TH_Q   = 0.0
N_TH_R   = 0.0
GAMMA_PHI = 0.0

N_G     = 40
G_OVER_KAPPA = np.logspace(-1, 1.3, N_G)   # 0.1 to 20
G_VALUES     = G_OVER_KAPPA * KAPPA

T_END   = 600
N_STEPS = 600
N_MAX   = 20

TAG = f"nth{N_TH_Q:.3f}_gphi{GAMMA_PHI:.3f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    os.makedirs('../data/JC', exist_ok=True)
    os.makedirs('../figures/JC', exist_ok=True)

    n_ss_arr  = np.zeros(N_G)
    tot_ph    = np.zeros(N_G)
    peak_Pe   = np.zeros(N_G)
    peak_coh  = np.zeros(N_G)
    purcell   = np.zeros(N_G)

    for i, g in enumerate(tqdm(G_VALUES, desc="g/kappa sweep")):
        res = evolve(wq=WQ, wr=WR, g=g, gamma=GAMMA, kappa=KAPPA,
                     gamma_phi=GAMMA_PHI, n_th_q=N_TH_Q, n_th_r=N_TH_R,
                     N_max=N_MAX, t_end=T_END, n_steps=N_STEPS,
                     qubit_init='e', n_photons_init=0)

        n_ss_arr[i] = np.mean(res['n_photons'][int(0.8*N_STEPS):])
        tot_ph[i]   = res['photons_emitted'][-1]
        peak_Pe[i]  = res['P_e_q'].max()
        peak_coh[i] = res['coherence_qr'].max()
        purcell[i]  = purcell_rate(g, KAPPA, WQ-WR)

    np.savez(f'../data/JC/JC_sweep_g_kappa_{TAG}.npz',
             g_over_kappa=G_OVER_KAPPA, g_values=G_VALUES,
             n_ss=n_ss_arr, tot_photons=tot_ph,
             peak_Pe=peak_Pe, peak_coh=peak_coh,
             purcell_rate=purcell,
             kappa=KAPPA, gamma=GAMMA,
             n_th_q=N_TH_Q, gamma_phi=GAMMA_PHI)
    print(f"\nData → data/JC/JC_sweep_g_kappa_{TAG}.npz")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0,0]
    ax.semilogx(G_OVER_KAPPA, peak_Pe, 'o-', lw=2, color='C0')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\kappa$')
    ax.set_xlabel('$g/\\kappa$'); ax.set_ylabel('Peak $P_e^q$')
    ax.set_title('Peak qubit population'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[0,1]
    ax.loglog(G_OVER_KAPPA, n_ss_arr, 'o-', lw=2, color='C1',
              label='Numeric $\\langle n\\rangle_{ss}$')
    ax.loglog(G_OVER_KAPPA, purcell/KAPPA, 's--', lw=2, color='gray',
              label='$\\gamma_P/\\kappa$ (Purcell)')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xlabel('$g/\\kappa$'); ax.set_ylabel('$\\langle n\\rangle_{ss}$')
    ax.set_title('Steady-state photon number'); ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

    ax = axes[1,0]
    ax.semilogx(G_OVER_KAPPA, tot_ph, 'o-', lw=2, color='C2')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.axhline(1.0, color='gray', ls=':', lw=1, label='1 photon (conservation)')
    ax.set_xlabel('$g/\\kappa$'); ax.set_ylabel('Total photons emitted')
    ax.set_title('Cumulative emitted photons')
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1,1]
    ax.semilogx(G_OVER_KAPPA, peak_coh, 'o-', lw=2, color='purple')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\kappa$')
    ax.set_xlabel('$g/\\kappa$')
    ax.set_ylabel(r'Peak $|\langle a\sigma_+\rangle|$')
    ax.set_title('Q-R coherence onset (mirrors TLS coherence)')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(
        rf'JC sweep $g/\kappa$: Solomon breakdown at $g/\kappa\sim1$  '
        rf'($n_{{th}}$={N_TH_Q}, $\gamma_\phi$={GAMMA_PHI})',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../figures/JC/JC_sweep_g_kappa_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure → figures/JC/JC_sweep_g_kappa_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
