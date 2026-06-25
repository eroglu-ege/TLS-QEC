"""
11_JC_regimes.py
================
Four regimes of Jaynes-Cummings at the Solomon breakdown boundary.

g/kappa = 0.5, 1.0, 5.0, 10.0
(Solomon breaks down at g/kappa ~ 0.5, same physics as g/gamma_t ~ 0.5)

Initial state: qubit |e>, resonator |0>

Plots: P_e_q, n_photons, photon_flux, cumulative photons, coherence_qr

Run from: TLS-QEC/analysis/
    python 11_JC_regimes.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from qubit_tls.jaynes_cummings import evolve, purcell_rate
from utils.physics import n_thermal

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
WR    = 1.0
GAMMA = 0.01
KAPPA = 0.01
N_TH_Q   = 0.0       # n_thermal(5.0, 0.020) for realistic
N_TH_R   = 0.0
GAMMA_PHI = 0.0

T_END   = 800
N_STEPS = 800
N_MAX   = 20

# g values at Solomon breakdown boundary and beyond
SCENARIOS = [
    (0.005, 'Bad cavity   $g/\\kappa=0.5$',  'C0'),
    (0.01,  'Threshold    $g/\\kappa=1.0$',  'C1'),
    (0.05,  'Strong coup  $g/\\kappa=5.0$',  'C2'),
    (0.10,  'Deep Rabi    $g/\\kappa=10.0$', 'C3'),
]

TAG = f"nth{N_TH_Q:.3f}_gphi{GAMMA_PHI:.3f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    os.makedirs('../data/JC', exist_ok=True)
    os.makedirs('../figures/JC', exist_ok=True)

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.4)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(3)]

    print(f"{'g':>8} {'g/kappa':>10} {'Purcell':>12} "
          f"{'max_n':>8} {'tot_ph':>10} {'peak_P_e':>10}")

    saved = {}
    for g, label, color in SCENARIOS:
        res = evolve(wq=WQ, wr=WR, g=g, gamma=GAMMA, kappa=KAPPA,
                     gamma_phi=GAMMA_PHI, n_th_q=N_TH_Q, n_th_r=N_TH_R,
                     N_max=N_MAX, t_end=T_END, n_steps=N_STEPS,
                     qubit_init='e', n_photons_init=0)

        gk  = g / KAPPA
        gP  = purcell_rate(g, KAPPA, delta=WQ-WR)
        print(f"{g:>8.4f} {gk:>10.2f} {gP:>12.6f} "
              f"{res['n_photons'].max():>8.4f} "
              f"{res['photons_emitted'][-1]:>10.4f} "
              f"{res['P_e_q'].max():>10.4f}")

        saved[f'g{g:.4f}'] = {
            't': res['t'], 'P_e_q': res['P_e_q'],
            'n_photons': res['n_photons'],
            'photon_flux': res['photon_flux'],
            'photons_emitted': res['photons_emitted'],
        }

        axes[0].plot(res['t'], res['P_e_q'],          color=color, lw=2, label=label)
        axes[1].plot(res['t'], res['n_photons'],       color=color, lw=2, label=label)
        axes[2].plot(res['t'], res['photon_flux'],     color=color, lw=2, label=label)
        axes[3].plot(res['t'], res['photons_emitted'], color=color, lw=2, label=label)
        axes[4].semilogy(res['t'],
                         np.where(res['coherence_qr']>0,
                                  res['coherence_qr'],1e-16),
                         color=color, lw=2, label=label)

    # Save
    np.savez(f'../data/JC/JC_regimes_{TAG}.npz',
             **{k: v['n_photons'] for k,v in saved.items()},
             t=res['t'], g_values=[s[0] for s in SCENARIOS])
    print(f"\nData → data/JC/JC_regimes_{TAG}.npz")

    # Format axes
    titles  = ['Qubit $P_e(t)$',
               r'Intracavity $\langle a^\dagger a\rangle$',
               r'Photon flux $\kappa\langle n\rangle$',
               'Cumulative emitted photons',
               r'Q-R coherence $|\langle a\sigma_+\rangle|$']
    ylabels = ['$P_e$','$\\langle n\\rangle$','Flux',
               'Total photons','Coherence']

    for ax,title,ylabel in zip(axes[:5], titles, ylabels):
        ax.set_xlabel('Time'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Parameter info
    axes[5].axis('off')
    info = ("PURCELL RATES\n\n"
            + "\n".join([
                f"g/κ={g/KAPPA:.1f}: γ_P={purcell_rate(g,KAPPA,WQ-WR):.5f}"
                for g,_,_ in SCENARIOS])
            + f"\n\nn_th_q={N_TH_Q}\nn_th_r={N_TH_R}\nγ_φ={GAMMA_PHI}")
    axes[5].text(0.05, 0.95, info, transform=axes[5].transAxes,
                fontsize=10, va='top', fontfamily='monospace')

    fig.suptitle(
        rf'Jaynes-Cummings regimes at Solomon breakdown boundary, '
        rf'$n_{{th}}$={N_TH_Q}, $\gamma_\phi$={GAMMA_PHI}',
        fontsize=13)
    plt.savefig(f'../figures/JC/JC_regimes_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure → figures/JC/JC_regimes_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
