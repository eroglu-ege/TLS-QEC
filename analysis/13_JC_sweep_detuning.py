"""
13_JC_sweep_detuning.py
=======================
Sweep detuning Delta=wq-wr and plot n_photons(t) on log-t axis.

This is the "swap spectroscopy" figure: as Delta increases the qubit
moves off-resonance with the resonator, photon transfer slows down
and the resonator stays empty longer.

Physical picture:
    Delta=0   (resonant):  fast exchange, resonator fills quickly
    Delta>0   (dispersive): slow exchange, Lorentzian suppression by Delta^2
    Very large Delta: qubit and resonator decouple, no photons at all

Mirrors sweep 04 (g/Delta) from the TLS model but now shows the
full time-domain picture on log-t.

Run from: TLS-QEC/analysis/
    python 13_JC_sweep_detuning.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors

from qubit_tls.jaynes_cummings import evolve, purcell_rate
from utils.physics import n_thermal

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
G     = 0.05
GAMMA = 0.01
KAPPA = 0.01
N_TH_Q   = 0.0
N_TH_R   = 0.0
GAMMA_PHI = 0.0

# Detuning sweep: Delta = wq - wr
# Range chosen so g/Delta goes from dispersive to resonant
DELTA_VALUES = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
WR_VALUES    = WQ - DELTA_VALUES   # resonator frequency for each detuning

T_END   = 1200
N_STEPS = 800
N_MAX   = 20

TAG = f"g{G:.3f}_nth{N_TH_Q:.3f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    os.makedirs('../data/JC', exist_ok=True)
    os.makedirs('../figures/JC', exist_ok=True)

    fig = plt.figure(figsize=(16, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0])   # n_photons vs log t (colormap lines)
    ax2 = fig.add_subplot(gs[1])   # P_e_q vs log t
    ax3 = fig.add_subplot(gs[2])   # peak n_photons vs Delta (Lorentzian)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DELTA_VALUES)))

    peak_n   = []
    tot_n    = []
    purcells = []
    all_data = {}

    for wr, delta, color in zip(WR_VALUES, DELTA_VALUES, colors):
        label = f'$\\Delta$={delta:.3f} ($g/\\Delta$={G/delta:.1f})' \
                if delta > 0 else '$\\Delta=0$ (resonant)'

        res = evolve(wq=WQ, wr=wr, g=G, gamma=GAMMA, kappa=KAPPA,
                     gamma_phi=GAMMA_PHI, n_th_q=N_TH_Q, n_th_r=N_TH_R,
                     N_max=N_MAX, t_end=T_END, n_steps=N_STEPS,
                     qubit_init='e', n_photons_init=0)

        t = res['t']
        n = res['n_photons']
        P = res['P_e_q']

        peak_n.append(n.max())
        tot_n.append(res['photons_emitted'][-1])
        purcells.append(purcell_rate(G, KAPPA, delta))

        all_data[f'delta{delta:.4f}'] = {
            'n_photons': n, 'P_e_q': P, 't': t}

        ax1.semilogx(t[1:], n[1:], color=color, lw=2, label=label)
        ax2.semilogx(t[1:], P[1:], color=color, lw=2, label=label)

    # Lorentzian fit to peak_n vs Delta
    delta_range  = np.linspace(0, DELTA_VALUES.max()*1.2, 300)
    # Purcell/Lorentzian: n_peak ~ g^2/(Delta^2 + kappa^2/4)
    n_lorentz = peak_n[0] * (KAPPA/2)**2 / (delta_range**2 + (KAPPA/2)**2)
    ax3.plot(delta_range, n_lorentz, 'k--', lw=2, label='Lorentzian')
    ax3.plot(DELTA_VALUES, peak_n, 'o', ms=10, color='darkred',
             label='Simulation peak $\\langle n\\rangle$')
    ax3.plot(DELTA_VALUES, purcells, 's', ms=8, color='blue',
             label='Purcell rate $\\gamma_P$', alpha=0.7)
    ax3.set_xlabel('Detuning $\\Delta$', fontsize=12)
    ax3.set_ylabel('Peak value', fontsize=12)
    ax3.set_title('Lorentzian resonance in detuning', fontsize=12)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    ax1.set_xlabel('Time (log scale)', fontsize=12)
    ax1.set_ylabel('$\\langle a^\\dagger a\\rangle$', fontsize=12)
    ax1.set_title('Photon buildup vs detuning (log $t$)', fontsize=12)
    ax1.legend(fontsize=7); ax1.grid(True, which='both', alpha=0.3)

    ax2.set_xlabel('Time (log scale)', fontsize=12)
    ax2.set_ylabel('Qubit $P_e$', fontsize=12)
    ax2.set_title('Qubit decay vs detuning (log $t$)', fontsize=12)
    ax2.legend(fontsize=7); ax2.grid(True, which='both', alpha=0.3)

    # Save
    np.savez(f'../data/JC/JC_sweep_detuning_{TAG}.npz',
             delta_values=DELTA_VALUES,
             peak_n=np.array(peak_n),
             tot_n=np.array(tot_n),
             purcell=np.array(purcells),
             t=res['t'],
             g=G, kappa=KAPPA, gamma=GAMMA)
    print(f"\nData → data/JC/JC_sweep_detuning_{TAG}.npz")

    fig.suptitle(
        rf'JC detuning sweep: $P_e(t)$ and $\langle n\rangle(t)$ on log $t$  '
        rf'($g={G}$, $\kappa={KAPPA}$, $\gamma={GAMMA}$, $n_{{th}}$={N_TH_Q})',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../figures/JC/JC_sweep_detuning_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure → figures/JC/JC_sweep_detuning_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
