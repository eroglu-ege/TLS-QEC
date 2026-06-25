"""
15_JC_threshold_sweep.py
=========================
Fine sweep around the JC Solomon breakdown threshold: g/kappa in [0.5, 1.5]
with realistic thermal population and pure dephasing.

Mirrors 08_fine_threshold_scan.py from the TLS model, but for the JC system.

REALISTIC PARAMETERS:
  gamma     = 0.01   (T1 = 100 in simulation units)
  kappa     = 0.01   (resonator lifetime = 100)
  n_th_q    = 0.05   (~50 mK for 5 GHz qubit)
  n_th_r    = 0.05   (same temperature for resonator)
  gamma_phi = 0.03   (T2 = T1/2 = 50, charge/flux noise dominated)

TRACKED METRICS:
  peak_Pe      : max qubit population during evolution
  peak_n       : max intracavity photon number
  peak_coh     : max qubit-resonator coherence |<a*sigma+>|
  tot_photons  : total photons emitted to ADC (efficiency)
  purcell_rate : analytic Purcell rate gamma_P = 4g^2/kappa
  n_ss         : steady-state photon number (long-time mean)

TWO-REGIME COMPARISON:
  T=0, no dephasing  (ideal)
  nth=0.05, T2=T1/2  (realistic)

Run from: TLS-QEC/analysis/
    python 15_JC_threshold_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from qubit_tls.jaynes_cummings import evolve, purcell_rate
from utils.physics import gamma_phi_from_T2

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
WR    = 1.0       # resonant: Delta=0
GAMMA = 0.01      # T1_q = 100
KAPPA = 0.01      # resonator decay = 100

# Two scenarios to compare
SCENARIOS = {
    'ideal':     {'n_th_q': 0.0,  'n_th_r': 0.0,  'gamma_phi': 0.0},
    'realistic': {'n_th_q': 0.05, 'n_th_r': 0.05,
                  'gamma_phi': gamma_phi_from_T2(0.01, 50.0)},  # T2=50=T1/2
}

# Dense linear sweep around threshold
N_G          = 50
G_OVER_KAPPA = np.linspace(0.5, 1.5, N_G)
G_VALUES     = G_OVER_KAPPA * KAPPA

T_END   = 800
N_STEPS = 600
N_MAX   = 15

TAG = "threshold_0.5-1.5"
# ──────────────────────────────────────────────────────────────────────────────


def run_scenario(scenario_name, params):
    print(f"\n  Running {scenario_name}: "
          f"n_th={params['n_th_q']:.3f}  "
          f"gamma_phi={params['gamma_phi']:.4f}")

    peak_Pe  = np.zeros(N_G)
    peak_n   = np.zeros(N_G)
    peak_coh = np.zeros(N_G)
    tot_ph   = np.zeros(N_G)
    n_ss     = np.zeros(N_G)
    purcells = np.zeros(N_G)

    for i, g in enumerate(tqdm(G_VALUES, desc=f"    {scenario_name}")):
        res = evolve(wq=WQ, wr=WR, g=g, gamma=GAMMA, kappa=KAPPA,
                     gamma_phi=params['gamma_phi'],
                     n_th_q=params['n_th_q'],
                     n_th_r=params['n_th_r'],
                     N_max=N_MAX, t_end=T_END, n_steps=N_STEPS,
                     qubit_init='e', n_photons_init=0)

        peak_Pe[i]  = res['P_e_q'].max()
        peak_n[i]   = res['n_photons'].max()
        peak_coh[i] = res['coherence_qr'].max()
        tot_ph[i]   = res['photons_emitted'][-1]
        n_ss[i]     = np.mean(res['n_photons'][int(0.85*N_STEPS):])
        purcells[i] = purcell_rate(g, KAPPA, WQ-WR)

    return {
        'peak_Pe': peak_Pe, 'peak_n': peak_n,
        'peak_coh': peak_coh, 'tot_photons': tot_ph,
        'n_ss': n_ss, 'purcell_rate': purcells,
    }


def run():
    os.makedirs('../data/JC', exist_ok=True)
    os.makedirs('../figures/JC', exist_ok=True)

    print(f"JC threshold sweep: g/kappa in [0.5, 1.5]  ({N_G} points)")
    print(f"gamma={GAMMA}  kappa={KAPPA}")
    print(f"T2_realistic = 50 = T1/2  ->  gamma_phi = "
          f"{SCENARIOS['realistic']['gamma_phi']:.4f}")

    results = {}
    for name, params in SCENARIOS.items():
        results[name] = run_scenario(name, params)

    # Save
    np.savez(f'../data/JC/JC_threshold_sweep_{TAG}.npz',
             g_over_kappa=G_OVER_KAPPA,
             g_values=G_VALUES,
             **{f"{sc}_{k}": v
                for sc, r in results.items()
                for k, v in r.items()},
             gamma=GAMMA, kappa=KAPPA,
             **{f"nth_{sc}": SCENARIOS[sc]['n_th_q']
                for sc in SCENARIOS},
             **{f"gphi_{sc}": SCENARIOS[sc]['gamma_phi']
                for sc in SCENARIOS})
    print(f"\nData → data/JC/JC_threshold_sweep_{TAG}.npz")

    # Print summary table
    print(f"\n{'g/κ':>6} | {'peak_n ideal':>13} {'peak_n real':>13} "
          f"{'peak_coh ideal':>15} {'peak_coh real':>15} "
          f"{'tot_ph ideal':>13} {'tot_ph real':>13}")
    print("-"*90)
    for i in range(0, N_G, 5):
        print(f"{G_OVER_KAPPA[i]:>6.2f} | "
              f"{results['ideal']['peak_n'][i]:>13.5f} "
              f"{results['realistic']['peak_n'][i]:>13.5f} "
              f"{results['ideal']['peak_coh'][i]:>15.5f} "
              f"{results['realistic']['peak_coh'][i]:>15.5f} "
              f"{results['ideal']['tot_photons'][i]:>13.5f} "
              f"{results['realistic']['tot_photons'][i]:>13.5f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.4)

    colors = {'ideal': 'C0', 'realistic': 'C1'}
    styles = {'ideal': '-', 'realistic': '--'}
    labels = {'ideal': 'Ideal ($T=0$, no dephasing)',
              'realistic': 'Realistic ($n_{{th}}=0.05$, $T_2=T_1/2$)'}

    ax1 = fig.add_subplot(gs[0,0])   # peak photon number
    ax2 = fig.add_subplot(gs[0,1])   # peak coherence
    ax3 = fig.add_subplot(gs[0,2])   # total photons (efficiency)
    ax4 = fig.add_subplot(gs[1,0])   # Purcell rate
    ax5 = fig.add_subplot(gs[1,1])   # coherence ratio (real/ideal)
    ax6 = fig.add_subplot(gs[1,2])   # efficiency ratio

    for sc in ['ideal', 'realistic']:
        r   = results[sc]
        c   = colors[sc]
        ls  = styles[sc]
        lbl = labels[sc]

        ax1.plot(G_OVER_KAPPA, r['peak_n'],   color=c, ls=ls, lw=2, label=lbl)
        ax2.plot(G_OVER_KAPPA, r['peak_coh'], color=c, ls=ls, lw=2, label=lbl)
        ax3.plot(G_OVER_KAPPA, r['tot_photons'], color=c, ls=ls, lw=2, label=lbl)

    # Purcell rate (same for both)
    ax4.semilogy(G_OVER_KAPPA, results['ideal']['purcell_rate'],
                 'k-', lw=2, label=r'$\gamma_P = 4g^2/\kappa$')
    ax4.axhline(GAMMA, color='blue', ls='--', lw=1.5,
               label=f'$\\gamma={GAMMA}$')
    ax4.axhline(GAMMA*5, color='orange', ls=':', lw=1.5,
               label='$5\\gamma$')

    # Coherence suppression by dephasing
    coh_ideal = results['ideal']['peak_coh']
    coh_real  = results['realistic']['peak_coh']
    coh_ratio = np.where(coh_ideal>0, coh_real/coh_ideal, 0)
    ax5.plot(G_OVER_KAPPA, coh_ratio, 'o-', lw=2, color='purple', ms=4)
    ax5.axhline(1.0, color='gray', ls=':', lw=1)

    # Photon efficiency change
    tot_ideal = results['ideal']['tot_photons']
    tot_real  = results['realistic']['tot_photons']
    ax6.plot(G_OVER_KAPPA, tot_real - tot_ideal, 'o-', lw=2,
             color='darkgreen', ms=4)
    ax6.axhline(0, color='gray', ls=':', lw=1)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\kappa$')
        ax.axvline(0.5, color='orange', ls=':', lw=1.2,
                  label='Solomon limit')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax.set_xlabel('$g/\\kappa$')

    for ax in [ax5, ax6]:
        ax.axvline(1.0, color='red', ls='--', lw=1.5)
        ax.axvline(0.5, color='orange', ls=':', lw=1.2)
        ax.grid(True, alpha=0.3); ax.set_xlabel('$g/\\kappa$')

    ax1.set_ylabel('Peak $\\langle a^\\dagger a\\rangle$')
    ax1.set_title('Peak intracavity photons', fontsize=11)

    ax2.set_ylabel(r'Peak $|\langle a\sigma_+\rangle|$')
    ax2.set_title('Qubit-resonator coherence onset', fontsize=11)

    ax3.set_ylabel('Total photons emitted')
    ax3.set_title('ADC efficiency (fraction of $E_{qubit}$)', fontsize=11)

    ax4.set_ylabel('$\\gamma_P$')
    ax4.set_title('Purcell rate vs $g/\\kappa$', fontsize=11)

    ax5.set_ylabel('Coherence ratio (real/ideal)')
    ax5.set_title('Dephasing suppression of coherence', fontsize=11)

    ax6.set_ylabel('$\\Delta$(total photons) = real $-$ ideal')
    ax6.set_title('Thermal photon contribution to ADC', fontsize=11)

    fig.suptitle(
        rf'JC threshold sweep $g/\kappa \in [0.5, 1.5]$: '
        rf'ideal vs realistic ($n_{{th}}=0.05$, $T_2=T_1/2$)',
        fontsize=14)
    plt.savefig(f'../figures/JC/JC_threshold_sweep_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure → figures/JC/JC_threshold_sweep_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
