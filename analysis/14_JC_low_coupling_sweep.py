"""
14_JC_low_coupling_sweep.py
============================
Low coupling sweep: g/kappa from 0.1 to 1.1 with realistic
thermal population and pure dephasing.

This is the physically interesting region because:
  - g/kappa = 0.1: deep Purcell regime, Solomon exact
  - g/kappa = 0.5: Solomon starts breaking down (matches TLS result)
  - g/kappa = 1.1: just past threshold, coherent effects emerging

PHYSICAL PARAMETERS (realistic 5 GHz transmon at ~50 mK):
  gamma     = 0.01   → T1 = 100  (in simulation units)
  kappa     = 0.01   → resonator lifetime = 100
  n_th_q    = 0.05   → ~50 mK thermal population
  gamma_phi = 0.03   → T2 = 2/(0.01+0.03) = 50 = T1/2 (realistic)
  T2_analytic = 50 (set by charge/flux noise)

Purcell rate at resonance: gamma_P = 4*g^2/kappa
  At g/kappa=0.1: gamma_P = 4*0.001^2/0.01 = 0.0004  (4% of gamma)
  At g/kappa=0.5: gamma_P = 4*0.005^2/0.01 = 0.01    (= gamma)
  At g/kappa=1.1: gamma_P = 4*0.011^2/0.01 = 0.0484  (5x gamma)

Tracked per g value:
  P_e_q(t), n_photons(t), photon_flux(t), coherence_qr(t)

Also plots colormap: n_photons(t) vs g/kappa (like coupling_tevo for TLS)

Run from: TLS-QEC/analysis/
    python 14_JC_low_coupling_sweep.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from qubit_tls.jaynes_cummings import evolve, purcell_rate, n_ss_analytic
from utils.physics import n_thermal, gamma_phi_from_T2

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
WR    = 1.0     # resonant: Delta=0
GAMMA = 0.01    # T1 = 100
KAPPA = 0.01    # resonator lifetime = 100

# Realistic thermal population — 5 GHz qubit at ~50 mK
N_TH_Q = 0.05   # n_thermal(5.0, 0.050) ~ 0.05
N_TH_R = 0.05   # same temperature for resonator

# Realistic pure dephasing — T2 = T1/2 (charge/flux noise dominated)
T1_Q      = 1.0 / GAMMA       # = 100
T2_Q      = T1_Q / 2.0        # = 50
GAMMA_PHI = gamma_phi_from_T2(GAMMA, T2_Q)   # = 2/50 - 0.01 = 0.03

# Low coupling sweep: g/kappa from 0.1 to 1.1
N_G          = 40
G_OVER_KAPPA = np.linspace(0.1, 1.1, N_G)
G_VALUES     = G_OVER_KAPPA * KAPPA

T_END   = 800
N_STEPS = 600
N_MAX   = 15

TAG = f"low_gk0.1-1.1_nth{N_TH_Q:.3f}_T2q{T2_Q:.0f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    os.makedirs('../data/JC', exist_ok=True)
    os.makedirs('../figures/JC', exist_ok=True)

    print(f"Physical parameters:")
    print(f"  gamma={GAMMA}  T1={T1_Q:.0f}")
    print(f"  kappa={KAPPA}")
    print(f"  n_th_q={N_TH_Q}  n_th_r={N_TH_R}")
    print(f"  gamma_phi={GAMMA_PHI:.4f}  T2={T2_Q:.0f} = T1/2")
    print(f"  P_ss_qubit = {N_TH_Q/(2*N_TH_Q+1):.4f}  (thermal steady state)")
    print(f"\nPurcell rates at resonance:")
    for gk in [0.1, 0.5, 1.0, 1.1]:
        g = gk * KAPPA
        gP = purcell_rate(g, KAPPA, 0.0)
        print(f"  g/kappa={gk:.1f}: gamma_P={gP:.5f}  "
              f"(gamma_P/gamma={gP/GAMMA:.3f})")
    print()

    # Storage
    P_e_grid  = np.zeros((N_G, N_STEPS))
    n_ph_grid = np.zeros((N_G, N_STEPS))
    flux_grid = np.zeros((N_G, N_STEPS))
    coh_grid  = np.zeros((N_G, N_STEPS))
    peak_Pe   = np.zeros(N_G)
    peak_n    = np.zeros(N_G)
    peak_coh  = np.zeros(N_G)
    tot_ph    = np.zeros(N_G)
    purcells  = np.zeros(N_G)
    t_array   = None

    for i, g in enumerate(tqdm(G_VALUES, desc="g/kappa sweep")):
        res = evolve(wq=WQ, wr=WR, g=g, gamma=GAMMA, kappa=KAPPA,
                     gamma_phi=GAMMA_PHI,
                     n_th_q=N_TH_Q, n_th_r=N_TH_R,
                     N_max=N_MAX, t_end=T_END, n_steps=N_STEPS,
                     qubit_init='e', n_photons_init=0)

        P_e_grid[i]  = res['P_e_q']
        n_ph_grid[i] = res['n_photons']
        flux_grid[i] = res['photon_flux']
        coh_grid[i]  = res['coherence_qr']
        peak_Pe[i]   = res['P_e_q'].max()
        peak_n[i]    = res['n_photons'].max()
        peak_coh[i]  = res['coherence_qr'].max()
        tot_ph[i]    = res['photons_emitted'][-1]
        purcells[i]  = purcell_rate(g, KAPPA, WQ-WR)
        if t_array is None:
            t_array = res['t']

    # Save
    np.savez(f'../data/JC/JC_low_coupling_{TAG}.npz',
             g_over_kappa=G_OVER_KAPPA, g_values=G_VALUES, t=t_array,
             P_e_grid=P_e_grid, n_ph_grid=n_ph_grid,
             flux_grid=flux_grid, coh_grid=coh_grid,
             peak_Pe=peak_Pe, peak_n=peak_n, peak_coh=peak_coh,
             tot_photons=tot_ph, purcell_rate=purcells,
             gamma=GAMMA, kappa=KAPPA, gamma_phi=GAMMA_PHI,
             n_th_q=N_TH_Q, n_th_r=N_TH_R, T2_q=T2_Q)
    print(f"\nData → data/JC/JC_low_coupling_{TAG}.npz")

    # ── Figure 1: colormap n_photons(t) vs g/kappa ───────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 5))

    for ax, grid, title in zip(
        axes1,
        [n_ph_grid, flux_grid],
        [r'Intracavity $\langle a^\dagger a\rangle$',
         r'Photon flux $\kappa\langle a^\dagger a\rangle$']
    ):
        im = ax.pcolormesh(t_array, G_OVER_KAPPA, grid,
                           cmap='RdYlBu_r', shading='gouraud')
        fig1.colorbar(im, ax=ax)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('$g/\\kappa$', fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.axhline(1.0, color='black', lw=2, ls='-',
                  label='$g=\\kappa$ (threshold)')
        ax.axhline(0.5, color='white', lw=1.5, ls='--',
                  label='$g/\\kappa=0.5$ (Solomon)')
        ax.legend(fontsize=8, facecolor='white', framealpha=0.8)

    fig1.suptitle(
        rf'JC low coupling colormap: $g/\kappa \in [0.1, 1.1]$  '
        rf'($n_{{th}}$={N_TH_Q}, $T_2={T2_Q:.0f}=T_1/2$, '
        rf'$\gamma_\phi$={GAMMA_PHI:.3f})',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(f'../figures/JC/JC_low_coupling_colormap_{TAG}.png',
                dpi=150, bbox_inches='tight')

    # ── Figure 2: line cuts + scalar metrics ─────────────────────────────────
    fig2 = plt.figure(figsize=(16, 9))
    gs   = gridspec.GridSpec(2, 3, wspace=0.35, hspace=0.4)

    # Line cuts at 4 g values
    cut_gk = [0.1, 0.5, 0.86, 1.1]
    colors  = ['C0','C1','C2','C3']

    ax_Pe  = fig2.add_subplot(gs[0,0])
    ax_n   = fig2.add_subplot(gs[0,1])
    ax_coh = fig2.add_subplot(gs[0,2])

    for gk, c in zip(cut_gk, colors):
        idx = np.argmin(np.abs(G_OVER_KAPPA - gk))
        actual_gk = G_OVER_KAPPA[idx]
        lbl = f'$g/\\kappa$={actual_gk:.2f}'
        ax_Pe.plot(t_array,  P_e_grid[idx],  color=c, lw=2, label=lbl)
        ax_n.plot(t_array,   n_ph_grid[idx], color=c, lw=2, label=lbl)
        ax_coh.semilogy(t_array,
                        np.where(coh_grid[idx]>0, coh_grid[idx], 1e-16),
                        color=c, lw=2, label=lbl)

    P_ss = N_TH_Q / (2*N_TH_Q + 1)
    ax_Pe.axhline(P_ss, color='gray', ls=':', lw=1,
                 label=f'$P_{{ss}}$={P_ss:.3f}')

    for ax, title, ylabel in zip(
        [ax_Pe, ax_n, ax_coh],
        ['Qubit $P_e(t)$',
         r'$\langle a^\dagger a\rangle(t)$',
         r'Q-R coherence $|\langle a\sigma_+\rangle|$'],
        ['$P_e$', '$\\langle n\\rangle$', 'Coherence']
    ):
        ax.set_xlabel('Time'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Scalar metrics vs g/kappa
    ax_pkPe  = fig2.add_subplot(gs[1,0])
    ax_pkn   = fig2.add_subplot(gs[1,1])
    ax_purc  = fig2.add_subplot(gs[1,2])

    ax_pkPe.plot(G_OVER_KAPPA, peak_Pe, 'o-', lw=2, color='C0')
    ax_pkPe.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\kappa$')
    ax_pkPe.axvline(0.5, color='orange', ls='--', lw=1.5,
                   label='Solomon limit')
    ax_pkPe.set_xlabel('$g/\\kappa$'); ax_pkPe.set_ylabel('Peak $P_e^q$')
    ax_pkPe.set_title('Peak qubit population')
    ax_pkPe.legend(fontsize=8); ax_pkPe.grid(True, alpha=0.3)

    ax_pkn.plot(G_OVER_KAPPA, peak_n, 'o-', lw=2, color='C1',
               label='Peak $\\langle n\\rangle$')
    ax_pkn.plot(G_OVER_KAPPA, peak_coh, 's-', lw=2, color='purple',
               label='Peak coherence')
    ax_pkn.axvline(1.0, color='red', ls='--', lw=1.5)
    ax_pkn.set_xlabel('$g/\\kappa$')
    ax_pkn.set_title('Peak photon number + coherence')
    ax_pkn.legend(fontsize=8); ax_pkn.grid(True, alpha=0.3)

    # Purcell rate vs g/kappa
    gk_range = np.linspace(0.05, 1.2, 200)
    gP_range = purcell_rate(gk_range*KAPPA, KAPPA, 0.0)
    ax_purc.semilogy(gk_range, gP_range, 'k-', lw=2,
                    label=r'Analytic $\gamma_P=4g^2/\kappa$')
    ax_purc.semilogy(G_OVER_KAPPA, purcells, 'ro', ms=6, label='Sweep')
    ax_purc.axhline(GAMMA, color='blue', ls='--', lw=1.5,
                   label=f'$\\gamma$={GAMMA}')
    ax_purc.axvline(1.0, color='red', ls='--', lw=1.5)
    ax_purc.set_xlabel('$g/\\kappa$')
    ax_purc.set_ylabel('$\\gamma_P$')
    ax_purc.set_title('Purcell rate vs $g/\\kappa$')
    ax_purc.legend(fontsize=8); ax_purc.grid(True, which='both', alpha=0.3)

    fig2.suptitle(
        rf'JC low coupling analysis: $g/\kappa \in [0.1,1.1]$, '
        rf'$n_{{th}}$={N_TH_Q}, $T_2$={T2_Q:.0f}=$T_1/2$',
        fontsize=13)
    plt.savefig(f'../figures/JC/JC_low_coupling_analysis_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figures → figures/JC/JC_low_coupling_*.png")
    plt.show()


if __name__ == "__main__":
    run()
