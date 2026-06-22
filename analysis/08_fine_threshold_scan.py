"""
08_fine_threshold_scan.py
==========================
High-resolution scan around g=gamma_t with trace distance.

METRICS (5):
  1. L_max, S_max     : peak qubit population
  2. ISE               : integrated squared error
  3. coherence_max     : peak |rho_eg,ge|
  4. osc_amp           : oscillation amplitude
  5. D_max             : peak trace distance (rigorous significance)

TWO THRESHOLDS:
  - onset_x       : 10x noise floor (sensitivity)
  - significant_x : D > 0.05 (substantial disagreement)

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
from utils.metrics_extra import trace_distance_trajectory, find_two_thresholds
from utils.physics import make_params

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,
    gamma_t = 0.005,
    T2_q    = None,
    T2_t    = None,
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)
FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}

# Two-tier sampling
G_OVER_GAMMA_T_COARSE = np.geomspace(0.05, 10, 30)
G_OVER_GAMMA_T_ZOOM   = np.linspace(0.5, 2.0, 60)
G_OVER_GAMMA_T = np.unique(np.concatenate(
    [G_OVER_GAMMA_T_COARSE, G_OVER_GAMMA_T_ZOOM]))
G_OVER_GAMMA_T.sort()
G_VALUES = G_OVER_GAMMA_T * FIXED["gamma_t"]
N_G      = len(G_VALUES)

T_END   = 4000
N_STEPS = 1200
TAG     = f"nth{N_TH:.4f}_T2q{PARAMS['T2_q']:.0f}_T2t{PARAMS['T2_t']:.0f}"
# ──────────────────────────────────────────────────────────────────────────────


def run():
    print(f"Fine threshold scan: {N_G} points  N_TH={N_TH}")
    print(f"  T2_q={PARAMS['T2_q']:.0f}  T2_t={PARAMS['T2_t']:.0f}\n")

    L_max   = np.zeros(N_G)
    S_max   = np.zeros(N_G)
    osc_amp = np.zeros(N_G)
    ise_arr = np.zeros(N_G)
    coh_max = np.zeros(N_G)
    D_max   = np.zeros(N_G)

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

        P_S_q = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
        P_S_t = np.interp(res_L['t'], res_S['t'], res_S['P_e_t'])

        ise_arr[i] = ISE(res_L['t'], res_L['P_e_q'], P_S_q)
        coh_max[i] = coherence_metrics(res_L['t'], res_L['coherence'])['max']
        D = trace_distance_trajectory(res_L['states'], P_S_q, P_S_t)
        D_max[i] = D.max()

    np.savez(f'../data/sweeps/fine_threshold_{TAG}.npz',
             g_values=G_VALUES, g_over_gamma_t=G_OVER_GAMMA_T,
             L_max=L_max, S_max=S_max, osc_amp=osc_amp,
             ISE=ise_arr, coherence_max=coh_max, D_max=D_max)
    print(f"\nData saved → data/sweeps/fine_threshold_{TAG}.npz")

    # Thresholds using baseline from g/gamma_t < 0.1
    baseline_mask = G_OVER_GAMMA_T < 0.1
    thresholds = find_two_thresholds(
        G_OVER_GAMMA_T, D_max,
        baseline_mask=baseline_mask,
        onset_multiplier=10.0,
        significance_level=0.05,
    )
    onset = thresholds['onset']
    onset_str = f"{onset['onset_x']:.4f}" if onset['onset_x'] is not None else "not found"
    sig_str   = f"{thresholds['significant_x']:.4f}" if thresholds['significant_x'] is not None else "not found"

    print("\n" + "="*55)
    print("TWO-THRESHOLD RESULT (trace distance based)")
    print("="*55)
    print(f"  Baseline D:             {onset['baseline']:.5f}")
    print(f"  Sensitivity onset:      g/γt = {onset_str}")
    print(f"  Significance (D>0.05):  g/γt = {sig_str}")
    print("="*55)

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0, 0]
    ax.plot(G_OVER_GAMMA_T, L_max, 'o-', lw=1.5, ms=3, label='Lindblad')
    ax.plot(G_OVER_GAMMA_T, S_max, 's-', lw=1.5, ms=3, label='Solomon')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('$g/\\gamma_t$')
    ax.set_ylabel('Peak qubit $P_e$'); ax.set_title('Peak population')
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(G_OVER_GAMMA_T, ise_arr, 'o-', lw=1.5, ms=3, color='black')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('ISE')
    ax.set_title('Integrated Squared Error')

    ax = axes[0, 2]
    ax.plot(G_OVER_GAMMA_T, coh_max, 'o-', lw=1.5, ms=3, color='purple')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('$g/\\gamma_t$')
    ax.set_ylabel('Peak coherence'); ax.set_title('Coherence buildup')

    ax = axes[1, 0]
    ax.plot(G_OVER_GAMMA_T, osc_amp, 'o-', lw=1.5, ms=3, color='darkgreen')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel('$g/\\gamma_t$')
    ax.set_ylabel('Oscillation amplitude'); ax.set_title('Rabi onset')

    ax = axes[1, 1]
    ax.plot(G_OVER_GAMMA_T, D_max, 'o-', lw=2, ms=3, color='darkred')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\gamma_t$')
    if onset['onset_x']:
        ax.axvline(onset['onset_x'], color='blue', ls=':', lw=1.5,
                  label=f"onset: {onset_str}")
    if thresholds['significant_x']:
        ax.axvline(thresholds['significant_x'], color='green', ls=':', lw=1.5,
                  label=f"sig: {sig_str}")
    ax.axhline(0.05, color='gray', ls=':', lw=1, alpha=0.7)
    ax.set_xscale('log'); ax.set_xlabel('$g/\\gamma_t$')
    ax.set_ylabel('Max trace distance $D$')
    ax.set_title('Trace distance (rigorous)'); ax.legend(fontsize=7)

    ax = axes[1, 2]
    ax.axis('off')
    summary = (f"TWO-THRESHOLD SUMMARY\n\n"
               f"Sensitivity onset:\n  g/γt = {onset_str}\n"
               f"  (10x noise floor)\n\n"
               f"Significance:\n  g/γt = {sig_str}\n"
               f"  (D > 0.05)\n\n"
               f"Baseline D: {onset['baseline']:.5f}\n"
               f"Max D: {D_max.max():.4f}\n\n"
               f"T2_q={PARAMS['T2_q']:.0f}  T2_t={PARAMS['T2_t']:.0f}\n"
               f"n_th={N_TH:.3f}")
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
           fontsize=11, va='top', fontfamily='monospace')

    fig.suptitle(rf'Fine threshold scan, $n_{{th}}$={N_TH}, '
                 rf'$T_2^q$={PARAMS["T2_q"]:.0f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/fine_threshold_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/fine_threshold_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
