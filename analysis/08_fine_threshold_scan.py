"""
08_fine_threshold_scan.py
==========================
High-resolution scan zoomed tightly around the threshold g = gamma_t,
now including TRACE DISTANCE as the rigorous significance metric
alongside the original sensitivity-based diagnostics.

PHYSICS: g = gamma_t marks where two competing timescales cross:
  - Rabi period       T_rabi = pi/g           (coherent exchange time)
  - TLS decoherence   1/gamma_t               (how long coherence survives)

METRICS TRACKED (5 total):
  1. L_max, S_max     : peak qubit population (Lindblad vs Solomon)
  2. ISE               : integrated squared error
  3. coherence_max     : peak |rho_eg,ge(t)|  (mechanistic cause)
  4. osc_amp            : oscillation amplitude (Rabi onset signature)
  5. D_max              : peak trace distance (rigorous significance)

TWO THRESHOLDS COMPUTED (via find_two_thresholds):
  - onset_x       : first detectable departure (10x noise floor)
  - significant_x : first point with D > 0.05 (substantial disagreement,
                     standard quantum information convention)

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
from utils.metrics_extra import (
    trace_distance_trajectory, find_onset, find_two_thresholds
)

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

# Build consistent params from physical timescales.
# T2=None means no pure dephasing (T2 = 2*T1, Lindblad limit).
# Set T2_q < 2*T1_q to add realistic pure dephasing.
PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,    # T1_q = 100
    gamma_t = 0.005,   # T1_t = 200
    T2_q    = None,    # None => T2 = 2*T1 (no pure dephasing)
    T2_t    = None,    # None => T2 = 2*T1 (no pure dephasing)
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)
FIXED = {k: PARAMS[k] for k in
         ["wq","wt","gamma_q","gamma_t","gamma_phi_q","gamma_phi_t",
          "n_th_q","n_th_t"]}
# ──────────────────────────────────────────────────────────────────────────

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

        P_S_q_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
        P_S_t_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_t'])

        ise_arr[i] = ISE(res_L['t'], res_L['P_e_q'], P_S_q_interp)
        coh_max[i] = coherence_metrics(res_L['t'], res_L['coherence'])['max']

        D = trace_distance_trajectory(res_L['states'], P_S_q_interp, P_S_t_interp)
        D_max[i] = D.max()

    np.savez(f'../data/sweeps/fine_threshold_{TAG}.npz',
             g_values=G_VALUES, g_over_gamma_t=G_OVER_GAMMA_T,
             L_max=L_max, S_max=S_max,
             osc_amp=osc_amp, ISE=ise_arr, coherence_max=coh_max,
             D_max=D_max)
    print(f"\nData saved → data/sweeps/fine_threshold_{TAG}.npz")

    # ── Compute the two-number thesis claim using trace distance ───────────────
    # Restrict baseline to the genuinely flat region (g/gt < 0.1), since
    # divergence is already underway by g/gt ~ 0.07-0.3 per earlier scans —
    # including those points in the baseline would inflate the noise floor.
    baseline_mask = G_OVER_GAMMA_T < 0.1
    thresholds = find_two_thresholds(
        G_OVER_GAMMA_T, D_max,
        baseline_mask=baseline_mask,
        onset_multiplier=10.0,
        significance_level=0.05,   # D > 0.05 = substantial disagreement
    )
    print("\n" + "="*55)
    print("TWO-THRESHOLD RESULT (trace distance based)")
    print("="*55)
    onset = thresholds['onset']
    print(f"  Baseline D (noise floor): {onset['baseline']:.5f}")
    onset_print = f"{onset['onset_x']:.4f}" if onset['onset_x'] is not None else "not found in scanned range"
    sig_print = f"{thresholds['significant_x']:.4f}" if thresholds['significant_x'] is not None else "not found in scanned range"
    print(f"  Sensitivity onset (10x baseline): g/γt = {onset_print}")
    print(f"  Significance threshold (D>0.05): g/γt = {sig_print}")
    print("="*55)

    # ── Figure: 5-panel diagnostic ──────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    ax = axes[0, 0]
    ax.plot(G_OVER_GAMMA_T, L_max, 'o-', lw=1.5, ms=3, label='Lindblad max $P_q$')
    ax.plot(G_OVER_GAMMA_T, S_max, 's-', lw=1.5, ms=3, label='Solomon max $P_q$')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\gamma_t$')
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Peak qubit population')
    ax.set_title('Peak population vs coupling'); ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(G_OVER_GAMMA_T, ise_arr, 'o-', lw=1.5, ms=3, color='black')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('ISE')
    ax.set_title('Integrated Squared Error')

    ax = axes[0, 2]
    ax.plot(G_OVER_GAMMA_T, coh_max, 'o-', lw=1.5, ms=3, color='purple')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Peak coherence')
    ax.set_title('Coherence buildup')

    ax = axes[1, 0]
    ax.plot(G_OVER_GAMMA_T, osc_amp, 'o-', lw=1.5, ms=3, color='darkgreen')
    ax.axvline(1.0, color='red', ls='--', lw=1.5)
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Oscillation amplitude')
    ax.set_title('Rabi oscillation onset')

    # NEW: trace distance panel with both thresholds marked
    ax = axes[1, 1]
    ax.plot(G_OVER_GAMMA_T, D_max, 'o-', lw=2, ms=3, color='darkred')
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$g=\\gamma_t$')
    if onset['onset_x'] is not None:
        ax.axvline(onset['onset_x'], color='blue', ls=':', lw=1.5,
                  label=f"onset: {onset['onset_x']:.3f}")
    if thresholds['significant_x'] is not None:
        ax.axvline(thresholds['significant_x'], color='green', ls=':', lw=1.5,
                  label=f"significant: {thresholds['significant_x']:.3f}")
    ax.axhline(0.05, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('$g/\\gamma_t$'); ax.set_ylabel('Max trace distance $D$')
    ax.set_title('Trace distance (rigorous significance)')
    ax.legend(fontsize=7)

    # Summary text panel
    ax = axes[1, 2]
    ax.axis('off')
    onset_str = f"{onset['onset_x']:.4f}" if onset['onset_x'] is not None else "not found"
    sig_str = f"{thresholds['significant_x']:.4f}" if thresholds['significant_x'] is not None else "not found"
    summary = (
        f"TWO-THRESHOLD SUMMARY\n\n"
        f"Sensitivity onset:\n"
        f"  g/γt = {onset_str}\n"
        f"  (10x noise floor, D-based)\n\n"
        f"Significance threshold:\n"
        f"  g/γt = {sig_str}\n"
        f"  (D > 0.05, substantial)\n\n"
        f"Baseline D: {onset['baseline']:.5f}\n"
        f"Max D in scan: {D_max.max():.4f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(
        rf'Fine threshold scan with trace distance, $n_{{th}}$={N_TH}',
        fontsize=14
    )
    plt.tight_layout()
    plt.savefig(f'../figures/sweeps/fine_threshold_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/sweeps/fine_threshold_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
