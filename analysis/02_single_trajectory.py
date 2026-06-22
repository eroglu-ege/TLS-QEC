"""
02_single_trajectory.py
=======================
Full comparison of Lindblad vs Solomon at a single parameter point.
Now includes T2 (pure dephasing) and coherence decay fit.

CONFIGURE HERE — set T2_q and T2_t to control dephasing.
N_TH controls thermal population.

Run from: TLS-QEC/analysis/
    python 02_single_trajectory.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.metrics import full_comparison, print_summary
from utils.physics import make_params, n_thermal
from utils.io import save

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
N_TH = 0.1

# Build params from physical timescales
# T2=None means T2=2*T1 (no pure dephasing, Lindblad limit)
# T2=T1 means equal relaxation and dephasing (realistic SC qubit)
PARAMS = make_params(
    wq      = 1.0,
    wt      = 1.0,
    gamma_q = 0.01,    # T1_q = 100
    gamma_t = 0.005,   # T1_t = 200
    T2_q    = None,    # None = 2*T1 = 200 (no pure dephasing)
    T2_t    = None,    # None = 2*T1 = 400 (no pure dephasing)
    n_th_q  = N_TH,
    n_th_t  = N_TH,
)
# Add coupling — not part of make_params since it's a sweep variable here
G = 0.1
T_END   = 600
N_STEPS = 1000
# ──────────────────────────────────────────────────────────────────────────────

TAG = (f"nth{N_TH:.4f}_"
       f"T2q{PARAMS['T2_q']:.0f}_"
       f"T2t{PARAMS['T2_t']:.0f}")


def fit_coherence_decay(t, coh):
    """Fit |rho_eg,ge(t)| to A*exp(-t/T2_eff) after initial transient."""
    def exp_decay(t, A, T2):
        return A * np.exp(-t / T2)
    # Start fit after first 10% of trajectory (skip initial transient)
    start = int(0.1 * len(t))
    coh_safe = np.where(coh > 1e-10, coh, 1e-10)
    try:
        popt, _ = curve_fit(exp_decay, t[start:], coh_safe[start:],
                            p0=[coh_safe[start], t[-1]/3],
                            bounds=([0, 1], [1, 1e6]))
        return popt[1]  # T2_eff
    except Exception:
        return None


def run():
    p = PARAMS.copy()
    p['g'] = G

    print(f"Parameters:")
    print(f"  T1_q={p['T1_q']:.0f}  T2_q={p['T2_q']:.0f}  "
          f"gamma_phi_q={p['gamma_phi_q']:.5f}")
    print(f"  T1_t={p['T1_t']:.0f}  T2_t={p['T2_t']:.0f}  "
          f"gamma_phi_t={p['gamma_phi_t']:.5f}")
    print(f"  g={G:.3f}  n_th={N_TH:.4f}\n")

    lindblad_params = {k: p[k] for k in
        ['wq','wt','gamma_q','gamma_t','gamma_phi_q','gamma_phi_t',
         'n_th_q','n_th_t']}
    solomon_params = lindblad_params.copy()

    print("Running Lindblad...")
    res_L = lindblad_evolve(**lindblad_params, g=G,
                             t_end=T_END, n_steps=N_STEPS,
                             save_path=f'../data/single_trajectory/lindblad_{TAG}.h5')

    print("Running Solomon...")
    res_S = solomon_evolve(**solomon_params, g=G,
                            t_end=T_END, n_steps=N_STEPS,
                            save_path=f'../data/single_trajectory/solomon_{TAG}.h5')

    m = full_comparison(
        res_L['t'], res_L['P_e_q'], res_L['coherence'],
        res_S['t'], res_S['P_e_q'],
    )
    print_summary(m, {**lindblad_params, 'g': G})

    # Fit T2_eff from coherence decay
    T2_eff = fit_coherence_decay(res_L['t'], res_L['coherence'])
    T2_analytic = 2.0 / (p['gamma_q'] + p['gamma_phi_q'] +
                          p['gamma_t'] + p['gamma_phi_t'])
    print(f"\n  T2_eff (fitted):    {T2_eff:.2f}" if T2_eff else "\n  T2 fit failed")
    print(f"  T2 analytic limit:  {T2_analytic:.2f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 5))
    gs  = gridspec.GridSpec(1, 5, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(res_L['t'], res_L['P_e_q'], lw=2, label='Lindblad')
    ax1.plot(res_S['t'], res_S['P_e_q'], lw=2, ls='--', label='Solomon')
    ax1.axhline(res_L['P_e_q_ss'], color='gray', ls=':', lw=1,
               label=f"$P_{{ss}}$={res_L['P_e_q_ss']:.3f}")
    ax1.set_xlabel('Time'); ax1.set_ylabel('Population')
    ax1.set_title('Qubit $P_e$'); ax1.legend(fontsize=8)
    ax1.set_ylim(-0.02, 1.05)

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(res_L['t'], res_L['P_e_t'], lw=2, color='C1', label='Lindblad')
    ax2.plot(res_S['t'], res_S['P_e_t'], lw=2, ls='--', color='C3', label='Solomon')
    ax2.set_xlabel('Time'); ax2.set_title('TLS $P_e$')
    ax2.legend(fontsize=8); ax2.set_ylim(-0.02, 1.05)

    ax3 = fig.add_subplot(gs[2])
    coh = np.where(res_L['coherence'] > 0, res_L['coherence'], 1e-16)
    ax3.semilogy(res_L['t'], coh, color='purple', lw=2, label='Coherence')
    if T2_eff:
        t_fit = res_L['t']
        ax3.semilogy(t_fit, coh[0]*np.exp(-t_fit/T2_eff), 'k--', lw=1,
                    label=f'$T_2^{{eff}}$={T2_eff:.1f}')
    ax3.axvline(p['T2_q'], color='blue', ls=':', lw=1,
               label=f"$T_2^q$={p['T2_q']:.0f}")
    ax3.set_xlabel('Time'); ax3.set_ylabel(r'$|\rho_{eg,ge}|$')
    ax3.set_title('Coherence + $T_2$ fit'); ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(gs[3])
    P_S_interp = np.interp(res_L['t'], res_S['t'], res_S['P_e_q'])
    diff = np.abs(res_L['P_e_q'] - P_S_interp)
    ax4.semilogy(res_L['t'], np.where(diff>0, diff, 1e-16), color='black', lw=2)
    ax4.set_xlabel('Time'); ax4.set_ylabel(r'$|P_L-P_S|$')
    ax4.set_title(f'Discrepancy  ISE={m["ISE"]:.3e}')

    # Parameter box
    ax5 = fig.add_subplot(gs[4])
    ax5.axis('off')
    info = (f"PARAMETERS\n\n"
            f"g = {G:.4f}\n"
            f"$n_{{th}}$ = {N_TH:.4f}\n\n"
            f"Qubit:\n"
            f"  $T_1^q$ = {p['T1_q']:.0f}\n"
            f"  $T_2^q$ = {p['T2_q']:.0f}\n"
            f"  $\\gamma_\\phi^q$ = {p['gamma_phi_q']:.5f}\n\n"
            f"TLS:\n"
            f"  $T_1^t$ = {p['T1_t']:.0f}\n"
            f"  $T_2^t$ = {p['T2_t']:.0f}\n"
            f"  $\\gamma_\\phi^t$ = {p['gamma_phi_t']:.5f}\n\n"
            f"Fitted $T_2^{{eff}}$ = {T2_eff:.1f}" if T2_eff else "T2 fit failed")
    ax5.text(0.05, 0.95, info, transform=ax5.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')

    fig.suptitle(
        rf"Lindblad vs Solomon  ($\Delta$={p['wq']-p['wt']:.2f}, "
        rf"$g$={G:.3f}, $T_2^q$={p['T2_q']:.0f}, $n_{{th}}$={N_TH:.3f})",
        fontsize=13
    )
    plt.savefig(f'../figures/single_trajectory/lindblad_vs_solomon_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"\nFigure saved → figures/single_trajectory/lindblad_vs_solomon_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
