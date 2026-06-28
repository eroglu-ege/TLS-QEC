"""
20_TLS_lifetime_sweeps.py
==========================
Comprehensive TLS lifetime prediction sweeps.

All sweeps are analytic (no ODE solver needed) except sweep 4
which validates the analytic picture with Lindblad simulation.

SWEEPS:
  1. T1_TLS vs g          (at Delta=0, fixed T2_q)
  2. T1_TLS vs T2_q       (at Delta=0, fixed g)
  3. T1_TLS vs T (temperature, via n_th broadening)
  4. T1_TLS vs Delta at several g values
     - Lindblad simulation of qubit T1_eff vs Delta (experimentally accessible)
     - Overlay Solomon prediction for comparison
  5. Qubit T1_eff vs Delta (what you'd actually measure in the lab)

PHYSICAL CONTEXT:
  TLS decays ONLY via qubit (no direct bath coupling).
  Effective TLS decay rate (reversed Purcell):
    Gamma_TLS(Delta) = 2*g^2 * gamma_q_eff / (Delta^2 + gamma_q_eff^2)
  where gamma_q_eff = gamma_q/2 + gamma_phi = 1/T2_q

  Setting g ~ gamma_phi puts us at Solomon breakdown threshold.

Run from: TLS-QEC/analysis/
    python 20_TLS_lifetime_sweeps.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.physics import make_params, gamma_phi_from_T2, n_thermal

# ─── BASELINE PARAMETERS ──────────────────────────────────────────────────────
GAMMA_Q   = 0.01      # qubit T1 decay rate  (T1_q = 100)
T2_Q      = 50.0      # qubit T2             (T2 = T1/2)
GAMMA_PHI = gamma_phi_from_T2(GAMMA_Q, T2_Q)   # = 0.030
GAMMA_Q_EFF = 1.0 / T2_Q                        # = 0.020

# g at Solomon threshold: g = gamma_phi
G_BASE    = GAMMA_PHI   # = 0.030

# Unknown intrinsic TLS decay (saturation)
GAMMA_T_INT = 0.002    # baseline assumption: T1_TLS_int = 500

N_TH_BASE = 0.05       # baseline thermal population

TAG = f"baseline_T2q{T2_Q:.0f}_g{G_BASE:.4f}"
# ──────────────────────────────────────────────────────────────────────────────


# ── Analytic formulas ─────────────────────────────────────────────────────────

def gamma_q_eff(gamma_q, gamma_phi):
    """Total qubit linewidth = 1/T2_q."""
    return gamma_q / 2.0 + gamma_phi


def gamma_TLS_mediated(delta, g, gq_eff):
    """Qubit-mediated TLS decay rate."""
    return 2.0 * g**2 * gq_eff / (delta**2 + gq_eff**2)


def T1_TLS_total(delta, g, gq_eff, gamma_t_int=0.0):
    """Total TLS lifetime including intrinsic decay."""
    rate = gamma_TLS_mediated(delta, g, gq_eff) + gamma_t_int
    return 1.0 / np.where(rate > 1e-30, rate, 1e-30)


def gq_eff_thermal(gamma_q, gamma_phi, n_th):
    """Thermally broadened qubit linewidth."""
    return gamma_q * (2*n_th + 1) / 2.0 + gamma_phi


def run():
    os.makedirs('../figures/sweeps', exist_ok=True)
    os.makedirs('../data/sweeps', exist_ok=True)

    print(f"Baseline: gamma_q={GAMMA_Q}, T2_q={T2_Q}")
    print(f"  gamma_phi={GAMMA_PHI:.4f}, gamma_q_eff={GAMMA_Q_EFF:.4f}")
    print(f"  g=gamma_phi={G_BASE:.4f}  (g/gamma_q_eff={G_BASE/GAMMA_Q_EFF:.2f})")
    print(f"  T1_TLS(Delta=0) = {T2_Q/(2*G_BASE**2):.1f}")

    fig = plt.figure(figsize=(20, 16))
    gs  = gridspec.GridSpec(3, 3, wspace=0.38, hspace=0.45)

    # ── SWEEP 1: T1_TLS vs g ─────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    g_range = np.logspace(-3, -1, 300)
    gq = gamma_q_eff(GAMMA_Q, GAMMA_PHI)

    for gamma_t_int, ls, lbl in [
        (0.0,   '-',  'No intrinsic decay'),
        (GAMMA_T_INT, '--', f'$\\gamma_t^{{int}}$={GAMMA_T_INT:.3f}'),
    ]:
        T1 = T1_TLS_total(0.0, g_range, gq, gamma_t_int)
        ax1.loglog(g_range, T1, ls=ls, lw=2, label=lbl)

    # Power law: T1 ~ 1/g^2
    ax1.loglog(g_range, gq/(2*g_range**2), 'k:', lw=1.5,
               label=r'$T_2^q/(2g^2)$ (analytic)')
    ax1.axvline(G_BASE, color='red', ls='--', lw=1.5,
               label=f'$g=\\gamma_\\phi$={G_BASE:.3f}')
    ax1.axvline(GAMMA_Q, color='blue', ls=':', lw=1.5,
               label=f'$g=\\gamma_q$={GAMMA_Q:.3f}')
    ax1.set_xlabel('Coupling $g$', fontsize=11)
    ax1.set_ylabel('$T_1^{TLS}$ at $\\Delta=0$', fontsize=11)
    ax1.set_title('Sweep 1: $T_1^{TLS}$ vs $g$\n'
                  r'($\propto 1/g^2$ power law)', fontsize=11)
    ax1.legend(fontsize=7); ax1.grid(True, which='both', alpha=0.3)

    # ── SWEEP 2: T1_TLS vs T2_q ──────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    T2q_range = np.linspace(10, 200, 300)

    for g_val, color, lbl in [
        (G_BASE*0.5, 'C0', f'$g={G_BASE*0.5:.3f}$ (below threshold)'),
        (G_BASE,     'C1', f'$g={G_BASE:.3f}=\\gamma_\\phi$ (threshold)'),
        (G_BASE*2.0, 'C2', f'$g={G_BASE*2:.3f}$ (above threshold)'),
    ]:
        gq_vals = 1.0 / T2q_range
        T1 = T1_TLS_total(0.0, g_val, gq_vals, GAMMA_T_INT)
        ax2.semilogy(T2q_range, T1, color=color, lw=2, label=lbl)

    ax2.axvline(T2_Q, color='gray', ls='--', lw=1.5,
               label=f'Baseline $T_2^q$={T2_Q:.0f}')
    ax2.set_xlabel('Qubit $T_2^q$', fontsize=11)
    ax2.set_ylabel('$T_1^{TLS}$ at $\\Delta=0$', fontsize=11)
    ax2.set_title('Sweep 2: $T_1^{TLS}$ vs $T_2^q$\n'
                  r'(longer $T_2$ $\Rightarrow$ longer $T_1^{TLS}$!)', fontsize=11)
    ax2.legend(fontsize=7); ax2.grid(True, which='both', alpha=0.3)

    # ── SWEEP 3: T1_TLS vs Temperature ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    freq_GHz = 5.0
    T_mK_range = np.linspace(5, 200, 300)
    T_K_range  = T_mK_range * 1e-3
    n_th_range = np.array([n_thermal(freq_GHz, T) for T in T_K_range])

    for g_val, color, lbl in [
        (G_BASE*0.5, 'C0', f'$g={G_BASE*0.5:.3f}$'),
        (G_BASE,     'C1', f'$g={G_BASE:.3f}$ (threshold)'),
        (G_BASE*2.0, 'C2', f'$g={G_BASE*2:.3f}$'),
    ]:
        gq_th = np.array([gq_eff_thermal(GAMMA_Q, GAMMA_PHI, n)
                          for n in n_th_range])
        T1 = T1_TLS_total(0.0, g_val, gq_th, GAMMA_T_INT)
        ax3.semilogy(T_mK_range, T1, color=color, lw=2, label=lbl)

    ax3.axvline(20,  color='gray', ls=':',  lw=1.5, label='20 mK (dilution fridge)')
    ax3.axvline(50,  color='gray', ls='--', lw=1.5, label='50 mK')
    ax3.axvline(100, color='gray', ls='-',  lw=1.0, label='100 mK')
    ax3.set_xlabel('Temperature (mK)', fontsize=11)
    ax3.set_ylabel('$T_1^{TLS}$ at $\\Delta=0$', fontsize=11)
    ax3.set_title(f'Sweep 3: $T_1^{{TLS}}$ vs Temperature\n'
                  f'({freq_GHz} GHz qubit)', fontsize=11)
    ax3.legend(fontsize=7); ax3.grid(True, which='both', alpha=0.3)

    # ── SWEEP 4: T1_TLS vs Delta at several g values ─────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    DELTA = np.linspace(-0.5, 0.5, 500)
    gq    = gamma_q_eff(GAMMA_Q, GAMMA_PHI)

    g_values_sweep4 = [G_BASE*0.3, G_BASE*0.5, G_BASE, G_BASE*2, G_BASE*5]
    colors4 = plt.cm.viridis(np.linspace(0.1, 0.9, len(g_values_sweep4)))

    for g_val, color in zip(g_values_sweep4, colors4):
        ratio = g_val / GAMMA_PHI
        T1 = T1_TLS_total(DELTA, g_val, gq, GAMMA_T_INT)
        ax4.semilogy(DELTA, T1, color=color, lw=2,
                    label=f'$g={g_val:.4f}$ ($g/\\gamma_\\phi$={ratio:.1f})')

    ax4.axvline( gq, color='gray', ls=':', lw=1.5, label=f'$1/T_2^q$={gq:.3f}')
    ax4.axvline(-gq, color='gray', ls=':', lw=1.5)
    ax4.set_xlabel('Detuning $\\Delta$', fontsize=11)
    ax4.set_ylabel('$T_1^{TLS}$ (log)', fontsize=11)
    ax4.set_title('Sweep 4: $T_1^{TLS}$ vs $\\Delta$ at several $g$\n'
                  'Deeper dip = stronger coupling, same saturation plateau',
                  fontsize=11)
    ax4.legend(fontsize=8, ncol=2); ax4.grid(True, which='both', alpha=0.3)

    # ── SWEEP 5: Qubit T1_eff vs Delta (what lab actually measures) ───────────
    print("\nRunning Lindblad simulation for qubit T1_eff vs Delta...")
    ax5 = fig.add_subplot(gs[1, 2])

    delta_sim_vals = np.array([-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2])
    gamma_t_sim    = 0.005   # TLS intrinsic decay rate for simulation
    T_END_SIM      = 600
    N_STEPS_SIM    = 400

    gamma1_L = []
    gamma1_S = []

    for delta in tqdm(delta_sim_vals, desc="  Qubit T1_eff vs Delta"):
        p = make_params(wq=1.0, wt=1.0-delta,
                        gamma_q=GAMMA_Q, gamma_t=gamma_t_sim,
                        T2_q=T2_Q, n_th_q=N_TH_BASE, n_th_t=N_TH_BASE)
        fixed = {k: p[k] for k in
                 ["wq","wt","gamma_q","gamma_t","gamma_phi_q",
                  "gamma_phi_t","n_th_q","n_th_t"]}

        res_L = lindblad_evolve(**fixed, g=G_BASE, t_end=T_END_SIM,
                                n_steps=N_STEPS_SIM)
        res_S = solomon_evolve(**fixed, g=G_BASE, t_end=T_END_SIM,
                               n_steps=N_STEPS_SIM)

        # Fit effective qubit decay rate: P_e(t) ~ exp(-gamma1_eff * t)
        from scipy.optimize import curve_fit
        def exp_decay(t, A, gamma1, C):
            return A * np.exp(-gamma1 * t) + C

        try:
            pL, _ = curve_fit(exp_decay, res_L['t'], res_L['P_e_q'],
                              p0=[1.0, GAMMA_Q*2, 0.05], maxfev=5000)
            gamma1_L.append(pL[1])
        except Exception:
            gamma1_L.append(np.nan)

        try:
            pS, _ = curve_fit(exp_decay, res_S['t'], res_S['P_e_q'],
                              p0=[1.0, GAMMA_Q*2, 0.05], maxfev=5000)
            gamma1_S.append(pS[1])
        except Exception:
            gamma1_S.append(np.nan)

    gamma1_L = np.array(gamma1_L)
    gamma1_S = np.array(gamma1_S)

    # Analytic qubit T1_eff: qubit gets Purcell enhancement from TLS
    delta_an  = np.linspace(-0.3, 0.3, 300)
    gamma_P_an = 2*G_BASE**2*gamma_t_sim / (delta_an**2 + gamma_t_sim**2)
    gamma1_an  = GAMMA_Q + gamma_P_an

    ax5.semilogy(delta_an, 1.0/gamma1_an, 'k-', lw=2,
                label='Analytic $T_1^q$ (Purcell)')
    ax5.semilogy(delta_sim_vals, 1.0/gamma1_L, 'o-', lw=2,
                color='C0', ms=8, label='Lindblad (fitted)')
    ax5.semilogy(delta_sim_vals, 1.0/gamma1_S, 's--', lw=2,
                color='C1', ms=8, label='Solomon (fitted)')
    ax5.axhline(1.0/GAMMA_Q, color='gray', ls=':', lw=1.5,
               label=f'$T_1^q$ (bare)={1/GAMMA_Q:.0f}')
    ax5.set_xlabel('Detuning $\\Delta$', fontsize=11)
    ax5.set_ylabel('Effective qubit $T_1^q$ (log)', fontsize=11)
    ax5.set_title('Sweep 5: Qubit $T_1^q$ vs $\\Delta$\n'
                  '(experimentally measurable, dip at resonance)',
                  fontsize=11)
    ax5.legend(fontsize=7); ax5.grid(True, which='both', alpha=0.3)

    # ── SUMMARY PANEL ─────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    T1_at_res  = T2_Q / (2*G_BASE**2)
    T1_sat     = 1.0 / GAMMA_T_INT

    summary = (
        "SUMMARY: TLS LIFETIME PREDICTION\n\n"
        f"  g (coupling)                    = {G_BASE:.4f}  (= gamma_phi, Solomon threshold)\n"
        f"  gamma_q_eff = 1/T2_q            = {GAMMA_Q_EFF:.4f}  (qubit linewidth seen by TLS)\n"
        f"  T1_TLS at Delta=0               = {T1_at_res:.1f}  (minimum lifetime)\n"
        f"  T1_TLS saturation (large Delta) = {T1_sat:.1f}  (unknown intrinsic plateau)\n"
        f"  Ratio T1_sat / T1_min           = {T1_sat/T1_at_res:.2f}\n\n"
        "KEY RESULTS:\n"
        "  Sweep 1: T1_TLS ~ 1/g^2  -- stronger coupling = shorter TLS lifetime\n"
        "  Sweep 2: T1_TLS ~ T2_q   -- better qubit = LONGER TLS lifetime!\n"
        "  Sweep 3: T1_TLS decreases with T -- thermal broadening drains TLS faster\n"
        "  Sweep 4: Lorentzian dip, same saturation plateau for all g\n"
        "  Sweep 5: Qubit T1_eff dip -- L and S disagree at g ~ gamma_phi"
    )
    ax6.text(0.01, 0.95, summary, transform=ax6.transAxes,
            fontsize=10, va='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig.suptitle(
        rf'TLS Lifetime Prediction: 5 sweeps  '
        rf'($g=\gamma_\phi={G_BASE:.3f}$, $T_1^q={1/GAMMA_Q:.0f}$, '
        rf'$T_2^q={T2_Q:.0f}$, $n_{{th}}={N_TH_BASE}$)',
        fontsize=14)

    plt.savefig('../figures/sweeps/TLS_lifetime_all_sweeps.png',
                dpi=150, bbox_inches='tight')
    print("\nFigure → figures/sweeps/TLS_lifetime_all_sweeps.png")
    plt.show()


if __name__ == "__main__":
    run()
