"""
19_TLS_lifetime_vs_detuning.py
================================
TLS lifetime T1_TLS vs detuning Delta, when TLS decays ONLY via the qubit.

PHYSICAL PICTURE:
  The TLS has no direct bath coupling. Its only decay channel is:
    TLS excited -> emit into qubit -> qubit decays via gamma_q

  This is the "reversed Purcell effect": the qubit is the lossy cavity,
  the TLS is the emitter. The effective TLS decay rate is:

    Gamma_eff(Delta) = 2*g^2 * (1/T2_q) / (Delta^2 + (1/T2_q)^2)

  where 1/T2_q = gamma_q/2 + gamma_phi is the FULL qubit linewidth
  (T1 AND T2 both broaden the qubit resonance that the TLS sees).

  Setting g ~ gamma_phi (as requested) puts us in the regime where
  coherent exchange competes with qubit dephasing.

SATURATION AT LARGE DELTA:
  At large detuning, qubit-mediated decay vanishes. The TLS has some
  unknown intrinsic decay gamma_t_int (phonons, charge noise in substrate).
  This sets the saturation plateau. We show a family of curves for
  different assumed gamma_t_int values to show the uncertainty.

  Full formula:
    1/T1_TLS = Gamma_eff(Delta) + gamma_t_int

GRAPH:
  x: detuning Delta
  y: T1_TLS (in units of 1/gamma_q)
  Shows: Lorentzian dip at Delta=0, rising on both sides, saturating
  at 1/gamma_t_int at large Delta.

Run from: TLS-QEC/analysis/
    python 19_TLS_lifetime_vs_detuning.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils.physics import make_params, gamma_phi_from_T2

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
GAMMA_Q   = 0.01     # qubit T1 decay rate
T2_Q      = 50.0     # qubit T2 (T2 = T1/2 => gamma_phi = 0.03)

# Pure dephasing rate
GAMMA_PHI = gamma_phi_from_T2(GAMMA_Q, T2_Q)   # = 2/T2 - gamma_q = 0.03

# Qubit linewidth seen by TLS = 1/T2_q (includes both T1 and T2)
GAMMA_Q_EFF = 1.0 / T2_Q    # = 0.02

# Coupling: g ~ gamma_phi (as requested)
# gamma_phi = 0.03, so g ~ 0.03
G = GAMMA_PHI    # g exactly equal to gamma_phi

# Detuning sweep
DELTA_MAX = 1.0
N_DELTA   = 500
DELTA     = np.linspace(-DELTA_MAX, DELTA_MAX, N_DELTA)

# Unknown intrinsic TLS decay rates (saturation uncertainty)
# These are what's NOT known experimentally
GAMMA_T_INT_VALUES = [0.0, 0.001, 0.005, 0.01, 0.05]
# 0.0 = no intrinsic decay (pure qubit-mediated)
# 0.001 = very long intrinsic T1_TLS = 1000
# 0.05  = short intrinsic T1_TLS = 20
# ──────────────────────────────────────────────────────────────────────────────


def gamma_eff(delta, g, gamma_q_eff):
    """Qubit-mediated TLS decay rate (reversed Purcell)."""
    return 2 * g**2 * gamma_q_eff / (delta**2 + gamma_q_eff**2)


def T1_TLS(delta, g, gamma_q_eff, gamma_t_int=0.0):
    """Total TLS lifetime including intrinsic decay."""
    rate = gamma_eff(delta, g, gamma_q_eff) + gamma_t_int
    return 1.0 / np.where(rate > 0, rate, 1e-30)


def run():
    os.makedirs('../figures/sweeps', exist_ok=True)

    print(f"TLS lifetime vs detuning")
    print(f"  gamma_q   = {GAMMA_Q:.4f}  (T1_q = {1/GAMMA_Q:.0f})")
    print(f"  T2_q      = {T2_Q:.1f}     (gamma_phi = {GAMMA_PHI:.4f})")
    print(f"  gamma_q_eff = 1/T2_q = {GAMMA_Q_EFF:.4f}")
    print(f"  g = gamma_phi = {G:.4f}  (g/gamma_q_eff = {G/GAMMA_Q_EFF:.2f})")
    print()
    print(f"  At Delta=0 (resonant):")
    print(f"    Gamma_eff = 2*g^2/gamma_q_eff = {2*G**2/GAMMA_Q_EFF:.6f}")
    print(f"    T1_TLS(0) = {1/(2*G**2/GAMMA_Q_EFF):.2f}  (qubit-mediated only)")
    print()
    print(f"  At large Delta: T1_TLS -> 1/gamma_t_int (unknown saturation)")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.4)

    ax1 = fig.add_subplot(gs[0, :])   # main figure: T1_TLS vs Delta, all curves
    ax2 = fig.add_subplot(gs[1, 0])   # zoom near Delta=0
    ax3 = fig.add_subplot(gs[1, 1])   # Gamma_eff vs Delta (Lorentzian shape)

    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(GAMMA_T_INT_VALUES)))

    for gamma_t_int, color in zip(GAMMA_T_INT_VALUES, colors):
        T1 = T1_TLS(DELTA, G, GAMMA_Q_EFF, gamma_t_int)

        if gamma_t_int == 0.0:
            lbl = 'No intrinsic decay\n(pure qubit-mediated)'
            ls  = '-'
            lw  = 3.0
        else:
            T1_sat = 1.0 / gamma_t_int
            lbl = (f'$\\gamma_t^{{int}}$={gamma_t_int:.3f}  '
                   f'($T_1^{{sat}}$={T1_sat:.0f})')
            ls  = '--'
            lw  = 2.0

        ax1.semilogy(DELTA, T1, color=color, ls=ls, lw=lw, label=lbl)
        ax2.semilogy(DELTA, T1, color=color, ls=ls, lw=lw)

        # Mark saturation
        if gamma_t_int > 0:
            ax1.axhline(1.0/gamma_t_int, color=color, ls=':', lw=1, alpha=0.5)

    # Annotate key features on ax1
    T1_min = 1.0 / (2*G**2/GAMMA_Q_EFF)
    ax1.annotate(
        f'Minimum $T_1^{{TLS}}$ = {T1_min:.1f}\n'
        f'(at $\\Delta=0$, no intrinsic decay)',
        xy=(0, T1_min), xytext=(0.15, T1_min*0.3),
        fontsize=9, color='black',
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Mark qubit linewidth
    for ax in [ax1, ax2]:
        ax.axvline( GAMMA_Q_EFF, color='gray', ls=':', lw=1.5,
                   label=f'$1/T_2^q$={GAMMA_Q_EFF:.3f}')
        ax.axvline(-GAMMA_Q_EFF, color='gray', ls=':', lw=1.5)
        ax.axvline( G, color='blue', ls='--', lw=1.5,
                   label=f'$g=\\gamma_\\phi$={G:.3f}')
        ax.axvline(-G, color='blue', ls='--', lw=1.5)

    ax1.set_xlabel('Detuning $\\Delta = \\omega_q - \\omega_{{TLS}}$', fontsize=12)
    ax1.set_ylabel('$T_1^{TLS}$ (log scale)', fontsize=12)
    ax1.set_title(
        rf'TLS lifetime vs detuning: dip at resonance, saturation at large $\Delta$'
        '\n'
        rf'($g=\gamma_\phi={G:.3f}$, $1/T_2^q={GAMMA_Q_EFF:.3f}$, '
        rf'TLS decays ONLY via qubit)',
        fontsize=12)
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(True, which='both', alpha=0.3)
    ax1.set_xlim(-DELTA_MAX, DELTA_MAX)

    # Zoom
    zoom = 5 * GAMMA_Q_EFF
    ax2.set_xlim(-zoom, zoom)
    ax2.set_xlabel('Detuning $\\Delta$', fontsize=11)
    ax2.set_ylabel('$T_1^{TLS}$ (log scale)', fontsize=11)
    ax2.set_title(f'Zoom near $\\Delta=0$\n($\\pm 5/T_2^q$)', fontsize=11)
    ax2.grid(True, which='both', alpha=0.3)

    # Gamma_eff Lorentzian shape
    Geff = gamma_eff(DELTA, G, GAMMA_Q_EFF)
    ax3.plot(DELTA, Geff, 'k-', lw=2.5)
    ax3.axvline( GAMMA_Q_EFF, color='gray', ls=':', lw=1.5,
                label=f'HWHM = $1/T_2^q$ = {GAMMA_Q_EFF:.3f}')
    ax3.axvline(-GAMMA_Q_EFF, color='gray', ls=':', lw=1.5)
    ax3.axvline( G, color='blue', ls='--', lw=1.5,
                label=f'$g={G:.3f}$')
    ax3.axvline(-G, color='blue', ls='--', lw=1.5)
    ax3.fill_between(DELTA, 0, Geff,
                     where=np.abs(DELTA) < GAMMA_Q_EFF,
                     alpha=0.15, color='red',
                     label='Resonant window')
    ax3.set_xlabel('Detuning $\\Delta$', fontsize=11)
    ax3.set_ylabel('$\\Gamma_{eff}(\\Delta)$', fontsize=11)
    ax3.set_title(
        'Qubit-mediated TLS decay rate\n'
        r'$\Gamma_{eff} = 2g^2(1/T_2^q)/(\Delta^2+(1/T_2^q)^2)$',
        fontsize=11)
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    fig.suptitle(
        rf'TLS lifetime — qubit-mediated decay only  '
        rf'($g \approx \gamma_\phi$, '
        rf'$T_1^q={1/GAMMA_Q:.0f}$, $T_2^q={T2_Q:.0f}$)',
        fontsize=14)

    plt.savefig('../figures/sweeps/TLS_lifetime_vs_detuning.png',
                dpi=150, bbox_inches='tight')
    print("\nFigure → figures/sweeps/TLS_lifetime_vs_detuning.png")
    plt.show()


if __name__ == "__main__":
    run()
