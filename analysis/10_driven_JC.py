"""
10_driven_JC.py
===============
Continuously driven Jaynes-Cummings: does photon number diverge?

Instead of initializing once, the qubit is continuously driven by a
coherent microwave tone at rate Omega. This models keeping the qubit
repeatedly excited and watching photons build up in the resonator.

Drive Hamiltonian (rotating frame):
    H = delta_q/2*sz + delta_r*a†a + g*(a*sp+a†*sm) + Omega/2*(sp+sm)

The drive pumps the qubit continuously. The qubit emits into the
resonator. The resonator leaks to the ADC at rate kappa.

ANSWER: Photon number does NOT diverge. It saturates at:
    <n>_ss ~ Omega^2 / (kappa*(gamma+kappa))   [resonant, weak drive]

This is a Lorentzian in drive frequency — standard cavity response.
Leakage (kappa) is exactly what prevents divergence.

Run from: TLS-QEC/analysis/
    python 10_driven_JC.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import qutip as qt

from qubit_tls.jaynes_cummings import build_operators

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
WR    = 1.0
WD    = 1.0       # drive on resonance
G     = 0.05
GAMMA = 0.01
KAPPA = 0.01
N_MAX = 30        # needs to be larger for driven case

T_END   = 2000
N_STEPS = 1000

OMEGA_VALUES = [0.005, 0.01, 0.05, 0.1]
# ──────────────────────────────────────────────────────────────────────────────


def build_H_driven(wq, wr, wd, g, Omega, N_max):
    """JC + coherent drive in rotating frame."""
    o = build_operators(N_max)
    dq = wq - wd
    dr = wr - wd
    return ((dq/2)*o['sz'] + dr*o['n_r']
            + g*(o['a']*o['sp'] + o['adag']*o['sm'])
            + (Omega/2)*(o['sp'] + o['sm']))


def run():
    o    = build_operators(N_MAX)
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(N_MAX+1, 0))
    rho0 = qt.ket2dm(psi0)
    c_ops = [np.sqrt(GAMMA)*o['sm'], np.sqrt(KAPPA)*o['a']]
    t     = np.linspace(0, T_END, N_STEPS)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    ss_photons = []

    for Omega, color in zip(OMEGA_VALUES, ['C0','C1','C2','C3']):
        print(f"Omega={Omega:.3f}  (Omega/gamma={Omega/GAMMA:.1f})...")

        H      = build_H_driven(WQ, WR, WD, G, Omega, N_MAX)
        result = qt.mesolve(H, rho0, t, c_ops,
                            e_ops=[o['sz'], o['n_r']],
                            options={"nsteps":10000})

        P_e_q     = (np.array(result.expect[0]) + 1) / 2
        n_photons = np.array(result.expect[1])
        n_ss      = np.mean(n_photons[int(0.9*N_STEPS):])
        n_ss_an   = Omega**2 / (KAPPA*(GAMMA+KAPPA))
        ss_photons.append(n_ss)

        if n_photons.max() > 0.8*N_MAX:
            print(f"  WARNING: max<n>={n_photons.max():.1f} near N_max={N_MAX}")

        lbl = (f'$\\Omega$={Omega:.3f}  '
               f'$\\langle n\\rangle_{{ss}}$={n_ss:.3f} '
               f'(analytic:{n_ss_an:.3f})')

        ax1.plot(t, n_photons, color=color, lw=2, label=lbl)
        ax2.plot(t, P_e_q,     color=color, lw=2,
                label=f'$\\Omega$={Omega:.3f}')
        ax3.semilogx(t[1:], n_photons[1:], color=color, lw=2, label=lbl)
        print(f"  <n>_ss numeric={n_ss:.4f}  analytic={n_ss_an:.4f}")

    # Analytic saturation curve
    Om_range = np.linspace(0.001, 0.15, 300)
    ax4.plot(Om_range, Om_range**2/(KAPPA*(GAMMA+KAPPA)),
             'k-', lw=2, label='Analytic')
    ax4.plot(OMEGA_VALUES, ss_photons, 'ro', ms=10, label='Simulation')
    ax4.set_xlabel('Drive $\\Omega$', fontsize=12)
    ax4.set_ylabel('$\\langle n\\rangle_{ss}$', fontsize=12)
    ax4.set_title('Saturation: $\\langle n\\rangle_{ss}\\propto\\Omega^2$',
                  fontsize=12)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    ax1.set_xlabel('Time'); ax1.set_ylabel('$\\langle a^\\dagger a\\rangle$')
    ax1.set_title('Photon buildup and saturation', fontsize=12)
    ax1.legend(fontsize=7); ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time'); ax2.set_ylabel('$P_e$')
    ax2.set_title('Qubit population under drive', fontsize=12)
    ax2.legend(fontsize=8); ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel('log Time'); ax3.set_ylabel('$\\langle a^\\dagger a\\rangle$')
    ax3.set_title('Buildup on log timescale', fontsize=12)
    ax3.legend(fontsize=7); ax3.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        rf'Driven JC: continuous pump, $g/\kappa$={G/KAPPA:.1f}, '
        rf'photon number saturates at $\langle n\rangle_{{ss}}\propto\Omega^2/\kappa$',
        fontsize=13)

    os.makedirs('../figures/single_trajectory', exist_ok=True)
    plt.savefig('../figures/single_trajectory/driven_JC_photons.png',
                dpi=150, bbox_inches='tight')
    print("\nFigure → figures/single_trajectory/driven_JC_photons.png")
    plt.show()


if __name__ == "__main__":
    run()
