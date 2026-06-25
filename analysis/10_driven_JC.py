"""
10_driven_JC.py
===============
Continuously driven Jaynes-Cummings: photon saturation vs drive strength.

PHYSICS: Continuous coherent drive on qubit → photons build up in
resonator → leak to ADC at rate kappa. Does <n> diverge? NO — it
saturates at:
    <n>_ss = g^2*Omega^2 / (4*(g^2 + gamma*kappa/4)^2)

This is the correct formula for qubit-mediated drive (NOT direct
resonator drive). The g^2 in numerator means: no coupling = no photons.

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
from utils.physics import n_thermal, make_params

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
WQ    = 1.0
WR    = 1.0
WD    = 1.0
G     = 0.05
GAMMA = 0.01
KAPPA = 0.01
N_TH_Q = 0.0        # n_thermal(5.0, 0.020) for realistic
N_TH_R = 0.0
GAMMA_PHI = 0.0     # set > 0 for pure dephasing, e.g. 0.01 for T2=T1

N_MAX   = 30
T_END   = 2000
N_STEPS = 1000

OMEGA_VALUES = [0.005, 0.01, 0.05, 0.1]
TAG = f"g{G:.3f}_kappa{KAPPA:.3f}_nth{N_TH_Q:.3f}"
# ──────────────────────────────────────────────────────────────────────────────


def n_ss_analytic(g, Omega, gamma, kappa):
    """Correct steady-state photon number for qubit-mediated drive."""
    return g**2 * Omega**2 / (4 * (g**2 + gamma*kappa/4)**2)


def build_H_driven(wq, wr, wd, g, Omega, N_max):
    """JC + coherent drive in rotating frame."""
    o  = build_operators(N_max)
    dq = wq - wd
    dr = wr - wd
    return ((dq/2)*o['sz'] + dr*o['n_r']
            + g*(o['a']*o['sp'] + o['adag']*o['sm'])
            + (Omega/2)*(o['sp'] + o['sm']))


def run():
    os.makedirs('../data/JC', exist_ok=True)
    os.makedirs('../figures/JC', exist_ok=True)

    o    = build_operators(N_MAX)
    psi0 = qt.ket2dm(qt.tensor(qt.basis(2,1), qt.basis(N_MAX+1,0)))
    t    = np.linspace(0, T_END, N_STEPS)

    # Jump operators with thermal and dephasing
    c_ops = [np.sqrt(GAMMA*(1+N_TH_Q))*o['sm']]
    if N_TH_Q > 0: c_ops.append(np.sqrt(GAMMA*N_TH_Q)*o['sp'])
    c_ops.append(np.sqrt(KAPPA*(1+N_TH_R))*o['a'])
    if N_TH_R > 0: c_ops.append(np.sqrt(KAPPA*N_TH_R)*o['adag'])
    if GAMMA_PHI > 0: c_ops.append(np.sqrt(GAMMA_PHI/2)*o['sz'])

    print(f"Driven JC: g={G}, kappa={KAPPA}, gamma={GAMMA}")
    print(f"  n_th_q={N_TH_Q}, n_th_r={N_TH_R}, gamma_phi={GAMMA_PHI}\n")
    print(f"{'Omega':>8} {'n_ss_num':>12} {'n_ss_an':>12} {'ratio':>8} {'P_e_ss':>8}")

    fig = plt.figure(figsize=(16,9))
    gs  = gridspec.GridSpec(2, 2, wspace=0.35, hspace=0.4)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])

    ss_photons  = []
    all_n       = {}
    all_Pe      = {}

    for Omega, color in zip(OMEGA_VALUES, ['C0','C1','C2','C3']):
        H = build_H_driven(WQ, WR, WD, G, Omega, N_MAX)
        r = qt.mesolve(H, psi0, t, c_ops,
                       e_ops=[o['sz'], o['n_r']],
                       options={"nsteps":10000})

        P_e   = (np.array(r.expect[0]) + 1) / 2
        n_ph  = np.array(r.expect[1])
        n_ss  = np.mean(n_ph[int(0.9*N_STEPS):])
        P_ss  = np.mean(P_e[int(0.9*N_STEPS):])
        n_an  = n_ss_analytic(G, Omega, GAMMA, KAPPA)

        ss_photons.append(n_ss)
        all_n[Omega] = n_ph
        all_Pe[Omega] = P_e

        if n_ph.max() > 0.8*N_MAX:
            print(f"  WARNING: max<n>={n_ph.max():.1f} near N_max={N_MAX}")

        ratio = n_ss/n_an if n_an > 0 else float('nan')
        print(f"{Omega:>8.4f} {n_ss:>12.5f} {n_an:>12.5f} {ratio:>8.4f} {P_ss:>8.4f}")

        lbl = f'$\\Omega$={Omega:.3f}  $\\langle n\\rangle_{{ss}}$={n_ss:.4f}'
        ax1.plot(t, n_ph, color=color, lw=2, label=lbl)
        ax2.plot(t, P_e,  color=color, lw=2, label=f'$\\Omega$={Omega:.3f}')
        ax3.semilogx(t[1:], n_ph[1:], color=color, lw=2, label=lbl)

    # Save data
    np.savez(f'../data/JC/driven_JC_{TAG}.npz',
             t=t, Omega_values=np.array(OMEGA_VALUES),
             ss_photons=np.array(ss_photons),
             g=G, kappa=KAPPA, gamma=GAMMA,
             n_th_q=N_TH_Q, n_th_r=N_TH_R, gamma_phi=GAMMA_PHI)
    print(f"\nData saved → data/JC/driven_JC_{TAG}.npz")

    # Analytic saturation curve
    Om_range = np.linspace(0.001, 0.15, 300)
    n_an_curve = n_ss_analytic(G, Om_range, GAMMA, KAPPA)
    ax4.plot(Om_range, n_an_curve, 'k-', lw=2,
             label=r'$g^2\Omega^2/4(g^2+\gamma\kappa/4)^2$')
    ax4.plot(OMEGA_VALUES, ss_photons, 'ro', ms=10, label='Simulation')
    ax4.set_xlabel('Drive $\\Omega$', fontsize=12)
    ax4.set_ylabel('$\\langle n\\rangle_{ss}$', fontsize=12)
    ax4.set_title('Saturation: $\\Omega^2$ scaling', fontsize=12)
    ax4.legend(fontsize=9); ax4.grid(True, alpha=0.3)

    for ax,title,ylabel in zip(
        [ax1,ax2,ax3],
        ['Photon buildup and saturation',
         'Qubit population under drive',
         'Photon buildup (log time)'],
        ['$\\langle a^\\dagger a\\rangle$','$P_e$',
         '$\\langle a^\\dagger a\\rangle$']
    ):
        ax.set_xlabel('Time'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax2.set_ylim(-0.02, 0.6)

    fig.suptitle(
        rf'Driven JC: $g/\kappa$={G/KAPPA:.1f}, '
        rf'$\gamma/\kappa$={GAMMA/KAPPA:.1f}, '
        rf'$n_{{th}}$={N_TH_Q}, $\gamma_\phi$={GAMMA_PHI}',
        fontsize=13)
    plt.savefig(f'../figures/JC/driven_JC_{TAG}.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved → figures/JC/driven_JC_{TAG}.png")
    plt.show()


if __name__ == "__main__":
    run()
