"""
jaynes_cummings.py
==================
Qubit coupled to a harmonic oscillator resonator (Jaynes-Cummings model).

DIFFERENT from qubit+TLS model:
  TLS model: qubit ⊗ TLS      (4-dim Hilbert space, both two-level)
  JC model:  qubit ⊗ resonator (2*(N_max+1) dim, resonator is bosonic)

Hamiltonian (RWA, hbar=1):
    H = (wq/2)*sz + wr*a†a + g*(a*sigma+ + a†*sigma-)
         qubit       resonator    exchange coupling

Dissipation channels:
    gamma    : qubit relaxation rate (1/T1_q)
    kappa    : resonator photon loss rate (leakage to transmission line)
    gamma_phi: qubit pure dephasing

Tracked observables:
    P_e_q          : qubit excited state population
    n_photons      : intracavity photon number <a†a>(t)
    photon_flux    : kappa*<a†a>(t)  [emitted photons/time, what ADC sees]
    photons_emitted: cumulative integral of flux = total photons out

Physical regimes (g/kappa):
    < 1  Bad cavity (Purcell): kappa >> g, qubit decays exponentially
                               Purcell rate: gamma_P = g^2/kappa
    ~ 1  Threshold: timescales comparable, onset of oscillations
    > 1  Strong coupling: vacuum Rabi oscillations in photon flux
    >> 1 Deep Rabi: many oscillations before photon escapes

Connection to Solomon breakdown:
    g/kappa plays the same role as g/gamma_t in the qubit+TLS model.
    Both mark the boundary between incoherent transfer (Solomon valid)
    and coherent oscillation (Solomon fails, Rabi regime).

Reference: Blais et al. Rev. Mod. Phys. 93, 025005 (2021), Sec. VI
"""

import numpy as np
import qutip as qt
from scipy.integrate import cumulative_trapezoid
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.io import save


def build_operators(N_max: int = 10) -> dict:
    """
    Build all operators for the qubit ⊗ resonator Hilbert space.

    Convention: qt.tensor(qubit_op, resonator_op)
    Basis: {|e,0>, |e,1>, ..., |e,N>, |g,0>, ..., |g,N>}

    Parameters
    ----------
    N_max : int
        Maximum photon number (Fock space truncation).
        WARNING: if <a†a> approaches N_max during simulation,
        increase N_max — truncation causes unphysical reflection.
        Rule of thumb: N_max >= 3 * max expected <a†a>.
    """
    I_q = qt.qeye(2)
    I_r = qt.qeye(N_max + 1)

    sz   = qt.tensor(qt.sigmaz(), I_r)
    sp   = qt.tensor(qt.sigmap(), I_r)
    sm   = qt.tensor(qt.sigmam(), I_r)
    a    = qt.tensor(I_q, qt.destroy(N_max + 1))
    adag = a.dag()
    n_r  = adag * a

    return {'sz': sz, 'sp': sp, 'sm': sm,
            'a': a, 'adag': adag, 'n_r': n_r}


def build_H_JC(wq: float, wr: float, g: float,
               N_max: int = 10) -> qt.Qobj:
    """
    Jaynes-Cummings Hamiltonian under RWA.

    H = (wq/2)*sz + wr*a†a + g*(a*sigma+ + a†*sigma-)

    The interaction a†*sigma- raises the resonator (creates a photon)
    while lowering the qubit (qubit emits). a*sigma+ does the reverse.
    Energy is conserved: excitations swap between qubit and resonator.
    """
    o = build_operators(N_max)
    return ((wq / 2) * o['sz']
            + wr * o['n_r']
            + g * (o['a'] * o['sp'] + o['adag'] * o['sm']))


def initial_state(qubit: str = 'e', n_photons: int = 0,
                  N_max: int = 10) -> qt.Qobj:
    """
    Build initial density matrix as a product state.

    This represents an INSTANTANEOUS initialization at t=0:
    no drive pulse is simulated. Physically this corresponds to
    an ideal pi-pulse that is much faster than 1/g (fast init limit).

    Parameters
    ----------
    qubit     : 'e' (excited) or 'g' (ground)
    n_photons : initial resonator Fock state (0 = vacuum)
    N_max     : Fock truncation
    """
    q_basis = {'e': qt.basis(2, 0), 'g': qt.basis(2, 1)}
    psi0 = qt.tensor(q_basis[qubit], qt.basis(N_max + 1, n_photons))
    return qt.ket2dm(psi0)


def purcell_rate(g: float, kappa: float, delta: float = 0.0) -> float:
    """
    Analytic Purcell decay rate in the bad-cavity limit (kappa >> g).

    gamma_Purcell = g^2 * kappa / (delta^2 + (kappa/2)^2)

    At resonance (delta=0): gamma_P = 4*g^2/kappa
    In dispersive limit:    gamma_P = g^2*kappa/delta^2

    This is the enhanced qubit decay rate caused by the resonator
    providing an additional decay channel. Solomon is valid in this
    regime (kappa >> g, i.e. g/kappa << 1).
    """
    return g**2 * kappa / (delta**2 + (kappa / 2)**2)


def evolve(
    wq:             float,
    wr:             float,
    g:              float,
    gamma:          float,
    kappa:          float,
    gamma_phi:      float = 0.0,
    n_th_q:         float = 0.0,
    n_th_r:         float = 0.0,
    N_max:          int   = 10,
    t_end:          float = None,
    n_steps:        int   = 500,
    qubit_init:     str   = 'e',
    n_photons_init: int   = 0,
    save_path:      str   = None,
) -> dict:
    """
    Solve the Jaynes-Cummings master equation.

    Parameters
    ----------
    wq, wr      : qubit and resonator frequencies
    g           : qubit-resonator coupling strength
    gamma       : qubit relaxation rate (1/T1_q)
    kappa       : resonator photon loss rate
    gamma_phi   : qubit pure dephasing (default 0)
    n_th_q      : mean thermal photon number at qubit frequency
    n_th_r      : mean thermal photon number at resonator frequency
    N_max       : Fock space truncation — increase if n_photons -> N_max
    t_end       : total simulation time (default 8/gamma)
    n_steps     : number of output time points
    qubit_init  : 'e' or 'g' — initial qubit state
    n_photons_init : initial resonator Fock state

    Returns
    -------
    dict:
        't'              : time array
        'P_e_q'          : qubit excited state population
        'n_photons'      : intracavity <a†a>(t)
        'photon_flux'    : kappa*<a†a>(t) — emitted photon rate
        'photons_emitted': cumulative total photons emitted
        'coherence_qr'   : |<a*sigma+>|(t) — qubit-resonator coherence
        'Purcell_rate'   : analytic Purcell rate (bad-cavity limit)
        'delta'          : detuning wq - wr
        'params'         : all input parameters
    """
    if t_end is None:
        t_end = 8.0 / max(gamma, 1e-9)

    delta = wq - wr

    H    = build_H_JC(wq, wr, g, N_max)
    o    = build_operators(N_max)
    t    = np.linspace(0, t_end, n_steps)
    rho0 = initial_state(qubit_init, n_photons_init, N_max)

    # ── Jump operators ────────────────────────────────────────────────────────
    c_ops = []

    # Qubit: decay + thermal excitation
    c_ops.append(np.sqrt(gamma * (1.0 + n_th_q)) * o['sm'])
    if n_th_q > 0:
        c_ops.append(np.sqrt(gamma * n_th_q) * o['sp'])

    # Resonator: photon loss + thermal creation
    c_ops.append(np.sqrt(kappa * (1.0 + n_th_r)) * o['a'])
    if n_th_r > 0:
        c_ops.append(np.sqrt(kappa * n_th_r) * o['adag'])

    # Qubit pure dephasing
    if gamma_phi > 0:
        c_ops.append(np.sqrt(gamma_phi / 2.0) * o['sz'])

    # ── Solve ─────────────────────────────────────────────────────────────────
    result = qt.mesolve(
        H, rho0, t, c_ops,
        e_ops=[o['sz'], o['n_r']],
        options={"nsteps": 10000, "store_states": True},
    )

    P_e_q       = (np.array(result.expect[0]) + 1.0) / 2.0
    n_photons   = np.array(result.expect[1])
    photon_flux = kappa * n_photons

    # Cumulative emitted photons (integral of flux)
    photons_emitted = cumulative_trapezoid(photon_flux, t, initial=0)

    # ── N_max saturation check ────────────────────────────────────────────────
    if n_photons.max() > 0.8 * N_max:
        print(f"WARNING: max <n> = {n_photons.max():.2f} is close to "
              f"N_max={N_max}. Increase N_max to avoid truncation artifacts.")

    # ── Qubit-resonator coherence |<a*sigma+>| ────────────────────────────────
    op_qr = o['a'] * o['sp']
    coherence_qr = np.array([
        abs(qt.expect(op_qr, rho)) for rho in result.states
    ])

    out = {
        't':               t,
        'P_e_q':           P_e_q,
        'n_photons':       n_photons,
        'photon_flux':     photon_flux,
        'photons_emitted': photons_emitted,
        'coherence_qr':    coherence_qr,
        'states':          result.states,
        'Purcell_rate':    purcell_rate(g, kappa, delta),
        'delta':           delta,
        'params': {
            'wq': wq, 'wr': wr, 'g': g, 'delta': delta,
            'gamma': gamma, 'kappa': kappa, 'gamma_phi': gamma_phi,
            'n_th_q': n_th_q, 'n_th_r': n_th_r,
            'N_max': N_max, 'qubit_init': qubit_init,
            'g_over_kappa': g / kappa,
            'Purcell_rate': purcell_rate(g, kappa, delta),
        },
    }

    if save_path:
        save(out, save_path)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Sweep g/kappa from 0.5 (Solomon breakdown onset) upward
    base = dict(wq=1.0, wr=1.0, gamma=0.01, kappa=0.01,
                N_max=20, t_end=800, n_steps=800)

    scenarios = [
        (0.005, 'Bad cavity   $g/\\kappa=0.5$',  'C0'),
        (0.01,  'Threshold    $g/\\kappa=1.0$',  'C1'),
        (0.05,  'Strong coup  $g/\\kappa=5.0$',  'C2'),
        (0.10,  'Deep Rabi    $g/\\kappa=10.0$', 'C3'),
    ]

    fig = plt.figure(figsize=(15, 9))
    gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.4)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    print(f"{'g':>8} {'g/kappa':>10} {'Purcell':>12} {'max_n':>8} {'tot_phot':>10}")

    for g, label, color in scenarios:
        res = evolve(**base, g=g)
        t   = res['t']
        gk  = g / base['kappa']

        print(f"{g:>8.4f} {gk:>10.2f} "
              f"{res['Purcell_rate']:>12.6f} "
              f"{res['n_photons'].max():>8.4f} "
              f"{res['photons_emitted'][-1]:>10.4f}")

        axes[0].plot(t, res['P_e_q'],          color=color, lw=2, label=label)
        axes[1].plot(t, res['n_photons'],       color=color, lw=2, label=label)
        axes[2].plot(t, res['photon_flux'],     color=color, lw=2, label=label)
        axes[3].plot(t, res['photons_emitted'], color=color, lw=2, label=label)
        axes[4].semilogy(t,
                         np.where(res['coherence_qr']>0,
                                  res['coherence_qr'], 1e-16),
                         color=color, lw=2, label=label)

    titles  = ['Qubit $P_e(t)$',
               r'Intracavity $\langle a^\dagger a\rangle$',
               r'Photon flux $\kappa\langle a^\dagger a\rangle$',
               'Cumulative emitted photons',
               r'Q-R coherence $|\langle a\sigma_+\rangle|$']
    ylabels = ['$P_e$', '$\\langle n\\rangle$', 'Flux',
               'Total photons', 'Coherence']

    for ax, title, ylabel in zip(axes[:5], titles, ylabels):
        ax.set_xlabel('Time'); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[5].axis('off')
    info = ("PURCELL RATES\n\n"
            + "\n".join([f"g/κ={g/base['kappa']:.1f}:  γ_P={purcell_rate(g,base['kappa']):.5f}"
                         for g, _, _ in scenarios]))
    axes[5].text(0.05, 0.95, info, transform=axes[5].transAxes,
                fontsize=10, va='top', fontfamily='monospace')

    plt.suptitle(
        r'Jaynes-Cummings: qubit $|e\rangle$ + resonator $|0\rangle$, '
        r'sweeping $g/\kappa$ from Solomon threshold upward',
        fontsize=13)
    plt.savefig('../../figures/single_trajectory/jaynes_cummings_regimes.png',
                dpi=150, bbox_inches='tight')
    plt.show()
