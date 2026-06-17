"""
lindblad.py
===========
Open-system evolution via the Lindblad master equation,
with optional thermal population at finite temperature.

PHYSICS SUMMARY
---------------
The Lindblad master equation governs how a density matrix rho evolves
when the system is coupled to an environment (bath):

    drho/dt = -i[H, rho]                          <- coherent evolution
              + sum_k D[L_k](rho)                  <- dissipation

where the dissipator is:
    D[L](rho) = L*rho*L† - (1/2){L†L, rho}

AT ZERO TEMPERATURE (n_th = 0):
    Each system can only lose energy to the bath.
    Jump operators: sqrt(gamma) * sigma_-  (decay only)
    Steady state: |gg><gg| (both ground)

AT FINITE TEMPERATURE (n_th > 0):
    The bath also drives the system upward (thermal excitation).
    Each decay channel gamma splits into TWO processes:

        DECAY:      rate = gamma * (1 + n_th)    operator = sigma_-
        EXCITATION: rate = gamma * n_th           operator = sigma_+

    Physical meaning:
        - (1 + n_th): stimulated + spontaneous emission
        - n_th: absorption from thermal bath

    Steady-state excited population:
        P_e^ss = n_th / (2*n_th + 1)

    Examples:
        n_th = 0    -> P_e^ss = 0      (zero temperature)
        n_th = 0.01 -> P_e^ss = 0.005  (cold: ~20 mK superconducting qubit)
        n_th = 0.1  -> P_e^ss = 0.083  (warm)
        n_th -> inf -> P_e^ss -> 0.5   (infinite temperature, fully mixed)

Reference: Breuer & Petruccione (2002), Ch. 3
           Blais et al. Rev. Mod. Phys. 93, 025005 (2021), Sec. IV.C
"""

import numpy as np
import qutip as qt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qubit_tls.hamiltonian import build_H, state, ops
from utils.io import save


# ─── Thermal helpers ──────────────────────────────────────────────────────────

def n_thermal(freq_GHz: float, temp_K: float) -> float:
    """
    Mean thermal photon number at given frequency and temperature.

    n_th = 1 / (exp(hbar*omega / kB*T) - 1)

    The conversion hbar/kB = 47.99 mK/GHz means:
        hbar*omega/kB [K] = freq_GHz * 47.99e-3

    Parameters
    ----------
    freq_GHz : float  transition frequency in GHz
    temp_K   : float  temperature in Kelvin

    Returns
    -------
    n_th : float  (dimensionless, >= 0)

    Examples
    --------
    n_thermal(5.0, 0.020) -> 0.0049  (5 GHz qubit at 20 mK)
    n_thermal(5.0, 0.100) -> 0.078   (5 GHz qubit at 100 mK)
    n_thermal(1.0, 0.050) -> 0.181   (1 GHz TLS at 50 mK)
    """
    if temp_K <= 0:
        return 0.0
    x = freq_GHz * 47.99e-3 / temp_K  # hbar*omega / kB*T
    return 1.0 / (np.exp(x) - 1.0)


def thermal_steady_state(n_th: float) -> float:
    """
    Excited-state population at thermal equilibrium.

    P_e^ss = n_th / (2*n_th + 1)

    Derivation: at steady state dP/dt = 0 with decay rate gamma*(1+n_th)
    and excitation rate gamma*n_th gives this Boltzmann result.

    Limits:
        n_th = 0   -> 0.0   (ground state at T=0)
        n_th = 0.1 -> 0.083
        n_th -> inf -> 0.5  (fully mixed at T=inf)
    """
    return n_th / (2.0 * n_th + 1.0)


# ─── Jump operators ───────────────────────────────────────────────────────────

def build_jump_ops(
    gamma_q:     float = 0.0,
    gamma_t:     float = 0.0,
    gamma_phi_q: float = 0.0,
    gamma_phi_t: float = 0.0,
    n_th_q:      float = 0.0,
    n_th_t:      float = 0.0,
) -> list:
    """
    Build Lindblad jump operators with thermal population.

    At n_th=0 (zero temperature):
        Only decay operators: sqrt(gamma) * sigma_-

    At n_th>0 (finite temperature):
        Decay:      sqrt(gamma * (1 + n_th)) * sigma_-
        Excitation: sqrt(gamma * n_th)       * sigma_+

    WHY sqrt(rate)?
    QuTiP computes D[L](rho) = L*rho*L† - (1/2){L†L, rho}.
    The rate appears as L†L, so L = sqrt(rate)*operator gives
    the correct rate in the dissipator.

    Parameters
    ----------
    gamma_q, gamma_t     : relaxation rates (1/T1)
    gamma_phi_q, _t      : pure dephasing rates
    n_th_q, n_th_t       : thermal photon numbers (0 = zero temperature)

    Returns
    -------
    list of qt.Qobj jump operators
    """
    o = ops()
    c_ops = []

    # ── Qubit: decay + thermal excitation ─────────────────────────────────────
    if gamma_q > 0:
        # DECAY: qubit falls from |e> to |g>, emitting a photon
        # Rate = gamma_q * (1 + n_th_q)
        # At n_th=0: just gamma_q (standard T1)
        # At n_th>0: faster because bath stimulates additional emission
        c_ops.append(np.sqrt(gamma_q * (1.0 + n_th_q)) * o['sm_q'])

        # EXCITATION: qubit absorbs a thermal photon, |g> -> |e>
        # Rate = gamma_q * n_th_q
        # Only exists at finite temperature
        # Uses sigma_+ (raising operator) because population goes UP
        if n_th_q > 0:
            c_ops.append(np.sqrt(gamma_q * n_th_q) * o['sp_q'])

    # ── TLS: decay + thermal excitation ───────────────────────────────────────
    if gamma_t > 0:
        # Same structure as qubit.
        # n_th_t can differ from n_th_q if TLS frequency differs from qubit,
        # since n_th = 1/(exp(hbar*omega/kT)-1) depends on omega.
        c_ops.append(np.sqrt(gamma_t * (1.0 + n_th_t)) * o['sm_t'])
        if n_th_t > 0:
            c_ops.append(np.sqrt(gamma_t * n_th_t) * o['sp_t'])

    # ── Pure dephasing: temperature independent ────────────────────────────────
    # Dephasing destroys coherences (off-diagonal rho elements)
    # without changing populations (diagonal elements).
    # It is NOT modified by temperature in this model —
    # temperature affects energy relaxation, not pure dephasing.
    # Uses sigma_z because it commutes with sigma_z*rho*sigma_z = rho
    # for diagonal elements, but flips sign of off-diagonal elements.
    if gamma_phi_q > 0:
        c_ops.append(np.sqrt(gamma_phi_q / 2.0) * o['sz_q'])
    if gamma_phi_t > 0:
        c_ops.append(np.sqrt(gamma_phi_t / 2.0) * o['sz_t'])

    return c_ops


# ─── Main solver ──────────────────────────────────────────────────────────────

def evolve(
    wq:          float,
    wt:          float,
    g:           float,
    gamma_q:     float,
    gamma_t:     float,
    gamma_phi_q: float = 0.0,
    gamma_phi_t: float = 0.0,
    n_th_q:      float = 0.0,
    n_th_t:      float = 0.0,
    t_end:       float = None,
    n_steps:     int   = 500,
    rho0:        qt.Qobj = None,
    save_path:   str   = None,
) -> dict:
    """
    Solve the Lindblad master equation for qubit + TLS.

    Parameters
    ----------
    wq, wt      : transition frequencies
    g           : coupling strength
    gamma_q     : qubit relaxation rate (1/T1_q)
    gamma_t     : TLS relaxation rate   (1/T1_t)
    gamma_phi_q : qubit pure dephasing (default 0)
    gamma_phi_t : TLS pure dephasing   (default 0)
    n_th_q      : thermal photon number at qubit frequency (default 0)
    n_th_t      : thermal photon number at TLS frequency   (default 0)
                  Compute from physical units: n_thermal(freq_GHz, temp_K)
    t_end       : total time (default: 8/gamma_q)
    n_steps     : time points
    rho0        : initial density matrix (default: |eg><eg|)
    save_path   : if given, save to .h5 file

    Returns
    -------
    dict:
        't'         : time array
        'P_e_q'     : qubit excited-state population
        'P_e_t'     : TLS excited-state population
        'coherence' : |rho_{eg,ge}(t)|
        'states'    : list of density matrices rho(t)
        'P_e_q_ss'  : expected thermal steady state (qubit)
        'P_e_t_ss'  : expected thermal steady state (TLS)
        'params'    : all input parameters
    """
    if rho0 is None:
        rho0 = qt.ket2dm(state('e', 'g'))

    if t_end is None:
        t_end = 8.0 / max(gamma_q, 1e-9)

    H     = build_H(wq, wt, g)
    c_ops = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t,
                           n_th_q, n_th_t)
    t     = np.linspace(0, t_end, n_steps)
    o     = ops()

    # mesolve: master equation solver
    # Integrates the full Lindblad equation numerically
    # e_ops: track <sz_q> and <sz_t> at every time step
    # store_states=True: keep full rho(t) — needed for coherence extraction
    result = qt.mesolve(
        H, rho0, t, c_ops,
        e_ops=[o['sz_q'], o['sz_t']],
        options={"nsteps": 10000, "store_states": True},
    )

    # Convert <sz> in [-1,+1] to population in [0,1]
    # P_e = (<sz> + 1) / 2
    P_e_q = (np.array(result.expect[0]) + 1.0) / 2.0
    P_e_t = (np.array(result.expect[1]) + 1.0) / 2.0

    # Extract off-diagonal coherence |rho_{eg,ge}|
    # Basis ordering: |ee>=0, |eg>=1, |ge>=2, |gg>=3
    # rho[1,2] is the coherence between |eg> and |ge>
    # This is what Solomon sets to zero — large values = Solomon invalid
    coherence = np.array([
        abs(rho.full()[1, 2]) for rho in result.states
    ])

    out = {
        't':         t,
        'P_e_q':     P_e_q,
        'P_e_t':     P_e_t,
        'coherence': coherence,
        'states':    result.states,
        'P_e_q_ss':  thermal_steady_state(n_th_q),
        'P_e_t_ss':  thermal_steady_state(n_th_t),
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'gamma_phi_q': gamma_phi_q, 'gamma_phi_t': gamma_phi_t,
            'n_th_q': n_th_q, 'n_th_t': n_th_t,
            't_end': t_end,
        },
    }

    if save_path:
        save(out, save_path)

    return out


# ─── Steady state ─────────────────────────────────────────────────────────────

def steady_state(
    wq: float, wt: float, g: float,
    gamma_q: float, gamma_t: float,
    gamma_phi_q: float = 0.0, gamma_phi_t: float = 0.0,
    n_th_q: float = 0.0, n_th_t: float = 0.0,
) -> qt.Qobj:
    """
    Compute the steady-state density matrix rho_ss (drho/dt = 0).

    At n_th=0: rho_ss approaches |gg><gg|
    At n_th>0: rho_ss is mixed with P_e^ss = n_th/(2*n_th+1)

    Useful for:
      1. Verifying the solver reaches the correct steady state
      2. Using as initial state for a 'pre-thermalised' simulation
    """
    H     = build_H(wq, wt, g)
    c_ops = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t,
                           n_th_q, n_th_t)
    return qt.steadystate(H, c_ops)


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)

    scenarios = [
        (0.0,  'n_th=0 (T=0)',      'C0'),
        (0.01, 'n_th=0.01 (cold)',  'C1'),
        (0.1,  'n_th=0.1  (warm)',  'C2'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for n_th, label, color in scenarios:
        res  = evolve(**base, n_th_q=n_th, n_th_t=n_th, t_end=600, n_steps=800)
        P_ss = thermal_steady_state(n_th)

        axes[0].plot(res['t'], res['P_e_q'], label=label, color=color, lw=2)
        axes[0].axhline(P_ss, color=color, ls=':', lw=1)

        axes[1].plot(res['t'], res['P_e_t'], label=label, color=color, lw=2)

        coh = np.where(res['coherence'] > 0, res['coherence'], 1e-16)
        axes[2].semilogy(res['t'], coh, label=label, color=color, lw=2)

        print(f"n_th={n_th:.2f}: final P_e_q={res['P_e_q'][-1]:.5f}, "
              f"expected={P_ss:.5f}")

    for ax, title in zip(axes, ['Qubit P_e (dotted=steady state)',
                                  'TLS P_e', 'Coherence (log)']):
        ax.set_xlabel('Time'); ax.set_title(title); ax.legend(fontsize=9)
    axes[0].set_ylim(-0.02, 1.05)
    axes[1].set_ylim(-0.02, 1.05)

    plt.suptitle('Lindblad: thermal population effect', fontsize=13)
    plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/lindblad_thermal.png', dpi=150)
    plt.show()
