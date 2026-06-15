"""
lindblad.py
===========
Open-system evolution of the qubit + TLS via the Lindblad master equation.

The full equation is (hbar=1):

    drho/dt = -i[H, rho]
              + gamma_q  * D[sm_q](rho)     # qubit T1
              + gamma_t  * D[sm_t](rho)     # TLS T1
              + gamma_phi_q/2 * D[sz_q](rho)  # qubit pure dephasing
              + gamma_phi_t/2 * D[sz_t](rho)  # TLS pure dephasing

where the Lindblad dissipator is:
    D[L](rho) = L*rho*L† - (1/2)*{L†*L, rho}

This promotes the closed-system state vector |psi> to a density matrix
rho (4x4 complex matrix for our 4-dimensional Hilbert space).

Physical meaning of each rate:
    gamma_q   = 1/T1_q    : qubit energy relaxation
    gamma_t   = 1/T1_t    : TLS energy relaxation
    gamma_phi_q = 2/T_phi_q : qubit pure dephasing (T2 decay beyond T1)
    gamma_phi_t = 2/T_phi_t : TLS pure dephasing

The total dephasing rate 1/T2 = 1/(2*T1) + 1/T_phi.

Reference: Breuer & Petruccione, The Theory of Open Quantum Systems (2002)
           Blais et al., Rev. Mod. Phys. 93, 025005 (2021), Sec. IV.C
"""

import numpy as np
import qutip as qt
from hamiltonian import build_H, state, ops


# ─── Jump operators ───────────────────────────────────────────────────────────

def build_jump_ops(
    gamma_q:     float = 0.0,
    gamma_t:     float = 0.0,
    gamma_phi_q: float = 0.0,
    gamma_phi_t: float = 0.0,
) -> list:
    """
    Build the list of Lindblad jump operators.

    Parameters
    ----------
    gamma_q     : qubit relaxation rate (1/T1_q)
    gamma_t     : TLS relaxation rate   (1/T1_t)
    gamma_phi_q : qubit pure dephasing rate (2/T_phi_q)
    gamma_phi_t : TLS pure dephasing rate

    Returns
    -------
    List of qt.Qobj jump operators L_k (already scaled by sqrt(rate)).
    QuTiP expects: D[L_k](rho) with L_k = sqrt(rate) * operator.
    """
    o = ops()
    c_ops = []

    # sqrt(rate) * operator — QuTiP convention
    if gamma_q > 0:
        c_ops.append(np.sqrt(gamma_q) * o['sm_q'])

    if gamma_t > 0:
        c_ops.append(np.sqrt(gamma_t) * o['sm_t'])

    if gamma_phi_q > 0:
        # Pure dephasing: D[sz/2] — factor of 1/2 included here
        c_ops.append(np.sqrt(gamma_phi_q / 2) * o['sz_q'])

    if gamma_phi_t > 0:
        c_ops.append(np.sqrt(gamma_phi_t / 2) * o['sz_t'])

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
    t_end:       float = None,
    n_steps:     int   = 500,
    rho0:        qt.Qobj = None,
) -> dict:
    """
    Solve the Lindblad master equation for qubit + TLS.

    Parameters
    ----------
    wq, wt      : qubit and TLS frequencies
    g           : coupling
    gamma_q     : qubit relaxation rate
    gamma_t     : TLS relaxation rate
    gamma_phi_q : qubit pure dephasing (default 0)
    gamma_phi_t : TLS pure dephasing (default 0)
    t_end       : total time (default: 5 / min(gamma_q, gamma_t))
    n_steps     : number of time points
    rho0        : initial density matrix (default: |eg><eg|)

    Returns
    -------
    dict with keys:
        't'          : time array
        'P_e_q'      : qubit excited-state population
        'P_e_t'      : TLS excited-state population
        'coherence'  : |rho_{eg,ge}|, off-diagonal coherence magnitude
        'states'     : list of density matrices rho(t)
        'params'     : input parameters
    """
    # Default initial state: qubit excited, TLS ground
    if rho0 is None:
        psi0 = state('e', 'g')
        rho0 = qt.ket2dm(psi0)   # |eg><eg|

    # Default time: run until both systems have decayed (~5 lifetimes)
    if t_end is None:
        min_rate = max(gamma_q, gamma_t, 1e-6)
        t_end = 5.0 / min_rate

    H      = build_H(wq, wt, g)
    c_ops  = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t)
    t      = np.linspace(0, t_end, n_steps)

    # Operators to track expectation values
    o      = ops()
    e_ops  = [o['sz_q'], o['sz_t']]

    # mesolve: master equation solver (Lindblad)
    result = qt.mesolve(
        H,
        rho0,
        t,
        c_ops,
        e_ops,
        options={"nsteps": 10000},
    )

    # <sz> to population
    P_e_q = (np.array(result.expect[0]) + 1) / 2
    P_e_t = (np.array(result.expect[1]) + 1) / 2

    # Extract coherence |rho_{eg,ge}| from density matrices
    # Basis ordering in QuTiP tensor: |ee>=0, |eg>=1, |ge>=2, |gg>=3
    coherence = np.array([
        abs(rho.full()[1, 2]) for rho in result.states
    ])

    return {
        't':         t,
        'P_e_q':     P_e_q,
        'P_e_t':     P_e_t,
        'coherence': coherence,
        'states':    result.states,
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'gamma_phi_q': gamma_phi_q, 'gamma_phi_t': gamma_phi_t,
        },
    }


# ─── Steady state ─────────────────────────────────────────────────────────────

def steady_state(
    wq: float, wt: float, g: float,
    gamma_q: float, gamma_t: float,
    gamma_phi_q: float = 0.0, gamma_phi_t: float = 0.0,
) -> qt.Qobj:
    """
    Compute the steady-state density matrix rho_ss such that drho/dt = 0.

    For a system with only decay (no coherent driving), the steady state
    is always |gg><gg| (both ground). This function is useful to verify
    the solver is working correctly.
    """
    H     = build_H(wq, wt, g)
    c_ops = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t)
    return qt.steadystate(H, c_ops)


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters: resonant, weakly damped
    params = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)

    print("Running Lindblad solver...")
    res = evolve(**params, t_end=200)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: populations
    ax = axes[0]
    ax.plot(res['t'], res['P_e_q'], label='Qubit $P_e$', lw=2)
    ax.plot(res['t'], res['P_e_t'], label='TLS $P_e$',   lw=2, ls='--')
    ax.set_xlabel('Time')
    ax.set_ylabel('Excited state population')
    ax.set_title(f"Lindblad: $g$={params['g']}, "
                 f"$\\gamma_q$={params['gamma_q']}, "
                 f"$\\gamma_t$={params['gamma_t']}")
    ax.legend()

    # Right: coherence decay
    ax2 = axes[1]
    ax2.plot(res['t'], res['coherence'], color='purple', lw=2)
    ax2.set_xlabel('Time')
    ax2.set_ylabel(r'$|\rho_{eg,ge}|$')
    ax2.set_title('Off-diagonal coherence')
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('../../figures/lindblad_evolution.png', dpi=150)
    print("Saved to figures/lindblad_evolution.png")
    plt.show()

    # Verify steady state
    rho_ss = steady_state(**params)
    print(f"\nSteady state (should be |gg><gg|):")
    print(np.round(rho_ss.full(), 4))
