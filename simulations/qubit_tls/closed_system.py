"""
closed_system.py
================
Closed-system (unitary) time evolution of the qubit + TLS.

No dissipation here. The state evolves as:
    |psi(t)> = U(t)|psi(0)>,   U(t) = exp(-i*H*t)

This corresponds to the theoretical result derived in the notes:

    P_e(t) = 1 - (g/Omega_R)^2 * sin^2(Omega_R * t)

where Omega_R = (1/2)*sqrt(Delta^2 + 4*g^2) is the generalised Rabi
frequency and Delta = wq - wt is the detuning.

We use QuTiP's sesolve() (Schrodinger equation solver) which integrates
    i * d/dt |psi> = H |psi>
using an adaptive ODE integrator. This is exact up to numerical precision.

Reference: Haroche & Raimond, Exploring the Quantum (2006), Ch. 3
"""

import numpy as np
import qutip as qt
from hamiltonian import build_H, state


# ─── Analytic result (for validation) ────────────────────────────────────────

def rabi_analytic(t: np.ndarray, wq: float, wt: float, g: float) -> np.ndarray:
    """
    Analytic qubit excited-state population starting from |eg>.

    P_e(t) = 1 - (g/Omega_R)^2 * sin^2(Omega_R * t)

    Parameters
    ----------
    t   : array of times
    wq  : qubit frequency
    wt  : TLS frequency
    g   : coupling

    Returns
    -------
    P_e : array, same shape as t
    """
    delta   = wq - wt
    Omega_R = 0.5 * np.sqrt(delta**2 + 4 * g**2)
    return 1.0 - (g / Omega_R)**2 * np.sin(Omega_R * t)**2


# ─── Numerical solver ─────────────────────────────────────────────────────────

def evolve(
    wq:    float,
    wt:    float,
    g:     float,
    t_end: float,
    n_steps: int = 500,
    psi0:  qt.Qobj = None,
) -> dict:
    """
    Solve the Schrodinger equation for qubit + TLS (no dissipation).

    Parameters
    ----------
    wq      : qubit frequency
    wt      : TLS frequency
    g       : coupling strength
    t_end   : total evolution time
    n_steps : number of time points
    psi0    : initial state (default: |eg>, qubit excited TLS ground)

    Returns
    -------
    dict with keys:
        't'        : time array
        'P_e_q'    : qubit excited-state population <e|rho_q|e>
        'P_e_t'    : TLS excited-state population
        'coherence': |<eg|psi>|^2 - population in |eg> state
        'states'   : list of qt.Qobj states at each time step
        'P_e_analytic': analytic result for |eg> initial state
    """
    if psi0 is None:
        psi0 = state('e', 'g')   # qubit excited, TLS ground

    H  = build_H(wq, wt, g)
    t  = np.linspace(0, t_end, n_steps)

    # sesolve: Schrodinger equation solver
    # e_ops: list of operators whose expectation values to track
    sz_q = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz_t = qt.tensor(qt.qeye(2), qt.sigmaz())

    result = qt.sesolve(
        H,
        psi0,
        t,
        e_ops=[sz_q, sz_t],   # track <sz_q> and <sz_t>
        options={"nsteps": 10000},
    )

    # Convert <sz> to population: P_e = (<sz> + 1) / 2
    P_e_q = (np.array(result.expect[0]) + 1) / 2
    P_e_t = (np.array(result.expect[1]) + 1) / 2

    return {
        't':            t,
        'P_e_q':        P_e_q,
        'P_e_t':        P_e_t,
        'states':       result.states,
        'P_e_analytic': rabi_analytic(t, wq, wt, g),
        'params':       {'wq': wq, 'wt': wt, 'g': g},
    }


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # --- Resonant case: Delta = 0, full Rabi oscillations ---
    print("Running resonant case (Delta=0)...")
    res = evolve(wq=1.0, wt=1.0, g=0.1, t_end=4*np.pi/0.1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: resonant
    ax = axes[0]
    ax.plot(res['t'], res['P_e_q'],        label=r'Qubit $P_e$ (numeric)', lw=2)
    ax.plot(res['t'], res['P_e_t'],        label=r'TLS $P_e$ (numeric)',   lw=2, ls='--')
    ax.plot(res['t'], res['P_e_analytic'], label='Analytic',               lw=1, ls=':', color='k')
    ax.set_title(r'Resonant: $\Delta=0$, $g=0.1$')
    ax.set_xlabel('Time')
    ax.set_ylabel('Excited state population')
    ax.legend()
    ax.set_ylim(-0.05, 1.05)

    # Right: dispersive case
    print("Running dispersive case (Delta >> g)...")
    res2 = evolve(wq=1.0, wt=0.5, g=0.05, t_end=4*np.pi/0.05)

    ax2 = axes[1]
    ax2.plot(res2['t'], res2['P_e_q'],        label=r'Qubit $P_e$ (numeric)', lw=2)
    ax2.plot(res2['t'], res2['P_e_t'],        label=r'TLS $P_e$ (numeric)',   lw=2, ls='--')
    ax2.plot(res2['t'], res2['P_e_analytic'], label='Analytic',               lw=1, ls=':', color='k')
    ax2.set_title(r'Dispersive: $\Delta=0.5$, $g=0.05$')
    ax2.set_xlabel('Time')
    ax2.legend()
    ax2.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig('../../figures/closed_system_rabi.png', dpi=150)
    print("Saved to figures/closed_system_rabi.png")
    plt.show()
