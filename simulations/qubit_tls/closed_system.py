"""
closed_system.py
================
Closed-system (unitary) time evolution of the qubit + TLS.

No dissipation. State evolves as:
    |psi(t)> = U(t)|psi(0)>,   U(t) = exp(-i*H*t)

Analytic result for initial state |eg>:
    P_e(t) = 1 - (g/Omega_R)^2 * sin^2(Omega_R * t)
    Omega_R = (1/2) * sqrt(Delta^2 + 4*g^2)

Reference: Haroche & Raimond, Exploring the Quantum (2006), Ch. 3
"""

import numpy as np
import qutip as qt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qubit_tls.hamiltonian import build_H, state
from utils.io import save


# ─── Analytic result ──────────────────────────────────────────────────────────

def rabi_analytic(t: np.ndarray, wq: float, wt: float, g: float) -> np.ndarray:
    """
    Analytic qubit excited-state population starting from |eg>.
    P_e(t) = 1 - (g/Omega_R)^2 * sin^2(Omega_R * t)
    """
    delta   = wq - wt
    Omega_R = 0.5 * np.sqrt(delta**2 + 4 * g**2)
    return 1.0 - (g / Omega_R)**2 * np.sin(Omega_R * t)**2


# ─── Solver ───────────────────────────────────────────────────────────────────

def evolve(
    wq:      float,
    wt:      float,
    g:       float,
    t_end:   float,
    n_steps: int = 500,
    psi0:    qt.Qobj = None,
    save_path: str = None,
) -> dict:
    """
    Solve the Schrodinger equation for qubit + TLS (no dissipation).

    Parameters
    ----------
    wq, wt    : qubit and TLS frequencies
    g         : coupling strength
    t_end     : total evolution time
    n_steps   : number of time points
    psi0      : initial state (default: |eg>)
    save_path : if given, save result to this .h5 file

    Returns
    -------
    dict:
        't'            : time array
        'P_e_q'        : qubit excited-state population
        'P_e_t'        : TLS excited-state population
        'P_e_analytic' : analytic P_e(t) for |eg> initial state
        'states'       : list of qt.Qobj state vectors
        'params'       : input parameters
    """
    if psi0 is None:
        psi0 = state('e', 'g')

    H  = build_H(wq, wt, g)
    t  = np.linspace(0, t_end, n_steps)

    sz_q = qt.tensor(qt.sigmaz(), qt.qeye(2))
    sz_t = qt.tensor(qt.qeye(2), qt.sigmaz())

    result = qt.sesolve(
        H, psi0, t,
        e_ops=[sz_q, sz_t],
        options={"nsteps": 10000},
    )

    P_e_q = (np.array(result.expect[0]) + 1) / 2
    P_e_t = (np.array(result.expect[1]) + 1) / 2

    out = {
        't':            t,
        'P_e_q':        P_e_q,
        'P_e_t':        P_e_t,
        'P_e_analytic': rabi_analytic(t, wq, wt, g),
        'states':       result.states,
        'params':       {'wq': wq, 'wt': wt, 'g': g, 't_end': t_end},
    }

    if save_path:
        save(out, save_path)

    return out


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    res = evolve(
        wq=1.0, wt=1.0, g=0.1,
        t_end=4*np.pi/0.1,
        save_path='../../data/single_trajectory/closed_resonant.h5'
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res['t'], res['P_e_q'],        label='Qubit (numeric)', lw=2)
    ax.plot(res['t'], res['P_e_t'],        label='TLS (numeric)',   lw=2, ls='--')
    ax.plot(res['t'], res['P_e_analytic'], label='Analytic',        lw=1, ls=':', color='k')
    ax.set_xlabel('Time'); ax.set_ylabel('Population')
    ax.set_title('Closed system — resonant')
    ax.legend(); ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig('../../figures/validation/closed_system_resonant.png', dpi=150)
    plt.show()
