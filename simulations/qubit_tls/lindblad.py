"""
lindblad.py
===========
Open-system evolution via the Lindblad master equation.

drho/dt = -i[H, rho]
          + gamma_q     * D[sm_q](rho)
          + gamma_t     * D[sm_t](rho)
          + gamma_phi_q * D[sz_q/2](rho)
          + gamma_phi_t * D[sz_t/2](rho)

D[L](rho) = L*rho*L† - (1/2){L†L, rho}

Reference: Breuer & Petruccione (2002), Blais et al. RMP (2021) Sec. IV.C
"""

import numpy as np
import qutip as qt
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from qubit_tls.hamiltonian import build_H, state, ops
from utils.io import save


# ─── Jump operators ───────────────────────────────────────────────────────────

def build_jump_ops(
    gamma_q:     float = 0.0,
    gamma_t:     float = 0.0,
    gamma_phi_q: float = 0.0,
    gamma_phi_t: float = 0.0,
) -> list:
    """Build Lindblad jump operators scaled by sqrt(rate)."""
    o = ops()
    c_ops = []
    if gamma_q     > 0: c_ops.append(np.sqrt(gamma_q)       * o['sm_q'])
    if gamma_t     > 0: c_ops.append(np.sqrt(gamma_t)       * o['sm_t'])
    if gamma_phi_q > 0: c_ops.append(np.sqrt(gamma_phi_q/2) * o['sz_q'])
    if gamma_phi_t > 0: c_ops.append(np.sqrt(gamma_phi_t/2) * o['sz_t'])
    return c_ops


# ─── Solver ───────────────────────────────────────────────────────────────────

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
    save_path:   str   = None,
) -> dict:
    """
    Solve the Lindblad master equation for qubit + TLS.

    Parameters
    ----------
    wq, wt      : frequencies
    g           : coupling
    gamma_q     : qubit relaxation rate (1/T1_q)
    gamma_t     : TLS relaxation rate   (1/T1_t)
    gamma_phi_q : qubit pure dephasing (default 0)
    gamma_phi_t : TLS pure dephasing   (default 0)
    t_end       : total time (default: 8/gamma_q)
    n_steps     : time points
    rho0        : initial density matrix (default: |eg><eg|)
    save_path   : if given, save result to this .h5 file

    Returns
    -------
    dict:
        't'          : time array
        'P_e_q'      : qubit excited-state population
        'P_e_t'      : TLS excited-state population
        'coherence'  : |rho_{eg,ge}(t)|
        'states'     : list of density matrices rho(t)
        'params'     : all input parameters
    """
    if rho0 is None:
        rho0 = qt.ket2dm(state('e', 'g'))

    if t_end is None:
        t_end = 8.0 / max(gamma_q, 1e-9)

    H     = build_H(wq, wt, g)
    c_ops = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t)
    t     = np.linspace(0, t_end, n_steps)
    o     = ops()

    result = qt.mesolve(
        H, rho0, t, c_ops,
        e_ops=[o['sz_q'], o['sz_t']],
        options={"nsteps": 10000},
    )

    P_e_q = (np.array(result.expect[0]) + 1) / 2
    P_e_t = (np.array(result.expect[1]) + 1) / 2

    # Off-diagonal coherence: |rho_{eg,ge}|
    # Basis: |ee>=0, |eg>=1, |ge>=2, |gg>=3
    coherence = np.array([abs(rho.full()[1, 2]) for rho in result.states])

    out = {
        't':         t,
        'P_e_q':     P_e_q,
        'P_e_t':     P_e_t,
        'coherence': coherence,
        'states':    result.states,
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'gamma_phi_q': gamma_phi_q, 'gamma_phi_t': gamma_phi_t,
            't_end': t_end,
        },
    }

    if save_path:
        save(out, save_path)

    return out


def steady_state(wq, wt, g, gamma_q, gamma_t,
                 gamma_phi_q=0., gamma_phi_t=0.) -> qt.Qobj:
    """Compute steady-state density matrix (should be |gg><gg|)."""
    H     = build_H(wq, wt, g)
    c_ops = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t)
    return qt.steadystate(H, c_ops)


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)
    res = evolve(**params, t_end=300,
                 save_path='../../data/single_trajectory/lindblad_resonant.h5')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(res['t'], res['P_e_q'], label='Qubit', lw=2)
    axes[0].plot(res['t'], res['P_e_t'], label='TLS',   lw=2, ls='--')
    axes[0].set_xlabel('Time'); axes[0].set_ylabel('Population')
    axes[0].set_title('Lindblad — populations'); axes[0].legend()

    axes[1].semilogy(res['t'], np.where(res['coherence']>0, res['coherence'], 1e-16),
                     color='purple', lw=2)
    axes[1].set_xlabel('Time'); axes[1].set_ylabel(r'$|\rho_{eg,ge}|$')
    axes[1].set_title('Off-diagonal coherence')

    plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/lindblad_resonant.png', dpi=150)
    plt.show()
