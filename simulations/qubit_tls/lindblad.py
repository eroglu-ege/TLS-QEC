"""
lindblad.py — Lindblad master equation with thermal population.

drho/dt = -i[H, rho]
          + gamma_q*(1+n_th_q) * D[sm_q](rho)   decay
          + gamma_q*n_th_q     * D[sp_q](rho)   thermal excitation
          + gamma_t*(1+n_th_t) * D[sm_t](rho)
          + gamma_t*n_th_t     * D[sp_t](rho)
          + gamma_phi_q/2      * D[sz_q](rho)   pure dephasing
          + gamma_phi_t/2      * D[sz_t](rho)

Steady state: P_e^ss = n_th / (2*n_th + 1)

Reference: Breuer & Petruccione (2002); Blais et al. RMP 93, 025005 (2021)
"""

import numpy as np
import qutip as qt
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from qubit_tls.hamiltonian import build_H, state, ops
from utils.io import save


def n_thermal(freq_GHz: float, temp_K: float) -> float:
    """
    Mean thermal photon number: n_th = 1/(exp(hbar*w/kT) - 1)
    Uses hbar/kB = 47.99 mK/GHz.
    Examples: n_thermal(5.0, 0.020) -> 0.005  (5GHz qubit at 20mK)
              n_thermal(5.0, 0.100) -> 0.078  (5GHz qubit at 100mK)
    """
    if temp_K <= 0:
        return 0.0
    x = freq_GHz * 47.99e-3 / temp_K
    return 1.0 / (np.exp(x) - 1.0)


def thermal_steady_state(n_th: float) -> float:
    """P_e^ss = n_th / (2*n_th + 1). At n_th=0: 0. At n_th->inf: 0.5."""
    return n_th / (2.0 * n_th + 1.0)


def build_jump_ops(
    gamma_q:     float = 0.0,
    gamma_t:     float = 0.0,
    gamma_phi_q: float = 0.0,
    gamma_phi_t: float = 0.0,
    n_th_q:      float = 0.0,
    n_th_t:      float = 0.0,
) -> list:
    """
    Build jump operators with thermal excitation.

    Zero temperature (n_th=0):
        sqrt(gamma) * sigma_-          (decay only)

    Finite temperature (n_th>0):
        sqrt(gamma*(1+n_th)) * sigma_- (decay, slightly faster)
        sqrt(gamma*n_th)     * sigma_+ (thermal excitation, new)

    Pure dephasing is temperature-independent.
    """
    o = ops()
    c_ops = []

    if gamma_q > 0:
        c_ops.append(np.sqrt(gamma_q * (1.0 + n_th_q)) * o['sm_q'])
        if n_th_q > 0:
            c_ops.append(np.sqrt(gamma_q * n_th_q) * o['sp_q'])

    if gamma_t > 0:
        c_ops.append(np.sqrt(gamma_t * (1.0 + n_th_t)) * o['sm_t'])
        if n_th_t > 0:
            c_ops.append(np.sqrt(gamma_t * n_th_t) * o['sp_t'])

    if gamma_phi_q > 0:
        c_ops.append(np.sqrt(gamma_phi_q / 2.0) * o['sz_q'])
    if gamma_phi_t > 0:
        c_ops.append(np.sqrt(gamma_phi_t / 2.0) * o['sz_t'])

    return c_ops


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
    n_th_q      : thermal photon number at qubit frequency (default 0=T=0)
    n_th_t      : thermal photon number at TLS frequency   (default 0=T=0)
    t_end       : total time (default 8/gamma_q)
    n_steps     : time points
    rho0        : initial density matrix (default |eg><eg|)
    save_path   : if given, save to .h5

    Returns
    -------
    dict: t, P_e_q, P_e_t, coherence, states, P_e_q_ss, P_e_t_ss, params
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

    result = qt.mesolve(
        H, rho0, t, c_ops,
        e_ops=[o['sz_q'], o['sz_t']],
        options={"nsteps": 10000, "store_states": True},
    )

    P_e_q = (np.array(result.expect[0]) + 1.0) / 2.0
    P_e_t = (np.array(result.expect[1]) + 1.0) / 2.0
    coherence = np.array([abs(rho.full()[1, 2]) for rho in result.states])

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


def steady_state(wq, wt, g, gamma_q, gamma_t,
                 gamma_phi_q=0., gamma_phi_t=0.,
                 n_th_q=0., n_th_t=0.) -> qt.Qobj:
    """Compute steady-state density matrix (drho/dt = 0)."""
    H     = build_H(wq, wt, g)
    c_ops = build_jump_ops(gamma_q, gamma_t, gamma_phi_q, gamma_phi_t,
                           n_th_q, n_th_t)
    return qt.steadystate(H, c_ops)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)
    scenarios = [
        (0.0,  'n_th=0 (T=0)',     'C0'),
        (0.01, 'n_th=0.01 (cold)', 'C1'),
        (0.1,  'n_th=0.1 (warm)',  'C2'),
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
        print(f"n_th={n_th:.2f}: P_ss={P_ss:.5f}, final={res['P_e_q'][-1]:.5f}")

    for ax, title in zip(axes, ['Qubit P_e', 'TLS P_e', 'Coherence']):
        ax.set_xlabel('Time'); ax.set_title(title); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/lindblad_thermal.png', dpi=150)
    plt.show()
