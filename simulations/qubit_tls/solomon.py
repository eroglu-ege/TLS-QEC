"""
solomon.py
==========
Solomon equations: classical rate-equation limit of Lindblad.

dP_q/dt = -gamma_q * P_q - Gamma * (P_q - P_t)
dP_t/dt = -gamma_t * P_t + Gamma * (P_q - P_t)

Gamma = 2*g^2*gamma_t / (Delta^2 + gamma_t^2)   [exchange rate]

Emerges from Lindblad when off-diagonal coherences rho_{eg,ge} = 0.
Valid when: |Delta| >> g  OR  gamma_t >> g

Reference: Solomon Phys.Rev.99,559(1955); Ashhab et al. PRB 74,184415(2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.io import save


def exchange_rate(g: float, delta: float, gamma_t: float) -> float:
    """Gamma = 2*g^2*gamma_t / (Delta^2 + gamma_t^2)"""
    return 2 * g**2 * gamma_t / (delta**2 + gamma_t**2)


def evolve(
    wq:      float,
    wt:      float,
    g:       float,
    gamma_q: float,
    gamma_t: float,
    t_end:   float = None,
    n_steps: int   = 500,
    P_q0:    float = 1.0,
    P_t0:    float = 0.0,
    save_path: str = None,
) -> dict:
    """
    Solve the Solomon equations.

    Parameters
    ----------
    wq, wt  : frequencies (only Delta = wq-wt matters)
    g       : coupling
    gamma_q : qubit relaxation (1/T1_q)
    gamma_t : TLS relaxation   (1/T1_t)
    t_end   : total time
    n_steps : time points
    P_q0    : initial qubit population (default 1.0)
    P_t0    : initial TLS population   (default 0.0)
    save_path: if given, save to .h5

    Returns
    -------
    dict: 't', 'P_e_q', 'P_e_t', 'Gamma', 'params'
    """
    delta = wq - wt
    if t_end is None:
        t_end = 8.0 / max(gamma_q, 1e-9)

    Gamma  = exchange_rate(g, delta, gamma_t)
    t_eval = np.linspace(0, t_end, n_steps)

    def rhs(t, y):
        P_q, P_t = y
        return [
            -gamma_q * P_q - Gamma * (P_q - P_t),
            -gamma_t * P_t + Gamma * (P_q - P_t),
        ]

    sol = solve_ivp(rhs, (0, t_end), [P_q0, P_t0],
                    t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)

    out = {
        't':      sol.t,
        'P_e_q':  sol.y[0],
        'P_e_t':  sol.y[1],
        'Gamma':  Gamma,
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'delta': delta, 'Gamma': Gamma, 't_end': t_end,
        },
    }

    if save_path:
        save(out, save_path)

    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)
    res = evolve(**params, t_end=500,
                 save_path='../../data/single_trajectory/solomon_resonant.h5')

    print(f"Exchange rate Gamma = {res['Gamma']:.5f}")
    print(f"Gamma/gamma_q = {res['Gamma']/params['gamma_q']:.2f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(res['t'], res['P_e_q'], label='Qubit', lw=2)
    ax.plot(res['t'], res['P_e_t'], label='TLS',   lw=2, ls='--')
    ax.set_xlabel('Time'); ax.set_ylabel('Population')
    ax.set_title(f"Solomon equations  Γ={res['Gamma']:.4f}")
    ax.legend()
    plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/solomon_resonant.png', dpi=150)
    plt.show()
