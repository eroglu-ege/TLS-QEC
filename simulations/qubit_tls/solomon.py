"""
solomon.py — Solomon equations with thermal population.

dP_q/dt = -gamma_q*(2*n_th_q+1)*(P_q - P_q^ss) - Gamma*(P_q - P_t)
dP_t/dt = -gamma_t*(2*n_th_t+1)*(P_t - P_t^ss) + Gamma*(P_q - P_t)

P_ss = n_th/(2*n_th+1)   [thermal steady state]
Gamma = 2*g^2*gamma_t_eff / (Delta^2 + gamma_t_eff^2)
gamma_t_eff = gamma_t*(2*n_th_t+1)   [thermally broadened TLS linewidth]

Reference: Solomon Phys.Rev.99,559(1955); Ashhab et al. PRB 74,184415(2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.io import save


def thermal_steady_state(n_th: float) -> float:
    """P_ss = n_th/(2*n_th+1). At n_th=0: 0. At n_th->inf: 0.5."""
    return n_th / (2.0 * n_th + 1.0)


def exchange_rate(g, delta, gamma_t, n_th_t=0.0):
    """
    Gamma = 2*g^2*gamma_t_eff / (Delta^2 + gamma_t_eff^2)

    At finite temperature the TLS linewidth broadens:
        gamma_t_eff = gamma_t*(2*n_th_t+1)

    Effect: at Delta=0, Gamma DECREASES (broader but shorter peak).
            at large Delta, Gamma INCREASES (broader tail reaches further).
    """
    gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)
    return 2.0 * g**2 * gamma_t_eff / (delta**2 + gamma_t_eff**2)


def _rhs(t, y, gamma_q, gamma_t, Gamma, n_th_q, n_th_t):
    """
    RHS of thermal Solomon equations.

    Written as displacement from thermal steady state:
        dP/dt = -gamma_eff*(P - P_ss) - exchange_term

    At n_th=0: P_ss=0, gamma_eff=gamma → original Solomon equations.
    """
    P_q, P_t = y
    P_q_ss = thermal_steady_state(n_th_q)
    P_t_ss = thermal_steady_state(n_th_t)
    gamma_q_eff = gamma_q * (2.0 * n_th_q + 1.0)
    gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)

    dP_q = -gamma_q_eff * (P_q - P_q_ss) - Gamma * (P_q - P_t)
    dP_t = -gamma_t_eff * (P_t - P_t_ss) + Gamma * (P_q - P_t)
    return [dP_q, dP_t]


def evolve(
    wq:      float,
    wt:      float,
    g:       float,
    gamma_q: float,
    gamma_t: float,
    n_th_q:  float = 0.0,
    n_th_t:  float = 0.0,
    t_end:   float = None,
    n_steps: int   = 500,
    P_q0:    float = 1.0,
    P_t0:    float = 0.0,
    save_path: str = None,
) -> dict:
    """
    Solve the thermal Solomon equations.

    Parameters
    ----------
    wq, wt   : frequencies (only delta=wq-wt matters)
    g        : coupling
    gamma_q  : qubit relaxation (1/T1_q)
    gamma_t  : TLS relaxation   (1/T1_t)
    n_th_q   : thermal photon number at qubit frequency (default 0)
    n_th_t   : thermal photon number at TLS frequency   (default 0)
    t_end    : total time (default 8/gamma_q)
    n_steps  : time points
    P_q0     : initial qubit population (default 1.0)
    P_t0     : initial TLS population   (default 0.0)
    save_path: if given, save to .h5

    Returns
    -------
    dict: t, P_e_q, P_e_t, Gamma, P_e_q_ss, P_e_t_ss, params
    """
    delta = wq - wt
    if t_end is None:
        t_end = 8.0 / max(gamma_q, 1e-9)

    Gamma  = exchange_rate(g, delta, gamma_t, n_th_t)
    t_eval = np.linspace(0, t_end, n_steps)

    sol = solve_ivp(
        fun    = _rhs,
        t_span = (0, t_end),
        y0     = [P_q0, P_t0],
        args   = (gamma_q, gamma_t, Gamma, n_th_q, n_th_t),
        t_eval = t_eval,
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )

    out = {
        't':        sol.t,
        'P_e_q':    sol.y[0],
        'P_e_t':    sol.y[1],
        'Gamma':    Gamma,
        'P_e_q_ss': thermal_steady_state(n_th_q),
        'P_e_t_ss': thermal_steady_state(n_th_t),
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'n_th_q': n_th_q, 'n_th_t': n_th_t,
            'delta': delta, 'Gamma': Gamma, 't_end': t_end,
        },
    }

    if save_path:
        save(out, save_path)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)
    scenarios = [
        (0.0,  'n_th=0 (T=0)',     'C0'),
        (0.01, 'n_th=0.01 (cold)', 'C1'),
        (0.1,  'n_th=0.1 (warm)',  'C2'),
    ]

    fig, ax = plt.subplots(figsize=(8, 4))
    for n_th, label, color in scenarios:
        res  = evolve(**base, n_th_q=n_th, n_th_t=n_th, t_end=600)
        P_ss = thermal_steady_state(n_th)
        Gamma = exchange_rate(0.1, 0.0, 0.005, n_th)
        ax.plot(res['t'], res['P_e_q'], label=label, color=color, lw=2)
        ax.axhline(P_ss, color=color, ls=':', lw=1)
        print(f"n_th={n_th:.2f}: Gamma={Gamma:.4f}, "
              f"P_ss={P_ss:.5f}, final={res['P_e_q'][-1]:.5f}")

    ax.set_xlabel('Time'); ax.set_ylabel('Qubit P_e')
    ax.set_title('Solomon: thermal population (dotted=steady state)')
    ax.legend(); plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/solomon_thermal.png', dpi=150)
    plt.show()
