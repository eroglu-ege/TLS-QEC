"""
solomon.py
==========
Solomon equations with thermal population AND pure dephasing.

dP_q/dt = -gamma_q*(2*n_th_q+1)*(P_q - P_q^ss) - Gamma*(P_q - P_t)
dP_t/dt = -gamma_t*(2*n_th_t+1)*(P_t - P_t^ss) + Gamma*(P_q - P_t)

P_ss = n_th/(2*n_th+1)

Exchange rate with pure dephasing (gamma_phi broadens the TLS linewidth):
    gamma_t_total = gamma_t*(2*n_th_t+1) + gamma_phi_t
    Gamma = 2*g^2*gamma_t_total / (Delta^2 + gamma_t_total^2)

Physical meaning of gamma_phi in Solomon:
    Pure dephasing broadens the TLS resonance linewidth without changing
    its population decay rate. A broader linewidth means the TLS resonance
    condition is relaxed (exchange possible further off-resonance) but the
    peak exchange rate decreases. This HELPS Solomon's validity — faster
    dephasing kills coherences sooner, making the incoherent rate-equation
    picture more accurate.

Reference: Solomon Phys.Rev.99,559(1955);
           Ashhab, Johansson, Nori PRB 74,184415(2006)
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.io import save
from utils.physics import thermal_steady_state, exchange_rate


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
    P_q0:        float = 1.0,
    P_t0:        float = 0.0,
    save_path:   str   = None,
) -> dict:
    """
    Solve the Solomon equations with thermal population and pure dephasing.

    Parameters
    ----------
    wq, wt      : transition frequencies
    g           : coupling strength
    gamma_q     : qubit relaxation rate (1/T1_q)
    gamma_t     : TLS relaxation rate   (1/T1_t)
    gamma_phi_q : qubit pure dephasing rate (default 0)
    gamma_phi_t : TLS pure dephasing rate   (default 0)
    n_th_q      : thermal photon number at qubit frequency (default 0)
    n_th_t      : thermal photon number at TLS frequency   (default 0)
    t_end       : total time (default 8/gamma_q)
    n_steps     : time points
    P_q0        : initial qubit population (default 1.0)
    P_t0        : initial TLS population   (default 0.0)
    save_path   : if given, save to .h5

    Returns
    -------
    dict: t, P_e_q, P_e_t, Gamma, P_e_q_ss, P_e_t_ss, params
    """
    delta = wq - wt

    if t_end is None:
        t_end = 8.0 / max(gamma_q, 1e-9)

    # Exchange rate: TLS linewidth includes BOTH relaxation AND dephasing
    # gamma_t_eff = gamma_t*(2*n_th+1) + gamma_phi_t
    gamma_t_total = gamma_t * (2.0 * n_th_t + 1.0) + gamma_phi_t
    Gamma = 2.0 * g**2 * gamma_t_total / (delta**2 + gamma_t_total**2)

    P_q_ss = thermal_steady_state(n_th_q)
    P_t_ss = thermal_steady_state(n_th_t)
    t_eval = np.linspace(0, t_end, n_steps)

    def rhs(t, y):
        P_q, P_t = y
        gamma_q_eff = gamma_q * (2.0 * n_th_q + 1.0)
        gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)
        dP_q = -gamma_q_eff * (P_q - P_q_ss) - Gamma * (P_q - P_t)
        dP_t = -gamma_t_eff * (P_t - P_t_ss) + Gamma * (P_q - P_t)
        return [dP_q, dP_t]

    sol = solve_ivp(rhs, (0, t_end), [P_q0, P_t0],
                    t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)

    out = {
        't':        sol.t,
        'P_e_q':    sol.y[0],
        'P_e_t':    sol.y[1],
        'Gamma':    Gamma,
        'P_e_q_ss': P_q_ss,
        'P_e_t_ss': P_t_ss,
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'gamma_phi_q': gamma_phi_q, 'gamma_phi_t': gamma_phi_t,
            'n_th_q': n_th_q, 'n_th_t': n_th_t,
            'delta': delta, 'Gamma': Gamma, 't_end': t_end,
        },
    }

    if save_path:
        save(out, save_path)
    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.physics import make_params, T2_from_rates

    # Compare T2=2*T1 (no dephasing) vs T2=T1 (moderate dephasing)
    base = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005,
                n_th_q=0.1, n_th_t=0.1)

    fig, ax = plt.subplots(figsize=(9, 4))
    for gamma_phi, label in [(0.0, '$T_2=2T_1$ (no dephasing)'),
                              (0.01, '$T_2=T_1$'),
                              (0.03, '$T_2=T_1/2$')]:
        T2 = T2_from_rates(base['gamma_q'], gamma_phi)
        res = evolve(**base, gamma_phi_q=gamma_phi, gamma_phi_t=gamma_phi/2,
                     t_end=800, n_steps=600, P_q0=1.0, P_t0=0.0)
        ax.plot(res['t'], res['P_e_q'], lw=2,
                label=f'{label}  ($T_2$={T2:.0f})')

    ax.set_xlabel('Time'); ax.set_ylabel('Qubit $P_e$')
    ax.set_title('Solomon: effect of pure dephasing on qubit population')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/solomon_T2.png', dpi=150)
    plt.show()
