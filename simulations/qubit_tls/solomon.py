"""
solomon.py
==========
Solomon equations: classical rate-equation limit of Lindblad,
with optional thermal population at finite temperature.

PHYSICS SUMMARY
---------------
The Solomon equations describe qubit + TLS populations as coupled
rate equations, with NO quantum coherences:

    dP_q/dt = -gamma_q*(2*n_th_q+1)*(P_q - P_q^ss) - Gamma*(P_q - P_t)
    dP_t/dt = -gamma_t*(2*n_th_t+1)*(P_t - P_t^ss) + Gamma*(P_q - P_t)

where:
    P_q^ss = n_th_q / (2*n_th_q + 1)   thermal steady state (qubit)
    P_t^ss = n_th_t / (2*n_th_t + 1)   thermal steady state (TLS)

AT ZERO TEMPERATURE (n_th=0):
    P_q^ss = P_t^ss = 0
    Equations reduce to the original Solomon (1955) form:
        dP_q/dt = -gamma_q * P_q - Gamma*(P_q - P_t)
        dP_t/dt = -gamma_t * P_t + Gamma*(P_q - P_t)

AT FINITE TEMPERATURE (n_th>0):
    The effective decay rate becomes gamma*(2*n_th+1) — faster decay
    because both spontaneous and stimulated emission contribute.
    The system no longer relaxes to zero but to the thermal population.

THE EXCHANGE RATE Gamma:
    At finite temperature, the TLS linewidth broadens:
        gamma_t_eff = gamma_t * (2*n_th_t + 1)

    This modifies the Lorentzian exchange rate:
        Gamma = 2*g^2*gamma_t_eff / (Delta^2 + gamma_t_eff^2)

    Physical meaning: a broader TLS linewidth means the qubit can
    exchange energy with it even further off resonance, but the
    peak exchange rate at Delta=0 decreases.

HOW SOLOMON EMERGES FROM LINDBLAD:
    Start from the Lindblad master equation with thermal operators.
    Write equations of motion for rho_{ee}, rho_{eg,eg}, rho_{ge,ge},
    and the coherence rho_{eg,ge}.
    In the secular approximation (average over fast Rabi oscillations)
    AND setting rho_{eg,ge} = 0 (no coherences), the population
    equations decouple and become exactly these Solomon equations.

Reference: Solomon, Phys. Rev. 99, 559 (1955) — original NMR derivation
           Ashhab, Johansson, Nori, PRB 74, 184415 (2006) — qubit+TLS
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.io import save


# ─── Thermal helpers ──────────────────────────────────────────────────────────

def thermal_steady_state(n_th: float) -> float:
    """
    Thermal equilibrium excited-state population.

    P_ss = n_th / (2*n_th + 1)

    This is where each subsystem relaxes to in the absence of coupling.
    At n_th=0: P_ss=0. At n_th->inf: P_ss->0.5.
    """
    return n_th / (2.0 * n_th + 1.0)


def exchange_rate(
    g:       float,
    delta:   float,
    gamma_t: float,
    n_th_t:  float = 0.0,
) -> float:
    """
    Qubit-TLS exchange rate Gamma (Lorentzian in detuning Delta).

    At zero temperature:
        Gamma = 2*g^2*gamma_t / (Delta^2 + gamma_t^2)

    At finite temperature, the TLS linewidth broadens to:
        gamma_t_eff = gamma_t * (2*n_th_t + 1)

    So the full expression is:
        Gamma = 2*g^2*gamma_t_eff / (Delta^2 + gamma_t_eff^2)

    Physical effect of temperature on Gamma:
        - Broader linewidth: TLS resonance condition relaxed
        - At Delta=0: Gamma DECREASES with n_th (peak is lower, broader)
        - At large Delta: Gamma can INCREASE with n_th (broader tail)
        - The product Gamma * gamma_t_eff is conserved (spectral weight)

    Parameters
    ----------
    g       : coupling strength
    delta   : detuning = wq - wt
    gamma_t : TLS relaxation rate (1/T1_t)
    n_th_t  : thermal photon number at TLS frequency (default 0)

    Returns
    -------
    Gamma : float, the exchange rate
    """
    gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)
    return 2.0 * g**2 * gamma_t_eff / (delta**2 + gamma_t_eff**2)


# ─── Right-hand side ──────────────────────────────────────────────────────────

def _rhs(t, y, gamma_q, gamma_t, Gamma, n_th_q, n_th_t):
    """
    RHS of the thermal Solomon equations for scipy.solve_ivp.

    y = [P_q, P_t]

    The equations are written in terms of displacement from steady state
    to make the thermal generalisation transparent:

        dP_q/dt = -gamma_q*(2n+1)*(P_q - P_q^ss) - Gamma*(P_q - P_t)
        dP_t/dt = -gamma_t*(2n+1)*(P_t - P_t^ss) + Gamma*(P_q - P_t)

    Expanding the (2*n_th+1) factors:
        Effective qubit decay rate  = gamma_q * (2*n_th_q + 1)
        Effective TLS decay rate    = gamma_t * (2*n_th_t + 1)

    At n_th=0: reduces to original Solomon equations.

    The Gamma*(P_q - P_t) term:
        When P_q > P_t: qubit loses population to TLS (positive for TLS)
        When P_q < P_t: TLS loses population to qubit (negative for TLS)
        No coherent oscillation — just diffusion between populations.
        This is the key difference from Lindblad.
    """
    P_q, P_t = y

    P_q_ss = thermal_steady_state(n_th_q)
    P_t_ss = thermal_steady_state(n_th_t)

    # Effective decay rates at finite temperature
    gamma_q_eff = gamma_q * (2.0 * n_th_q + 1.0)
    gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)

    dP_q = -gamma_q_eff * (P_q - P_q_ss) - Gamma * (P_q - P_t)
    dP_t = -gamma_t_eff * (P_t - P_t_ss) + Gamma * (P_q - P_t)

    return [dP_q, dP_t]


# ─── Main solver ──────────────────────────────────────────────────────────────

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
    wq, wt   : transition frequencies (only delta=wq-wt matters)
    g        : coupling strength
    gamma_q  : qubit relaxation rate (1/T1_q)
    gamma_t  : TLS relaxation rate   (1/T1_t)
    n_th_q   : thermal photon number at qubit frequency (default 0)
    n_th_t   : thermal photon number at TLS frequency   (default 0)
    t_end    : total time (default: 8/gamma_q)
    n_steps  : time points
    P_q0     : initial qubit population (default 1.0 = qubit excited)
    P_t0     : initial TLS population   (default 0.0 = TLS ground)
    save_path: if given, save result to .h5

    Returns
    -------
    dict:
        't'        : time array
        'P_e_q'    : qubit population P_q(t)
        'P_e_t'    : TLS population P_t(t)
        'Gamma'    : exchange rate used
        'P_e_q_ss' : thermal steady state population (qubit)
        'P_e_t_ss' : thermal steady state population (TLS)
        'params'   : all input parameters
    """
    delta = wq - wt

    if t_end is None:
        t_end = 8.0 / max(gamma_q, 1e-9)

    # Compute exchange rate (thermally modified)
    Gamma  = exchange_rate(g, delta, gamma_t, n_th_t)
    t_eval = np.linspace(0, t_end, n_steps)

    # Thermal steady states — where populations will end up
    P_q_ss = thermal_steady_state(n_th_q)
    P_t_ss = thermal_steady_state(n_th_t)

    # Solve ODEs using RK45 (Runge-Kutta 4th/5th order)
    # Tight tolerances ensure ODE solver error << physics we're measuring
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
        'P_e_q_ss': P_q_ss,
        'P_e_t_ss': P_t_ss,
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


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)

    scenarios = [
        (0.0,  'n_th=0 (T=0)',      'C0'),
        (0.01, 'n_th=0.01 (cold)',  'C1'),
        (0.1,  'n_th=0.1  (warm)',  'C2'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for n_th, label, color in scenarios:
        res   = evolve(**base, n_th_q=n_th, n_th_t=n_th, t_end=600, n_steps=800)
        P_ss  = thermal_steady_state(n_th)
        Gamma = res['Gamma']

        axes[0].plot(res['t'], res['P_e_q'], label=label, color=color, lw=2)
        axes[0].axhline(P_ss, color=color, ls=':', lw=1)

        axes[1].plot(res['t'], res['P_e_t'], label=label, color=color, lw=2)

        print(f"n_th={n_th:.2f}: Gamma={Gamma:.4f}, "
              f"P_q^ss={P_ss:.4f}, "
              f"final P_e_q={res['P_e_q'][-1]:.5f}")

    for ax, title in zip(axes, ['Qubit P_e (dotted=steady state)', 'TLS P_e']):
        ax.set_xlabel('Time'); ax.set_title(title)
        ax.legend(fontsize=9); ax.set_ylim(-0.02, 1.05)

    plt.suptitle('Solomon equations: thermal population effect', fontsize=13)
    plt.tight_layout()
    plt.savefig('../../figures/single_trajectory/solomon_thermal.png', dpi=150)
    plt.show()
