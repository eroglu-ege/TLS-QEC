"""
solomon.py
==========
The Solomon equations: the classical (incoherent) rate-equation limit
of the full Lindblad master equation.

The Solomon equations for a qubit coupled to a TLS are:

    dP_q/dt = -gamma_q * P_q - Gamma * (P_q - P_t)
    dP_t/dt = -gamma_t * P_t + Gamma * (P_q - P_t)

where:
    P_q = qubit excited-state population
    P_t = TLS excited-state population
    gamma_q = 1/T1_q   (qubit intrinsic decay)
    gamma_t = 1/T1_t   (TLS intrinsic decay)
    Gamma   = exchange rate between qubit and TLS

The exchange rate Gamma is a Lorentzian in the detuning Delta:

    Gamma = 2 * g^2 * gamma_t / (Delta^2 + gamma_t^2)

Physical meaning: Gamma is the rate at which the qubit transfers its
excitation to the TLS. It is maximal at resonance (Delta=0) and
suppressed off-resonance.

Derivation from Lindblad:
--------------------------
The Solomon equations emerge from the full Lindblad equation by:
1. Moving to the secular approximation (averaging over Rabi oscillations)
2. Setting all off-diagonal density matrix elements to zero: rho_{eg,ge} = 0

This is valid when:
- Dispersive regime: |Delta| >> g  (coherences oscillate fast, average out)
- Overdamped regime: gamma_t >> g  (TLS dephases before coherences build up)

Task 3 of the thesis is to find where this approximation BREAKS DOWN,
i.e., where the full Lindblad solution differs significantly from Solomon.

Reference:
  Solomon, Phys. Rev. 99, 559 (1955) — original derivation for NMR
  Ashhab, Johansson, Nori, Phys. Rev. B 74, 184415 (2006) — qubit+TLS context
"""

import numpy as np
from scipy.integrate import solve_ivp


# ─── Exchange rate ────────────────────────────────────────────────────────────

def exchange_rate(g: float, delta: float, gamma_t: float) -> float:
    """
    Compute the qubit-TLS exchange rate Gamma (Lorentzian lineshape).

    Gamma = 2 * g^2 * gamma_t / (Delta^2 + gamma_t^2)

    Parameters
    ----------
    g       : coupling strength
    delta   : detuning = wq - wt
    gamma_t : TLS relaxation rate (1/T1_t)

    Returns
    -------
    Gamma : float
    """
    return 2 * g**2 * gamma_t / (delta**2 + gamma_t**2)


# ─── Right-hand side of Solomon ODEs ─────────────────────────────────────────

def _solomon_rhs(t, y, gamma_q, gamma_t, Gamma):
    """
    RHS of the Solomon equations for scipy.solve_ivp.

    y = [P_q, P_t]
    """
    P_q, P_t = y
    dP_q = -gamma_q * P_q - Gamma * (P_q - P_t)
    dP_t = -gamma_t * P_t + Gamma * (P_q - P_t)
    return [dP_q, dP_t]


# ─── Main solver ──────────────────────────────────────────────────────────────

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
) -> dict:
    """
    Solve the Solomon equations for qubit + TLS populations.

    Parameters
    ----------
    wq, wt  : qubit and TLS frequencies (only their difference matters)
    g       : coupling strength
    gamma_q : qubit relaxation rate (1/T1_q)
    gamma_t : TLS relaxation rate   (1/T1_t)
    t_end   : total time (default: 5/gamma_q)
    n_steps : number of time points
    P_q0    : initial qubit population (default 1.0 = qubit excited)
    P_t0    : initial TLS population   (default 0.0 = TLS ground)

    Returns
    -------
    dict with keys:
        't'      : time array
        'P_e_q'  : qubit excited-state population P_q(t)
        'P_e_t'  : TLS excited-state population P_t(t)
        'Gamma'  : exchange rate used
        'params' : input parameters
    """
    delta = wq - wt

    if t_end is None:
        t_end = 5.0 / max(gamma_q, 1e-9)

    Gamma = exchange_rate(g, delta, gamma_t)

    t_eval = np.linspace(0, t_end, n_steps)

    sol = solve_ivp(
        fun    = _solomon_rhs,
        t_span = (0, t_end),
        y0     = [P_q0, P_t0],
        args   = (gamma_q, gamma_t, Gamma),
        t_eval = t_eval,
        method = 'RK45',
        rtol   = 1e-8,
        atol   = 1e-10,
    )

    return {
        't':      sol.t,
        'P_e_q':  sol.y[0],
        'P_e_t':  sol.y[1],
        'Gamma':  Gamma,
        'params': {
            'wq': wq, 'wt': wt, 'g': g,
            'gamma_q': gamma_q, 'gamma_t': gamma_t,
            'delta': delta,
        },
    }


# ─── Analytic solution ────────────────────────────────────────────────────────

def analytic(
    t: np.ndarray,
    gamma_q: float,
    gamma_t: float,
    Gamma: float,
    P_q0: float = 1.0,
    P_t0: float = 0.0,
) -> tuple:
    """
    Analytic solution of the Solomon equations (biexponential decay).

    The Solomon equations are linear ODEs with constant coefficients.
    The solution is a sum of two exponentials with rates:

    lambda_± = -( (gamma_q + gamma_t)/2 + Gamma )
               ± sqrt( ((gamma_q - gamma_t)/2)^2 + Gamma^2 )

    Returns
    -------
    P_q(t), P_t(t)  as numpy arrays
    """
    A = -(gamma_q + 2*Gamma + gamma_t) / 2
    B = np.sqrt(((gamma_q - gamma_t) / 2)**2 + Gamma**2)

    lam_p = A + B
    lam_m = A - B

    # Solve for coefficients from initial conditions
    # P_q = c1*exp(lam_p*t) + c2*exp(lam_m*t)
    # P_t = d1*exp(lam_p*t) + d2*exp(lam_m*t)
    # Using the eigenstructure of the 2x2 matrix
    M = np.array([
        [-(gamma_q + Gamma), Gamma],
        [Gamma,              -(gamma_t + Gamma)],
    ])
    vals, vecs = np.linalg.eig(M)
    idx = np.argsort(vals)[::-1]   # sort descending
    vals, vecs = vals[idx], vecs[:, idx]

    # Solve initial condition: vecs @ c = [P_q0, P_t0]
    c = np.linalg.solve(vecs, [P_q0, P_t0])

    P_q = c[0] * vecs[0, 0] * np.exp(vals[0]*t) + c[1] * vecs[0, 1] * np.exp(vals[1]*t)
    P_t = c[0] * vecs[1, 0] * np.exp(vals[0]*t) + c[1] * vecs[1, 1] * np.exp(vals[1]*t)

    return np.real(P_q), np.real(P_t)


# ─── Quick run ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    params = dict(wq=1.0, wt=1.0, g=0.1, gamma_q=0.01, gamma_t=0.005)

    print("Running Solomon equations...")
    res = evolve(**params, t_end=500)

    Gamma = res['Gamma']
    print(f"Exchange rate Gamma = {Gamma:.5f}")
    print(f"gamma_q = {params['gamma_q']}, gamma_t = {params['gamma_t']}")
    print(f"Gamma/gamma_q = {Gamma/params['gamma_q']:.2f}  "
          f"(>1 means exchange dominates)")

    # Analytic solution
    P_q_an, P_t_an = analytic(
        res['t'], params['gamma_q'], params['gamma_t'], Gamma
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(res['t'], res['P_e_q'], label='Qubit (numeric)',   lw=2)
    ax.plot(res['t'], res['P_e_t'], label='TLS (numeric)',     lw=2, ls='--')
    ax.plot(res['t'], P_q_an,       label='Qubit (analytic)',  lw=1, ls=':', color='C0')
    ax.plot(res['t'], P_t_an,       label='TLS (analytic)',    lw=1, ls=':', color='C1')
    ax.set_xlabel('Time')
    ax.set_ylabel('Excited state population')
    ax.set_title(f'Solomon equations: $\\Gamma$={Gamma:.4f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('../../figures/solomon_evolution.png', dpi=150)
    print("Saved to figures/solomon_evolution.png")
    plt.show()
