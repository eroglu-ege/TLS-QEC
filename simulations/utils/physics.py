"""
physics.py
==========
Physical helper functions shared across simulation files.

Includes:
  - n_thermal: mean thermal photon number
  - thermal_steady_state: equilibrium population
  - exchange_rate: Solomon exchange rate (thermally modified)
  - T2_from_rates: compute T2 from T1 and pure dephasing rate
  - gamma_phi_from_T2: compute pure dephasing rate from T1, T2
  - make_params: build a consistent parameter dict from physical timescales
"""

import numpy as np


def n_thermal(freq_GHz: float, temp_K: float) -> float:
    """
    Mean thermal photon number: n_th = 1/(exp(hbar*2pi*f/kT) - 1)
    Uses hbar*2pi/kB = 0.04802 K/GHz.

    Examples:
        n_thermal(5.0, 0.020) -> ~6e-6  (5 GHz qubit at 20 mK)
        n_thermal(5.0, 0.100) -> ~0.10  (5 GHz qubit at 100 mK)
    """
    if temp_K <= 0:
        return 0.0
    x = freq_GHz * 0.04802 / temp_K
    return 1.0 / (np.exp(x) - 1.0)


def thermal_steady_state(n_th: float) -> float:
    """
    Thermal equilibrium excited-state population.
    P_ss = n_th / (2*n_th + 1)
    At n_th=0: 0.0. At n_th->inf: 0.5.
    """
    return n_th / (2.0 * n_th + 1.0)


def exchange_rate(g: float, delta: float,
                  gamma_t: float, n_th_t: float = 0.0) -> float:
    """
    Solomon exchange rate (thermally broadened Lorentzian).
    Gamma = 2*g^2*gamma_t_eff / (Delta^2 + gamma_t_eff^2)
    gamma_t_eff = gamma_t*(2*n_th_t+1)
    """
    gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)
    return 2.0 * g**2 * gamma_t_eff / (delta**2 + gamma_t_eff**2)


def T2_from_rates(gamma: float, gamma_phi: float) -> float:
    """
    Compute T2 from T1 decay rate and pure dephasing rate.

    1/T2 = gamma/2 + gamma_phi/2
    -> T2 = 2 / (gamma + gamma_phi)

    At gamma_phi=0: T2 = 2*T1 (Lindblad limit, maximum T2)
    At gamma_phi>0: T2 < 2*T1

    Parameters
    ----------
    gamma     : relaxation rate (1/T1)
    gamma_phi : pure dephasing rate (1/T_phi)
    """
    return 2.0 / (gamma + gamma_phi)


def gamma_phi_from_T2(gamma: float, T2: float) -> float:
    """
    Compute pure dephasing rate from T1 decay rate and desired T2.

    gamma_phi = 2/T2 - gamma

    Raises ValueError if T2 > 2*T1 (physically impossible).

    Parameters
    ----------
    gamma : relaxation rate (1/T1)
    T2    : desired total dephasing time
    """
    T1 = 1.0 / gamma
    if T2 > 2 * T1:
        raise ValueError(
            f"T2={T2:.2f} > 2*T1={2*T1:.2f} — physically impossible. "
            f"Maximum T2 is 2*T1.")
    return 2.0 / T2 - gamma


def make_params(
    wq:          float,
    wt:          float,
    gamma_q:     float,
    gamma_t:     float,
    T2_q:        float = None,
    T2_t:        float = None,
    n_th_q:      float = 0.0,
    n_th_t:      float = 0.0,
) -> dict:
    """
    Build a consistent parameter dict from physical timescales.

    If T2_q is given, computes gamma_phi_q automatically.
    If T2_q is None, assumes gamma_phi=0 (T2 = 2*T1, Lindblad limit).

    Parameters
    ----------
    wq, wt    : transition frequencies
    gamma_q   : qubit relaxation rate (1/T1_q)
    gamma_t   : TLS relaxation rate   (1/T1_t)
    T2_q      : qubit total dephasing time (optional)
    T2_t      : TLS total dephasing time   (optional)
    n_th_q    : thermal photon number at qubit frequency
    n_th_t    : thermal photon number at TLS frequency

    Returns
    -------
    dict with keys: wq, wt, gamma_q, gamma_t, gamma_phi_q, gamma_phi_t,
                    n_th_q, n_th_t, T1_q, T1_t, T2_q, T2_t
    """
    T1_q = 1.0 / gamma_q
    T1_t = 1.0 / gamma_t

    if T2_q is not None:
        gamma_phi_q = gamma_phi_from_T2(gamma_q, T2_q)
    else:
        gamma_phi_q = 0.0
        T2_q = 2 * T1_q   # Lindblad limit

    if T2_t is not None:
        gamma_phi_t = gamma_phi_from_T2(gamma_t, T2_t)
    else:
        gamma_phi_t = 0.0
        T2_t = 2 * T1_t

    return {
        'wq':          wq,
        'wt':          wt,
        'gamma_q':     gamma_q,
        'gamma_t':     gamma_t,
        'gamma_phi_q': gamma_phi_q,
        'gamma_phi_t': gamma_phi_t,
        'n_th_q':      n_th_q,
        'n_th_t':      n_th_t,
        'T1_q':        T1_q,
        'T1_t':        T1_t,
        'T2_q':        T2_q,
        'T2_t':        T2_t,
    }


def params_label(p: dict) -> str:
    """
    One-line label for a parameter dict — for figure titles and filenames.
    """
    return (f"T1q={p['T1_q']:.0f} T2q={p['T2_q']:.0f} "
            f"T1t={p['T1_t']:.0f} T2t={p['T2_t']:.0f} "
            f"nth={p['n_th_q']:.3f}")
