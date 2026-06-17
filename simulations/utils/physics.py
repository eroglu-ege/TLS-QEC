"""
physics.py
==========
Physical helper functions shared across simulation files.
"""

import numpy as np


def n_thermal(freq_GHz: float, temp_K: float) -> float:
    """
    Mean thermal photon number: n_th = 1/(exp(hbar*w/kT) - 1)
    Uses hbar/kB = 47.99 mK/GHz.

    Examples:
        n_thermal(5.0, 0.020) -> 0.0049  (5 GHz qubit at 20 mK)
        n_thermal(5.0, 0.100) -> 0.078   (5 GHz qubit at 100 mK)
        n_thermal(1.0, 0.050) -> 0.181   (1 GHz TLS at 50 mK)
    """
    if temp_K <= 0:
        return 0.0
    x = freq_GHz * 47.99e-3 / temp_K
    return 1.0 / (np.exp(x) - 1.0)


def thermal_steady_state(n_th: float) -> float:
    """
    Excited-state population at thermal equilibrium.
    P_ss = n_th / (2*n_th + 1)
    At n_th=0: 0.0. At n_th->inf: 0.5.
    """
    return n_th / (2.0 * n_th + 1.0)


def exchange_rate(g: float, delta: float,
                  gamma_t: float, n_th_t: float = 0.0) -> float:
    """
    Qubit-TLS exchange rate (Solomon model).
    Gamma = 2*g^2*gamma_t_eff / (Delta^2 + gamma_t_eff^2)
    gamma_t_eff = gamma_t*(2*n_th_t+1)  [thermally broadened]
    """
    gamma_t_eff = gamma_t * (2.0 * n_th_t + 1.0)
    return 2.0 * g**2 * gamma_t_eff / (delta**2 + gamma_t_eff**2)
