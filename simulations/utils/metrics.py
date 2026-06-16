"""
metrics.py
==========
All quantitative metrics for comparing Lindblad vs Solomon.

Every metric returns a plain float or dict — no plotting, no I/O.
This is pure physics computation.

Metrics defined here:
    - ISE          : integrated squared error
    - max_diff     : peak absolute difference
    - coherence_max: peak off-diagonal density matrix element
    - coherence_integral: total coherence weight
    - fit_biexp    : biexponential decay fit → effective rates Gamma1, Gamma2
    - solomon_validity: composite score (0=invalid, 1=perfect)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid


# ─── Individual metrics ───────────────────────────────────────────────────────

def ISE(t: np.ndarray, P_L: np.ndarray, P_S: np.ndarray) -> float:
    """
    Integrated Squared Error between Lindblad and Solomon qubit populations.

    ISE = integral( (P_L(t) - P_S(t))^2 dt )

    This is the primary scalar summary of total disagreement.
    Use this for sweeps and the phase diagram.
    """
    integrand = (P_L - P_S)**2
    return float(trapezoid(integrand, t))


def max_diff(t: np.ndarray, P_L: np.ndarray, P_S: np.ndarray) -> dict:
    """
    Peak absolute difference and the time at which it occurs.

    Returns
    -------
    dict:
        'value' : float, max|P_L - P_S|
        't_peak': float, time of peak
    """
    diff = np.abs(P_L - P_S)
    idx  = np.argmax(diff)
    return {
        'value':  float(diff[idx]),
        't_peak': float(t[idx]),
    }


def coherence_metrics(t: np.ndarray, coherence: np.ndarray) -> dict:
    """
    Metrics for the off-diagonal coherence |rho_{eg,ge}(t)|.

    The coherence is the quantity Solomon sets to zero.
    Large coherence → Solomon approximation is unjustified.

    Returns
    -------
    dict:
        'max'      : peak |rho_{eg,ge}|
        't_peak'   : time of peak coherence
        'integral' : total coherence weight (trapezoid rule)
    """
    idx = np.argmax(coherence)
    return {
        'max':      float(coherence[idx]),
        't_peak':   float(t[idx]),
        'integral': float(trapezoid(coherence, t)),
    }


def fit_biexp(t: np.ndarray, P: np.ndarray) -> dict:
    """
    Fit population decay to a biexponential:
        P(t) = A1 * exp(-Gamma1 * t) + A2 * exp(-Gamma2 * t)

    Both Lindblad and Solomon predict biexponential decay (two eigenvalues
    of the Liouvillian). When the fitted rates disagree between the two
    models, Solomon has failed.

    Returns
    -------
    dict:
        'A1', 'Gamma1': fast component amplitude and rate
        'A2', 'Gamma2': slow component amplitude and rate
        'success'     : bool, whether fit converged
        'residual'    : RMS fit residual
    """
    def biexp(t, A1, G1, A2, G2):
        return A1 * np.exp(-G1 * t) + A2 * np.exp(-G2 * t)

    # Initial guess: two rates bracketing the overall decay
    t_half = t[np.argmin(np.abs(P - 0.5))] if np.any(P < 0.5) else t[-1]/2
    p0 = [0.7, 2/t_half, 0.3, 0.5/t_half]
    bounds = ([0, 0, 0, 0], [1, 100, 1, 100])

    try:
        popt, _ = curve_fit(biexp, t, P, p0=p0, bounds=bounds,
                            maxfev=10000)
        A1, G1, A2, G2 = popt
        # Sort so Gamma1 >= Gamma2 (fast then slow)
        if G1 < G2:
            A1, G1, A2, G2 = A2, G2, A1, G1

        residual = float(np.sqrt(np.mean((biexp(t, *popt) - P)**2)))
        return {
            'A1': float(A1), 'Gamma1': float(G1),
            'A2': float(A2), 'Gamma2': float(G2),
            'success': True,
            'residual': residual,
        }
    except RuntimeError:
        return {
            'A1': np.nan, 'Gamma1': np.nan,
            'A2': np.nan, 'Gamma2': np.nan,
            'success': False,
            'residual': np.nan,
        }


def rate_discrepancy(fit_L: dict, fit_S: dict) -> dict:
    """
    Compare effective decay rates between Lindblad and Solomon fits.

    Returns fractional discrepancy in each rate:
        delta_Gamma1 = |Gamma1_L - Gamma1_S| / Gamma1_S
        delta_Gamma2 = |Gamma2_L - Gamma2_S| / Gamma2_S

    Large discrepancy → Solomon gives wrong decay rates.
    """
    if not (fit_L['success'] and fit_S['success']):
        return {'delta_Gamma1': np.nan, 'delta_Gamma2': np.nan}

    dG1 = abs(fit_L['Gamma1'] - fit_S['Gamma1']) / (fit_S['Gamma1'] + 1e-12)
    dG2 = abs(fit_L['Gamma2'] - fit_S['Gamma2']) / (fit_S['Gamma2'] + 1e-12)

    return {
        'delta_Gamma1': float(dG1),
        'delta_Gamma2': float(dG2),
        'Gamma1_L': fit_L['Gamma1'],
        'Gamma1_S': fit_S['Gamma1'],
        'Gamma2_L': fit_L['Gamma2'],
        'Gamma2_S': fit_S['Gamma2'],
    }


# ─── Composite analysis ───────────────────────────────────────────────────────

def full_comparison(
    t_L:       np.ndarray,
    P_e_q_L:   np.ndarray,
    coherence: np.ndarray,
    t_S:       np.ndarray,
    P_e_q_S:   np.ndarray,
) -> dict:
    """
    Run all metrics on a Lindblad vs Solomon comparison.

    Parameters
    ----------
    t_L, P_e_q_L, coherence : Lindblad results
    t_S, P_e_q_S            : Solomon results

    Returns
    -------
    dict with all metrics — ready to print or save
    """
    # Interpolate Solomon onto Lindblad time grid
    P_S_interp = np.interp(t_L, t_S, P_e_q_S)

    ise    = ISE(t_L, P_e_q_L, P_S_interp)
    mdiff  = max_diff(t_L, P_e_q_L, P_S_interp)
    coh    = coherence_metrics(t_L, coherence)
    fit_L  = fit_biexp(t_L, P_e_q_L)
    fit_S  = fit_biexp(t_S, P_e_q_S)
    rdiff  = rate_discrepancy(fit_L, fit_S)

    return {
        'ISE':                ise,
        'max_diff':           mdiff['value'],
        't_peak_diff':        mdiff['t_peak'],
        'coherence_max':      coh['max'],
        'coherence_t_peak':   coh['t_peak'],
        'coherence_integral': coh['integral'],
        'fit_lindblad':       fit_L,
        'fit_solomon':        fit_S,
        'rate_discrepancy':   rdiff,
    }


def print_summary(metrics: dict, params: dict = None):
    """Pretty-print a metrics summary."""
    sep = "─" * 50
    print(sep)
    print("COMPARISON METRICS: Lindblad vs Solomon")
    if params:
        print(f"  g={params.get('g','?'):.4f}  "
              f"Δ={params.get('wq',1)-params.get('wt',1):.4f}  "
              f"γ_q={params.get('gamma_q','?'):.4f}  "
              f"γ_t={params.get('gamma_t','?'):.4f}")
    print(sep)
    print(f"  ISE (total disagreement)  : {metrics['ISE']:.4e}")
    print(f"  Max |P_L - P_S|           : {metrics['max_diff']:.4f}  "
          f"at t={metrics['t_peak_diff']:.2f}")
    print(f"  Peak coherence            : {metrics['coherence_max']:.4f}  "
          f"at t={metrics['coherence_t_peak']:.2f}")
    print(f"  Coherence integral        : {metrics['coherence_integral']:.4f}")
    print()
    rd = metrics['rate_discrepancy']
    if not np.isnan(rd['delta_Gamma1']):
        print(f"  Γ1: Lindblad={rd['Gamma1_L']:.4f}  "
              f"Solomon={rd['Gamma1_S']:.4f}  "
              f"discrepancy={rd['delta_Gamma1']:.1%}")
        print(f"  Γ2: Lindblad={rd['Gamma2_L']:.4f}  "
              f"Solomon={rd['Gamma2_S']:.4f}  "
              f"discrepancy={rd['delta_Gamma2']:.1%}")
    print(sep)
