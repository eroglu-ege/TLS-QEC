"""
metrics_extra.py
=================
Additional metrics for quantifying Solomon model breakdown rigorously:
  - find_onset: generic "first detectable departure" finder
  - trace_distance: proper quantum distinguishability measure

These complement utils/metrics.py (ISE, max_diff, coherence_metrics,
fit_biexp, rate_discrepancy) with two new pieces:

1. A standardised way to find "where does X first become significant"
   for ANY metric array, anchored to a noise floor rather than an
   arbitrary percentage.

2. Trace distance D(rho_L, rho_S) — the proper quantum-mechanical
   measure of how distinguishable two states are, bounded in [0,1],
   with operational meaning (related to optimal measurement
   distinguishability). This is more rigorous than comparing
   populations alone because it uses the FULL density matrix.

Reference: Nielsen & Chuang, Quantum Computation and Quantum
           Information, Ch. 9 (distance measures for quantum states)
"""

import numpy as np


# ─── Generic onset finder ──────────────────────────────────────────────────────

def find_onset(
    x:          np.ndarray,
    metric:     np.ndarray,
    baseline_mask: np.ndarray = None,
    multiplier: float = 10.0,
) -> dict:
    """
    Find the first point where a metric exceeds (multiplier x baseline).

    This gives an OBJECTIVE "first detectable departure" threshold,
    anchored to the metric's own noise floor rather than an arbitrary
    percentage choice.

    Parameters
    ----------
    x : array
        The swept parameter (e.g. g/gamma_t), assumed sorted ascending.
    metric : array
        The metric values at each x (e.g. ISE, coherence_max).
        Must be non-negative and same length as x.
    baseline_mask : boolean array, optional
        Which points to use for estimating the baseline (noise floor).
        Default: the first 20% of points (assumed deep in the
        well-behaved regime).
    multiplier : float
        How many times above baseline counts as "detectable".
        Default 10x — a common convention for "clearly above noise".

    Returns
    -------
    dict:
        'baseline'      : estimated baseline (mean of masked region)
        'threshold'     : baseline * multiplier
        'onset_x'       : x value of first detectable departure
                          (None if metric never exceeds threshold)
        'onset_idx'     : index of onset_x in the x array
    """
    x      = np.asarray(x)
    metric = np.asarray(metric)

    if baseline_mask is None:
        n_baseline = max(3, int(0.2 * len(x)))
        baseline_mask = np.zeros(len(x), dtype=bool)
        baseline_mask[:n_baseline] = True

    baseline  = np.mean(metric[baseline_mask])
    threshold = baseline * multiplier

    above = np.where(metric > threshold)[0]
    if len(above) == 0:
        return {
            'baseline':  float(baseline),
            'threshold': float(threshold),
            'onset_x':   None,
            'onset_idx': None,
        }

    idx = above[0]
    return {
        'baseline':  float(baseline),
        'threshold': float(threshold),
        'onset_x':   float(x[idx]),
        'onset_idx': int(idx),
    }


def find_two_thresholds(
    x:               np.ndarray,
    metric:          np.ndarray,
    baseline_mask:   np.ndarray = None,
    onset_multiplier: float = 10.0,
    significance_level: float = 0.05,
) -> dict:
    """
    Compute both the SENSITIVITY threshold (first detectable departure,
    anchored to noise floor) and the SIGNIFICANCE threshold (first point
    exceeding a fixed physical/relative-error cutoff).

    This is the synthesis function — gives you the two-number thesis
    claim: "detectable at g/gt=X, significant at g/gt=Y".

    Parameters
    ----------
    x      : swept parameter array
    metric : relative-error-like array, e.g. |L_max - S_max| / S_max
             Should be a FRACTION (0 to 1), not a percentage.
    significance_level : fraction, default 0.05 = 5% relative error

    Returns
    -------
    dict:
        'onset'       : result of find_onset() — sensitivity threshold
        'significant_x': x value where metric first exceeds significance_level
        'significant_idx': index of that point
    """
    onset_result = find_onset(x, metric, baseline_mask, onset_multiplier)

    x = np.asarray(x)
    metric = np.asarray(metric)
    above_sig = np.where(metric > significance_level)[0]

    sig_x   = float(x[above_sig[0]]) if len(above_sig) > 0 else None
    sig_idx = int(above_sig[0]) if len(above_sig) > 0 else None

    return {
        'onset':            onset_result,
        'significant_x':    sig_x,
        'significant_idx':  sig_idx,
    }


# ─── Trace distance ───────────────────────────────────────────────────────────

def trace_distance(rho_L, rho_S_diag) -> float:
    """
    Trace distance between two density matrices.

    D(rho1, rho2) = (1/2) * Tr|rho1 - rho2|
                  = (1/2) * sum(|eigenvalues of (rho1 - rho2)|)

    Bounded in [0, 1]:
        D = 0  -> states are identical
        D = 1  -> states are perfectly distinguishable

    Operational meaning: D is the maximum probability of correctly
    distinguishing the two states in a single optimal measurement,
    above random guessing.

    Convention (standard in quantum information, e.g. Nielsen & Chuang):
        D < 0.01        : negligible difference
        0.01 < D < 0.1   : noticeable but small
        D > 0.1          : substantial, models meaningfully disagree

    Parameters
    ----------
    rho_L      : qt.Qobj, the full Lindblad density matrix at some time t
    rho_S_diag : qt.Qobj, the "Solomon-implied" density matrix —
                 constructed as diagonal in the population basis
                 (no coherences), built from Solomon's P_q(t), P_t(t)

    Returns
    -------
    D : float, trace distance
    """
    diff = (rho_L - rho_S_diag).full()
    # Eigenvalues of a Hermitian matrix (rho_L - rho_S_diag is Hermitian
    # since both rho's are Hermitian)
    eigenvalues = np.linalg.eigvalsh(diff)
    return float(0.5 * np.sum(np.abs(eigenvalues)))


def build_solomon_density_matrix(P_q: float, P_t: float):
    """
    Construct the density matrix implied by Solomon's populations,
    assuming NO coherences (diagonal in the {ee, eg, ge, gg} basis).

    This lets us compare Solomon "as if it were a full density matrix"
    against the true Lindblad rho via trace distance.

    Basis ordering (matches hamiltonian.py / lindblad.py convention):
        |ee> = 0, |eg> = 1, |ge> = 2, |gg> = 3

    Since Solomon only tracks P_q and P_t (not joint ee/gg populations),
    we assume independence: P(ee) = P_q*P_t, P(eg) = P_q*(1-P_t),
    P(ge) = (1-P_q)*P_t, P(gg) = (1-P_q)*(1-P_t).
    This is the natural "no correlation, no coherence" assumption
    consistent with Solomon's classical rate-equation picture.

    Parameters
    ----------
    P_q : float, qubit excited population from Solomon
    P_t : float, TLS excited population from Solomon

    Returns
    -------
    qt.Qobj, 4x4 diagonal density matrix
    """
    import qutip as qt

    P_ee = P_q * P_t
    P_eg = P_q * (1 - P_t)
    P_ge = (1 - P_q) * P_t
    P_gg = (1 - P_q) * (1 - P_t)

    diag = np.array([P_ee, P_eg, P_ge, P_gg])
    return qt.Qobj(np.diag(diag), dims=[[2, 2], [2, 2]])


def trace_distance_trajectory(states_L: list, P_q_S: np.ndarray,
                               P_t_S: np.ndarray) -> np.ndarray:
    """
    Compute trace distance D(t) over an entire trajectory.

    Parameters
    ----------
    states_L : list of qt.Qobj, Lindblad density matrices rho_L(t)
    P_q_S    : array, Solomon qubit population P_q(t)
    P_t_S    : array, Solomon TLS population P_t(t)
               (must be same length / time grid as states_L)

    Returns
    -------
    D : array, trace distance at each time point
    """
    D = np.zeros(len(states_L))
    for i, rho_L in enumerate(states_L):
        rho_S = build_solomon_density_matrix(P_q_S[i], P_t_S[i])
        D[i] = trace_distance(rho_L, rho_S)
    return D
