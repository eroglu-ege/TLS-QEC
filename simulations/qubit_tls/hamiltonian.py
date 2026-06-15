"""
hamiltonian.py
==============
Builds the qubit + TLS Hamiltonian and all operators.

This is the single source of truth for the model.
Every other file imports from here — nothing is ever
redefined elsewhere.

Physical model (hbar = 1 throughout):
--------------------------------------
H = (wq/2) * sz_q  +  (wt/2) * sz_t  +  g * (sp_q*sm_t + sm_q*sp_t)

    |___________|     |_____________|     |__________________________|
    qubit bare H      TLS bare H          exchange interaction (RWA)

The interaction term sp_q*sm_t + sm_q*sp_t is the
Jaynes-Cummings coupling under the rotating-wave approximation (RWA).
It conserves total excitation number, so H is block-diagonal:
  - N=0: |gg>              (1x1 block, trivial)
  - N=1: {|eg>, |ge>}      (2x2 block, all the action)
  - N=2: |ee>              (1x1 block, trivial)

Reference: Blais et al., Rev. Mod. Phys. 93, 025005 (2021), Sec. III.A
"""

import numpy as np
import qutip as qt


# ─── Single-qubit Pauli operators ────────────────────────────────────────────

def _sz() -> qt.Qobj:
    """Pauli Z: |e><e| - |g><g|  (basis: [|e>, |g>])"""
    return qt.sigmaz()

def _sp() -> qt.Qobj:
    """Raising operator sigma_+: |e><g|"""
    return qt.sigmap()

def _sm() -> qt.Qobj:
    """Lowering operator sigma_-: |g><e|"""
    return qt.sigmam()

def _id() -> qt.Qobj:
    """2x2 identity"""
    return qt.qeye(2)


# ─── Two-qubit operators (qubit ⊗ TLS) ───────────────────────────────────────
# Convention: first index = qubit, second = TLS
# qt.tensor(A, B) gives A ⊗ B

def ops():
    """
    Return all two-body operators as a dictionary.

    Returns
    -------
    dict with keys:
        'sz_q'  : sigma_z on qubit, identity on TLS
        'sz_t'  : identity on qubit, sigma_z on TLS
        'sp_q'  : sigma_+ on qubit
        'sm_q'  : sigma_- on qubit
        'sp_t'  : sigma_+ on TLS
        'sm_t'  : sigma_- on TLS
        'n_exc' : total excitation number operator (sz_q + sz_t)/2 + 1
    """
    sz_q = qt.tensor(_sz(), _id())
    sz_t = qt.tensor(_id(), _sz())
    sp_q = qt.tensor(_sp(), _id())
    sm_q = qt.tensor(_sm(), _id())
    sp_t = qt.tensor(_id(), _sp())
    sm_t = qt.tensor(_id(), _sm())

    # Total excitation number: eigenvalues 0, 1, 2
    # N_exc = (sz_q + sz_t)/2 + 1  (maps {-1,+1} eigenvalues to {0,1})
    n_exc = (sz_q + sz_t) / 2 + qt.tensor(_id(), _id())

    return {
        'sz_q':  sz_q,
        'sz_t':  sz_t,
        'sp_q':  sp_q,
        'sm_q':  sm_q,
        'sp_t':  sp_t,
        'sm_t':  sm_t,
        'n_exc': n_exc,
    }


# ─── Hamiltonian ──────────────────────────────────────────────────────────────

def build_H(wq: float, wt: float, g: float) -> qt.Qobj:
    """
    Build the Jaynes-Cummings Hamiltonian (RWA) for qubit + TLS.

    H = (wq/2)*sz_q + (wt/2)*sz_t + g*(sp_q*sm_t + sm_q*sp_t)

    Parameters
    ----------
    wq : float
        Qubit transition frequency [rad/s, or dimensionless if normalised].
    wt : float
        TLS transition frequency.
    g  : float
        Exchange coupling strength.
        RWA is valid when g << wq, wt.

    Returns
    -------
    H : qt.Qobj  (4x4 matrix, dims=[[2,2],[2,2]])
    """
    o = ops()

    H_qubit = (wq / 2) * o['sz_q']
    H_tls   = (wt / 2) * o['sz_t']
    H_int   = g * (o['sp_q'] * o['sm_t'] + o['sm_q'] * o['sp_t'])

    return H_qubit + H_tls + H_int


def build_H_rotating(wq: float, wt: float, g: float) -> qt.Qobj:
    """
    Hamiltonian in the frame rotating at wq (qubit frequency).

    In this frame the qubit term vanishes and the TLS term picks up
    the detuning Delta = wq - wt:

    H_rot = -(Delta/2)*sz_t + g*(sp_q*sm_t + sm_q*sp_t)

    This frame removes the fast wq oscillation, making numerics cheaper
    and the dispersive regime transparent.

    Reference: Blais et al. (2021), Appendix B.
    """
    delta = wq - wt
    o = ops()

    H_tls_rot = -(delta / 2) * o['sz_t']
    H_int     = g * (o['sp_q'] * o['sm_t'] + o['sm_q'] * o['sp_t'])

    return H_tls_rot + H_int


# ─── Initial states ───────────────────────────────────────────────────────────

def state(qubit: str, tls: str) -> qt.Qobj:
    """
    Return a product state |qubit, tls>.

    Parameters
    ----------
    qubit, tls : 'e' or 'g'

    Examples
    --------
    state('e', 'g')  →  |eg>  (qubit excited, TLS ground)
    state('g', 'g')  →  |gg>  (both ground)
    """
    basis = {'e': qt.basis(2, 0), 'g': qt.basis(2, 1)}
    if qubit not in basis or tls not in basis:
        raise ValueError("qubit and tls must be 'e' or 'g'")
    return qt.tensor(basis[qubit], basis[tls])


# ─── Quick sanity checks ──────────────────────────────────────────────────────

def print_H(wq: float, wt: float, g: float):
    """Print the Hamiltonian matrix and its eigenvalues."""
    H = build_H(wq, wt, g)
    print("H matrix (basis: |ee>, |eg>, |ge>, |gg>):")
    print(np.round(H.full(), 4))
    print("\nEigenvalues:", np.round(H.eigenenergies(), 4))

    delta = wq - wt
    Omega_R = 0.5 * np.sqrt(delta**2 + 4*g**2)
    print(f"\nAnalytic single-excitation eigenvalues: ±{Omega_R:.4f}")
    print(f"Generalised Rabi frequency Omega_R = {Omega_R:.4f}")


if __name__ == "__main__":
    # --- Example: resonant case (wq = wt = 1.0, g = 0.1) ---
    print("=== Resonant case: wq=wt=1.0, g=0.1 ===")
    print_H(wq=1.0, wt=1.0, g=0.1)

    print("\n=== Dispersive case: wq=1.0, wt=0.8, g=0.05 ===")
    print_H(wq=1.0, wt=0.8, g=0.05)
