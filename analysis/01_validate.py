"""
01_validate.py
==============
Step 1: Validate that the numeric closed-system solver agrees with
the analytic Rabi formula to machine precision.

Tests:
  1. Resonant case (Delta=0): full Rabi oscillations
  2. Dispersive case (Delta >> g): incomplete oscillations
  3. Intermediate detuning

Pass criterion: max|P_numeric - P_analytic| < 1e-4 for all cases.

Run from: TLS-QEC/analysis/
    python 01_validate.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
from qubit_tls.closed_system import evolve


# ─── Test cases ───────────────────────────────────────────────────────────────

CASES = [
    {'label': 'Resonant',     'wq': 1.0, 'wt': 1.0, 'g': 0.1},
    {'label': 'Near-resonant','wq': 1.0, 'wt': 0.9, 'g': 0.1},
    {'label': 'Dispersive',   'wq': 1.0, 'wt': 0.5, 'g': 0.05},
    {'label': 'Weak coupling','wq': 1.0, 'wt': 1.0, 'g': 0.01},
]

PASS_THRESHOLD = 1e-4


def run_validation():
    print("=" * 55)
    print("VALIDATION: Numeric vs Analytic (closed system)")
    print("=" * 55)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    all_passed = True

    for i, case in enumerate(CASES):
        wq, wt, g = case['wq'], case['wt'], case['g']
        delta   = wq - wt
        Omega_R = 0.5 * np.sqrt(delta**2 + 4*g**2)
        t_end   = 3 * np.pi / Omega_R   # 1.5 Rabi periods

        res = evolve(wq=wq, wt=wt, g=g, t_end=t_end, n_steps=1000,
                     save_path=f'../data/single_trajectory/validate_{i}.h5')

        error    = np.abs(res['P_e_q'] - res['P_e_analytic'])
        max_err  = np.max(error)
        passed   = max_err < PASS_THRESHOLD

        status = "PASS ✓" if passed else "FAIL ✗"
        if not passed:
            all_passed = False

        print(f"\n  {case['label']}: Δ={delta:.2f}, g={g:.3f}, "
              f"Ω_R={Omega_R:.4f}")
        print(f"    Max error: {max_err:.2e}   [{status}]")

        # Plot
        ax = axes[i]
        ax.plot(res['t'], res['P_e_q'],        lw=2,   label='Numeric')
        ax.plot(res['t'], res['P_e_analytic'], lw=1.5, ls='--', label='Analytic')
        ax2 = ax.twinx()
        ax2.plot(res['t'], error, color='red', lw=1, alpha=0.5, label='Error')
        ax2.set_ylabel('|Error|', color='red', fontsize=9)
        ax2.tick_params(colors='red')
        ax.set_title(f"{case['label']}  [{status}]  max err={max_err:.1e}")
        ax.set_xlabel('Time'); ax.set_ylabel('P_e (qubit)')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('Validation: Numeric vs Analytic', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig('../figures/validation/numeric_vs_analytic.png',
                dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved → figures/validation/numeric_vs_analytic.png")

    print("\n" + "=" * 55)
    if all_passed:
        print("  ALL TESTS PASSED — solver is validated ✓")
    else:
        print("  SOME TESTS FAILED — check solver settings ✗")
    print("=" * 55)


if __name__ == "__main__":
    run_validation()
