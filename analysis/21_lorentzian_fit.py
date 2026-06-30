"""
21_lorentzian_fit.py
=====================
Lorentzian fit to simulated qubit T1_eff vs detuning.

QUESTIONS BEING ANSWERED (per Mathieu's feedback):
  Q1: Is the dip shape Lorentzian in simulation (not just analytically)?
  Q2: Do fitted parameters match input g and gamma_t?
  Q3: Does the Lorentzian shape hold across coupling regimes
      (below threshold, at threshold, near strong coupling)?

PROTOCOL:
  For each g/gamma_t value:
    1. Run Lindblad simulation at N_DELTA detuning values
    2. Fit P_e_q(t) to A*exp(-gamma_eff*t) + C to extract gamma_eff(Delta)
    3. Fit gamma_eff(Delta) - gamma_q to Lorentzian:
         L(Delta) = A_fit / (Delta^2 + w_fit^2)
    4. Compare: A_fit vs 2*g^2*gamma_t (expected)
                w_fit vs gamma_t         (expected)
    5. Compute pull statistics: (fitted - expected) / uncertainty

PULL STATISTIC:
  pull = (x_fit - x_true) / sigma_fit
  |pull| < 1: fitted value within 1 sigma of truth (good fit)
  |pull| > 3: significant deviation (shape not Lorentzian or wrong model)

Run from: TLS-QEC/analysis/
    python 21_lorentzian_fit.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'simulations'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from tqdm import tqdm

from qubit_tls.lindblad import evolve as lindblad_evolve
from qubit_tls.solomon  import evolve as solomon_evolve
from utils.physics import make_params, gamma_phi_from_T2

# ─── CONFIGURE HERE ───────────────────────────────────────────────────────────
GAMMA_Q   = 0.01
GAMMA_T   = 0.005
T2_Q      = 50.0
N_TH      = 0.1

PARAMS = make_params(
    wq=1.0, wt=1.0,
    gamma_q=GAMMA_Q, gamma_t=GAMMA_T,
    T2_q=T2_Q, n_th_q=N_TH, n_th_t=N_TH,
)
GAMMA_PHI = PARAMS['gamma_phi_q']

# Three coupling regimes to compare
G_OVER_GAMMA_T = [0.5, 1.0, 2.0]
G_VALUES       = [r * GAMMA_T for r in G_OVER_GAMMA_T]

# Detuning sweep — denser near zero, sparser far out
DELTA_FINE   = np.linspace(-0.05, 0.05, 15)
DELTA_COARSE = np.array([-0.15, -0.10, 0.10, 0.15])
DELTA_VALUES = np.unique(np.concatenate([DELTA_FINE, DELTA_COARSE]))
DELTA_VALUES.sort()

T_END   = 800
N_STEPS = 600

TAG = f"T2q{T2_Q:.0f}_nth{N_TH:.3f}"
# ──────────────────────────────────────────────────────────────────────────────

FIXED_BASE = {k: PARAMS[k] for k in
              ['gamma_q','gamma_t','gamma_phi_q','gamma_phi_t',
               'n_th_q','n_th_t']}


def exp_decay(t, A, gamma_eff, C):
    """Biexponential with offset: A*exp(-gamma_eff*t) + C"""
    return A * np.exp(-gamma_eff * t) + C


def lorentzian(delta, A_lor, w):
    """Lorentzian in detuning: A/(Delta^2 + w^2)"""
    return A_lor / (delta**2 + w**2)


def fit_gamma_eff(t, P_e_q, gamma_q):
    """Fit decay curve, return effective decay rate and uncertainty."""
    try:
        P_ss = N_TH / (2*N_TH + 1)
        popt, pcov = curve_fit(
            exp_decay, t, P_e_q,
            p0=[1.0 - P_ss, gamma_q * 2, P_ss],
            bounds=([0, gamma_q*0.5, 0], [2.0, gamma_q*50, 0.5]),
            maxfev=5000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt[1], perr[1]   # gamma_eff, sigma_gamma_eff
    except Exception:
        return np.nan, np.nan


def run_one_regime(g, g_ratio):
    """Run detuning sweep for one g value, fit Lorentzian."""
    print(f"\n  g/gamma_t={g_ratio:.1f}  (g={g:.5f})")

    gamma_eff_vals = np.zeros(len(DELTA_VALUES))
    gamma_eff_errs = np.zeros(len(DELTA_VALUES))
    gamma_eff_S    = np.zeros(len(DELTA_VALUES))

    for i, delta in enumerate(tqdm(DELTA_VALUES, desc=f"    Delta sweep")):
        wq = 1.0
        wt = wq - delta

        res_L = lindblad_evolve(wq=wq, wt=wt, g=g, **FIXED_BASE,
                                t_end=T_END, n_steps=N_STEPS)
        res_S = solomon_evolve(wq=wq, wt=wt, g=g, **FIXED_BASE,
                               t_end=T_END, n_steps=N_STEPS)

        geff, gerr = fit_gamma_eff(res_L['t'], res_L['P_e_q'], GAMMA_Q)
        gamma_eff_vals[i] = geff
        gamma_eff_errs[i] = gerr

        geff_S, _ = fit_gamma_eff(res_S['t'], res_S['P_e_q'], GAMMA_Q)
        gamma_eff_S[i] = geff_S

    # Purcell enhancement = gamma_eff - gamma_q
    purcell_vals = gamma_eff_vals - GAMMA_Q
    purcell_errs = gamma_eff_errs
    purcell_S    = gamma_eff_S - GAMMA_Q

    # Fit Lorentzian to purcell_vals vs DELTA_VALUES
    try:
        A_expected = 2 * g**2 * GAMMA_T
        w_expected = GAMMA_T

        popt_L, pcov_L = curve_fit(
            lorentzian, DELTA_VALUES, purcell_vals,
            p0=[A_expected, w_expected],
            sigma=purcell_errs,
            absolute_sigma=True,
            maxfev=5000
        )
        perr_L = np.sqrt(np.diag(pcov_L))

        popt_S, pcov_S = curve_fit(
            lorentzian, DELTA_VALUES, purcell_S,
            p0=[A_expected, w_expected],
            maxfev=5000
        )
        perr_S = np.sqrt(np.diag(pcov_S))

    except Exception as e:
        print(f"    Fit failed: {e}")
        popt_L = perr_L = popt_S = perr_S = [np.nan, np.nan]

    # Pull statistics
    A_fit_L, w_fit_L = popt_L
    A_err_L, w_err_L = perr_L

    pull_A = (A_fit_L - A_expected) / A_err_L if A_err_L > 0 else np.nan
    pull_w = (w_fit_L - w_expected) / w_err_L if w_err_L > 0 else np.nan

    print(f"    Expected:  A = {A_expected:.6f}  w = {w_expected:.5f}")
    print(f"    Fitted L:  A = {A_fit_L:.6f} ± {A_err_L:.6f}  "
          f"w = {w_fit_L:.5f} ± {w_err_L:.5f}")
    print(f"    Pull:      A_pull = {pull_A:.3f}   w_pull = {pull_w:.3f}")
    print(f"    {'GOOD FIT' if abs(pull_A)<2 and abs(pull_w)<2 else 'POOR FIT'}")

    return {
        'delta':         DELTA_VALUES,
        'gamma_eff_L':   gamma_eff_vals,
        'gamma_eff_err': gamma_eff_errs,
        'gamma_eff_S':   gamma_eff_S,
        'purcell_L':     purcell_vals,
        'purcell_S':     purcell_S,
        'purcell_err':   purcell_errs,
        'popt_L':        popt_L,  'perr_L': perr_L,
        'popt_S':        popt_S,  'perr_S': perr_S,
        'A_expected':    A_expected,
        'w_expected':    w_expected,
        'pull_A':        pull_A,
        'pull_w':        pull_w,
        'g':             g,
        'g_ratio':       g_ratio,
    }


def run():
    os.makedirs('../data/sweeps', exist_ok=True)
    os.makedirs('../figures/sweeps', exist_ok=True)

    print(f"Lorentzian fit: qubit T1_eff vs detuning")
    print(f"  gamma_q={GAMMA_Q}  gamma_t={GAMMA_T}  T2_q={T2_Q}")
    print(f"  gamma_phi={GAMMA_PHI:.4f}  n_th={N_TH}")
    print(f"  {len(DELTA_VALUES)} detuning points, 3 coupling regimes")

    all_results = {}
    for g, ratio in zip(G_VALUES, G_OVER_GAMMA_T):
        all_results[ratio] = run_one_regime(g, ratio)

    # Save
    save_dict = {}
    for ratio, r in all_results.items():
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                save_dict[f"r{ratio:.1f}_{k}"] = v
            elif not isinstance(v, dict):
                save_dict[f"r{ratio:.1f}_{k}"] = v
    np.savez(f'../data/sweeps/lorentzian_fit_{TAG}.npz', **save_dict,
             delta_values=DELTA_VALUES, g_ratios=G_OVER_GAMMA_T)
    print(f"\nData → data/sweeps/lorentzian_fit_{TAG}.npz")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 3, wspace=0.38, hspace=0.45)

    delta_fine = np.linspace(DELTA_VALUES.min(), DELTA_VALUES.max(), 300)
    colors = {'L': 'C0', 'S': 'C1'}

    for col, (ratio, r) in enumerate(all_results.items()):
        g = r['g']

        # Row 0: gamma_eff(Delta) with Lorentzian fit
        ax = fig.add_subplot(gs[0, col])
        ax.errorbar(r['delta'], r['gamma_eff_L'], yerr=r['gamma_eff_err'],
                   fmt='o', color='C0', ms=5, capsize=3, label='Lindblad')
        ax.plot(r['delta'], r['gamma_eff_S'], 's', color='C1', ms=5,
               label='Solomon')

        if not np.isnan(r['popt_L'][0]):
            fit_curve = lorentzian(delta_fine, *r['popt_L']) + GAMMA_Q
            ax.plot(delta_fine, fit_curve, 'k-', lw=2,
                   label=f'Lor. fit\n$w$={r["popt_L"][1]:.4f}')
        # Analytic
        an_curve = lorentzian(delta_fine, r['A_expected'], r['w_expected']) + GAMMA_Q
        ax.plot(delta_fine, an_curve, 'r--', lw=1.5, label='Analytic')
        ax.axhline(GAMMA_Q, color='gray', ls=':', lw=1, label='$\\gamma_q$')
        ax.set_title(f'$g/\\gamma_t$={ratio:.1f}', fontsize=12)
        ax.set_xlabel('$\\Delta$'); ax.set_ylabel('$\\gamma_{{eff}}$')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        # Row 1: Purcell enhancement (gamma_eff - gamma_q)
        ax2 = fig.add_subplot(gs[1, col])
        ax2.errorbar(r['delta'], r['purcell_L'], yerr=r['purcell_err'],
                    fmt='o', color='C0', ms=5, capsize=3, label='Lindblad')
        ax2.plot(r['delta'], r['purcell_S'], 's', color='C1', ms=5,
                label='Solomon')
        if not np.isnan(r['popt_L'][0]):
            ax2.plot(delta_fine, lorentzian(delta_fine, *r['popt_L']),
                    'k-', lw=2, label=(
                        f'Fit: A={r["popt_L"][0]:.4f}\n'
                        f'Expected: {r["A_expected"]:.4f}'))
        ax2.plot(delta_fine, lorentzian(delta_fine, r['A_expected'], r['w_expected']),
                'r--', lw=1.5, label='Analytic')
        ax2.set_xlabel('$\\Delta$')
        ax2.set_ylabel('$\\gamma_{{eff}} - \\gamma_q$ (Purcell)')
        ax2.legend(fontsize=7); ax2.grid(True, alpha=0.3)

        # Row 2: Residuals + pull
        ax3 = fig.add_subplot(gs[2, col])
        if not np.isnan(r['popt_L'][0]):
            fitted = lorentzian(r['delta'], *r['popt_L'])
            residuals = r['purcell_L'] - fitted
            pulls = residuals / r['purcell_err']
            ax3.bar(r['delta'], pulls, width=0.003, color='C0', alpha=0.7)
            ax3.axhline(0,  color='black', lw=1)
            ax3.axhline( 2, color='red', ls='--', lw=1, label='$\\pm 2\\sigma$')
            ax3.axhline(-2, color='red', ls='--', lw=1)
            ax3.set_xlabel('$\\Delta$')
            ax3.set_ylabel('Pull = residual/$\\sigma$')
            ax3.set_title(
                f'Pulls  |pull_A|={abs(r["pull_A"]):.2f}  '
                f'|pull_w|={abs(r["pull_w"]):.2f}',
                fontsize=10)
            ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

    fig.suptitle(
        rf'Lorentzian fit: qubit $T_1^{{eff}}$ vs $\Delta$  '
        rf'($\gamma_q={GAMMA_Q}$, $\gamma_t={GAMMA_T}$, '
        rf'$T_2^q={T2_Q}$, $n_{{th}}={N_TH}$)',
        fontsize=13)

    plt.savefig(f'../figures/sweeps/lorentzian_fit_{TAG}.png',
                dpi=150, bbox_inches='tight')
    print(f"Figure → figures/sweeps/lorentzian_fit_{TAG}.png")

    # Pull summary table
    print(f"\n{'='*60}")
    print(f"PULL SUMMARY")
    print(f"{'='*60}")
    print(f"{'g/γt':>8} {'A_exp':>10} {'A_fit':>10} {'pull_A':>8} "
          f"{'w_exp':>10} {'w_fit':>10} {'pull_w':>8}")
    print("-"*60)
    for ratio, r in all_results.items():
        print(f"{ratio:>8.1f} {r['A_expected']:>10.5f} "
              f"{r['popt_L'][0]:>10.5f} {r['pull_A']:>8.3f} "
              f"{r['w_expected']:>10.5f} {r['popt_L'][1]:>10.5f} "
              f"{r['pull_w']:>8.3f}")

    plt.show()


if __name__ == "__main__":
    run()
