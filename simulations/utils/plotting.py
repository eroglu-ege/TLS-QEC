"""
plotting.py
===========
All figure generation lives here.

Each function takes result dicts from lindblad.py or solomon.py
and produces a matplotlib figure. Nothing is computed here —
this file only handles visualisation.

Usage:
    from utils.plotting import plot_comparison
    fig = plot_comparison(lindblad_result, solomon_result)
    fig.savefig('figures/comparison.png', dpi=150)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─── Style ───────────────────────────────────────────────────────────────────

COLORS = {
    'qubit_lindblad':  '#1f77b4',   # blue
    'tls_lindblad':    '#ff7f0e',   # orange
    'qubit_solomon':   '#2ca02c',   # green
    'tls_solomon':     '#d62728',   # red
    'coherence':       '#9467bd',   # purple
    'analytic':        'black',
}


def _style_ax(ax, xlabel='Time', ylabel='', title=''):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.02, 1.05)


# ─── Individual plots ─────────────────────────────────────────────────────────

def plot_rabi(closed_result: dict, save_path: str = None) -> plt.Figure:
    """
    Plot closed-system Rabi oscillations with analytic overlay.
    """
    r   = closed_result
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(r['t'], r['P_e_q'], color=COLORS['qubit_lindblad'],
            label='Qubit $P_e$ (numeric)', lw=2)
    ax.plot(r['t'], r['P_e_t'], color=COLORS['tls_lindblad'],
            label='TLS $P_e$ (numeric)', lw=2, ls='--')
    ax.plot(r['t'], r['P_e_analytic'], color=COLORS['analytic'],
            label='Analytic', lw=1, ls=':')

    p = r['params']
    delta = p['wq'] - p['wt']
    Omega_R = 0.5 * np.sqrt(delta**2 + 4*p['g']**2)
    _style_ax(ax,
              ylabel='Excited-state population',
              title=rf"Closed system: $\Delta$={delta:.2f}, $g$={p['g']:.3f}, "
                    rf"$\Omega_R$={Omega_R:.3f}")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_lindblad(lindblad_result: dict, save_path: str = None) -> plt.Figure:
    """
    Plot Lindblad populations and coherence decay.
    """
    r   = lindblad_result
    fig = plt.figure(figsize=(12, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    # Left: populations
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(r['t'], r['P_e_q'], color=COLORS['qubit_lindblad'],
             label='Qubit $P_e$', lw=2)
    ax1.plot(r['t'], r['P_e_t'], color=COLORS['tls_lindblad'],
             label='TLS $P_e$', lw=2, ls='--')
    p = r['params']
    _style_ax(ax1,
              ylabel='Population',
              title=rf"Lindblad: $g$={p['g']:.3f}, "
                    rf"$\gamma_q$={p['gamma_q']:.4f}, "
                    rf"$\gamma_t$={p['gamma_t']:.4f}")

    # Right: coherence on log scale
    ax2 = fig.add_subplot(gs[1])
    coh = r['coherence']
    coh_safe = np.where(coh > 0, coh, 1e-16)   # avoid log(0)
    ax2.semilogy(r['t'], coh_safe, color=COLORS['coherence'], lw=2)
    ax2.set_xlabel('Time', fontsize=11)
    ax2.set_ylabel(r'$|\rho_{eg,ge}|$', fontsize=11)
    ax2.set_title('Off-diagonal coherence (log scale)')
    ax2.tick_params(labelsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_comparison(
    lindblad_result: dict,
    solomon_result:  dict,
    save_path: str = None,
) -> plt.Figure:
    """
    Side-by-side comparison of Lindblad (full quantum) vs Solomon (rate eqs).

    This is the key figure for Task 3: visualise where the two models agree
    and where they diverge.
    """
    r_L = lindblad_result
    r_S = solomon_result

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # ── Left: qubit population ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(r_L['t'], r_L['P_e_q'],
             color=COLORS['qubit_lindblad'], lw=2,
             label='Lindblad (full quantum)')
    ax1.plot(r_S['t'], r_S['P_e_q'],
             color=COLORS['qubit_solomon'], lw=2, ls='--',
             label='Solomon (rate eqs)')
    _style_ax(ax1, ylabel='Qubit $P_e$', title='Qubit population')

    # ── Middle: TLS population ────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(r_L['t'], r_L['P_e_t'],
             color=COLORS['tls_lindblad'], lw=2,
             label='Lindblad')
    ax2.plot(r_S['t'], r_S['P_e_t'],
             color=COLORS['tls_solomon'], lw=2, ls='--',
             label='Solomon')
    _style_ax(ax2, ylabel='TLS $P_e$', title='TLS population')

    # ── Right: difference |Lindblad - Solomon| ────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    # Interpolate to common time grid
    t_common = r_L['t']
    P_S_interp = np.interp(t_common, r_S['t'], r_S['P_e_q'])
    diff = np.abs(r_L['P_e_q'] - P_S_interp)

    ax3.semilogy(t_common, np.where(diff > 0, diff, 1e-16),
                 color='black', lw=2)
    ax3.set_xlabel('Time', fontsize=11)
    ax3.set_ylabel(r'$|P_e^{\rm Lindblad} - P_e^{\rm Solomon}|$', fontsize=11)
    ax3.set_title('Discrepancy (log scale)')
    ax3.tick_params(labelsize=10)

    # Add parameter info
    p = r_L['params']
    fig.suptitle(
        rf"$\Delta$={p['wq']-p['wt']:.3f}, $g$={p['g']:.3f}, "
        rf"$\gamma_q$={p['gamma_q']:.4f}, $\gamma_t$={p['gamma_t']:.4f}",
        fontsize=12, y=1.01
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_regime_sweep(
    results: list,
    sweep_param: str,
    sweep_values: list,
    save_path: str = None,
) -> plt.Figure:
    """
    Plot Lindblad vs Solomon discrepancy as a function of a swept parameter.

    Used for Task 3: identify where Solomon breaks down.

    Parameters
    ----------
    results      : list of (lindblad_result, solomon_result) tuples
    sweep_param  : label for x-axis, e.g. 'g/Delta' or 'g/gamma_t'
    sweep_values : values of the swept parameter
    """
    max_diffs = []
    for r_L, r_S in results:
        t = r_L['t']
        P_S = np.interp(t, r_S['t'], r_S['P_e_q'])
        max_diffs.append(np.max(np.abs(r_L['P_e_q'] - P_S)))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(sweep_values, max_diffs, 'o-', lw=2, color='black')
    ax.set_xlabel(sweep_param, fontsize=12)
    ax.set_ylabel(r'max$|P_e^{\rm Lindblad} - P_e^{\rm Solomon}|$', fontsize=11)
    ax.set_title('Solomon approximation breakdown', fontsize=13)
    ax.grid(True, which='both', alpha=0.3)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
