#!/usr/bin/env python3
"""
Figure 6: Energy Balance for pH-Gradient-Driven CO2 Reduction.

Two-panel figure:
(a) Waterfall chart of energy budget at deltapH=6
(b) Feasibility map: net margin vs deltapH for mackinawite and greigite
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTDIR = r'D:\home\ignat\project-third-matter\paper\figures'

# Physical constants
R = 8.314       # J/(mol*K)
T = 298.15      # K
F = 96485       # C/mol
RT_F = R * T / F  # ~0.02569 V


def dE_pH(delta_pH):
    """Nernst potential from pH gradient (V)."""
    return RT_F * np.log(10) * delta_pH  # ~0.05916 * delta_pH


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # ═══════════════════════════════════════════════════════════════
    # Panel (a): Waterfall chart at deltapH = 6
    # ═══════════════════════════════════════════════════════════════
    ax1.set_title('(a) Energy budget ($\\Delta$pH = 6)', fontsize=12,
                  fontweight='bold', pad=10)

    # Values (V)
    dE_total = dE_pH(6)  # 0.3550 V
    eta_mack = 0.023     # mackinawite overpotential (Panico 2025, 23 mV)
    IR_sol = 0.020       # solution IR drop
    IR_mem = 0.006       # membrane IR drop
    dG_req = 0.171       # thermodynamic requirement (DeltaG0 / 2F)

    # Waterfall items
    labels = [
        '$\\Delta E_{pH}$\n(available)',
        '$\\eta_{mack}$\n(overpotential)',
        '$IR_{solution}$',
        '$IR_{membrane}$',
        'Remaining\n(after losses)',
        '$\\Delta G^0/2F$\n(required)',
        'Net\nmargin',
    ]

    remaining = dE_total - eta_mack - IR_sol - IR_mem
    net_margin = remaining - dG_req
    margin_pct = 100 * net_margin / dG_req

    values = [dE_total, -eta_mack, -IR_sol, -IR_mem, remaining, -dG_req, net_margin]

    # Compute bar positions for waterfall
    bottoms = [0, dE_total, dE_total - eta_mack,
               dE_total - eta_mack - IR_sol,
               0,
               remaining,
               0]

    bar_colors = ['#4CAF50', '#F44336', '#F44336', '#F44336',
                  '#2196F3', '#FF9800', '#4CAF50']
    edge_colors = ['#2E7D32', '#C62828', '#C62828', '#C62828',
                   '#1565C0', '#E65100', '#2E7D32']
    alphas = [0.85, 0.75, 0.75, 0.75, 0.85, 0.85, 0.85]

    x = np.arange(len(labels))

    for i in range(len(labels)):
        h = abs(values[i])
        b = bottoms[i] if values[i] >= 0 else bottoms[i] - h
        ax1.bar(x[i], h, bottom=b, color=bar_colors[i],
                edgecolor=edge_colors[i], linewidth=1.2, alpha=alphas[i],
                width=0.65, zorder=3)

        # Value label
        label_y = b + h + 0.008 if values[i] >= 0 else b - 0.008
        va = 'bottom' if values[i] >= 0 else 'top'
        sign = '+' if values[i] > 0 and i > 0 else ''
        if i == 4:
            sign = ''
        ax1.text(x[i], label_y,
                 f'{sign}{values[i]:+.3f} V' if i not in [0, 4, 6]
                 else f'{values[i]:.3f} V',
                 ha='center', va=va, fontsize=8.5, fontweight='bold',
                 color=edge_colors[i])

    # Connecting lines between waterfall bars
    for i in range(3):
        level = dE_total - sum([eta_mack, IR_sol, IR_mem][:i + 1])
        ax1.plot([x[i] + 0.35, x[i + 1] - 0.35], [level, level],
                 '--', color='gray', lw=0.8, alpha=0.5, zorder=2)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8, rotation=0)
    ax1.set_ylabel('Voltage (V)', fontsize=12)
    ax1.set_ylim(-0.05, dE_total + 0.08)
    ax1.axhline(0, color='black', lw=0.5)
    ax1.tick_params(direction='in', top=True, right=True)

    # Margin percentage
    ax1.text(0.98, 0.95,
             f'Net margin: {net_margin:.3f} V ({margin_pct:+.0f}%)',
             transform=ax1.transAxes, fontsize=10, ha='right', va='top',
             fontweight='bold', color='#2E7D32',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                       edgecolor='#4CAF50'))

    # ═══════════════════════════════════════════════════════════════
    # Panel (b): Feasibility map
    # ═══════════════════════════════════════════════════════════════
    ax2.set_title('(b) Feasibility vs $\\Delta$pH', fontsize=12,
                  fontweight='bold', pad=10)

    dpH = np.linspace(3, 12, 500)

    # Mackinawite: eta = 23 mV (Panico 2025, L-480)
    eta_mack_val = 0.023
    total_loss_mack = eta_mack_val + IR_sol + IR_mem
    available_mack = dE_pH(dpH) - total_loss_mack
    margin_mack = (available_mack - dG_req) / dG_req * 100

    # Greigite: eta = 400 mV
    eta_greig = 0.400
    total_loss_greig = eta_greig + IR_sol + IR_mem
    available_greig = dE_pH(dpH) - total_loss_greig
    margin_greig = (available_greig - dG_req) / dG_req * 100

    # Find crossover points
    def find_crossover(dpH_arr, margin_arr):
        for i in range(len(margin_arr) - 1):
            if margin_arr[i] < 0 and margin_arr[i + 1] >= 0:
                # Linear interpolation
                frac = -margin_arr[i] / (margin_arr[i + 1] - margin_arr[i])
                return dpH_arr[i] + frac * (dpH_arr[i + 1] - dpH_arr[i])
        return None

    cross_mack = find_crossover(dpH, margin_mack)
    cross_greig = find_crossover(dpH, margin_greig)

    # Plot
    ax2.plot(dpH, margin_mack, '-', color='#4CAF50', lw=2.5,
             label=f'Mackinawite ($\\eta$ = {eta_mack_val * 1000:.0f} mV)')
    ax2.plot(dpH, margin_greig, '--', color='#F44336', lw=2.5,
             label=f'Greigite ($\\eta$ = {eta_greig * 1000:.0f} mV)')

    # Feasibility threshold
    ax2.axhline(0, color='black', lw=1.0, ls='-', zorder=2)
    ax2.text(3.2, 3, 'Feasible', fontsize=10, color='#2E7D32',
             fontweight='bold')
    ax2.text(3.2, -15, 'Not feasible', fontsize=10, color='#C62828',
             fontweight='bold')

    # Shade feasible regions
    ax2.fill_between(dpH, 0, margin_mack,
                     where=(margin_mack >= 0),
                     color='#4CAF50', alpha=0.1, zorder=0)

    # Crossover markers
    if cross_mack is not None:
        ax2.axvline(cross_mack, color='#4CAF50', ls=':', lw=1.0, alpha=0.7)
        ax2.annotate(f'$\\Delta$pH = {cross_mack:.1f}',
                     xy=(cross_mack, 0), xytext=(cross_mack + 0.5, -25),
                     fontsize=9, color='#2E7D32',
                     arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.0))

    if cross_greig is not None:
        ax2.axvline(cross_greig, color='#F44336', ls=':', lw=1.0, alpha=0.7)
        ax2.annotate(f'$\\Delta$pH = {cross_greig:.1f}',
                     xy=(cross_greig, 0), xytext=(cross_greig - 1.5, -25),
                     fontsize=9, color='#C62828',
                     arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.0))

    # Mark operating point (deltapH=6)
    margin_at_6_mack = np.interp(6, dpH, margin_mack)
    ax2.plot(6, margin_at_6_mack, 'o', color='#2E7D32', markersize=10,
             markeredgecolor='black', markeredgewidth=1.2, zorder=5)
    ax2.annotate(f'$\\Delta$pH=6\n{margin_at_6_mack:+.0f}%',
                 xy=(6, margin_at_6_mack),
                 xytext=(6.5, margin_at_6_mack + 20),
                 fontsize=9, fontweight='bold', color='#2E7D32',
                 arrowprops=dict(arrowstyle='->', color='#2E7D32', lw=1.2))

    ax2.set_xlabel('$\\Delta$pH', fontsize=12)
    ax2.set_ylabel('Net free energy margin (%)', fontsize=12)
    ax2.set_xlim(3, 12)
    ax2.set_ylim(-80, 150)
    ax2.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax2.tick_params(direction='in', top=True, right=True)

    # Grid
    ax2.grid(True, alpha=0.2, ls='--')

    fig.tight_layout()

    # Save
    out_base = f'{OUTDIR}/fig6_energy_balance'
    fig.savefig(f'{out_base}.png')
    fig.savefig(f'{out_base}.pdf')
    plt.close(fig)
    print('Saved: fig6_energy_balance.png')
    print('Saved: fig6_energy_balance.pdf')


if __name__ == '__main__':
    main()
