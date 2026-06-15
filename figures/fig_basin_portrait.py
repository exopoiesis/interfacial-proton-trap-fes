#!/usr/bin/env python3
"""
Basin-of-attraction portrait for TM6v3-min (Paper #2 SS3.4) -- the "kinetic trap".

Consilium (s-2026-06-12) recommendation (chemist): a numerical basin map with the
separatrix shows the REAL (large) basin of the active state and the threshold, far
more convincingly to a chemistry reviewer than a local quadratic Lyapunov certificate
(which would certify a physically empty ~1e-9 M ball). Two basins:
  * inactive trap E0 = (0, 0, S/kF)  -- A -> 0  (dead / point of no return)
  * active state  E+ (A* ~ 4.82 mM)  -- A stays > 0

Grid of initial conditions in (A0, M0) at Fe0 = S/kF; integrate; classify by final A.
ODE + params identical to fig5_protocell_model.py (verified).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

OUTDIR = r'D:\home\ignat\project-third-matter\paper\KineticTrap\figures'

PARAMS = dict(
    k1=1e-4, Ka=7e-4, f1=5e-3, Km_f=5e-4, km=3e-2,
    k_fe_gen=5e-5, fe_supply=5e-8, kd_A=1e-4, kd_m=3e-6, kd_fe=3e-4,
)
Fe0 = PARAMS['fe_supply'] / PARAMS['kd_fe']  # S/kF = E0 Fe value


def rhs(t, y, p):
    A, M, Fe = [max(v, 0.0) for v in y]
    f1_eff = p['f1'] * M / (p['Km_f'] + M)
    hill_a = A**2 / (p['Ka']**2 + A**2) if A > 0 else 0.0
    r1 = p['k1'] * f1_eff * hill_a
    r3m = p['km'] * Fe * A
    dA = r1 - p['kd_A'] * A
    dM = r3m - p['kd_m'] * M
    dFe = p['k_fe_gen'] * A + p['fe_supply'] - r3m - p['kd_fe'] * Fe
    return [dA, dM, dFe]


def final_A(A0, M0, t_end_h=4000.0):
    sol = solve_ivp(rhs, (0, t_end_h * 3600), [A0, M0, Fe0], args=(PARAMS,),
                    method='LSODA', rtol=1e-9, atol=1e-13, max_step=2000.0)
    return max(sol.y[0, -1], 0.0)


def main():
    nA, nM = 70, 70
    A_grid = np.linspace(0.0, 2.0e-3, nA)  # 0..2 mM seed product (zoom around threshold ~Ka)
    M_grid = np.linspace(0.0, 60e-3, nM)   # 0..60 mM membrane
    AF = np.zeros((nM, nA))
    A_star_guess = 4.82e-3
    thresh = 0.5 * A_star_guess            # classify active if final A > half of A*
    print(f"Integrating {nA}x{nM} = {nA*nM} ICs (Fe0={Fe0:.3e}) ...")
    for j, M0 in enumerate(M_grid):
        for i, A0 in enumerate(A_grid):
            AF[j, i] = final_A(A0, M0)
        print(f"  row {j+1}/{nM} done", flush=True)

    active = (AF > thresh).astype(float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel (a): basin map (binary) + separatrix contour
    ax1.pcolormesh(A_grid * 1e3, M_grid * 1e3, active, cmap='RdYlBu_r',
                   shading='auto', vmin=0, vmax=1, alpha=0.85)
    cs = ax1.contour(A_grid * 1e3, M_grid * 1e3, AF, levels=[thresh],
                     colors='black', linewidths=2.0, linestyles='--')
    ax1.clabel(cs, fmt='separatrix', fontsize=9)
    ax1.plot(0, 0, 'o', color='#08306b', ms=12, mec='white', mew=1.5,
             label='E0 (inactive trap)')
    ax1.axvline(PARAMS['Ka'] * 1e3, color='grey', ls=':', lw=1.2)
    ax1.text(PARAMS['Ka'] * 1e3 + 0.02, 3, 'K$_A$', color='grey', fontsize=9)
    ax1.set_xlabel('Initial product A$_0$ (mM)', fontsize=11)
    ax1.set_ylabel('Initial membrane M$_0$ (mM)', fontsize=11)
    ax1.set_title('(a) Basins of attraction (Fe$_0$=S/k$_F$; E+ active at A*$\\approx$4.8 mM, off-panel right)',
                  fontsize=10.5, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.text(0.15, 50, 'inactive trap basin\n(A$\\to$0: dead, absorbing)', color='#08306b',
             fontsize=9, ha='left', va='center')
    ax1.text(1.4, 18, 'active basin\n(A$\\to$A*)', color='#7f0000',
             fontsize=9, ha='center', va='center')

    # Panel (b): final A heatmap (continuous)
    pcm = ax2.pcolormesh(A_grid * 1e3, M_grid * 1e3, AF * 1e3, cmap='viridis',
                         shading='auto')
    ax2.contour(A_grid * 1e3, M_grid * 1e3, AF, levels=[thresh],
                colors='white', linewidths=1.5, linestyles='--')
    fig.colorbar(pcm, ax=ax2, label='final A (mM)')
    ax2.set_xlabel('Initial product A$_0$ (mM)', fontsize=11)
    ax2.set_ylabel('Initial membrane M$_0$ (mM)', fontsize=11)
    ax2.set_title('(b) Final A (steady state reached)', fontsize=12, fontweight='bold')

    fig.suptitle('TM6v3-min kinetic trap: basin separatrix between inactive E0 and active E+',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    base = f'{OUTDIR}/fig_basin_portrait'
    fig.savefig(f'{base}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    plt.close(fig)

    # Threshold structure (the physical content; basin FRACTION is box-dependent, not reported as physical)
    print(f"Saved {base}.png/.pdf")
    print("NOTE: basin fraction is box-dependent (arbitrary A0,M0 range) -- not a physical quantity.")
    print("Physical content = ignition-threshold curve A0*(M0) (separatrix) vs Ka:")
    for j in [0, nM // 4, nM // 2, 3 * nM // 4, nM - 1]:
        row = active[j]
        ign = A_grid[np.argmax(row > 0)] * 1e3 if row.any() else float('nan')
        print(f"  M0={M_grid[j]*1e3:5.1f} mM -> ignition threshold A0* ~ {ign:.3f} mM (Ka={PARAMS['Ka']*1e3:.3f} mM)")


if __name__ == '__main__':
    main()
