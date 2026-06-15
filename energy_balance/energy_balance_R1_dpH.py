#!/usr/bin/env python3
"""
Energy balance for R1: CO₂ + H₂O + 2e⁻ → HCOO⁻ + OH⁻
on mackinawite (FeS) membrane driven by pH gradient.

Project Third Matter, Task A3.
"""

import json
import os
import numpy as np

# === Constants ===
F = 96485.0       # Faraday constant, C/mol
R_gas = 8.314     # Gas constant, J/(mol·K)
T = 298.15        # Temperature, K

# === Thermodynamic data ===
DG0_R1 = 33000.0  # ΔG°(CO₂ → HCOO⁻) at pH 7, 25°C [J/mol]

# === Loss parameters ===
eta_mackinawite = 0.023   # overpotential mackinawite [V] (Panico 2025, L-480)
eta_greigite    = 0.400   # overpotential greigite [V]
IR_solution     = 0.020   # ohmic loss in solution [V]
IR_membrane     = 0.006   # ohmic loss through pentlandite 500 nm [V]
total_IR        = IR_solution + IR_membrane  # 0.026 V

# === Faradaic efficiencies ===
FE_values = [0.08, 0.10, 0.20]  # 8%, 10%, 20%

# === Currents (Barge data) ===
I_spontaneous = 5e-6    # 5 µA spontaneous
I_HS_driven   = 1.2e-3  # 1.2 mA with HS⁻ + NO₃⁻

# === Scan ΔpH from 2 to 10 ===
dpH_range = np.arange(2, 11, 1)

def calc_energy_balance(dpH, eta_cathode):
    """Calculate energy balance for given ΔpH and cathode overpotential."""
    # Nernst potential from pH gradient
    # For H⁺/e⁻ coupling: ΔE = 0.0592 × ΔpH (at 25°C, per electron for 1H⁺/1e⁻)
    # More precisely: 2.303 * R * T / F = 0.05916 V at 298.15 K
    nernst_factor = 2.303 * R_gas * T / F  # 0.05916 V
    dE_pH = nernst_factor * dpH  # available EMF [V]

    # Available electrical energy for 2-electron transfer
    dG_available = 2 * F * dE_pH  # [J/mol]

    # Total losses
    total_loss_V = eta_cathode + total_IR  # [V]

    # Net driving force
    dE_net = dE_pH - total_loss_V  # [V]
    dG_net = 2 * F * dE_net  # [J/mol], available after losses

    # Margin relative to ΔG° of R1
    # Positive margin = reaction is thermodynamically favorable
    margin_pct = (dG_net - DG0_R1) / DG0_R1 * 100.0

    return {
        'dpH': int(dpH),
        'dE_pH_V': round(dE_pH, 4),
        'dG_available_kJ': round(dG_available / 1000, 2),
        'losses_V': round(total_loss_V, 3),
        'dE_net_V': round(dE_net, 4),
        'dG_net_kJ': round(dG_net / 1000, 2),
        'margin_pct': round(margin_pct, 1),
        'feasible': bool(dG_net >= DG0_R1)
    }


def calc_formate_rate(I_A, FE, n_electrons=2):
    """
    Calculate formate production rate.
    rate = FE * I / (n * F)  [mol/s]
    """
    return FE * I_A / (n_electrons * F)


def find_min_dpH(eta_cathode, tolerance=0.01):
    """
    Find minimum ΔpH where ΔG_net ≥ ΔG°_R1.
    Solve: 2F × (0.0592 × ΔpH - η - IR) = ΔG°
    → ΔpH = (ΔG°/(2F) + η + IR) / 0.0592
    """
    nernst_factor = 2.303 * R_gas * T / F
    total_loss = eta_cathode + total_IR
    # 2F * (nernst * dpH - loss) = DG0
    # dpH = (DG0/(2F) + loss) / nernst
    dpH_min = (DG0_R1 / (2 * F) + total_loss) / nernst_factor
    return dpH_min


# === Main calculation ===
print("=" * 85)
print("ENERGY BALANCE: R1 (CO₂ + H₂O + 2e⁻ → HCOO⁻ + OH⁻) on mackinawite")
print("=" * 85)

# --- Table for mackinawite ---
print(f"\n{'─'*85}")
print(f"  Mackinawite (η = {eta_mackinawite*1000:.0f} mV), IR_total = {total_IR*1000:.0f} mV")
print(f"  ΔG°(R1) = {DG0_R1/1000:.1f} kJ/mol")
print(f"{'─'*85}")
print(f"  ΔpH │ ΔE(pH)  │ ΔG_avail │ Losses │ ΔE_net  │ ΔG_net  │ Margin  │ OK?")
print(f"       │   [V]   │ [kJ/mol] │  [mV]  │   [V]   │ [kJ/mol]│   [%]   │    ")
print(f"{'─'*85}")

results_mack = []
for dpH in dpH_range:
    r = calc_energy_balance(dpH, eta_mackinawite)
    results_mack.append(r)
    ok = "YES" if r['feasible'] else " no"
    print(f"   {r['dpH']:2d}  │ {r['dE_pH_V']:7.4f} │ {r['dG_available_kJ']:8.2f} │  {r['losses_V']*1000:4.0f}  │ {r['dE_net_V']:7.4f} │ {r['dG_net_kJ']:7.2f} │ {r['margin_pct']:+7.1f} │ {ok}")

# --- Table for greigite ---
print(f"\n{'─'*85}")
print(f"  Greigite (η = {eta_greigite*1000:.0f} mV), IR_total = {total_IR*1000:.0f} mV")
print(f"{'─'*85}")
print(f"  ΔpH │ ΔE(pH)  │ ΔG_avail │ Losses │ ΔE_net  │ ΔG_net  │ Margin  │ OK?")
print(f"       │   [V]   │ [kJ/mol] │  [mV]  │   [V]   │ [kJ/mol]│   [%]   │    ")
print(f"{'─'*85}")

results_grei = []
for dpH in dpH_range:
    r = calc_energy_balance(dpH, eta_greigite)
    results_grei.append(r)
    ok = "YES" if r['feasible'] else " no"
    print(f"   {r['dpH']:2d}  │ {r['dE_pH_V']:7.4f} │ {r['dG_available_kJ']:8.2f} │  {r['losses_V']*1000:4.0f}  │ {r['dE_net_V']:7.4f} │ {r['dG_net_kJ']:7.2f} │ {r['margin_pct']:+7.1f} │ {ok}")

# --- Minimum ΔpH ---
dpH_min_mack = find_min_dpH(eta_mackinawite)
dpH_min_grei = find_min_dpH(eta_greigite)

print(f"\n{'='*85}")
print(f"  MINIMUM ΔpH for R1 feasibility:")
print(f"    Mackinawite (η=23 mV): ΔpH_min = {dpH_min_mack:.2f}")
print(f"    Greigite   (η=400 mV): ΔpH_min = {dpH_min_grei:.2f}")
print(f"{'='*85}")

# --- Formate production rates ---
print(f"\n{'─'*85}")
print(f"  Formate production rate [mol/s] and [µmol/h]")
print(f"{'─'*85}")
print(f"  {'FE':>5s} │ I=5 µA (spont.) │ I=1.2 mA (HS⁻)")
print(f"  {'':>5s} │ mol/s   µmol/h  │ mol/s     µmol/h")
print(f"{'─'*85}")

rate_results = {}
for fe in FE_values:
    r_spont = calc_formate_rate(I_spontaneous, fe)
    r_hs = calc_formate_rate(I_HS_driven, fe)
    r_spont_umol_h = r_spont * 1e6 * 3600
    r_hs_umol_h = r_hs * 1e6 * 3600
    print(f"  {fe*100:4.0f}% │ {r_spont:.2e}  {r_spont_umol_h:7.4f}  │ {r_hs:.2e}   {r_hs_umol_h:7.2f}")
    rate_results[f'FE_{fe:.0%}'] = {
        'I_5uA_mol_s': f'{r_spont:.3e}',
        'I_5uA_umol_h': round(r_spont_umol_h, 5),
        'I_1200uA_mol_s': f'{r_hs:.3e}',
        'I_1200uA_umol_h': round(r_hs_umol_h, 3)
    }

# --- Key scenario: ΔpH=6, FE=10%, I=1.2 mA ---
print(f"\n{'='*85}")
print(f"  KEY SCENARIO: ΔpH=6, FE=10%, I=1.2 mA")
print(f"{'='*85}")
dpH6 = calc_energy_balance(6, eta_mackinawite)
rate_key = calc_formate_rate(I_HS_driven, 0.10)
rate_key_umol_h = rate_key * 1e6 * 3600
print(f"  ΔE(pH=6) = {dpH6['dE_pH_V']:.4f} V")
print(f"  ΔE_net   = {dpH6['dE_net_V']:.4f} V")
print(f"  ΔG_net   = {dpH6['dG_net_kJ']:.2f} kJ/mol (need ≥33.0)")
print(f"  Margin   = {dpH6['margin_pct']:+.1f}%")
print(f"  Formate rate = {rate_key:.3e} mol/s = {rate_key_umol_h:.3f} µmol/h")
# Accumulation in 10 µL chamber
V_chamber_L = 10e-6  # 10 µL = 10e-6 L
conc_rate_mM_h = (rate_key / V_chamber_L) * 1000 * 3600  # mM/h
print(f"  In 10 µL chamber: {conc_rate_mM_h:.2f} mM/h")
time_to_1mM = 1.0 / conc_rate_mM_h  # hours
time_to_50mM = 50.0 / conc_rate_mM_h
print(f"  Time to 1 mM: {time_to_1mM:.1f} h")
print(f"  Time to 50 mM: {time_to_50mM:.1f} h (without consumption)")

# --- Power budget ---
print(f"\n{'─'*85}")
print(f"  Power budget comparison")
print(f"{'─'*85}")
P_gross = I_HS_driven * dpH6['dE_pH_V']
P_net = I_HS_driven * dpH6['dE_net_V']
P_useful = 0.10 * P_net  # FE=10%
print(f"  Gross power (ΔpH=6): P = I×ΔE = {P_gross*1e6:.1f} µW")
print(f"  Net power (after losses): P = {P_net*1e6:.1f} µW")
print(f"  Useful power (FE=10%): P = {P_useful*1e6:.2f} µW")
print(f"  MEMORY.md reference: 1.43e-10 W needed, margin 142000×")
print(f"  Our P_useful = {P_useful:.2e} W >> 1.43e-10 W ✓")

# --- G3c architecture check ---
print(f"\n{'─'*85}")
print(f"  G3c Architecture: Chamber 1 (pH 2-3) — membrane — Chamber 2 (pH 8-9)")
print(f"{'─'*85}")
for pH1 in [2.0, 2.5, 3.0]:
    for pH2 in [8.0, 8.5, 9.0]:
        dpH_arch = pH2 - pH1
        r = calc_energy_balance(dpH_arch, eta_mackinawite)
        ok = "OK" if r['feasible'] else "NO"
        print(f"  pH1={pH1:.1f}, pH2={pH2:.1f} → ΔpH={dpH_arch:.1f}: "
              f"ΔG_net={r['dG_net_kJ']:+.1f} kJ/mol, margin={r['margin_pct']:+.1f}% [{ok}]")

# === Save results ===
# Convert numpy types to native Python for JSON serialization
def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

output = {
    'description': 'Energy balance R1: CO2 + H2O + 2e- -> HCOO- + OH- on mackinawite/greigite',
    'parameters': {
        'T_K': T,
        'DG0_R1_kJ_mol': DG0_R1 / 1000,
        'eta_mackinawite_mV': eta_mackinawite * 1000,
        'eta_greigite_mV': eta_greigite * 1000,
        'IR_solution_mV': IR_solution * 1000,
        'IR_membrane_mV': IR_membrane * 1000,
        'nernst_factor_V': round(2.303 * R_gas * T / F, 5)
    },
    'mackinawite_scan': results_mack,
    'greigite_scan': results_grei,
    'minimum_dpH': {
        'mackinawite': round(dpH_min_mack, 2),
        'greigite': round(dpH_min_grei, 2)
    },
    'formate_rates': rate_results,
    'key_scenario': {
        'dpH': 6,
        'FE': 0.10,
        'I_mA': 1.2,
        'dE_net_V': dpH6['dE_net_V'],
        'dG_net_kJ': dpH6['dG_net_kJ'],
        'margin_pct': dpH6['margin_pct'],
        'rate_mol_s': f'{rate_key:.3e}',
        'rate_umol_h': round(rate_key_umol_h, 3),
        'conc_rate_mM_h_10uL': round(conc_rate_mM_h, 2),
        'time_to_1mM_h': round(time_to_1mM, 1),
        'time_to_50mM_h': round(time_to_50mM, 1)
    }
}

results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(results_dir, exist_ok=True)
outpath = os.path.join(results_dir, 'energy_balance_R1.json')
with open(outpath, 'w', encoding='utf-8') as f:
    json.dump(to_native(output), f, indent=2, ensure_ascii=False)
print(f"\n[SAVED] {outpath}")
print("Done.")
