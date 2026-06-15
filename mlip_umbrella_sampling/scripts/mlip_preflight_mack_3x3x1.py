#!/usr/bin/env python3
"""
MLIP plain MD pre-flight для 3×3×1 mackinawite + 12 H2O + 1 H+ (s136 W2 consilium gate).

Цель: проверить, воспроизводят ли foundation MLIP (MACE-MP-0 / CHGNet-v0.3) Fe-chemi
кинетический капкан на этой системе (DFT s119 показал 100% Fe-chemi за 14 ps).

Вход: 3×3×1 mack + 12 H2O + H+ structure.xyz (73 атома, +1 charge → MLIP ignored).
Выход: trajectory.xyz + fe_h_distance.csv + summary.json (per-frame d_H_to_Fe, d_H_to_S, d_H_to_O).

Pass criterion (для US deploy): fraction(frames where d_min(H_38, Fe) < 1.8 Å) > 0.95.
Fail criterion (T6 paper claim): MLIP не reproducirует Fe-chemi.

Foundation MLIP caveat: charge=+1 не поддерживается → система treated as neutral.
Это методологическая часть paper benchmark.

Usage:
  python mlip_preflight_mack_3x3x1.py --backend mace --steps 10000
  python mlip_preflight_mack_3x3x1.py --backend chgnet --steps 10000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def setup_calculator(backend: str):
    """Foundation MLIP calculator (MACE-MP-0 или CHGNet-v0.3)."""
    if backend == "mace":
        from mace.calculators import mace_mp
        # MACE-MP-0 medium model — default foundation MLIP. Float32 для RTX 4070.
        calc = mace_mp(model="medium", default_dtype="float32", device="cuda")
        return calc, "mace-mp-0-medium"
    elif backend == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        calc = CHGNetCalculator(use_device="cuda")
        return calc, "chgnet-v0.3.0"
    else:
        raise ValueError(f"Unknown backend: {backend}")


def min_image_distance(p1, p2, cell):
    """Minimum-image distance с orthogonal cell."""
    delta = p1 - p2
    for i in range(3):
        a = cell[i, i]
        if abs(delta[i]) > a / 2:
            delta[i] -= np.sign(delta[i]) * a
    return np.linalg.norm(delta)


def analyze_frame(atoms, h_index: int):
    """Per-frame distances: H_38 → all Fe, S, O atoms (PBC).

    Returns dict с min(d_FeH), min(d_SH), min(d_OH), nearest atom indices.
    """
    cell = atoms.cell.array
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    h_pos = positions[h_index]

    fe_indices = [i for i, s in enumerate(symbols) if s == "Fe"]
    s_indices = [i for i, s in enumerate(symbols) if s == "S"]
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]

    d_fe = [(i, min_image_distance(h_pos, positions[i], cell)) for i in fe_indices]
    d_s = [(i, min_image_distance(h_pos, positions[i], cell)) for i in s_indices]
    d_o = [(i, min_image_distance(h_pos, positions[i], cell)) for i in o_indices]

    d_fe.sort(key=lambda x: x[1])
    d_s.sort(key=lambda x: x[1])
    d_o.sort(key=lambda x: x[1])

    # Coordination number с Fe (как в W2 metad CV): R_0=2.0, NN=6, ND=12
    cv = 0.0
    for _, d in d_fe:
        r = d / 2.0
        if abs(r - 1.0) < 1e-9:
            cv += 0.5  # L'Hopital limit
        else:
            cv += (1.0 - r ** 6) / (1.0 - r ** 18)

    return {
        "d_min_Fe": d_fe[0][1],
        "nearest_Fe_index": d_fe[0][0],
        "d_min_S": d_s[0][1],
        "nearest_S_index": d_s[0][0],
        "d_min_O": d_o[0][1],
        "nearest_O_index": d_o[0][0],
        "cv_coord_Fe": cv,
        "n_Fe_within_2.5A": sum(1 for _, d in d_fe if d < 2.5),
        "n_S_within_2.5A": sum(1 for _, d in d_s if d < 2.5),
        "n_O_within_2.5A": sum(1 for _, d in d_o if d < 2.5),
    }


def find_excess_h_index(atoms):
    """Find H_38 (1-based) = atom index 37 (0-based) — excess proton.
    From s121 / .inp: ATOMS_FROM 38 в COLVAR → 1-based index 38."""
    # Из structure.xyz: line 40 → atom 38 (1-based)
    # Verify: H atom при z близком к 5.0 Å (interlayer / PBC chemi region)
    symbols = atoms.get_chemical_symbols()
    expected_idx = 37  # 0-based
    if symbols[expected_idx] != "H":
        raise ValueError(
            f"Expected H at index 37 (1-based 38), got {symbols[expected_idx]}. "
            f"Structure file may have different ordering."
        )
    return expected_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mace", "chgnet"])
    ap.add_argument("--input", default="/workspace/structure.xyz")
    ap.add_argument("--out-dir", default="/workspace/mlip_preflight")
    ap.add_argument("--steps", type=int, default=10000, help="MD steps (0.5 fs each); 10000 = 5 ps")
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--friction", type=float, default=0.01, help="Langevin friction (1/fs)")
    ap.add_argument("--save-every", type=int, default=20, help="Save trajectory every N steps")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # Load structure
    print(f"[setup] Loading {args.input}", flush=True)
    atoms = read(args.input)
    # 3×3×1 mackinawite cell from grotthuss_metadyn_v2.inp:
    #   A 11.022, B 11.022, C 6.5 Å, PERIODIC XYZ
    if np.allclose(atoms.cell.diagonal(), 0):
        atoms.set_cell([11.022, 11.022, 6.5])
        print("[setup] Cell set: 11.022 × 11.022 × 6.5 Å (from grotthuss_metadyn_v2.inp)", flush=True)
    atoms.set_pbc(True)
    print(f"[setup] N_atoms={len(atoms)}, cell diag={atoms.cell.diagonal()}, pbc={atoms.pbc}", flush=True)

    # Find excess H
    h_idx = find_excess_h_index(atoms)
    print(f"[setup] Excess H index (0-based)={h_idx}, position={atoms.positions[h_idx]}", flush=True)

    # Initial structural check
    initial_metrics = analyze_frame(atoms, h_idx)
    print(f"[setup] Initial distances (H to Fe/S/O):", flush=True)
    print(f"  d_min(H,Fe) = {initial_metrics['d_min_Fe']:.3f} Å (atom #{initial_metrics['nearest_Fe_index']})", flush=True)
    print(f"  d_min(H,S)  = {initial_metrics['d_min_S']:.3f} Å (atom #{initial_metrics['nearest_S_index']})", flush=True)
    print(f"  d_min(H,O)  = {initial_metrics['d_min_O']:.3f} Å (atom #{initial_metrics['nearest_O_index']})", flush=True)
    print(f"  CV coord(Fe) = {initial_metrics['cv_coord_Fe']:.3f}", flush=True)
    print(f"  n_Fe_within_2.5A = {initial_metrics['n_Fe_within_2.5A']}", flush=True)

    # Setup calculator
    print(f"[setup] Loading {args.backend} foundation MLIP", flush=True)
    calc, model_label = setup_calculator(args.backend)
    atoms.calc = calc

    # Sanity check forces
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    fmax0 = np.max(np.abs(f0))
    print(f"[sanity] E_initial = {e0:.4f} eV ({e0/len(atoms):.4f} eV/atom)", flush=True)
    print(f"[sanity] fmax_initial = {fmax0:.4f} eV/Å", flush=True)
    if fmax0 > 50.0:
        print(f"[FAIL] Initial fmax {fmax0} > 50 eV/Å — geometry unstable in MLIP", flush=True)
        with open(out_dir / "summary.json", "w") as f:
            json.dump({"status": "FAIL_initial_unstable", "fmax_initial": float(fmax0),
                       "backend": model_label}, f, indent=2)
        sys.exit(1)

    # Initialize velocities at target T
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp_k, rng=np.random.default_rng(args.seed))

    # NVT Langevin
    dyn = Langevin(
        atoms,
        timestep=args.dt_fs * units.fs,
        temperature_K=args.temp_k,
        friction=args.friction / units.fs,
    )

    # Storage
    traj_path = out_dir / "trajectory.xyz"
    csv_path = out_dir / "fe_h_distance.csv"
    log_path = out_dir / "md.log"

    csv_f = open(csv_path, "w")
    csv_f.write("step,time_fs,d_min_Fe,nearest_Fe,d_min_S,nearest_S,d_min_O,nearest_O,cv_coord,n_Fe_2.5,n_S_2.5,n_O_2.5,T_K,E_pot_eV,fmax\n")

    log_f = open(log_path, "w")

    # Frame 0
    metrics = analyze_frame(atoms, h_idx)
    write(traj_path, atoms, append=False)
    csv_f.write(
        f"0,0.0,{metrics['d_min_Fe']:.4f},{metrics['nearest_Fe_index']},"
        f"{metrics['d_min_S']:.4f},{metrics['nearest_S_index']},"
        f"{metrics['d_min_O']:.4f},{metrics['nearest_O_index']},"
        f"{metrics['cv_coord_Fe']:.4f},{metrics['n_Fe_within_2.5A']},"
        f"{metrics['n_S_within_2.5A']},{metrics['n_O_within_2.5A']},"
        f"{atoms.get_kinetic_energy() / (1.5 * len(atoms) * units.kB):.2f},"
        f"{e0:.4f},{fmax0:.4f}\n"
    )
    csv_f.flush()

    # Watchdog state
    fmax_explosion_limit = 50.0
    e_explosion_window = 100.0
    explosion_count = 0
    e_history = [e0]

    print(f"[md] Starting {args.steps} steps × {args.dt_fs} fs = {args.steps * args.dt_fs / 1000:.2f} ps", flush=True)
    t_start = time.time()

    for step in range(1, args.steps + 1):
        try:
            dyn.run(1)
        except Exception as e:
            print(f"[FAIL] step {step}: {e}", flush=True)
            break

        if step % args.save_every == 0:
            write(traj_path, atoms, append=True)
            metrics = analyze_frame(atoms, h_idx)
            e = atoms.get_potential_energy()
            f = atoms.get_forces()
            fmax = float(np.max(np.abs(f)))
            T_now = atoms.get_kinetic_energy() / (1.5 * len(atoms) * units.kB)
            csv_f.write(
                f"{step},{step * args.dt_fs:.1f},{metrics['d_min_Fe']:.4f},{metrics['nearest_Fe_index']},"
                f"{metrics['d_min_S']:.4f},{metrics['nearest_S_index']},"
                f"{metrics['d_min_O']:.4f},{metrics['nearest_O_index']},"
                f"{metrics['cv_coord_Fe']:.4f},{metrics['n_Fe_within_2.5A']},"
                f"{metrics['n_S_within_2.5A']},{metrics['n_O_within_2.5A']},"
                f"{T_now:.2f},{e:.4f},{fmax:.4f}\n"
            )
            csv_f.flush()
            e_history.append(e)

            # Watchdog
            if fmax > fmax_explosion_limit:
                print(f"[FAIL] step {step}: fmax {fmax} > {fmax_explosion_limit} eV/Å — explosion", flush=True)
                explosion_count += 1
                if explosion_count >= 3:
                    print(f"[FAIL] 3 consecutive explosions — abort", flush=True)
                    break

            if len(e_history) > 50:
                e_mean = np.mean(e_history[-50:])
                if abs(e - e_mean) > e_explosion_window:
                    print(f"[FAIL] step {step}: |E - E_mean| {abs(e-e_mean):.1f} > {e_explosion_window} — explosion", flush=True)
                    break

        if step % args.log_every == 0:
            wallclock = time.time() - t_start
            steps_per_sec = step / wallclock
            eta_sec = (args.steps - step) / steps_per_sec
            log_msg = (
                f"step={step:6d} t={step*args.dt_fs:.1f} fs  "
                f"d_FeH={metrics['d_min_Fe']:.3f}  d_OH={metrics['d_min_O']:.3f}  "
                f"d_SH={metrics['d_min_S']:.3f}  CV={metrics['cv_coord_Fe']:.2f}  "
                f"T={T_now:.1f}  E={e:.2f}  fmax={fmax:.2f}  "
                f"speed={steps_per_sec:.1f} st/s  ETA={eta_sec/60:.1f} min"
            )
            print(log_msg, flush=True)
            log_f.write(log_msg + "\n")
            log_f.flush()

    csv_f.close()
    log_f.close()

    # Post-analysis: Pass/Fail
    import csv as csv_mod
    rows = []
    with open(csv_path) as f:
        rdr = csv_mod.DictReader(f)
        for row in rdr:
            rows.append(row)

    if not rows:
        print("[FAIL] no frames recorded", flush=True)
        sys.exit(1)

    d_fe_arr = np.array([float(r["d_min_Fe"]) for r in rows])
    d_o_arr = np.array([float(r["d_min_O"]) for r in rows])
    d_s_arr = np.array([float(r["d_min_S"]) for r in rows])
    cv_arr = np.array([float(r["cv_coord"]) for r in rows])

    frac_fe_chemi = float(np.mean(d_fe_arr < 1.8))
    frac_water_h3o = float(np.mean(d_o_arr < 1.2))  # H bonded к water O
    frac_s_chemi = float(np.mean(d_s_arr < 1.6))   # H_S formed (covalent S-H)
    frac_unbound = float(np.mean((d_fe_arr > 2.0) & (d_o_arr > 1.4) & (d_s_arr > 1.6)))

    summary = {
        "status": "DONE",
        "backend": model_label,
        "n_frames": len(rows),
        "n_md_steps": args.steps,
        "duration_ps": args.steps * args.dt_fs / 1000.0,
        "T_target_K": args.temp_k,
        "T_mean_K": float(np.mean([float(r["T_K"]) for r in rows])),
        "T_std_K": float(np.std([float(r["T_K"]) for r in rows])),
        "d_min_Fe": {
            "mean": float(np.mean(d_fe_arr)),
            "min": float(np.min(d_fe_arr)),
            "max": float(np.max(d_fe_arr)),
            "first": float(d_fe_arr[0]),
            "last": float(d_fe_arr[-1]),
        },
        "d_min_O": {
            "mean": float(np.mean(d_o_arr)),
            "min": float(np.min(d_o_arr)),
            "max": float(np.max(d_o_arr)),
        },
        "d_min_S": {
            "mean": float(np.mean(d_s_arr)),
            "min": float(np.min(d_s_arr)),
            "max": float(np.max(d_s_arr)),
        },
        "cv_coord_Fe": {
            "mean": float(np.mean(cv_arr)),
            "min": float(np.min(cv_arr)),
            "max": float(np.max(cv_arr)),
        },
        "fraction_fe_chemi (d_FeH<1.8)": frac_fe_chemi,
        "fraction_water_H3O (d_OH<1.2)": frac_water_h3o,
        "fraction_S_chemi (d_SH<1.6)": frac_s_chemi,
        "fraction_unbound": frac_unbound,
        "verdict_pre_flight": (
            "PASS_proceed_to_US" if frac_fe_chemi > 0.95 else
            ("FAIL_T6_failure_mode" if frac_fe_chemi < 0.5 else "AMBIGUOUS_review_needed")
        ),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Final print
    print("\n" + "=" * 70, flush=True)
    print(f"=== {model_label.upper()} pre-flight VERDICT ===", flush=True)
    print("=" * 70, flush=True)
    print(f"  duration: {summary['duration_ps']:.2f} ps ({summary['n_frames']} frames)", flush=True)
    print(f"  T_mean = {summary['T_mean_K']:.1f} K (target {args.temp_k:.0f}, std {summary['T_std_K']:.1f})", flush=True)
    print(f"  d_min(H,Fe): mean={summary['d_min_Fe']['mean']:.3f} Å, range [{summary['d_min_Fe']['min']:.3f}, {summary['d_min_Fe']['max']:.3f}]", flush=True)
    print(f"  d_min(H,O):  mean={summary['d_min_O']['mean']:.3f} Å", flush=True)
    print(f"  d_min(H,S):  mean={summary['d_min_S']['mean']:.3f} Å", flush=True)
    print(f"  CV mean = {summary['cv_coord_Fe']['mean']:.2f}", flush=True)
    print(f"\n  Fraction Fe-chemi (d_FeH<1.8 Å): {frac_fe_chemi:.1%}", flush=True)
    print(f"  Fraction water-H3O (d_OH<1.2 Å): {frac_water_h3o:.1%}", flush=True)
    print(f"  Fraction S-chemi (d_SH<1.6 Å):   {frac_s_chemi:.1%}", flush=True)
    print(f"  Fraction unbound:                {frac_unbound:.1%}", flush=True)
    print(f"\n  VERDICT: {summary['verdict_pre_flight']}", flush=True)
    print("=" * 70, flush=True)

    # Mark DONE
    Path(out_dir / "DONE").touch()


if __name__ == "__main__":
    main()
