#!/usr/bin/env python3
"""
MLIP Umbrella Sampling driver — ASE+MACE/CHGNet+native Python restraint.
PLUMED-free implementation для apples-to-apples MLIP↔MLIP US.

CV: smooth-min(d_FeH) ≡ -ln(Σ exp(-β·d_i)) / β, β=10 nm⁻¹ = same definition as PLUMED MIN.
Restraint: V = 0.5 · K · (CV - center)²
Forces: ∂V/∂x_atom = K · (CV - center) · ∂CV/∂x_atom
       ∂CV/∂x_H = Σ_i [exp(-β·d_i) / Σ_j exp(-β·d_j)] · (x_H - x_Fe_i) / d_i
       ∂CV/∂x_Fe_i = -[exp(-β·d_i) / Σ_j exp(-β·d_j)] · (x_H - x_Fe_i) / d_i

Validated против CP2K+PLUMED через single-point energy/force check (см. test_restraint.py).

Usage:
  python run_mlip_us_native.py --backend mace --window 0 \\
      --center 1.5 --kappa 1000 \\
      --steps 18000 --n-eq 2000 \\
      --out-dir /workspace/us_out/window_00
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
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# Atom indices (0-based) — H_38 (1-based) = index 37; 18 Fe atoms
H_INDEX = 37
FE_INDICES = [0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33]


def min_image_vec(p1: np.ndarray, p2: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Min-image displacement (orthogonal cell only)."""
    delta = p1 - p2
    diag = np.diag(cell)
    delta -= np.round(delta / diag) * diag
    return delta


def compute_cv_and_grad(positions: np.ndarray, cell: np.ndarray, beta: float = 10.0):
    """Smooth-min distance H_38 to 18 Fe + per-atom gradient ∂CV/∂x.

    Returns (cv, grad_array_natoms_3) in Å and Å/Å (=dimensionless).

    CV = -ln(Σ exp(-β·d_i)) / β, units of d_i (Å here).
    Gradient: ∂CV/∂d_k = exp(-β·d_k) / Σ_j exp(-β·d_j)  (= w_k, weight)
              ∂CV/∂x_atom = ∂CV/∂d_k × ∂d_k/∂x_atom
    """
    h_pos = positions[H_INDEX]
    n_atoms = len(positions)
    grad = np.zeros((n_atoms, 3))

    distances = []
    vecs = []
    for fe_i in FE_INDICES:
        v = min_image_vec(h_pos, positions[fe_i], cell)
        d = np.linalg.norm(v)
        distances.append(d)
        vecs.append(v)
    distances = np.array(distances)
    vecs = np.array(vecs)

    # numerically stable smooth-min: shift by min
    d_min_naive = distances.min()
    exp_factors = np.exp(-beta * (distances - d_min_naive))  # max = 1.0
    sum_exp = exp_factors.sum()
    cv = d_min_naive - np.log(sum_exp) / beta
    weights = exp_factors / sum_exp  # softmax weights

    # Gradient: ∂CV/∂x_H = Σ_i w_i · (x_H - x_Fe_i) / d_i
    # ∂CV/∂x_Fe_i = -w_i · (x_H - x_Fe_i) / d_i
    for k, fe_i in enumerate(FE_INDICES):
        gv = weights[k] * vecs[k] / distances[k]
        grad[H_INDEX] += gv
        grad[fe_i] -= gv

    return cv, grad, distances


class MinDistRestraintCalculator(Calculator):
    """Wrap base MLIP calculator + apply harmonic restraint on smooth-min(d_FeH)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, base_calc, center_A: float, kappa_kjmol_A2: float, beta: float = 10.0, **kwargs):
        super().__init__(**kwargs)
        self.base_calc = base_calc
        self.center_A = float(center_A)
        # Convert K_kJ/mol/Å² to eV/Å² (для ASE internal units)
        self.kappa_eV_A2 = float(kappa_kjmol_A2) * 0.0103642697
        self.beta = float(beta)
        # Diagnostics
        self.last_cv = None
        self.last_bias_eV = None

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        # Base MLIP energy + forces
        self.base_calc.calculate(atoms, properties, system_changes)
        e_base = self.base_calc.results["energy"]
        f_base = self.base_calc.results["forces"].copy()

        # Restraint
        positions = atoms.get_positions()
        cell = atoms.cell.array
        cv, dCV_dx, dists = compute_cv_and_grad(positions, cell, self.beta)
        delta = cv - self.center_A
        e_bias = 0.5 * self.kappa_eV_A2 * delta ** 2
        # Force = -dE/dx = -K · delta · dCV/dx
        f_bias = -self.kappa_eV_A2 * delta * dCV_dx

        self.results["energy"] = e_base + e_bias
        self.results["forces"] = f_base + f_bias
        self.last_cv = float(cv)
        self.last_bias_eV = float(e_bias)


def setup_base_calculator(backend: str):
    if backend == "mace":
        from mace.calculators import mace_mp
        return mace_mp(model="medium", default_dtype="float32", device="cuda")
    elif backend == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        return CHGNetCalculator(use_device="cuda")
    raise ValueError(backend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["mace", "chgnet"])
    ap.add_argument("--window", type=int, required=True)
    ap.add_argument("--center", type=float, required=True, help="Restraint center (Å)")
    ap.add_argument("--kappa", type=float, default=1000.0, help="Restraint K (kJ/mol/Å²)")
    ap.add_argument("--beta", type=float, default=10.0, help="Smooth-min β (Å⁻¹)")
    ap.add_argument("--input", default="/workspace/structure.xyz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=18000, help="Total MD steps")
    ap.add_argument("--n-eq", type=int, default=2000, help="Eq steps (skipped в WHAM)")
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--colvar-every", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed + args.window)

    # Load structure
    print(f"[setup] backend={args.backend}, window={args.window}, center={args.center} Å, K={args.kappa} kJ/mol/Å², β={args.beta} Å⁻¹", flush=True)
    atoms = read(args.input)
    if np.allclose(atoms.cell.diagonal(), 0):
        atoms.set_cell([11.022, 11.022, 6.5])
    atoms.set_pbc(True)
    print(f"[setup] N_atoms={len(atoms)}, cell={atoms.cell.diagonal()}", flush=True)

    # Verify H index
    if atoms.get_chemical_symbols()[H_INDEX] != "H":
        raise ValueError(f"Expected H at index {H_INDEX}, got {atoms.get_chemical_symbols()[H_INDEX]}")

    # Setup MLIP + restraint wrapper
    base_calc = setup_base_calculator(args.backend)
    calc = MinDistRestraintCalculator(
        base_calc=base_calc,
        center_A=args.center,
        kappa_kjmol_A2=args.kappa,
        beta=args.beta,
    )
    atoms.calc = calc

    # Sanity
    e0 = atoms.get_potential_energy()
    fmax0 = float(np.max(np.abs(atoms.get_forces())))
    cv0 = calc.last_cv
    bias0 = calc.last_bias_eV
    print(f"[sanity] CV={cv0:.4f} Å, bias={bias0:.4f} eV ({bias0/0.0103642697:.4f} kJ/mol)", flush=True)
    print(f"[sanity] E_total={e0:.3f} eV (base + bias), fmax={fmax0:.3f} eV/Å", flush=True)
    if fmax0 > 50.0:
        print(f"[FAIL] fmax {fmax0} > 50", flush=True)
        sys.exit(1)

    # Init velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp_k, rng=np.random.default_rng(args.seed + args.window))

    dyn = Langevin(
        atoms,
        timestep=args.dt_fs * units.fs,
        temperature_K=args.temp_k,
        friction=args.friction / units.fs,
    )

    traj_path = out_dir / "trajectory.xyz"
    colvar_path = out_dir / "colvar.dat"
    log_path = out_dir / "md.log"
    log_f = open(log_path, "w")

    # COLVAR header (apples-to-apples с PLUMED COLVAR format)
    cf = open(colvar_path, "w")
    cf.write("# step  time_fs  cv_FeH(A)  bias(kJ/mol)  d_min_FeH(A)  d_min_OH(A)  d_min_SH(A)\n")

    # Aux indices for diagnostics
    symbols = atoms.get_chemical_symbols()
    o_indices = [i for i, s in enumerate(symbols) if s == "O"]
    s_indices = [i for i, s in enumerate(symbols) if s == "S"]

    def aux_dists(positions, cell):
        h_pos = positions[H_INDEX]
        d_o = min(np.linalg.norm(min_image_vec(h_pos, positions[i], cell)) for i in o_indices)
        d_s = min(np.linalg.norm(min_image_vec(h_pos, positions[i], cell)) for i in s_indices)
        return d_o, d_s

    # Frame 0
    write(traj_path, atoms, append=False)
    d_o0, d_s0 = aux_dists(atoms.get_positions(), atoms.cell.array)
    cf.write(f"0 0.0 {cv0:.4f} {bias0/0.0103642697:.4f} {cv0:.4f} {d_o0:.4f} {d_s0:.4f}\n")
    cf.flush()

    print(f"[md] {args.steps} steps × {args.dt_fs} fs ({args.steps * args.dt_fs / 1000:.2f} ps)", flush=True)
    t_start = time.time()

    for step in range(1, args.steps + 1):
        try:
            dyn.run(1)
        except Exception as e:
            print(f"[FAIL] step {step}: {e}", flush=True)
            break

        if step % args.colvar_every == 0:
            cv = calc.last_cv
            bias = calc.last_bias_eV
            d_o, d_s = aux_dists(atoms.get_positions(), atoms.cell.array)
            cf.write(f"{step} {step*args.dt_fs:.1f} {cv:.4f} {bias/0.0103642697:.4f} {cv:.4f} {d_o:.4f} {d_s:.4f}\n")
            cf.flush()

        if step % args.save_every == 0:
            write(traj_path, atoms, append=True)

        if step % args.log_every == 0:
            wallclock = time.time() - t_start
            speed = step / wallclock
            eta_min = (args.steps - step) / speed / 60
            T_now = atoms.get_kinetic_energy() / (1.5 * len(atoms) * units.kB)
            cv = calc.last_cv
            log_msg = (
                f"step={step:6d} t={step*args.dt_fs:.1f} fs  CV={cv:.3f}  bias={calc.last_bias_eV/0.0103642697:.2f} kJ/mol  "
                f"T={T_now:.1f}  speed={speed:.1f} st/s  ETA={eta_min:.1f} min"
            )
            print(log_msg, flush=True)
            log_f.write(log_msg + "\n"); log_f.flush()

    cf.close()
    log_f.close()

    # Summary
    summary = {
        "status": "DONE",
        "backend": args.backend,
        "window_id": args.window,
        "center_A": args.center,
        "kappa_kJ_mol_A2": args.kappa,
        "beta_A_inv": args.beta,
        "n_steps": args.steps,
        "n_eq_steps": args.n_eq,
        "duration_ps": args.steps * args.dt_fs / 1000.0,
        "wallclock_sec": time.time() - t_start,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    Path(out_dir / "DONE").touch()
    print(f"\n[DONE] window {args.window} backend {args.backend} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
