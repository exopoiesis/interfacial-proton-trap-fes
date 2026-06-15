#!/usr/bin/env python3
"""
MLIP Umbrella Sampling driver — ASE+MACE/CHGNet+PLUMED for ONE window.

Apples-to-apples с CP2K+PLUMED через identical plumed.dat (только AT+KAPPA varies per window).

Usage:
  python run_mlip_us.py --backend mace --window 0 --plumed plumed.dat --steps 18000
  # window 0 = d_FeH=1.5 Å, ... window 17 = d_FeH=4.0 Å
  # 18000 steps × 0.5 fs = 9 ps = 1 ps eq + 8 ps prod
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
from ase.calculators.plumed import Plumed
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def setup_calculator(backend: str):
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
    ap.add_argument("--window", type=int, required=True, help="Window ID 0..17")
    ap.add_argument("--input", default="/workspace/structure.xyz")
    ap.add_argument("--plumed-dat", required=True, help="Path to plumed.dat for this window")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=18000, help="Total MD steps (1 ps eq + 8 ps prod = 9 ps default)")
    ap.add_argument("--n-eq", type=int, default=2000, help="Equilibration steps (excluded from PMF analysis)")
    ap.add_argument("--dt-fs", type=float, default=0.5)
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--friction", type=float, default=0.01)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # Load and configure structure
    atoms = read(args.input)
    if np.allclose(atoms.cell.diagonal(), 0):
        atoms.set_cell([11.022, 11.022, 6.5])
    atoms.set_pbc(True)
    print(f"[setup] window={args.window}, backend={args.backend}", flush=True)
    print(f"[setup] N_atoms={len(atoms)}, cell={atoms.cell.diagonal()}", flush=True)

    # Calculator: MLIP wrapped с PLUMED
    base_calc = setup_calculator(args.backend)

    # Read PLUMED input
    plumed_dat_text = Path(args.plumed_dat).read_text().splitlines()
    # ASE Plumed calculator expects list of strings
    plumed_dat = [l for l in plumed_dat_text if not l.startswith("#") and l.strip()]

    # Note: ASE Plumed calculator needs timestep, atoms (for masses), kT, log file
    calc = Plumed(
        calc=base_calc,
        input=plumed_dat,
        timestep=args.dt_fs * units.fs,
        atoms=atoms,
        kT=args.temp_k * units.kB,
        log=str(out_dir / "plumed.log"),
    )
    atoms.calc = calc

    # Sanity check
    e0 = atoms.get_potential_energy()
    fmax0 = float(np.max(np.abs(atoms.get_forces())))
    print(f"[sanity] E_initial={e0:.3f} eV, fmax={fmax0:.3f} eV/Å", flush=True)
    if fmax0 > 50.0:
        print(f"[FAIL] fmax {fmax0} > 50 — geometry unstable", flush=True)
        sys.exit(1)

    # Init velocities
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp_k, rng=np.random.default_rng(args.seed + args.window))

    # NVT Langevin
    dyn = Langevin(
        atoms,
        timestep=args.dt_fs * units.fs,
        temperature_K=args.temp_k,
        friction=args.friction / units.fs,
    )

    traj_path = out_dir / "trajectory.xyz"
    log_path = out_dir / "md.log"
    log_f = open(log_path, "w")

    print(f"[md] {args.steps} steps × {args.dt_fs} fs ({args.steps * args.dt_fs / 1000:.2f} ps)", flush=True)
    t_start = time.time()

    # Frame 0
    write(traj_path, atoms, append=False)

    for step in range(1, args.steps + 1):
        try:
            dyn.run(1)
        except Exception as e:
            print(f"[FAIL] step {step}: {e}", flush=True)
            break

        if step % args.save_every == 0:
            write(traj_path, atoms, append=True)

        if step % args.log_every == 0:
            wallclock = time.time() - t_start
            steps_per_sec = step / wallclock
            eta_min = (args.steps - step) / steps_per_sec / 60
            T_now = atoms.get_kinetic_energy() / (1.5 * len(atoms) * units.kB)
            log_msg = f"step={step:6d} t={step*args.dt_fs:.1f} fs  T={T_now:.1f}  speed={steps_per_sec:.1f} st/s  ETA={eta_min:.1f} min"
            print(log_msg, flush=True)
            log_f.write(log_msg + "\n"); log_f.flush()

    log_f.close()

    # Summary
    summary = {
        "status": "DONE",
        "backend": args.backend,
        "window": args.window,
        "n_steps": args.steps,
        "n_eq_steps": args.n_eq,
        "duration_ps": args.steps * args.dt_fs / 1000.0,
        "wallclock_sec": time.time() - t_start,
        "plumed_dat": str(args.plumed_dat),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    Path(out_dir / "DONE").touch()
    print(f"\n[DONE] window {args.window} backend {args.backend} → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
