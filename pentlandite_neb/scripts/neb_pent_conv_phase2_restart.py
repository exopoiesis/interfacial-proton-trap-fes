#!/usr/bin/env python3
"""
Phase 2 RESTART: CI-NEB for pentlandite H diffusion, conv cell, kpts=(2,2,2).

Context: Phase 1 converged (fmax=0.24), Phase 2 ran 8 steps (fmax=0.09->0.12).
Script restart caused Phase 1 re-run which diverged. This script loads Phase 2
checkpoint and continues CI-NEB with LBFGS (FIRE was oscillating).

Barrier at Phase 2 step 6: 0.414 eV (consistent with Gamma-only 0.442 eV).
Target: fmax < 0.05 eV/A.

Lessons applied:
  - _read_clean: strip ABACUS metadata from XYZ (nspins/nkpts crash ASE read)
  - NumpyEncoder: numpy types break json.dump
  - Checkpoint after every NEB step (SIGKILL resistance)
  - LBFGS not FIRE (FIRE oscillates near convergence for this system)
  - Verify loaded images: check H positions + barrier estimate
  - mpirun wrapper for ABACUS GPU (bare binary -> zombie orted)
"""

import json
import sys
import os
import re
import time
import tempfile
import traceback
import numpy as np
from pathlib import Path

from ase import Atom
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import LBFGS, FIRE
from ase.constraints import FixAtoms

# ABACUS ASE interface
ABACUS_ASE_PATH = "/opt/abacus-develop-3.9.0.26/interfaces/ASE_interface"
if ABACUS_ASE_PATH not in sys.path:
    sys.path.insert(0, ABACUS_ASE_PATH)
from abacuslite import Abacus, AbacusProfile

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORK_DIR = Path("/workspace/neb_pent_conv_kpts")
RESULTS = Path("/workspace/results")
PP_DIR = "/opt/sg15_pp"

# NEB parameters
N_IMAGES = 5
K_SPRING = 0.05
FMAX_TARGET = 0.05
MAX_STEPS = 200

# ---------------------------------------------------------------------------
# NumpyEncoder (lesson s72)
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ---------------------------------------------------------------------------
# _read_clean: strip ABACUS metadata from XYZ (lesson s76)
# ---------------------------------------------------------------------------
def _read_clean(path):
    """Read XYZ stripping ABACUS metadata (nspins, nkpts, etc.) that crash ASE."""
    KEEP = {'Lattice', 'Properties', 'pbc', 'energy', 'stress', 'free_energy'}
    with open(path) as f:
        lines = f.readlines()
    if len(lines) > 1:
        comment = lines[1]
        parts = []
        for m in re.finditer(r'(\w+)=(\"[^\"]*\"|\S+)', comment):
            if m.group(1) in KEEP:
                parts.append(m.group(0))
        lines[1] = ' '.join(parts) + '\n'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False, dir='/tmp') as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    atoms = read(tmp_path)
    Path(tmp_path).unlink()
    return atoms

# ---------------------------------------------------------------------------
# Checkpoint: save images after every NEB step
# ---------------------------------------------------------------------------
def save_images(images, tag="phase2_restart"):
    for k, img in enumerate(images):
        clean = img.copy()
        clean.calc = None
        write(str(WORK_DIR / f"neb_{tag}_img{k:02d}.xyz"), clean)

# ---------------------------------------------------------------------------
# ABACUS calculator (identical to original script)
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

profile = AbacusProfile(
    command="mpirun --allow-run-as-root -np 1 abacus",
    pseudo_dir=PP_DIR,
)

def make_calc(label="neb"):
    return Abacus(
        profile=profile,
        directory=str(WORK_DIR / label),
        pseudopotentials={
            "Fe": "Fe_ONCV_PBE-1.2.upf",
            "Ni": "Ni_ONCV_PBE-1.2.upf",
            "S":  "S_ONCV_PBE-1.2.upf",
            "H":  "H_ONCV_PBE-1.2.upf",
        },
        kpts={
            'nk': [2, 2, 2],
            'kshift': [0, 0, 0],
            'gamma-centered': True,
            'mode': 'mp-sampling',
        },
        inp={
            'basis_type': 'pw',
            'calculation': 'scf',
            'nspin': 1,
            'ecutwfc': 60,
            'smearing_method': 'gaussian',
            'smearing_sigma': 0.05,
            'scf_thr': 1e-6,
            'scf_nmax': 500,
            'mixing_type': 'broyden',
            'mixing_beta': 0.2,
            'mixing_ndim': 12,
            'cal_force': 1,
            'cal_stress': 0,
            'symmetry': 0,
        },
    )

# ---------------------------------------------------------------------------
# NEB step logger (checkpoint + stdout)
# ---------------------------------------------------------------------------
class NEBStepLogger:
    def __init__(self, neb, log_path, save_tag="phase2_restart"):
        self.neb = neb
        self.log_path = log_path
        self.save_tag = save_tag
        self.step = 0
        self.t0 = time.time()

    def __call__(self):
        images = self.neb.images
        forces = self.neb.get_forces()
        fmax = float(np.max(np.abs(forces)))

        energies = []
        for img in images:
            try:
                energies.append(img.get_potential_energy())
            except Exception:
                energies.append(float('nan'))

        e0 = energies[0] if energies else 0.0
        rel_e = [e - e0 for e in energies]
        barrier_est = max(rel_e) if rel_e else float('nan')

        dt = time.time() - self.t0
        line = (f"step={self.step:4d}  barrier_est={barrier_est:.4f} eV  "
                f"fmax={fmax:.4f}  t={dt:.0f}s")
        print(f"  [NEB] {line}", flush=True)

        with open(self.log_path, 'a') as f:
            f.write(line + "\n")

        # SIGKILL-resistant: save images after every step
        save_images(self.neb.images, tag=self.save_tag)
        self.step += 1

# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 70)
    print("  Phase 2 RESTART: CI-NEB pentlandite conv, kpts=(2,2,2)")
    print("  LBFGS optimizer (FIRE was oscillating at fmax~0.12)")
    print("  Target: fmax < 0.05 eV/A")
    print("=" * 70, flush=True)

    # -----------------------------------------------------------------------
    # 1. Load Phase 2 checkpoint (or phase2_restart if exists)
    # -----------------------------------------------------------------------
    n_total = N_IMAGES + 2  # endpoints + interior

    # Try restart checkpoint first, then original Phase 2
    restart_files = [WORK_DIR / f"neb_phase2_restart_img{k:02d}.xyz" for k in range(n_total)]
    phase2_files  = [WORK_DIR / f"neb_phase2_img{k:02d}.xyz" for k in range(n_total)]

    if all(f.exists() for f in restart_files):
        print("\n[RESUME] Loading from phase2_restart checkpoint...", flush=True)
        images = [_read_clean(f) for f in restart_files]
        source = "phase2_restart"
    elif all(f.exists() for f in phase2_files):
        print("\n[LOAD] Loading from Phase 2 checkpoint...", flush=True)
        images = [_read_clean(f) for f in phase2_files]
        source = "phase2"
    else:
        print("ERROR: No Phase 2 images found!")
        print(f"  Checked: {phase2_files[0]}")
        sys.exit(1)

    print(f"  Source: {source}")
    print(f"  Loaded {len(images)} images ({n_total} expected)")

    # -----------------------------------------------------------------------
    # 2. Verify loaded images
    # -----------------------------------------------------------------------
    for k, img in enumerate(images):
        h_idx = [i for i, s in enumerate(img.get_chemical_symbols()) if s == 'H']
        if len(h_idx) != 1:
            print(f"  ERROR: image {k} has {len(h_idx)} H atoms (expected 1)")
            sys.exit(1)
        h_pos = img.positions[h_idx[0]]
        print(f"  image {k}: {len(img)} atoms, H at ({h_pos[0]:.2f}, {h_pos[1]:.2f}, {h_pos[2]:.2f})")

    # Check atom count consistency
    n_atoms = len(images[0])
    for k, img in enumerate(images):
        if len(img) != n_atoms:
            print(f"  ERROR: image {k} has {len(img)} atoms (expected {n_atoms})")
            sys.exit(1)
    print(f"  All images: {n_atoms} atoms, 1 H each. OK.", flush=True)

    # -----------------------------------------------------------------------
    # 3. Fix atoms (all except H)
    # -----------------------------------------------------------------------
    for img in images:
        heavy = [i for i in range(len(img)) if img[i].symbol != 'H']
        img.set_constraint(FixAtoms(indices=heavy))

    # -----------------------------------------------------------------------
    # 4. Attach calculators to ALL images (including endpoints)
    # -----------------------------------------------------------------------
    for k in range(1, len(images) - 1):
        images[k].calc = make_calc(f"ph2r_img{k:02d}")
    # Endpoints also need calcs (for energy eval at harvest) — lesson s104
    images[0].calc = make_calc("ph2r_endA")
    images[-1].calc = make_calc("ph2r_endB")

    # -----------------------------------------------------------------------
    # 5. Create CI-NEB
    # -----------------------------------------------------------------------
    neb = NEB(images, climb=True, k=K_SPRING, allow_shared_calculator=False)
    print(f"\n  CI-NEB created: {len(images)} images, k={K_SPRING}, climb=True", flush=True)

    # -----------------------------------------------------------------------
    # 6. Run LBFGS
    # -----------------------------------------------------------------------
    log_path = WORK_DIR / "neb_phase2_restart.log"
    step_log = WORK_DIR / "neb_phase2_restart_step.log"
    logger = NEBStepLogger(neb, step_log, save_tag="phase2_restart")

    opt = LBFGS(neb, logfile=str(log_path))
    opt.attach(logger)

    print(f"\n=== Running CI-NEB LBFGS (fmax<{FMAX_TARGET}, max={MAX_STEPS}) ===", flush=True)
    t0 = time.time()
    converged = opt.run(fmax=FMAX_TARGET, steps=MAX_STEPS)
    dt = time.time() - t0

    # -----------------------------------------------------------------------
    # 7. Harvest results
    # -----------------------------------------------------------------------
    energies = [img.get_potential_energy() for img in images]
    e0 = energies[0]
    rel_e = [float(e - e0) for e in energies]
    barrier = max(rel_e)
    e_a_forward = float(max(energies) - min(energies[0], energies[-1]))
    e_a_reverse = float(max(energies) - max(energies[0], energies[-1]))

    forces = neb.get_forces()
    fmax_final = float(np.max(np.abs(forces)))

    result = {
        "mineral": "pentlandite",
        "cell": "conventional",
        "mechanism": "vacancy_hop_S-S",
        "code": "ABACUS",
        "mode": "PW GPU",
        "kpts": [2, 2, 2],
        "ecutwfc_ry": 60,
        "n_atoms": int(n_atoms),
        "n_images": N_IMAGES,
        "climb": True,
        "optimizer": "LBFGS",
        "k_spring": K_SPRING,
        "barrier_eV": float(barrier),
        "E_a_forward_eV": float(e_a_forward),
        "E_a_reverse_eV": float(e_a_reverse),
        "fmax_final": float(fmax_final),
        "converged": bool(converged),
        "steps": int(opt.nsteps),
        "time_s": float(dt),
        "energies_eV": [float(e) for e in energies],
        "relative_energies_eV": rel_e,
        "cross_verify": {
            "gamma_only_abacus": 0.442,
            "prim_gpaw": 1.115,
            "prim_abacus_lcao": 0.900,
        },
    }

    result_path = RESULTS / "neb_pentlandite_conv_kpts_result.json"
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'=' * 70}")
    print(f"  RESULT: E_a(forward) = {e_a_forward:.4f} eV")
    print(f"          E_a(reverse) = {e_a_reverse:.4f} eV")
    print(f"          barrier      = {barrier:.4f} eV")
    print(f"          fmax         = {fmax_final:.4f} eV/A")
    print(f"          converged    = {converged}")
    print(f"          steps        = {opt.nsteps}")
    print(f"          time         = {dt:.0f} s ({dt/3600:.1f} h)")
    print(f"  Cross-verify: Gamma-only = 0.442, prim GPAW = 1.115, prim ABACUS = 0.900")
    print(f"  Saved: {result_path}")
    print(f"{'=' * 70}", flush=True)

    # Write DONE flag
    done_msg = (f"PENT_CONV_KPTS_NEB: E_a={e_a_forward:.3f} eV, "
                f"fmax={fmax_final:.4f}, steps={opt.nsteps}, "
                f"t={dt/3600:.1f}h, converged={converged}")
    with open(RESULTS / "DONE_pent_conv_kpts", 'w') as f:
        f.write(done_msg + "\n")
    print(f"\n{done_msg}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}", flush=True)
        traceback.print_exc()
        with open(RESULTS / "crash_info", 'w') as f:
            f.write(f"{e}\n")
            traceback.print_exc(file=f)
        sys.exit(1)
