#!/usr/bin/env python3
"""
Sanity test: verify custom MinDistRestraintCalculator gives expected energy + forces
on a known geometry. Cross-validates analytical gradient via finite differences.
"""
import sys
from pathlib import Path
import numpy as np
from ase.io import read

sys.path.insert(0, str(Path(__file__).parent))
from run_mlip_us_native import compute_cv_and_grad, H_INDEX, FE_INDICES


def test_finite_diff(structure_path: str = "/workspace/structure.xyz"):
    """∂CV/∂x должно matching FD: (CV(x+h) - CV(x-h)) / (2h)."""
    atoms = read(structure_path)
    if np.allclose(atoms.cell.diagonal(), 0):
        atoms.set_cell([11.022, 11.022, 6.5])
    atoms.set_pbc(True)

    pos0 = atoms.get_positions().copy()
    cell = atoms.cell.array

    cv0, grad, dists = compute_cv_and_grad(pos0, cell, beta=10.0)
    print(f"Initial CV = {cv0:.6f} Å")
    print(f"min direct distance = {dists.min():.6f} Å")
    print(f"distances to 18 Fe (Å): {sorted(dists.tolist())[:5]}...")

    # FD test on H atom (most affected)
    h_eps = 1e-4
    print(f"\nFD test on H atom (index {H_INDEX}), h={h_eps}:")
    for axis in range(3):
        pos_p = pos0.copy(); pos_p[H_INDEX, axis] += h_eps
        pos_m = pos0.copy(); pos_m[H_INDEX, axis] -= h_eps
        cv_p, _, _ = compute_cv_and_grad(pos_p, cell, beta=10.0)
        cv_m, _, _ = compute_cv_and_grad(pos_m, cell, beta=10.0)
        fd = (cv_p - cv_m) / (2 * h_eps)
        an = grad[H_INDEX, axis]
        print(f"  axis {axis}: analytical={an:.6f}, FD={fd:.6f}, diff={abs(an-fd):.2e}")

    # FD test on closest Fe
    closest_fe_local_idx = int(np.argmin(dists))
    closest_fe_global_idx = FE_INDICES[closest_fe_local_idx]
    print(f"\nFD test on closest Fe atom (global index {closest_fe_global_idx}):")
    for axis in range(3):
        pos_p = pos0.copy(); pos_p[closest_fe_global_idx, axis] += h_eps
        pos_m = pos0.copy(); pos_m[closest_fe_global_idx, axis] -= h_eps
        cv_p, _, _ = compute_cv_and_grad(pos_p, cell, beta=10.0)
        cv_m, _, _ = compute_cv_and_grad(pos_m, cell, beta=10.0)
        fd = (cv_p - cv_m) / (2 * h_eps)
        an = grad[closest_fe_global_idx, axis]
        print(f"  axis {axis}: analytical={an:.6f}, FD={fd:.6f}, diff={abs(an-fd):.2e}")

    # Sanity: |grad| ≈ 1 для axis pointing along Fe-H direction (true min ≈ 1, smooth-min slightly < 1)
    grad_norm_h = np.linalg.norm(grad[H_INDEX])
    print(f"\n|grad CV w.r.t. H position| = {grad_norm_h:.4f} (expect close to 1.0 for true min, slightly less для smooth-min β=10)")


def test_restraint_at_center():
    """At CV=center, bias=0 and bias_force=0."""
    print("\n=== Restraint at center sanity ===")
    from run_mlip_us_native import MinDistRestraintCalculator

    # Mock base calc returning constant
    class ZeroCalc:
        def __init__(self):
            self.results = {}
        def calculate(self, atoms, properties=None, system_changes=None):
            self.results["energy"] = 0.0
            self.results["forces"] = np.zeros((len(atoms), 3))

    from ase import Atoms
    atoms = read("/workspace/structure.xyz")
    if np.allclose(atoms.cell.diagonal(), 0):
        atoms.set_cell([11.022, 11.022, 6.5])
    atoms.set_pbc(True)

    pos0 = atoms.get_positions()
    cv0, _, _ = compute_cv_and_grad(pos0, atoms.cell.array, beta=10.0)
    print(f"  Initial CV = {cv0:.4f} Å")

    # Set center = current CV → bias should be ~0
    calc = MinDistRestraintCalculator(
        base_calc=ZeroCalc(),
        center_A=cv0,
        kappa_kjmol_A2=1000.0,
        beta=10.0,
    )
    atoms.calc = calc
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    print(f"  Bias E at center: {e:.6e} eV (expect 0)")
    print(f"  Bias |F|_max at center: {np.max(np.abs(f)):.6e} eV/Å (expect 0)")

    # Offset by 0.1 Å → bias = 0.5 K (0.1)² = 5 kJ/mol = 0.0518 eV
    offset = 0.1
    calc.center_A = cv0 - offset
    atoms.calc = calc  # force re-attach
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    expected_bias_eV = 0.5 * 1000.0 * 0.0103642697 * offset ** 2
    print(f"  Bias E at offset {offset} Å: {e:.6e} eV (expect {expected_bias_eV:.6e} eV = {expected_bias_eV/0.0103642697:.2f} kJ/mol)")


if __name__ == "__main__":
    test_finite_diff()
    test_restraint_at_center()
