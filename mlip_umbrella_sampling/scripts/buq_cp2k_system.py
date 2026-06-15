"""BUQ CollectiveVariableSystem для CP2K+PLUMED Fe-H desorption (1D).

CV: smooth-min(d_FeH) с β=10 Å⁻¹ → 18 Fe atoms в 3×3×1 mack + H+ + 12 H2O.
Restraint: harmonic K=1000 kJ/mol/Å² centered на window position.
Mean force formula (Roux 1995, umbrella sampling):
    F(x_0) = -K * <CV - x_0>_prod
where <...> is time-average over production phase (skip eq).

API:
    system = CP2KPlumedFeHSystem(colvar_dir="results/us_2026-05-06/mace/windows")
    system.get_force(np.array([2.25]))  # returns mean force (eV/Å) at center=2.25 Å

Live deploy mode (TODO): integrate с SSH/scp orchestrator который launches CP2K window
run on Vast.ai instance, polls completion, fetches colvar.dat. For now this class is
**post-hoc adapter** — assumes colvar files уже existing.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from buq.systems import CollectiveVariableSystem


class CP2KPlumedFeHSystem(CollectiveVariableSystem):
    """1D CV system для Fe-H desorption через CP2K+PLUMED umbrella sampling.

    Args:
        colvar_dir: directory containing window_NN/colvar.dat files (one per window).
        K_kJ_mol_A2: restraint spring constant (default 1000, matches s136 setup).
        n_eq_steps: equilibration steps to skip (default 2000 = 1 ps at dt=0.5 fs).
        bounds: (min, max) CV in Å (default 1.5-4.0 матches W2 v2 wall).
        window_pattern: regex для extract window center from dirname (default "window_(\\d+)").
        window_centers: optional dict {window_id_int: center_A}. Default reads
            generate_windows.py CENTERS list.
    """

    DEFAULT_WINDOW_CENTERS_A = [
        1.50, 1.65, 1.80, 1.95, 2.10, 2.25, 2.40, 2.55,
        2.70, 2.85, 3.00, 3.15, 3.30, 3.45, 3.60, 3.75, 3.90, 4.00,
    ]

    def __init__(
        self,
        colvar_dir: str,
        K_kJ_mol_A2: float = 1000.0,
        n_eq_steps: int = 2000,
        bounds: tuple = (1.5, 4.0),
        window_centers: dict[int, float] | None = None,
    ):
        super().__init__(dim=1, bounds=bounds)
        self.colvar_dir = Path(colvar_dir)
        self.K_eV_A2 = K_kJ_mol_A2 / 96.485  # kJ/mol → eV (1 eV = 96.485 kJ/mol)
        self.n_eq_steps = n_eq_steps

        if window_centers is None:
            window_centers = {i: c for i, c in enumerate(self.DEFAULT_WINDOW_CENTERS_A)}
        self.window_centers = window_centers

        # Cache: window_id → (n_samples_prod, mean_cv, mean_force_eV_A)
        self._cache: dict[int, tuple[int, float, float]] = {}
        self._populate_cache()

    def _populate_cache(self):
        """Pre-compute mean force per window from existing colvar files."""
        if not self.colvar_dir.exists():
            return
        for window_id, center_a in self.window_centers.items():
            colvar_path = self.colvar_dir / f"window_{window_id:02d}" / "colvar.dat"
            if not colvar_path.exists():
                continue
            try:
                data = np.loadtxt(colvar_path, comments="#")
                if data.size == 0:
                    continue
                # Format: step time_fs cv_FeH bias d_min_FeH d_min_OH d_min_SH
                steps = data[:, 0].astype(int)
                cv_raw = data[:, 2]

                # s138 evening physicist Patch P4: unit detection.
                # PLUMED COLVAR default = nm; native Python run_mlip_us_native.py = Å.
                # Heuristic: if max(cv) < 1.0 → nm. Without this fix: 10× error в mean_force
                # for any DFT US live data via PLUMED.
                cv_factor_to_A = 10.0 if cv_raw.max() < 1.0 else 1.0
                cv = cv_raw * cv_factor_to_A

                # Skip equilibration
                mask = steps > self.n_eq_steps
                if not mask.any():
                    continue
                cv_prod = cv[mask]
                mean_cv = float(cv_prod.mean())
                # Mean force: F = -K * <CV - center>  (both в Å)
                mean_force = -self.K_eV_A2 * (mean_cv - center_a)
                self._cache[window_id] = (len(cv_prod), mean_cv, mean_force)
            except Exception as e:
                print(f"  [WARN] window {window_id:02d}: parse failed ({e})")

    def write_plumed_input(self, x: np.ndarray) -> None:
        """Generate plumed.dat для new window center=x[0]. TODO live mode."""
        # Live deploy: substitute {AT_VALUE_NM}, {KAPPA_NM2}, {WINDOW_ID} в template.
        # See generate_windows.py + plumed_window_template.dat.
        pass

    def run_simulation(self, x: np.ndarray) -> None:
        """Launch CP2K window run на remote Vast.ai instance. TODO live mode."""
        # Live deploy: SSH + scp + mpirun + poll + harvest colvar.dat.
        # Use tmp/go_smoke_w4_us.sh-style pattern.
        pass

    def get_force(self, x: np.ndarray) -> np.ndarray:
        """Return mean force dF/d(CV) at center x[0] Å.

        Strategy: find nearest pre-computed window. If exact match → return cached.
        If between windows → linear interpolation over mean_force samples.
        """
        center = float(x[0])
        if not self._cache:
            raise RuntimeError(
                f"No cached forces. Did colvar files exist в {self.colvar_dir}?"
            )

        # Sort by distance to center
        windows_by_dist = sorted(
            self._cache.items(),
            key=lambda kv: abs(self.window_centers[kv[0]] - center),
        )
        nearest_id, (n_samples, mean_cv, mean_force) = windows_by_dist[0]
        nearest_center = self.window_centers[nearest_id]

        if abs(nearest_center - center) < 1e-3:
            # Exact match
            return np.array([mean_force])

        # Linear interp between two nearest (for post-hoc analysis convenience)
        if len(windows_by_dist) >= 2:
            second_id, (_, _, second_force) = windows_by_dist[1]
            second_center = self.window_centers[second_id]
            # Linear interp на (center1, force1) - (center2, force2)
            if abs(second_center - nearest_center) > 1e-6:
                t = (center - nearest_center) / (second_center - nearest_center)
                interp_force = mean_force + t * (second_force - mean_force)
                return np.array([interp_force])

        return np.array([mean_force])

    def summary(self) -> str:
        """Print cached windows table — useful for sanity check."""
        lines = [
            f"CP2KPlumedFeHSystem: {len(self._cache)} windows cached",
            f"  bounds: {self.bounds[0]:.2f}-{self.bounds[1]:.2f} Å",
            f"  K = {self.K_eV_A2:.4f} eV/Å² ({self.K_eV_A2 * 96.485:.0f} kJ/mol/Å²)",
            f"  n_eq_steps = {self.n_eq_steps}",
            "",
            "  ID  center(Å)  N_prod  mean_cv(Å)  mean_force(eV/Å)",
            "  --  ---------  ------  ----------  ----------------",
        ]
        for wid in sorted(self._cache):
            n, mc, mf = self._cache[wid]
            c = self.window_centers[wid]
            lines.append(f"  {wid:02d}  {c:9.3f}  {n:6d}  {mc:10.3f}  {mf:+16.4f}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    colvar_dir = sys.argv[1] if len(sys.argv) > 1 else "results/us_2026-05-06/mace/windows"
    sys = CP2KPlumedFeHSystem(colvar_dir=colvar_dir)
    print(sys.summary())
