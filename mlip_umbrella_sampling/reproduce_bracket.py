#!/usr/bin/env python3
"""Reproduce the kinetic-trap bracket from the bundled per-window colvar data.

Self-contained, repo-relative, CPU-only (numpy). Runs WHAM on
`data/{mace,chgnet}/window_NN/colvar.dat` and prints the detachment-barrier
saddle for each foundation MLIP. Reproduces the manuscript headline:
    MACE-MP-0  ~ 0.80 eV
    CHGNet     ~ 0.32-0.40 eV   (monotonic rise -> read as a lower bound)
i.e. the bracket dF# in [0.32, 0.80] eV (§3.3).

This is the runnable reproducibility entry point (no GPU/DFT/MLIP needed).
`scripts/revision_uq_2d.py` is the fuller UQ + 2-D carrier reanalysis (provenance).

Usage:  python reproduce_bracket.py [--data-dir DIR]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

KB_KJ_MOL = 8.314e-3
T_K = 300.0
kT = KB_KJ_MOL * T_K
EV_PER_KJMOL = 0.0103642697
K_KAPPA = 1000.0            # kJ/(mol*A^2), umbrella force constant
N_EQ_ROWS = 200            # drop ~1 ps equilibration (stride 10)
RELIABLE = (1.5, 3.3)      # edge artifact sets in by ~3.3-3.4 A
SIGMA_KDE = 0.05
N_GRID = 300
N_WINDOWS = 18
CENTER0, DSTEP = 1.50, 0.15   # window centers: 1.50 + 0.15*NN


def parse_colvar(path: Path) -> np.ndarray:
    """colvar.dat columns: step time_fs cv_FeH bias d_FeH d_OH d_SH."""
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 3:
            try:
                rows.append(float(p[2]))   # cv_FeH
            except ValueError:
                pass
    return np.asarray(rows[N_EQ_ROWS:])    # production segment


def load_engine(data_dir: Path, engine: str):
    samples, centers = [], []
    for nn in range(N_WINDOWS):
        f = data_dir / engine / f"window_{nn:02d}" / "colvar.dat"
        if not f.exists():
            continue
        cv = parse_colvar(f)
        if cv.size > 10:
            samples.append(cv)
            centers.append(CENTER0 + DSTEP * nn)
    return samples, np.asarray(centers)


def wham(samples, centers, n_iter=2000, tol=1e-6):
    all_s = np.concatenate(samples)
    n_per = np.array([len(s) for s in samples])
    V = 0.5 * K_KAPPA * (all_s[:, None] - centers[None, :]) ** 2
    f = np.zeros(len(centers))
    for _ in range(n_iter):
        lw = -V / kT + f[None, :] / kT
        m = lw.max(axis=1)
        log_denom = m + np.log((n_per[None, :] * np.exp(lw - m[:, None])).sum(axis=1))
        lt = -V / kT - log_denom[:, None]
        mt = lt.max(axis=0)
        f_new = -kT * (mt + np.log(np.exp(lt - mt[None, :]).sum(axis=0)))
        f_new -= f_new[0]
        if np.abs(f_new - f).max() < tol:
            f = f_new
            break
        f = f_new
    return all_s, n_per, f


def pmf_saddle(samples, centers):
    all_s, n_per, f = wham(samples, centers)
    grid = np.linspace(1.4, 4.1, N_GRID)
    pmf = np.full(grid.size, np.nan)
    for i, x in enumerate(grid):
        ker = np.exp(-((all_s - x) ** 2) / (2 * SIGMA_KDE ** 2)).sum()
        denom = (n_per * np.exp(-0.5 * K_KAPPA * (x - centers) ** 2 / kT + f / kT)).sum()
        if denom > 0 and ker > 1e-15:
            pmf[i] = -kT * (np.log(ker / (len(all_s) * SIGMA_KDE * np.sqrt(2 * np.pi))) - np.log(denom))
    mask = (grid >= RELIABLE[0]) & (grid <= RELIABLE[1]) & ~np.isnan(pmf)
    pmf -= pmf[mask].min()
    idx = np.argmax(pmf[mask])
    return float(grid[mask][idx]), float(pmf[mask][idx] * EV_PER_KJMOL)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "data"),
                    help="dir with {mace,chgnet}/window_NN/colvar.dat (default: ./data)")
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)

    out = {}
    for engine in ("mace", "chgnet"):
        samples, centers = load_engine(data_dir, engine)
        if not samples:
            raise SystemExit(f"no colvar data for {engine} under {data_dir}")
        x, dF = pmf_saddle(samples, centers)
        out[engine] = (x, dF)
        print(f"{engine:7s}: dF# = {dF:.3f} eV  @ d_FeH = {x:.2f} A  ({len(samples)} windows)")

    lo, hi = sorted([out["mace"][1], out["chgnet"][1]])
    print(f"\nkinetic-trap bracket: dF# in [{lo:.2f}, {hi:.2f}] eV  (manuscript [0.32, 0.80], §3.3)")
    return out


if __name__ == "__main__":
    main()
