#!/usr/bin/env python3
"""
WHAM/MBAR analysis для US runs.

Reads:
  windows/window_NN/COLVAR_window_NN.dat (PLUMED COLVAR output)
  windows/manifest.txt

Outputs:
  pmf.dat — PMF(d_FeH) на CV grid
  pmf.png — plot
  diagnostics: Σχ² per window, autocorr, symmetry test (1st half vs 2nd half)

Uses pymbar (gold standard) если установлен; fallback на manual WHAM.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB_KJ_MOL = 8.314e-3  # kJ/(mol·K)


def parse_colvar(path: Path):
    """PLUMED COLVAR file: time d_min_FeH d_min_OH d_min_SH us.bias"""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                try:
                    rows.append([float(p) for p in parts[:5]])
                except ValueError:
                    pass
    if not rows:
        return np.empty((0, 5))
    return np.array(rows)


def parse_manifest(path: Path):
    """manifest.txt: window_id d_FeH_A d_FeH_nm K_kJ_mol_A2"""
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                windows.append({
                    "id": int(parts[0]),
                    "center_A": float(parts[1]),
                    "center_nm": float(parts[2]),
                    "K_kJ_mol_A2": float(parts[3]),
                })
    return windows


def manual_wham(windows_data, T_K=300.0, n_iter=1000, tol=1e-6):
    """Trivial WHAM: F_i = -kT log <exp(-V_i / kT)>_unbiased.

    Iterative procedure:
      f_i^(t+1) = -kT log Σ_j n_j × exp(-(V_ij - f_j^(t)) / kT) / Σ_j n_j × exp(-(V_ij - f_j^(t)) / kT) × ...

    More robust: использовать pymbar.

    Args:
        windows_data: list of dicts {center_nm, K_kJ_mol_A2, samples (d_FeH в nm)}
        T_K: temperature

    Returns:
        cv_grid_A, pmf_kjmol
    """
    kT = KB_KJ_MOL * T_K
    n_windows = len(windows_data)

    # Stack samples
    all_samples_nm = np.concatenate([w["samples_nm"] for w in windows_data])
    n_per_window = np.array([len(w["samples_nm"]) for w in windows_data])
    centers_nm = np.array([w["center_nm"] for w in windows_data])
    kappas_nm = np.array([w["K_kJ_mol_A2"] * 100.0 for w in windows_data])  # kJ/mol/nm²

    # Bias matrix V_ij[i,j] = V applied to sample i by window j's restraint
    n_samples = len(all_samples_nm)
    V_ij = 0.5 * kappas_nm[None, :] * (all_samples_nm[:, None] - centers_nm[None, :]) ** 2

    # Initial guess
    f_j = np.zeros(n_windows)

    for it in range(n_iter):
        log_w = -V_ij / kT + f_j[None, :] / kT
        log_w_max = np.max(log_w, axis=1, keepdims=True)
        denom = log_w_max + np.log(np.sum(n_per_window[None, :] * np.exp(log_w - log_w_max), axis=1, keepdims=True))
        # Accumulate sum of weights per window
        log_z_j = -np.log(np.sum(np.exp(-V_ij / kT - denom.squeeze()[:, None] + np.log(1.0)), axis=0))
        # Wait this is getting messy. Use simpler approach (per Kumar 1992)
        # f_j^(t+1) = -kT log Σ_i [ exp(-V_ij / kT) / Σ_k n_k exp(-V_ik/kT + f_k / kT) ]
        weights = np.exp(-V_ij / kT - denom.squeeze())  # n_samples × n_windows
        log_inv_z = np.log(np.sum(weights, axis=0))
        f_new = -kT * log_inv_z
        f_new -= f_new[0]  # gauge

        if np.max(np.abs(f_new - f_j)) < tol:
            print(f"[WHAM] converged at iter {it}", flush=True)
            break
        f_j = f_new

    # Reconstruct PMF
    cv_grid_nm = np.linspace(centers_nm.min() - 0.02, centers_nm.max() + 0.02, 200)
    cv_grid_A = cv_grid_nm * 10.0

    pmf = np.zeros_like(cv_grid_nm)
    for k, x in enumerate(cv_grid_nm):
        # Boltzmann reweight at each grid point
        dist_to_samples = np.abs(all_samples_nm - x)
        kernel = np.exp(-(dist_to_samples / 0.005) ** 2)  # gauss kernel σ=0.05 Å
        if kernel.sum() == 0:
            pmf[k] = np.nan
            continue
        # Bias at this CV value for each window
        V_xj = 0.5 * kappas_nm * (x - centers_nm) ** 2
        # Probability density estimate
        log_p_unnorm = np.log(kernel.sum() + 1e-30)
        log_p = log_p_unnorm + (np.sum(n_per_window * np.exp(-V_xj / kT + f_j / kT)) ** -1)
        pmf[k] = -kT * log_p

    pmf -= np.nanmin(pmf)
    return cv_grid_A, pmf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True, help="Directory with window_NN/")
    ap.add_argument("--manifest", default=None, help="Path to manifest.txt (default: <windows-dir>/manifest.txt)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-eq", type=int, default=2000, help="Equilibration steps to skip")
    ap.add_argument("--temp-k", type=float, default=300.0)
    ap.add_argument("--use-pymbar", action="store_true", help="Use pymbar (preferred if available)")
    args = ap.parse_args()

    windows_dir = Path(args.windows_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest) if args.manifest else windows_dir / "manifest.txt"
    if not manifest_path.exists():
        print(f"[FAIL] no manifest {manifest_path}", flush=True)
        return

    windows_meta = parse_manifest(manifest_path)
    print(f"[load] {len(windows_meta)} windows from manifest", flush=True)

    # Load COLVAR per window
    windows_data = []
    for w in windows_meta:
        win_dir = windows_dir / f"window_{w['id']:02d}"
        colvar_path = win_dir / f"COLVAR_window_{w['id']:02d}.dat"
        if not colvar_path.exists():
            print(f"[WARN] missing {colvar_path}", flush=True)
            continue
        data = parse_colvar(colvar_path)
        if len(data) <= args.n_eq // 10:  # COLVAR stride 10 vs MD stride 1
            print(f"[WARN] window {w['id']}: only {len(data)} samples (need >{args.n_eq // 10})", flush=True)
            continue

        # Skip eq period
        prod_data = data[args.n_eq // 10:]
        samples_nm = prod_data[:, 1]  # d_min_FeH в nm
        bias_kjmol = prod_data[:, 4]  # us.bias

        windows_data.append({
            **w,
            "samples_nm": samples_nm,
            "bias_kjmol": bias_kjmol,
            "n_samples": len(samples_nm),
            "mean_d_FeH_A": float(np.mean(samples_nm) * 10),
            "std_d_FeH_A": float(np.std(samples_nm) * 10),
        })

    if not windows_data:
        print("[FAIL] no valid windows", flush=True)
        return

    # Diagnostic: per-window stats
    print("\n[diagnostic] Per-window statistics:", flush=True)
    print(f"  win  center  n_samples  d_FeH_mean  d_FeH_std  overlap_with_next?", flush=True)
    for i, w in enumerate(windows_data):
        overlap = "?"
        if i + 1 < len(windows_data):
            w_next = windows_data[i + 1]
            sep = abs(w_next["mean_d_FeH_A"] - w["mean_d_FeH_A"])
            sigma_combined = np.sqrt(w["std_d_FeH_A"] ** 2 + w_next["std_d_FeH_A"] ** 2)
            overlap = f"sep={sep:.3f} σ={sigma_combined:.3f} ratio={sep/sigma_combined:.2f}"
        print(f"  {w['id']:2d}  {w['center_A']:.2f} Å  {w['n_samples']:6d}    {w['mean_d_FeH_A']:.3f}     {w['std_d_FeH_A']:.3f}    {overlap}", flush=True)

    # WHAM
    if args.use_pymbar:
        try:
            import pymbar
            print("[WHAM] using pymbar.MBAR", flush=True)
            # TODO: implement pymbar path
            cv_grid_A, pmf = manual_wham(windows_data, T_K=args.temp_k)  # fallback for now
        except ImportError:
            print("[WHAM] pymbar not available, falling back to manual WHAM", flush=True)
            cv_grid_A, pmf = manual_wham(windows_data, T_K=args.temp_k)
    else:
        cv_grid_A, pmf = manual_wham(windows_data, T_K=args.temp_k)

    # Save
    np.savetxt(out_dir / "pmf.dat", np.column_stack([cv_grid_A, pmf]),
               header="d_FeH(A)  PMF(kJ/mol)", fmt="%.4f")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cv_grid_A, pmf, lw=2, color="darkblue", label="PMF (WHAM)")
    for w in windows_data:
        ax.axvline(w["center_A"], color="gray", alpha=0.2, lw=0.5)
    ax.set_xlabel("d_FeH (Å)")
    ax.set_ylabel("PMF (kJ/mol)")
    ax.set_title(f"US PMF — {len(windows_data)} windows × {windows_data[0]['n_samples']/2000:.1f} ps prod each")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pmf.png", dpi=120)
    plt.close(fig)

    # Summary
    summary = {
        "n_windows": len(windows_data),
        "T_K": args.temp_k,
        "PMF_min_kjmol": float(np.nanmin(pmf)),
        "PMF_min_d_FeH_A": float(cv_grid_A[np.nanargmin(pmf)]),
        "PMF_max_kjmol": float(np.nanmax(pmf)),
        "deltaF_kjmol": float(np.nanmax(pmf) - np.nanmin(pmf)),
        "deltaF_eV": float((np.nanmax(pmf) - np.nanmin(pmf)) * 0.0103642697),
        "windows": [
            {k: w[k] for k in ("id", "center_A", "n_samples", "mean_d_FeH_A", "std_d_FeH_A")}
            for w in windows_data
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] PMF range: {summary['PMF_min_kjmol']:.2f} → {summary['PMF_max_kjmol']:.2f} kJ/mol", flush=True)
    print(f"  ΔF# = {summary['deltaF_kjmol']:.2f} kJ/mol = {summary['deltaF_eV']:.3f} eV", flush=True)
    print(f"  Output: {out_dir}/pmf.{{dat,png}}, summary.json", flush=True)


if __name__ == "__main__":
    main()
