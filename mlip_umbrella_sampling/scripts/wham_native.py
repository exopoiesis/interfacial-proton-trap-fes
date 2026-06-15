#!/usr/bin/env python3
"""
WHAM на MACE US data (custom colvar.dat format).

Format: # step  time_fs  cv_FeH(A)  bias(kJ/mol)  d_min_FeH(A)  d_min_OH(A)  d_min_SH(A)
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


def parse_native_colvar(path: Path):
    """Custom format: step time_fs cv bias d_FeH d_OH d_SH"""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    rows.append([float(p) for p in parts[:7]])
                except ValueError:
                    pass
    if not rows:
        return np.empty((0, 7))
    return np.array(rows)


def parse_manifest(path: Path):
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


def manual_wham(samples_per_window, centers_A, kappas_A2, T_K=300.0, n_iter=2000, tol=1e-7):
    """Trivial WHAM в Å units (samples = CV in Å, centers в Å, K в kJ/mol/Å²).

    f_i^(t+1) = -kT log Σ_n exp(-V_i(s_n) / kT) / Σ_j n_j exp(-V_j(s_n) / kT + f_j^(t) / kT)

    Reference: Kumar 1992, Roux 1995. WT-MetaD reweighting reference: Branduardi 2012.
    """
    kT = KB_KJ_MOL * T_K
    n_w = len(samples_per_window)

    # Stack
    all_samples = np.concatenate(samples_per_window)
    n_per = np.array([len(s) for s in samples_per_window])
    centers = np.asarray(centers_A)
    kappas = np.asarray(kappas_A2)

    # V_ij[i, j] = bias of window j evaluated at sample i
    n_total = len(all_samples)
    V_ij = 0.5 * kappas[None, :] * (all_samples[:, None] - centers[None, :]) ** 2  # kJ/mol

    # Initial f_j (offsets)
    f_j = np.zeros(n_w)

    for it in range(n_iter):
        # log(Σ_k n_k exp(-V_ki / kT + f_k / kT))
        log_w = -V_ij / kT + f_j[None, :] / kT  # n_total × n_w
        log_w_max = np.max(log_w, axis=1)
        log_denom = log_w_max + np.log(np.sum(n_per[None, :] * np.exp(log_w - log_w_max[:, None]), axis=1))
        # f_j^new = -kT log Σ_i exp(-V_ji - log_denom_i)
        log_t = -V_ij / kT - log_denom[:, None]  # n_total × n_w
        log_t_max = np.max(log_t, axis=0)
        f_new = -kT * (log_t_max + np.log(np.sum(np.exp(log_t - log_t_max[None, :]), axis=0)))
        f_new -= f_new[0]  # gauge

        diff = np.max(np.abs(f_new - f_j))
        f_j = f_new
        if diff < tol:
            print(f"[WHAM] converged at iter {it}, max|Δf|={diff:.4e}", flush=True)
            break
    else:
        print(f"[WHAM] reached max iter {n_iter}, max|Δf|={diff:.4e}", flush=True)

    # Reconstruct PMF на CV grid
    cv_grid = np.linspace(centers.min() - 0.05, centers.max() + 0.05, 300)

    pmf = np.zeros_like(cv_grid)
    for k, x in enumerate(cv_grid):
        # P(x) = (1/N) Σ_i δ(s_i - x) / Σ_j n_j exp(-V_j(x)/kT + f_j/kT)
        # Используем kernel density для δ
        # Kernel: gauss σ=0.02 Å
        kernel_sigma = 0.02
        weights = np.exp(-((all_samples - x) ** 2) / (2 * kernel_sigma ** 2))
        # Bias at x для каждого окна
        V_jx = 0.5 * kappas * (x - centers) ** 2
        # Denominator for unbiasing
        denom_x = np.sum(n_per * np.exp(-V_jx / kT + f_j / kT))
        if denom_x <= 0 or weights.sum() < 1e-10:
            pmf[k] = np.nan
        else:
            pmf[k] = -kT * (np.log(weights.sum() / (n_total * kernel_sigma * np.sqrt(2 * np.pi))) - np.log(denom_x))

    pmf -= np.nanmin(pmf)
    return cv_grid, pmf, f_j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-eq", type=int, default=2000)
    ap.add_argument("--temp-k", type=float, default=300.0)
    args = ap.parse_args()

    windows_dir = Path(args.windows_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = parse_manifest(Path(args.manifest))
    print(f"[load] {len(manifest)} windows from manifest", flush=True)

    samples_per_window = []
    centers_A = []
    kappas = []
    diag_rows = []

    for w in manifest:
        cv_path = windows_dir / f"window_{w['id']:02d}" / "colvar.dat"
        if not cv_path.exists():
            print(f"[SKIP] missing {cv_path}", flush=True)
            continue
        data = parse_native_colvar(cv_path)
        if len(data) < args.n_eq // 10:
            print(f"[SKIP] window {w['id']} too short ({len(data)} samples)", flush=True)
            continue
        # Skip eq period: colvar stride 10 vs MD stride 1 → n_eq_md=2000 → skip first n_eq_md/10 = 200 colvar lines
        prod = data[args.n_eq // 10:, 2]  # column 2 = cv_FeH(A)
        samples_per_window.append(prod)
        centers_A.append(w['center_A'])
        kappas.append(w['K_kJ_mol_A2'])
        diag_rows.append({
            "id": w['id'],
            "center_A": w['center_A'],
            "n_samples": len(prod),
            "mean_A": float(np.mean(prod)),
            "std_A": float(np.std(prod)),
        })

    print("\n[diagnostic] Per-window stats:")
    print("  win  center   n   mean_A   std_A   spacing_to_next/σ")
    for i, d in enumerate(diag_rows):
        if i + 1 < len(diag_rows):
            sep = abs(diag_rows[i + 1]["mean_A"] - d["mean_A"])
            sigma = np.sqrt(d["std_A"]**2 + diag_rows[i + 1]["std_A"]**2)
            ratio = sep / sigma if sigma > 0 else 0
            ovl = f"{ratio:.2f}"
        else:
            ovl = "—"
        print(f"  {d['id']:2d}   {d['center_A']:.2f}  {d['n_samples']:5d}  {d['mean_A']:.3f}    {d['std_A']:.3f}    {ovl}")

    # WHAM
    cv_grid, pmf, f_j = manual_wham(samples_per_window, centers_A, kappas, T_K=args.temp_k)

    # Save
    np.savetxt(out_dir / "pmf.dat", np.column_stack([cv_grid, pmf]),
               header="d_FeH(A)  PMF(kJ/mol)", fmt="%.6f")

    # Summary
    pmf_clean = pmf[~np.isnan(pmf)]
    if len(pmf_clean) == 0:
        print("[FAIL] all-NaN PMF", flush=True)
        return
    summary = {
        "n_windows": len(samples_per_window),
        "T_K": args.temp_k,
        "PMF_min_A": float(cv_grid[np.nanargmin(pmf)]),
        "PMF_min_kjmol": 0.0,  # by gauge
        "PMF_max_kjmol": float(np.nanmax(pmf)),
        "PMF_max_A": float(cv_grid[np.nanargmax(pmf)]),
        "deltaF_kjmol": float(np.nanmax(pmf) - np.nanmin(pmf)),
        "deltaF_eV": float((np.nanmax(pmf) - np.nanmin(pmf)) * 0.0103642697),
        "windows": diag_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cv_grid, pmf, lw=2, color="darkblue", label="PMF (WHAM)")
    for c in centers_A:
        ax.axvline(c, color="gray", alpha=0.15, lw=0.5)
    ax.set_xlabel("d_FeH (Å, smooth-min)")
    ax.set_ylabel("PMF (kJ/mol)")
    ax.set_title(f"MACE US PMF — {len(samples_per_window)} windows × ~9 ps prod each (W2 paper #1)")
    ax.grid(alpha=0.3)
    ax.legend()

    # Annotate min and max
    if not np.isnan(pmf).all():
        i_min = np.nanargmin(pmf)
        i_max = np.nanargmax(pmf)
        ax.plot(cv_grid[i_min], pmf[i_min], "o", color="green", ms=8, label=f"min @ {cv_grid[i_min]:.2f} Å")
        ax.plot(cv_grid[i_max], pmf[i_max], "s", color="red", ms=8, label=f"max @ {cv_grid[i_max]:.2f} Å, ΔF={pmf[i_max]:.1f} kJ/mol = {pmf[i_max]*0.0103642697:.2f} eV")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "pmf.png", dpi=120)
    plt.close(fig)

    print(f"\n[DONE]")
    print(f"  PMF min @ d_FeH = {summary['PMF_min_A']:.3f} Å (= 0 by gauge)")
    print(f"  PMF max @ d_FeH = {summary['PMF_max_A']:.3f} Å = {summary['PMF_max_kjmol']:.2f} kJ/mol = {summary['deltaF_eV']:.3f} eV")
    print(f"  ΔF# = {summary['deltaF_kjmol']:.2f} kJ/mol = {summary['deltaF_eV']:.3f} eV")
    print(f"\n  Outputs: {out_dir}/pmf.{{dat,png}}, summary.json")


if __name__ == "__main__":
    main()
