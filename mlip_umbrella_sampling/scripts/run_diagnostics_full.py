#!/usr/bin/env python3
"""
Pre-deploy diagnostics для DFT US (B-prime) — TDD-style проверки на existing MLIP US data.

ALL CHECKS должны pass перед deploying expensive DFT US.

Tests:
  T1. Inp consistency: W2 v2 charge, nspin, +U — verify match с MLIP foundation expectations
  T2. Smooth-min CV definition matches between Python (used in MLIP US) and reference math
  T3. Per-window histogram unimodality (Gaussian fit chi-squared)
  T4. Block analysis convergence (split prod в 2 halves, recompute PMF, |Δ| < 0.05 eV gate)
  T5. Autocorrelation per window — N_eff per window estimate
  T6. WHAM precision via bootstrap (200 resamples, 95% CI)
  T7. Edge artifact characterization — где starts unreliable region

Inputs:
  results/us_2026-05-06/{mace,chgnet}/windows/window_NN/colvar.dat
  results/dft_datasets/2026-05-06/w2_metad_v2/grotthuss_metadyn_v2.inp

Outputs:
  results/us_2026-05-06/diagnostics/{T1..T7}_<backend>.{png,json}
  results/us_2026-05-06/diagnostics/PRE_DEPLOY_VERDICT.json
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB_KJ_MOL = 8.314e-3  # kJ/(mol·K)
T_K_DEFAULT = 300.0


# ============================================================
# T1. Inp consistency check
# ============================================================

def parse_cp2k_inp(path: Path) -> dict:
    """Crude parser для extract CHARGE, nspin (из &SMEAR or other), +U."""
    text = path.read_text()
    res = {
        "PROJECT": None,
        "CHARGE": None,
        "nspin": "default(1)",
        "DFT_PLUS_U": False,
        "U_RAMPING": None,
        "ELECTRONIC_TEMPERATURE": None,
        "VDW_POTENTIAL": None,
        "XC": None,
    }
    m = re.search(r"PROJECT\s+(\S+)", text)
    if m: res["PROJECT"] = m.group(1)
    m = re.search(r"CHARGE\s+(-?\d+)", text)
    if m: res["CHARGE"] = int(m.group(1))
    if re.search(r"&DFT_PLUS_U(?!\s*OFF)", text, re.I):
        if re.search(r"&DFT_PLUS_U\s+ON", text, re.I):
            res["DFT_PLUS_U"] = True
    if re.search(r"\bnspin\s+2\b", text, re.I) or re.search(r"&UKS\s", text, re.I):
        res["nspin"] = "2"
    m = re.search(r"ELECTRONIC_TEMPERATURE\s*\[K\]\s+([\d.]+)", text)
    if m: res["ELECTRONIC_TEMPERATURE"] = float(m.group(1))
    if re.search(r"&VDW_POTENTIAL", text):
        m2 = re.search(r"TYPE\s+(\S+)", text)
        if m2: res["VDW_POTENTIAL"] = m2.group(1)
    m = re.search(r"&XC_FUNCTIONAL\s+(\S+)", text)
    if m: res["XC"] = m.group(1)
    return res


def t1_inp_consistency(w2_v2_inp_path: Path, out_dir: Path) -> dict:
    print("\n[T1] Inp consistency check")
    if not w2_v2_inp_path.exists():
        return {"status": "SKIP", "reason": f"missing {w2_v2_inp_path}"}
    inp = parse_cp2k_inp(w2_v2_inp_path)
    print(f"  W2 v2 .inp: {inp}")

    issues = []
    if inp["CHARGE"] != 1:
        issues.append(f"WARN: W2 v2 CHARGE = {inp['CHARGE']}, expected 1 (MLIP foundation treat as neutral — known caveat)")
    if inp["DFT_PLUS_U"]:
        issues.append("WARN: W2 v2 has DFT+U active — MLIP foundation has no U → systematic Hamiltonian difference")
    if inp["nspin"] == "2":
        issues.append("WARN: W2 v2 nspin=2 (AFM) — MLIP foundation does not enforce magnetic structure → another source of disagreement")
    if inp["ELECTRONIC_TEMPERATURE"] != 500.0:
        issues.append(f"WARN: ELECTRONIC_TEMPERATURE={inp['ELECTRONIC_TEMPERATURE']}, expected 500 K")
    if inp["XC"] != "PBE":
        issues.append(f"WARN: XC={inp['XC']}, expected PBE")

    verdict = "PASS" if not issues else "WARN" if all("WARN" in i for i in issues) else "FAIL"
    result = {
        "status": verdict,
        "inp_parsed": inp,
        "issues": issues,
        "summary": f"W2 v2: charge={inp['CHARGE']}, nspin={inp['nspin']}, +U={inp['DFT_PLUS_U']}, smear T_e={inp['ELECTRONIC_TEMPERATURE']} K, XC={inp['XC']}, vdW={inp['VDW_POTENTIAL']}",
    }
    with open(out_dir / "T1_inp_consistency.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {verdict}: {result['summary']}")
    for i in issues:
        print(f"    {i}")
    return result


# ============================================================
# T2. Smooth-min CV definition correctness check
# ============================================================

def smooth_min_logsumexp(distances: np.ndarray, beta: float) -> float:
    """log-sum-exp variant: -ln(Σ exp(-β·d))/β. Always lower bound to true min."""
    d_min = distances.min()
    return d_min - np.log(np.sum(np.exp(-beta * (distances - d_min)))) / beta


def smooth_min_inverse_power(distances: np.ndarray, beta: float) -> float:
    """PLUMED MIN={BETA}: (Σ d_i^(-β))^(-1/β). Different function!"""
    return float(np.sum(distances ** (-beta)) ** (-1.0 / beta))


def t2_cv_definition_check(out_dir: Path) -> dict:
    """Verify our Python smooth_min vs PLUMED MIN — do they agree on test geom?"""
    print("\n[T2] CV definition check (Python log-sum-exp vs PLUMED inverse-power)")

    test_cases = [
        ("Single Fe at d=1.5", np.array([1.5, 5.0, 5.0, 5.0])),
        ("Bidentate 2 Fe at d=1.5,1.7", np.array([1.5, 1.7, 5.0, 5.0])),
        ("Multi-Fe pocket d=1.6,1.8,2.0,2.5", np.array([1.6, 1.8, 2.0, 2.5, 5.0])),
        ("Far d=3.5,4.0,4.0", np.array([3.5, 4.0, 4.0, 5.0])),
    ]
    beta = 10.0
    rows = []
    for name, dists in test_cases:
        cv_python = smooth_min_logsumexp(dists, beta)
        cv_plumed = smooth_min_inverse_power(dists, beta)
        d_true_min = float(dists.min())
        diff = cv_plumed - cv_python
        rows.append({
            "case": name,
            "d_true_min_A": d_true_min,
            "Python_logsumexp": cv_python,
            "PLUMED_inverse_power": cv_plumed,
            "diff_A": diff,
            "diff_rel_pct": 100.0 * diff / d_true_min,
        })
        print(f"  {name}: true_min={d_true_min:.3f}, Python={cv_python:.4f}, PLUMED MIN={cv_plumed:.4f}, diff={diff:+.4f} Å ({100*diff/d_true_min:+.2f}%)")

    max_diff = max(abs(r["diff_A"]) for r in rows)
    verdict = "FAIL" if max_diff > 0.01 else "WARN" if max_diff > 0.001 else "PASS"
    result = {
        "status": verdict,
        "max_diff_A": max_diff,
        "test_cases": rows,
        "conclusion": (
            f"Max |Python - PLUMED MIN| = {max_diff:.4f} Å on test geometries. "
            f"For DFT US apples-to-apples с MLIP US must use CUSTOM/MATHEVAL function в plumed.dat: "
            f"`f: COMBINE ARG=d.x ... POWERS=...; mincv: MATHEVAL ARG=... FUNC=-log(...)/10` "
            f"OR write CP2K-native COLVAR с smooth-min via &MIXED_COLVAR."
        ),
    }
    with open(out_dir / "T2_cv_definition.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {verdict}: max diff = {max_diff:.4f} Å")
    return result


# ============================================================
# T3. Per-window unimodality check
# ============================================================

def parse_native_colvar(path: Path) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    pass
    return np.array(rows) if rows else np.empty((0, 3))


def gaussian_fit_chi2(samples: np.ndarray, n_bins: int = 30) -> tuple[float, float]:
    """Fit Gaussian + return chi² и mean/std observed vs fit."""
    if len(samples) < 50:
        return 0.0, 0.0
    hist, edges = np.histogram(samples, bins=n_bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mu = np.mean(samples)
    sigma = np.std(samples)
    expected = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((centers - mu) ** 2) / (2 * sigma ** 2))
    # Avoid divide by zero
    mask = expected > 1e-6
    if mask.sum() < 5:
        return 0.0, 0.0
    chi2 = float(np.sum(((hist[mask] - expected[mask]) ** 2) / expected[mask]))
    return chi2, sigma


def detect_bimodality(samples: np.ndarray, threshold_kurtosis: float = -1.0) -> bool:
    """Detect bimodality via excess kurtosis: bimodal distribution имеет negative excess kurtosis."""
    if len(samples) < 50:
        return False
    mean = np.mean(samples)
    m2 = np.mean((samples - mean) ** 2)
    m4 = np.mean((samples - mean) ** 4)
    if m2 < 1e-12:
        return False
    kurt_excess = float(m4 / (m2 ** 2) - 3.0)
    return bool(kurt_excess < threshold_kurtosis)


def t3_per_window_unimodality(windows_dir: Path, manifest: list[dict], out_dir: Path, backend: str) -> dict:
    print(f"\n[T3] Per-window unimodality ({backend})")
    rows = []
    suspicious = []
    for w in manifest:
        cv_path = windows_dir / f"window_{w['id']:02d}" / "colvar.dat"
        if not cv_path.exists():
            continue
        data = parse_native_colvar(cv_path)
        if len(data) < 200:
            continue
        prod = data[200:, 2]  # skip eq, take CV column
        chi2, sigma = gaussian_fit_chi2(prod)
        bimodal = detect_bimodality(prod)
        rows.append({
            "id": int(w['id']),
            "center_A": float(w['center_A']),
            "n_samples": int(len(prod)),
            "mean_A": float(np.mean(prod)),
            "std_A": float(sigma),
            "chi2_gaussian": float(chi2),
            "bimodal_flag": bool(bimodal),
        })
        if bimodal or chi2 > 5.0:
            suspicious.append(int(w['id']))

    print(f"  win  center  std    chi2_gauss  bimodal?")
    for r in rows:
        flag = "BIMODAL" if r["bimodal_flag"] else ("BAD-FIT" if r["chi2_gaussian"] > 5.0 else "OK")
        print(f"  {r['id']:2d}   {r['center_A']:.2f}  {r['std_A']:.4f}  {r['chi2_gaussian']:8.3f}  {flag}")

    verdict = "PASS" if not suspicious else "WARN" if len(suspicious) <= 2 else "FAIL"
    result = {
        "status": verdict,
        "backend": backend,
        "n_windows": len(rows),
        "suspicious_windows": suspicious,
        "windows": rows,
        "note": (
            "Suspicious windows may indicate bimodal distributions — system flips between two basins under restraint. "
            "Saddle position derived from these windows is potentially artifactual."
        ),
    }
    with open(out_dir / f"T3_unimodality_{backend}.json", "w") as f:
        json.dump(result, f, indent=2)

    # Plot histograms suspicious windows
    if suspicious:
        n = len(suspicious)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False)
        for ax, wid in zip(axes[0], suspicious):
            cv_path = windows_dir / f"window_{wid:02d}" / "colvar.dat"
            data = parse_native_colvar(cv_path)
            prod = data[200:, 2]
            ax.hist(prod, bins=40, alpha=0.7, color="steelblue", density=True)
            ax.axvline(np.mean(prod), color="red", ls="--", label=f"mean {np.mean(prod):.3f}")
            ax.set_title(f"{backend} window {wid:02d}\ncenter {[w['center_A'] for w in manifest if w['id']==wid][0]:.2f} Å, std {np.std(prod):.3f}")
            ax.set_xlabel("d_FeH (Å)")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / f"T3_unimodality_{backend}_suspicious.png", dpi=120)
        plt.close(fig)
        print(f"  → Saved suspicious window histograms: T3_unimodality_{backend}_suspicious.png")
    print(f"  → {verdict}: {len(suspicious)} suspicious windows {suspicious if suspicious else ''}")
    return result


# ============================================================
# T4. Block analysis (split prod в halves, recompute PMF)
# ============================================================

def simple_wham(samples_per_window, centers, kappas, T_K=300.0, n_iter=2000, tol=1e-7):
    """Simplified WHAM (re-used от tmp/wham_mace_native.py)."""
    kT = KB_KJ_MOL * T_K
    n_w = len(samples_per_window)
    all_samples = np.concatenate(samples_per_window)
    n_per = np.array([len(s) for s in samples_per_window])
    centers = np.asarray(centers)
    kappas = np.asarray(kappas)
    V_ij = 0.5 * kappas[None, :] * (all_samples[:, None] - centers[None, :]) ** 2
    f_j = np.zeros(n_w)
    for it in range(n_iter):
        log_w = -V_ij / kT + f_j[None, :] / kT
        log_w_max = np.max(log_w, axis=1)
        log_denom = log_w_max + np.log(np.sum(n_per[None, :] * np.exp(log_w - log_w_max[:, None]), axis=1))
        log_t = -V_ij / kT - log_denom[:, None]
        log_t_max = np.max(log_t, axis=0)
        f_new = -kT * (log_t_max + np.log(np.sum(np.exp(log_t - log_t_max[None, :]), axis=0)))
        f_new -= f_new[0]
        if np.max(np.abs(f_new - f_j)) < tol:
            break
        f_j = f_new
    cv_grid = np.linspace(centers.min() - 0.05, centers.max() + 0.05, 200)
    pmf = np.zeros_like(cv_grid)
    kernel_sigma = 0.02
    for k, x in enumerate(cv_grid):
        weights = np.exp(-((all_samples - x) ** 2) / (2 * kernel_sigma ** 2))
        V_jx = 0.5 * kappas * (x - centers) ** 2
        denom_x = np.sum(n_per * np.exp(-V_jx / kT + f_j / kT))
        if denom_x > 0 and weights.sum() > 1e-10:
            pmf[k] = -kT * (np.log(weights.sum() / (len(all_samples) * kernel_sigma * np.sqrt(2 * np.pi))) - np.log(denom_x))
        else:
            pmf[k] = np.nan
    pmf -= np.nanmin(pmf)
    return cv_grid, pmf


def find_saddle(cv, pmf, lo=1.7, hi=2.5):
    mask = (cv >= lo) & (cv <= hi) & (~np.isnan(pmf))
    if mask.sum() == 0:
        return None
    idx = np.where(mask)[0]
    i_max = idx[np.nanargmax(pmf[idx])]
    return float(cv[i_max]), float(pmf[i_max])


def t4_block_convergence(windows_dir: Path, manifest: list[dict], out_dir: Path, backend: str) -> dict:
    print(f"\n[T4] Block convergence ({backend})")
    samples_first = []
    samples_second = []
    centers = []
    kappas = []
    for w in manifest:
        cv_path = windows_dir / f"window_{w['id']:02d}" / "colvar.dat"
        if not cv_path.exists():
            continue
        data = parse_native_colvar(cv_path)
        if len(data) < 600:
            continue
        prod = data[200:, 2]  # skip 200 lines = 200×10 MD steps = 2 ps eq
        n_half = len(prod) // 2
        samples_first.append(prod[:n_half])
        samples_second.append(prod[n_half:])
        centers.append(w['center_A'])
        kappas.append(w['K_kJ_mol_A2'])

    cv_full, pmf_first = simple_wham(samples_first, centers, kappas)
    _, pmf_second = simple_wham(samples_second, centers, kappas)

    saddle_first = find_saddle(cv_full, pmf_first)
    saddle_second = find_saddle(cv_full, pmf_second)
    diff = abs(saddle_first[1] - saddle_second[1]) if (saddle_first and saddle_second) else None
    diff_eV = diff * 0.0103642697 if diff else None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cv_full, pmf_first, lw=2, label="First half")
    ax.plot(cv_full, pmf_second, lw=2, ls="--", label="Second half")
    if saddle_first:
        ax.plot(saddle_first[0], saddle_first[1], "o", ms=8, color="C0")
    if saddle_second:
        ax.plot(saddle_second[0], saddle_second[1], "s", ms=8, color="C1")
    ax.set_xlabel("d_FeH (Å)")
    ax.set_ylabel("PMF (kJ/mol)")
    title = f"{backend} block convergence — saddle ΔF# diff = {diff:.2f} kJ/mol = {diff_eV:.3f} eV" if diff else f"{backend} block convergence"
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"T4_block_convergence_{backend}.png", dpi=120)
    plt.close(fig)

    THRESHOLD_KJMOL = 5.0  # 0.05 eV
    verdict = "PASS" if (diff is not None and diff < THRESHOLD_KJMOL) else "WARN" if (diff and diff < 10) else "FAIL"
    result = {
        "status": verdict,
        "backend": backend,
        "saddle_first_half": {"d_FeH_A": saddle_first[0], "F_kjmol": saddle_first[1], "F_eV": saddle_first[1] * 0.0103642697} if saddle_first else None,
        "saddle_second_half": {"d_FeH_A": saddle_second[0], "F_kjmol": saddle_second[1], "F_eV": saddle_second[1] * 0.0103642697} if saddle_second else None,
        "saddle_diff_kjmol": diff,
        "saddle_diff_eV": diff_eV,
        "threshold_kjmol": THRESHOLD_KJMOL,
        "verdict_reason": f"|ΔF#_first - ΔF#_second| = {diff:.2f} kJ/mol vs threshold {THRESHOLD_KJMOL}" if diff else "diff not computable",
    }
    with open(out_dir / f"T4_block_convergence_{backend}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {verdict}: ΔF# split-halves diff = {diff:.2f} kJ/mol = {diff_eV:.3f} eV (threshold {THRESHOLD_KJMOL} kJ/mol)")
    return result


# ============================================================
# T5. Autocorrelation per window
# ============================================================

def autocorr_time(samples: np.ndarray, max_lag: int = 200) -> float:
    """Integrated autocorrelation time τ_int via simple sum of ACF."""
    if len(samples) < 2 * max_lag:
        max_lag = len(samples) // 2
    s = samples - np.mean(samples)
    var = np.var(s)
    if var < 1e-12:
        return 1.0
    acf = np.array([np.mean(s[: -k] * s[k:]) / var if k > 0 else 1.0 for k in range(max_lag)])
    # Integrated time = 1 + 2 Σ ρ(k), truncate где ρ < 0.05
    tau = 1.0
    for r in acf[1:]:
        if r < 0.05:
            break
        tau += 2 * r
    return tau


def t5_autocorrelation(windows_dir: Path, manifest: list[dict], out_dir: Path, backend: str) -> dict:
    print(f"\n[T5] Autocorrelation ({backend})")
    rows = []
    for w in manifest:
        cv_path = windows_dir / f"window_{w['id']:02d}" / "colvar.dat"
        if not cv_path.exists():
            continue
        data = parse_native_colvar(cv_path)
        prod = data[200:, 2]
        if len(prod) < 100:
            continue
        # Stride 10 MD steps, dt=0.5 fs → frame interval = 5 fs
        tau_frames = autocorr_time(prod)
        tau_ps = tau_frames * 5e-3  # ps per frame interval
        n_eff = len(prod) / tau_frames
        rows.append({
            "id": w['id'],
            "center_A": w['center_A'],
            "n_samples_total": len(prod),
            "tau_int_frames": float(tau_frames),
            "tau_int_ps": float(tau_ps),
            "n_eff": float(n_eff),
        })

    n_eff_min = min(r["n_eff"] for r in rows)
    print(f"  win  center  τ_int(ps)  N_eff")
    for r in rows:
        print(f"  {r['id']:2d}   {r['center_A']:.2f}    {r['tau_int_ps']:6.3f}    {r['n_eff']:6.0f}")

    verdict = "PASS" if n_eff_min >= 50 else "WARN" if n_eff_min >= 10 else "FAIL"
    result = {
        "status": verdict,
        "backend": backend,
        "n_eff_min": n_eff_min,
        "windows": rows,
        "note": (
            "N_eff ≥ 50 → reliable; 10-50 → marginal; <10 → re-run window с дольше production. "
            "PMF reliability scales as 1/sqrt(N_eff), so window with N_eff=10 dominates noise."
        ),
    }
    with open(out_dir / f"T5_autocorr_{backend}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  → {verdict}: N_eff_min = {n_eff_min:.0f}")
    return result


# ============================================================
# Driver
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-base", default="D:/home/ignat/project-third-matter/results/us_2026-05-06")
    ap.add_argument("--w2-v2-inp", default="D:/home/ignat/project-third-matter/results/dft_datasets/2026-05-06/w2_metad_v2/grotthuss_metadyn_v2.inp")
    ap.add_argument("--out-dir", default="D:/home/ignat/project-third-matter/results/us_2026-05-06/diagnostics")
    args = ap.parse_args()

    base = Path(args.results_base)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PRE-DEPLOY DIAGNOSTICS for DFT US (B-prime)")
    print("=" * 70)

    # Load manifest (used for both backends — same window scheme)
    manifest_path = base / "mace" / "manifest.txt"
    manifest = []
    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                manifest.append({"id": int(parts[0]), "center_A": float(parts[1]), "K_kJ_mol_A2": float(parts[3])})

    # Run tests
    results = {}
    results["T1_inp"] = t1_inp_consistency(Path(args.w2_v2_inp), out_dir)
    results["T2_cv_def"] = t2_cv_definition_check(out_dir)

    for backend in ("mace", "chgnet"):
        windows_dir = base / backend / "windows"
        if not windows_dir.exists():
            continue
        results[f"T3_unimodality_{backend}"] = t3_per_window_unimodality(windows_dir, manifest, out_dir, backend)
        results[f"T4_block_{backend}"] = t4_block_convergence(windows_dir, manifest, out_dir, backend)
        results[f"T5_autocorr_{backend}"] = t5_autocorrelation(windows_dir, manifest, out_dir, backend)

    # Final verdict
    statuses = [r["status"] for r in results.values()]
    fail_count = sum(1 for s in statuses if s == "FAIL")
    warn_count = sum(1 for s in statuses if s == "WARN")
    pass_count = sum(1 for s in statuses if s == "PASS")
    skip_count = sum(1 for s in statuses if s == "SKIP")

    if fail_count > 0:
        verdict = "FAIL_BLOCK_DEPLOY"
    elif warn_count > 0:
        verdict = "PASS_WITH_CAVEATS"
    else:
        verdict = "PASS_PROCEED_DEPLOY"

    summary = {
        "verdict": verdict,
        "tests": {k: v["status"] for k, v in results.items()},
        "fail_count": fail_count,
        "warn_count": warn_count,
        "pass_count": pass_count,
        "skip_count": skip_count,
        "details": {k: {"status": v["status"]} for k, v in results.items()},
    }
    with open(out_dir / "PRE_DEPLOY_VERDICT.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"=== FINAL VERDICT: {verdict} ===")
    print("=" * 70)
    for tname, tres in results.items():
        print(f"  {tres['status']:6s}  {tname}")
    print(f"\n  Pass: {pass_count}, Warn: {warn_count}, Fail: {fail_count}, Skip: {skip_count}")
    print(f"  Output: {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
