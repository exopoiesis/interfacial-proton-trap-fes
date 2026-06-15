#!/usr/bin/env python3
"""Comparison plot MACE vs CHGNet PMFs (W2 paper #1, s136)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_pmf(path):
    return np.loadtxt(path)


def find_plateau(cv, pmf, plateau_start_A=2.5, plateau_end_A=3.5):
    mask = (cv >= plateau_start_A) & (cv <= plateau_end_A) & (~np.isnan(pmf))
    if mask.sum() == 0:
        return None
    return float(np.mean(pmf[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mace-pmf", required=True)
    ap.add_argument("--chgnet-pmf", required=True)
    ap.add_argument("--mace-summary", required=True)
    ap.add_argument("--chgnet-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mace = load_pmf(args.mace_pmf)
    chgnet = load_pmf(args.chgnet_pmf)
    with open(args.mace_summary) as f:
        m_sum = json.load(f)
    with open(args.chgnet_summary) as f:
        c_sum = json.load(f)

    # Plateau values (intermediate water-bound region)
    m_plateau = find_plateau(mace[:, 0], mace[:, 1])
    c_plateau = find_plateau(chgnet[:, 0], chgnet[:, 1])

    # Saddle (max in [1.5, 2.5] window — first ascent)
    def find_saddle(cv, pmf, lo=1.7, hi=2.5):
        mask = (cv >= lo) & (cv <= hi) & (~np.isnan(pmf))
        if mask.sum() == 0:
            return None, None
        idx = np.where(mask)[0]
        i_max = idx[np.nanargmax(pmf[idx])]
        return float(cv[i_max]), float(pmf[i_max])

    m_saddle_x, m_saddle_y = find_saddle(mace[:, 0], mace[:, 1])
    c_saddle_x, c_saddle_y = find_saddle(chgnet[:, 0], chgnet[:, 1])

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full PMFs
    ax = axes[0]
    ax.plot(mace[:, 0], mace[:, 1], lw=2.5, color="darkblue", label="MACE-MP-0 medium")
    ax.plot(chgnet[:, 0], chgnet[:, 1], lw=2.5, color="darkred", label="CHGNet-v0.3.0", linestyle="--")
    if m_saddle_y is not None:
        ax.plot(m_saddle_x, m_saddle_y, "o", color="darkblue", ms=10)
    if c_saddle_y is not None:
        ax.plot(c_saddle_x, c_saddle_y, "s", color="darkred", ms=10)
    if m_plateau is not None:
        ax.axhline(m_plateau, color="darkblue", lw=0.7, ls=":", alpha=0.5)
    if c_plateau is not None:
        ax.axhline(c_plateau, color="darkred", lw=0.7, ls=":", alpha=0.5)
    ax.set_xlabel("d_FeH (Å, smooth-min)")
    ax.set_ylabel("PMF (kJ/mol)")
    ax.set_title("W2 US PMF — apples-to-apples MACE vs CHGNet")
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: focus на physical region (Fe-chemi → intermediate)
    ax = axes[1]
    ax.plot(mace[:, 0], mace[:, 1], lw=2.5, color="darkblue", label="MACE")
    ax.plot(chgnet[:, 0], chgnet[:, 1], lw=2.5, color="darkred", label="CHGNet", linestyle="--")
    ax.set_xlim(1.4, 3.6)
    ax.set_ylim(-5, 120)
    ax.set_xlabel("d_FeH (Å)")
    ax.set_ylabel("PMF (kJ/mol)")
    ax.set_title("Physical region (Fe-chemi → intermediate water-bound)")

    # Annotations
    if m_saddle_x and c_saddle_x:
        ax.annotate(
            f"MACE saddle: {m_saddle_x:.2f} Å, {m_saddle_y:.0f} kJ/mol = {m_saddle_y*0.0103642697:.2f} eV\n"
            f"CHGNet saddle: {c_saddle_x:.2f} Å, {c_saddle_y:.0f} kJ/mol = {c_saddle_y*0.0103642697:.2f} eV\n"
            f"MACE plateau ({m_plateau:.0f} kJ/mol = {m_plateau*0.0103642697:.2f} eV)\n"
            f"CHGNet plateau ({c_plateau:.0f} kJ/mol = {c_plateau*0.0103642697:.2f} eV)" if m_plateau and c_plateau else "",
            xy=(0.05, 0.95), xycoords="axes fraction", fontsize=9, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "pmf_compare.png", dpi=130)
    plt.close(fig)

    # Summary
    summary = {
        "mace": {
            "min_A": m_sum["PMF_min_A"],
            "max_A": m_sum["PMF_max_A"],
            "saddle_A": m_saddle_x,
            "saddle_kJmol": m_saddle_y,
            "saddle_eV": m_saddle_y * 0.0103642697 if m_saddle_y else None,
            "plateau_kJmol": m_plateau,
            "plateau_eV": m_plateau * 0.0103642697 if m_plateau else None,
            "deltaF_max_kJmol": m_sum["deltaF_kjmol"],
            "deltaF_max_eV": m_sum["deltaF_eV"],
        },
        "chgnet": {
            "min_A": c_sum["PMF_min_A"],
            "max_A": c_sum["PMF_max_A"],
            "saddle_A": c_saddle_x,
            "saddle_kJmol": c_saddle_y,
            "saddle_eV": c_saddle_y * 0.0103642697 if c_saddle_y else None,
            "plateau_kJmol": c_plateau,
            "plateau_eV": c_plateau * 0.0103642697 if c_plateau else None,
            "deltaF_max_kJmol": c_sum["deltaF_kjmol"],
            "deltaF_max_eV": c_sum["deltaF_eV"],
        },
        "agreement_diff_eV": {
            "saddle": abs((m_saddle_y - c_saddle_y) * 0.0103642697) if m_saddle_y and c_saddle_y else None,
            "plateau": abs((m_plateau - c_plateau) * 0.0103642697) if m_plateau and c_plateau else None,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("=== MLIP US Comparison Summary ===")
    print("=" * 70)
    print(f"  MACE-MP-0 medium:")
    print(f"    Fe-chemi min:  d = {m_sum['PMF_min_A']:.3f} Å")
    print(f"    Saddle:        d = {m_saddle_x:.3f} Å, F = {m_saddle_y:.2f} kJ/mol = {m_saddle_y*0.0103642697:.3f} eV")
    print(f"    Plateau:       F ≈ {m_plateau:.2f} kJ/mol = {m_plateau*0.0103642697:.3f} eV")
    print(f"    Edge max:      d = {m_sum['PMF_max_A']:.3f} Å, F = {m_sum['deltaF_kjmol']:.1f} kJ/mol = {m_sum['deltaF_eV']:.3f} eV")
    print(f"")
    print(f"  CHGNet-v0.3.0:")
    print(f"    Fe-chemi min:  d = {c_sum['PMF_min_A']:.3f} Å")
    print(f"    Saddle:        d = {c_saddle_x:.3f} Å, F = {c_saddle_y:.2f} kJ/mol = {c_saddle_y*0.0103642697:.3f} eV")
    print(f"    Plateau:       F ≈ {c_plateau:.2f} kJ/mol = {c_plateau*0.0103642697:.3f} eV")
    print(f"    Edge max:      d = {c_sum['PMF_max_A']:.3f} Å, F = {c_sum['deltaF_kjmol']:.1f} kJ/mol = {c_sum['deltaF_eV']:.3f} eV")
    print(f"")
    if summary["agreement_diff_eV"]["saddle"] is not None:
        print(f"  Agreement (saddle): |MACE - CHGNet| = {summary['agreement_diff_eV']['saddle']:.3f} eV")
    if summary["agreement_diff_eV"]["plateau"] is not None:
        print(f"  Agreement (plateau): |MACE - CHGNet| = {summary['agreement_diff_eV']['plateau']:.3f} eV")
    print()
    print(f"  Outputs: {out_dir}/pmf_compare.png, summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
