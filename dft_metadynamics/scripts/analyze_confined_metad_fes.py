#!/usr/bin/env python3
"""
W2 CP2K direct-DFT WT-MetaD FES reconstruction + convergence diagnostics.

Input:
  Directory containing CP2K HILLS-1_*_*.metadynLog + COLVAR-1_*.metadynLog files.

Approach:
  Native CP2K WT-MetaD (USE_PLUMED .FALSE.) writes one record per file:
    HILLS-1_<step>_<batch>.metadynLog : "time_fs  CV_at_deposit  sigma  height_Hartree"
    COLVAR-1_<step>.metadynLog        : "time_fs  CV  CV_velocity  bias  work  err"

  FES via sum-of-Gaussians:
    V_bias(s) = Σ_t h_t × exp(-(s - s_t)^2 / (2 σ^2))      [Hartree]
    F_WT(s)   = -(γ/(γ-1)) × V_bias(s)                     [Hartree]

  Convert to eV: F[eV] = F[Ha] × 27.2114.

System (W2 v2):
  3×3×1 mackinawite (Fe18 S18) + 12 H2O + 1 excess H+, 73 atoms.
  CV = COORDINATION number of atom 38 (first H of first H2O, per s139 rebrand)
       with all 18 surface Fe atoms.
  CP2K switching function: s_ij = (1 - (r/R0)^NN) / (1 - (r/R0)^(NN+ND))
       where R0 = 2.0 Å (input line 168), NN = 6, ND = 12 (input lines 169-170).
       Effective steepness exponent in denominator = NN + ND = 18 (NOT 12).
  CV → ~1.5-2 = Fe-chemi (bidentate-ish, weighted Fe shell occupancy).
  CV → 0   = water-bound (no Fe-H bonds within R0 cutoff).
  γ = 5 (WTGAMMA), σ = 0.05 (SCALE), WW = 5 kJ/mol = 0.001915 Ha.
  Wall at CV = 3.0 (prevents unphysical excursions).
  T = 300 K, k_B = 8.617e-5 eV/K, β = 1/(k_B T).
  Total MD: 15 ps (30000 steps × 0.5 fs).
  Spin: CP2K closed-shell RKS nspin=1 (mackinawite T_N=65K << 300K → PM surrogate, РЕШЕНИЕ-079).
"""
from __future__ import annotations
import sys, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HA_TO_EV = 27.211386245988
KJ_PER_MOL_TO_EV = 0.01036427
KB_EV = 8.617333262145e-5  # eV/K

# WT-MetaD params (from grotthuss_metadyn_v2.inp)
GAMMA = 5.0       # WTGAMMA
SIGMA = 0.05      # SCALE
WW_KJ = 5.0       # WW [kjmol]
T_MD  = 300.0     # K
GAMMA_FACTOR = GAMMA / (GAMMA - 1.0)   # = 1.25

DATA_DIR = Path("results/dft_datasets/2026-05-15/w2_metad_v2_final")
OUT_DIR  = Path("results/dft_datasets/2026-05-15/w2_metad_v2_final/analysis")
OUT_DIR.mkdir(exist_ok=True)


def load_hills(data_dir: Path) -> np.ndarray:
    """Parse all HILLS-1_<step>_<batch>.metadynLog files.

    Returns: array (N, 4) with columns [time_fs, CV, sigma, height_Ha], sorted by time.
    """
    pat = re.compile(r"-HILLS-1_(\d+)_(\d+)\.metadynLog$")
    hills = []
    for f in sorted(data_dir.glob("*-HILLS-1_*.metadynLog"),
                    key=lambda p: int(pat.search(p.name).group(1))):
        m = pat.search(f.name)
        if not m:
            continue
        step = int(m.group(1))
        text = f.read_text().strip()
        if not text:
            continue
        # Each file = ONE hill: "time_fs CV sigma height"
        parts = text.split()
        if len(parts) != 4:
            print(f"  WARN: {f.name} has {len(parts)} fields, skipping")
            continue
        t_fs, cv, sig, h = map(float, parts)
        hills.append((t_fs, cv, sig, h, step))
    arr = np.array([(t, cv, s, h) for t, cv, s, h, _ in hills], dtype=float)
    steps = np.array([step for *_, step in hills], dtype=int)
    print(f"Loaded {len(arr)} hills from {data_dir}")
    print(f"  time range: {arr[:,0].min():.1f} - {arr[:,0].max():.1f} fs")
    print(f"  CV range:   {arr[:,1].min():.3f} - {arr[:,1].max():.3f}")
    print(f"  σ (mean):   {arr[:,2].mean():.4f} (should be {SIGMA})")
    print(f"  height first/mid/last: {arr[0,3]:.5f} / {arr[len(arr)//2,3]:.5f} / {arr[-1,3]:.5f} Ha")
    return arr, steps


def load_colvar(data_dir: Path) -> np.ndarray:
    """Parse all COLVAR-1_<step>.metadynLog files.

    Returns: array (N, 6) [time_fs, CV, CV_vel, bias, work, err], sorted by time.
    """
    pat = re.compile(r"-COLVAR-1_(\d+)\.metadynLog$")
    rows = []
    for f in sorted(data_dir.glob("*-COLVAR-1_*.metadynLog"),
                    key=lambda p: int(pat.search(p.name).group(1))):
        m = pat.search(f.name)
        if not m:
            continue
        text = f.read_text().strip()
        if not text:
            continue
        parts = text.split()
        if len(parts) != 6:
            print(f"  WARN: {f.name} has {len(parts)} fields, skipping")
            continue
        rows.append(tuple(map(float, parts)))
    arr = np.array(rows, dtype=float)
    print(f"Loaded {len(arr)} COLVAR snapshots")
    print(f"  time range: {arr[:,0].min():.1f} - {arr[:,0].max():.1f} fs")
    print(f"  CV range:   {arr[:,1].min():.3f} - {arr[:,1].max():.3f}")
    return arr


def build_fes(hills: np.ndarray, cv_grid: np.ndarray, t_max_fs: float | None = None) -> np.ndarray:
    """Compute F(s) on cv_grid using hills with time ≤ t_max_fs.

    Convention: CP2K HILLS file stores pre-scaled height = W_0 × (γ/(γ-1)) × exp(-V_bias/(kT_ΔT)),
    i.e. the F-contribution per hill (NOT the raw deposit h_t).
    Reference: https://manual.cp2k.org/trunk/CP2K_INPUT/MOTION/FREE_ENERGY/METADYN/PRINT/HILLS.html
    Verification: observed first hill (V_bias≈0) = 65 meV = WW=5 kJ/mol × 1.25 = WW × γ/(γ-1) ✓.

    Therefore:
      F(s) = -Σ_t h_file × exp(-(s-s_t)^2 / (2 σ^2))     [Hartree], NO extra γ/(γ-1).
    """
    if t_max_fs is None:
        mask = np.ones(len(hills), dtype=bool)
    else:
        mask = hills[:, 0] <= t_max_fs
    sub = hills[mask]
    cv_t   = sub[:, 1][:, None]   # (N, 1)
    sigma_t = sub[:, 2][:, None]  # (N, 1)
    h_t    = sub[:, 3][:, None]   # (N, 1) — pre-scaled per CP2K convention
    grid   = cv_grid[None, :]      # (1, M)
    # Sum of Gaussians (vectorized): (N, M) → (M,)
    delta2 = (grid - cv_t) ** 2
    gauss  = np.exp(-delta2 / (2.0 * sigma_t ** 2))
    minus_F_Ha = np.sum(h_t * gauss, axis=0)   # = -F (because heights pre-scaled)
    F_Ha = -minus_F_Ha
    return F_Ha


def main():
    print("=" * 70)
    print("W2 CP2K direct-DFT WT-MetaD FES reconstruction (paper #2 anchor)")
    print("=" * 70)
    hills, hill_steps = load_hills(DATA_DIR)
    colvar = load_colvar(DATA_DIR)

    # CV grid covers physical range [0, 3] (wall at CV=3.0)
    cv_grid = np.linspace(0.0, 3.0, 601)  # step 0.005

    # Full FES (all hills)
    F_full_Ha = build_fes(hills, cv_grid, t_max_fs=None)
    F_full_eV = F_full_Ha * HA_TO_EV
    # Gauge: min(F) = 0
    F_full_eV -= F_full_eV.min()

    # Convergence: FES from first half / last half
    t_half = 7500.0  # fs (half of 15 ps)
    F_first_eV = build_fes(hills, cv_grid, t_max_fs=t_half) * HA_TO_EV
    F_first_eV -= F_first_eV.min()
    F_last_Ha  = build_fes(hills[hills[:,0] > t_half], cv_grid, t_max_fs=None) * HA_TO_EV
    F_last_Ha -= F_last_Ha.min()

    # Identify Fe-chemi minimum (CV in [1.5, 2.2]). For "saddle" and "water" we must FIRST
    # check that hills were actually deposited in those CV regions; otherwise F is a flat
    # plateau (unexplored) and argmax/argmin returns the mask edge — a TECHNICAL ARTIFACT,
    # not a physical saddle. Consilium s144 (chemist + physicist) explicitly flagged this.
    fe_chemi_mask = (cv_grid >= 1.5) & (cv_grid <= 2.2)
    water_mask    = (cv_grid >= 0.0) & (cv_grid <= 0.4)
    barrier_mask  = (cv_grid >= 0.3) & (cv_grid <= 1.5)  # transition region

    cv_hills_min = hills[:, 1].min()
    cv_hills_max = hills[:, 1].max()

    def explored(mask: np.ndarray, cv_grid_local: np.ndarray) -> bool:
        """Return True if any hill was deposited within this CV mask range."""
        lo = cv_grid_local[mask].min()
        hi = cv_grid_local[mask].max()
        return (hi >= cv_hills_min) and (lo <= cv_hills_max)

    idx_min   = np.argmin(F_full_eV[fe_chemi_mask]) + np.where(fe_chemi_mask)[0][0]
    F_min_eV  = F_full_eV[idx_min]
    cv_min    = cv_grid[idx_min]

    # Saddle: only report if at least one hill was placed inside the barrier mask
    barrier_explored = explored(barrier_mask, cv_grid)
    if barrier_explored:
        idx_saddle  = np.argmax(F_full_eV[barrier_mask]) + np.where(barrier_mask)[0][0]
        F_saddle_eV = F_full_eV[idx_saddle]
        cv_saddle   = cv_grid[idx_saddle]
        saddle_note = "argmax on explored region"
    else:
        idx_saddle  = -1
        F_saddle_eV = float("nan")
        cv_saddle   = float("nan")
        saddle_note = "PLATEAU ARTIFACT — no hills in [0.3, 1.5]; saddle NOT measured"

    # Water-bound min: same check
    water_explored = explored(water_mask, cv_grid)
    if water_explored:
        idx_water  = np.argmin(F_full_eV[water_mask]) + np.where(water_mask)[0][0]
        F_water_eV = F_full_eV[idx_water]
        cv_water   = cv_grid[idx_water]
        water_note = "argmin on explored region"
    else:
        idx_water  = -1
        F_water_eV = float("nan")
        cv_water   = float("nan")
        water_note = "PLATEAU ARTIFACT — no hills in [0.0, 0.4]; water minimum NOT measured"

    # Basin-depth lower bound: edge of EXPLORED region (where hills were deposited).
    # This is the only quantity we can defensibly extract from this run.
    cv_explored_low  = cv_hills_min
    cv_explored_high = cv_hills_max
    idx_edge_low  = int(np.argmin(np.abs(cv_grid - cv_explored_low)))
    idx_edge_high = int(np.argmin(np.abs(cv_grid - cv_explored_high)))
    F_edge_low_eV  = F_full_eV[idx_edge_low]
    F_edge_high_eV = F_full_eV[idx_edge_high]
    basin_depth_LB_low  = F_edge_low_eV  - F_min_eV
    basin_depth_LB_high = F_edge_high_eV - F_min_eV

    if barrier_explored and not np.isnan(F_saddle_eV):
        dF_barrier_fwd = F_saddle_eV - F_min_eV
    else:
        dF_barrier_fwd = float("nan")
    if water_explored and not np.isnan(F_water_eV):
        dF_barrier_rev = F_saddle_eV - F_water_eV if barrier_explored else float("nan")
        dF_reaction    = F_water_eV - F_min_eV
    else:
        dF_barrier_rev = float("nan")
        dF_reaction    = float("nan")

    print()
    print("=" * 70)
    print("FES features (full 15 ps)")
    print("=" * 70)
    print(f"Hill placement range:  CV ∈ [{cv_hills_min:.3f}, {cv_hills_max:.3f}]  ({len(hills)} hills)")
    print(f"Fe-chemi minimum:      CV = {cv_min:.3f}   F = {F_min_eV*1000:7.1f} meV ({F_min_eV:.4f} eV) (gauge)")
    print(f"Basin-depth LB (FWD):  edge of explored region @ CV = {cv_explored_low:.3f}, F = {F_edge_low_eV*1000:7.1f} meV → ΔF_basin ≥ {basin_depth_LB_low:.4f} eV")
    print(f"Basin-depth LB (REV):  edge of explored region @ CV = {cv_explored_high:.3f}, F = {F_edge_high_eV*1000:7.1f} meV → ΔF_basin ≥ {basin_depth_LB_high:.4f} eV")
    if barrier_explored:
        print(f"Transition saddle:     CV = {cv_saddle:.3f}   F = {F_saddle_eV*1000:7.1f} meV ({F_saddle_eV:.4f} eV)  [{saddle_note}]")
    else:
        print(f"Transition saddle:     NOT MEASURED — {saddle_note}")
    if water_explored:
        print(f"Water-bound min:       CV = {cv_water:.3f}   F = {F_water_eV*1000:7.1f} meV ({F_water_eV:.4f} eV)  [{water_note}]")
    else:
        print(f"Water-bound min:       NOT MEASURED — {water_note}")
    print()
    if barrier_explored:
        print(f"  ΔF‡ (Fe-chemi → water,  forward) = {dF_barrier_fwd:.4f} eV  ({dF_barrier_fwd*1000:.1f} meV)")
        print(f"  ΔF‡ (water → Fe-chemi,  reverse) = {dF_barrier_rev:.4f} eV  ({dF_barrier_rev*1000:.1f} meV)")
        print(f"  ΔF_rxn (Fe-chemi → water, state) = {dF_reaction:.4f} eV  ({dF_reaction*1000:.1f} meV)")
    else:
        print(f"  ΔF‡ saddle not measurable from this run (no hills crossed transition region).")
        print(f"  Reported basin-depth LB ≥ {basin_depth_LB_low:.4f} eV is depth at explored basin edge,")
        print(f"  not a saddle. The TRUE saddle along this CV is at higher F by an unknown amount.")

    # Convergence diagnostic: half-trajectory drift of BASIN DEPTH (not "saddle" — see fix above)
    # The first/last-half saddle drift was misleading per consilium (physicist Q4): late hills
    # are small by WT design, so ΔF(last) is mechanically small. Replace with basin-edge depth.
    idx_m_first = np.argmin(F_first_eV[fe_chemi_mask]) + np.where(fe_chemi_mask)[0][0]
    idx_m_last  = np.argmin(F_last_Ha[fe_chemi_mask]) + np.where(fe_chemi_mask)[0][0]
    basin_first = F_first_eV[idx_edge_low] - F_first_eV[idx_m_first]
    basin_last  = F_last_Ha[idx_edge_low]  - F_last_Ha[idx_m_last]
    basin_full  = basin_depth_LB_low

    print()
    print("=" * 70)
    print("Convergence diagnostics — basin-depth LB drift")
    print("=" * 70)
    print(f"Basin-depth LB from first 7.5 ps (150 hills): {basin_first:.4f} eV")
    print(f"Basin-depth LB from last  7.5 ps (150 hills): {basin_last:.4f} eV")
    print(f"Basin-depth LB from full 15.0 ps (300 hills): {basin_full:.4f} eV")
    print(f"|basin(first) - basin(last)| = {abs(basin_first - basin_last)*1000:.1f} meV")
    print(f"(NB: late hills are small by WT design → last-half ΔF is mechanically smaller;")
    print(f" this is not a strict convergence test. Use Tiwary-Parrinello reweighted PDF instead.)")
    print()

    # Hill height decay (well-tempered convergence indicator)
    n_hills = len(hills)
    bins = np.array_split(hills[:, 3], 6)
    bin_means = [b.mean() for b in bins]
    bin_means_eV = [b * HA_TO_EV for b in bin_means]
    print("Hill height decay (Ha, eV) — well-tempered convergence:")
    for i, (h_ha, h_ev) in enumerate(zip(bin_means, bin_means_eV)):
        t0 = i * 2500
        t1 = (i + 1) * 2500
        print(f"  t ∈ [{t0:5d}, {t1:5d}] fs:  <h> = {h_ha:.5f} Ha = {h_ev*1000:6.2f} meV")
    # Bonomi-Parrinello criterion: h_last / h_first (NOT h_last / W0_prefactor — that's a different metric)
    bp_ratio = bin_means[-1] / bin_means[0]
    W0_prescaled_eV = (WW_KJ * KJ_PER_MOL_TO_EV) * GAMMA_FACTOR
    h_last_to_W0_prescaled = bin_means_eV[-1] / W0_prescaled_eV
    print(f"  Bonomi-Parrinello ratio h_last/h_first (mean):  {bp_ratio:.4f} = {bp_ratio*100:.1f}%")
    print(f"    classical BP threshold: < 5% → {'PASS' if bp_ratio < 0.05 else 'FAIL (basin not self-consistent)'}")
    print(f"  Alternative metric h_last/(W0×γ/(γ-1)):  {h_last_to_W0_prescaled:.4f} = {h_last_to_W0_prescaled*100:.1f}%")
    print(f"    (different metric — relative to theoretical max prefactor, NOT BP)")
    print(f"  V_bias_typical_late ≈ -k_B·ΔT·ln(h_last/h_first_pre):")
    h_first_pre_eV = W0_prescaled_eV
    if bin_means_eV[-1] > 0 and h_first_pre_eV > 0:
        V_bias_typical_late = -KB_EV * (GAMMA-1) * T_MD * np.log(bin_means_eV[-1] / h_first_pre_eV)
        print(f"    V_bias_typical_late = {V_bias_typical_late:.4f} eV")
        print(f"    self-consistency ratio V_bias_typical_late / basin_depth_LB = {V_bias_typical_late / basin_full:.3f}")
        print(f"    WT asymptote target: (γ-1)/γ = {(GAMMA-1)/GAMMA:.3f}")
        if V_bias_typical_late / basin_full < 0.7:
            print(f"    → partial basin filling (far below WT asymptote 0.8); formally NOT converged")

    # COLVAR statistics
    print()
    print("=" * 70)
    print("CV trajectory statistics (3001 COLVAR snapshots × 5 fs)")
    print("=" * 70)
    cv_series = colvar[:, 1]
    print(f"  mean CV: {cv_series.mean():.3f}")
    print(f"  std CV:  {cv_series.std():.3f}")
    print(f"  CV range: [{cv_series.min():.3f}, {cv_series.max():.3f}]")
    print(f"  fraction CV > 1.5 (Fe-chemi region): {(cv_series > 1.5).mean()*100:.1f}%")
    print(f"  fraction CV ∈ [0.3, 1.5] (transition): {((cv_series >= 0.3) & (cv_series <= 1.5)).mean()*100:.1f}%")
    print(f"  fraction CV < 0.3 (water region): {(cv_series < 0.3).mean()*100:.1f}%")
    print(f"  wall hits (CV > 2.95): {(cv_series > 2.95).sum()} / {len(cv_series)} ({(cv_series > 2.95).mean()*100:.2f}%)")

    # CV PDF (histogram, biased trajectory)
    cv_hist, cv_edges = np.histogram(cv_series, bins=np.linspace(0, 3, 61), density=True)
    cv_centers = 0.5 * (cv_edges[:-1] + cv_edges[1:])

    # Plot 1: FES with HONEST annotations (no fake saddle)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    ax.plot(cv_grid, F_full_eV * 1000, lw=2, color="C0", label="FES (full 15 ps)")
    ax.plot(cv_grid, F_first_eV * 1000, lw=1.2, ls="--", color="C2", alpha=0.7, label="first 7.5 ps")
    ax.plot(cv_grid, F_last_Ha * 1000, lw=1.2, ls=":", color="C3", alpha=0.7, label="last 7.5 ps")
    # Shade UNEXPLORED regions (no hills, F plateau is artifact)
    ax.axvspan(0, cv_hills_min, color="gray", alpha=0.15, label=f"NOT EXPLORED (no hills, CV<{cv_hills_min:.2f})")
    ax.axvspan(cv_hills_max, 3.0, color="gray", alpha=0.15)
    ax.axvline(cv_min, color="C0", alpha=0.4, ls=":")
    ax.axvline(cv_explored_low, color="orange", alpha=0.6, ls="--", label=f"explored-edge CV={cv_explored_low:.2f}")
    ax.axvline(cv_explored_high, color="orange", alpha=0.6, ls="--")
    ax.annotate(f"Fe-chemi min\n(CV={cv_min:.2f}, F=0 gauge)",
                (cv_min, 0), xytext=(cv_min+0.1, 70), fontsize=9, color="C0")
    ax.annotate(f"basin-depth LB\nΔF_basin ≥ {basin_depth_LB_low*1000:.0f} meV\n(@ explored edge)",
                (cv_explored_low, F_edge_low_eV*1000),
                xytext=(cv_explored_low-0.5, F_edge_low_eV*1000+30),
                fontsize=9, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", alpha=0.7))
    ax.set_xlabel("CV = COORDINATION (H_38, 18 Fe; R₀=2.0 Å, NN=6, ND=12)")
    ax.set_ylabel("F (meV)")
    ax.set_title("W2 CP2K direct-DFT WT-MetaD FES (γ=5, WW=5 kJ/mol, 15 ps)\nbasin depth along CV — NOT desorption saddle (see SI)")
    ax.legend(loc="upper center", fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: hill height decay
    ax = axes[0, 1]
    ax.plot(hills[:, 0] / 1000, hills[:, 3] * 1000 * HA_TO_EV, ".", ms=3, alpha=0.7)
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("hill height (meV)")
    ax.set_yscale("log")
    ax.set_title("Hill height decay (WT convergence indicator)")
    ax.grid(alpha=0.3, which="both")

    # Plot 3: CV trajectory
    ax = axes[1, 0]
    ax.plot(colvar[:, 0] / 1000, colvar[:, 1], lw=0.5, alpha=0.6)
    ax.axhline(cv_min, color="C0", ls=":", label=f"Fe-chemi CV={cv_min:.2f}")
    ax.axhline(cv_explored_low, color="orange", ls="--", alpha=0.7, label=f"explored-edge CV={cv_explored_low:.2f}")
    ax.axhline(cv_explored_high, color="orange", ls="--", alpha=0.7)
    ax.axhline(3.0, color="black", ls="--", alpha=0.5, label="wall CV=3.0")
    ax.set_xlabel("t (ps)")
    ax.set_ylabel("CV (COORDINATION H_38–Fe, dimensionless)")
    ax.set_title("CV trajectory (biased sampling)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 4: CV PDF + hill placement histogram
    ax = axes[1, 1]
    ax.bar(cv_centers, cv_hist, width=cv_edges[1]-cv_edges[0], alpha=0.5, color="C0", label="biased CV PDF")
    # Overlay hill placement histogram (where bias was deposited)
    hill_hist, hill_edges = np.histogram(hills[:, 1], bins=np.linspace(0, 3, 61), density=True)
    hill_centers = 0.5 * (hill_edges[:-1] + hill_edges[1:])
    ax.bar(hill_centers, hill_hist, width=hill_edges[1]-hill_edges[0],
           alpha=0.5, color="orange", label=f"hill placements ({len(hills)} hills)")
    ax.axvline(cv_min, color="C0", ls=":")
    ax.axvline(cv_explored_low, color="orange", ls="--", alpha=0.7)
    ax.axvline(cv_explored_high, color="orange", ls="--", alpha=0.7)
    ax.set_xlabel("CV")
    ax.set_ylabel("density")
    ax.set_title("CV histogram + hill placement (gap = unexplored region)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fes_w2_metad_v2.png", dpi=150)
    print(f"\nSaved plot → {OUT_DIR / 'fes_w2_metad_v2.png'}")

    # Save FES data to txt
    np.savetxt(OUT_DIR / "fes_w2_metad_v2_full.dat",
               np.column_stack([cv_grid, F_full_eV * 1000]),
               header="CV  F(meV)", fmt="%.5f")
    np.savetxt(OUT_DIR / "fes_w2_metad_v2_first7p5ps.dat",
               np.column_stack([cv_grid, F_first_eV * 1000]),
               header="CV  F(meV)", fmt="%.5f")
    np.savetxt(OUT_DIR / "fes_w2_metad_v2_last7p5ps.dat",
               np.column_stack([cv_grid, F_last_Ha * 1000]),
               header="CV  F(meV)", fmt="%.5f")

    # Summary JSON — honest version (NaN-aware, no fake saddle)
    import json
    def _nan_to_none(x):
        return None if (isinstance(x, float) and np.isnan(x)) else float(x)

    summary = {
        "system": "3x3x1 mackinawite Fe18S18 + 12 H2O + 1 H+ (73 atoms, charge=+1 jellium)",
        "cv_definition": (
            "CP2K COORDINATION of atom 38 (first H of first H2O, s139 rebrand) with 18 surface Fe atoms; "
            "switching function s_ij = (1 - (r/R0)^NN) / (1 - (r/R0)^(NN+ND)), "
            "R0 = 2.0 Å, NN = 6, ND = 12 (effective steepness exponent NN+ND = 18)"
        ),
        "method": "CP2K WT-MetaD direct DFT (Quickstep PBE-D3(BJ), GAPW, GTH-PBE-q16, RKS nspin=1)",
        "params": {
            "gamma": GAMMA,
            "sigma_cv": SIGMA,
            "WW_kjmol": WW_KJ,
            "T_K": T_MD,
            "wall_cv": 3.0,
            "MD_steps": 30000,
            "dt_fs": 0.5,
            "total_time_ps": 15.0,
            "NT_HILLS": 100,
            "n_hills_deposited": len(hills),
            "cv_R0_A": 2.0,
            "cv_NN": 6,
            "cv_ND": 12,
            "cv_effective_m": 18,
            "spin_treatment": "RKS_nspin_1_PM_surrogate",
        },
        "hill_placement_range_cv": [float(cv_hills_min), float(cv_hills_max)],
        "basin_depth_LB_eV": {
            "Fe_chemi_min_cv": float(cv_min),
            "Fe_chemi_min_F_eV": float(F_min_eV),
            "explored_edge_low_cv": float(cv_explored_low),
            "explored_edge_high_cv": float(cv_explored_high),
            "basin_depth_LB_fwd_edge_eV": float(basin_depth_LB_low),
            "basin_depth_LB_rev_edge_eV": float(basin_depth_LB_high),
            "interpretation": "Lower bound on basin depth ALONG THIS CV; NOT a desorption saddle. The true saddle (if any along this CV) is at F >= basin_depth_LB. CV completeness vs full desorption coordinate is NOT validated by this run.",
        },
        "fes_features_eV_DEPRECATED_AS_SADDLE": {
            "WARNING": "These fields are reported for diagnostic transparency only. The 'saddle' below is the argmax on a flat plateau (no hills deposited in barrier_mask region) — see consilium s144. Use basin_depth_LB_eV above instead for paper-grade claims.",
            "barrier_mask_explored": bool(barrier_explored),
            "water_mask_explored": bool(water_explored),
            "saddle_cv":  _nan_to_none(cv_saddle),
            "saddle_F_eV": _nan_to_none(F_saddle_eV),
            "water_min_cv": _nan_to_none(cv_water),
            "water_min_F_eV": _nan_to_none(F_water_eV),
            "dF_forward_eV_argmax_on_mask": _nan_to_none(dF_barrier_fwd),
            "dF_reverse_eV_argmax_on_mask": _nan_to_none(dF_barrier_rev),
            "dF_reaction_eV_argmin_on_mask": _nan_to_none(dF_reaction),
        },
        "convergence": {
            "method": "basin-depth LB drift between time halves",
            "basin_LB_first_half_eV": float(basin_first),
            "basin_LB_last_half_eV": float(basin_last),
            "basin_LB_full_eV": float(basin_full),
            "abs_diff_first_last_meV": float(abs(basin_first - basin_last) * 1000),
            "WARNING": "First/last-half drift is misleading by WT design (late hills are small); use Tiwary-Parrinello reweighted CV PDF for strict convergence test.",
            "hill_height_first_bin_mean_meV": float(bin_means_eV[0] * 1000),
            "hill_height_last_bin_mean_meV":  float(bin_means_eV[-1] * 1000),
            "bonomi_parrinello_ratio_h_last_to_h_first": float(bp_ratio),
            "bonomi_parrinello_pass_5pct": bool(bp_ratio < 0.05),
            "h_last_to_W0_prescaled_ratio": float(h_last_to_W0_prescaled),
            "V_bias_typical_late_eV": float(V_bias_typical_late) if bin_means_eV[-1] > 0 else None,
            "V_bias_typical_late_over_basin_LB": float(V_bias_typical_late / basin_full) if bin_means_eV[-1] > 0 else None,
            "WT_asymptote_target_gamma_minus_1_over_gamma": (GAMMA-1)/GAMMA,
            "formally_converged": bool(bp_ratio < 0.05),
        },
        "cv_trajectory_stats": {
            "n_snapshots": int(len(colvar)),
            "mean": float(cv_series.mean()),
            "std": float(cv_series.std()),
            "cv_range_visited": [float(cv_series.min()), float(cv_series.max())],
            "frac_fe_chemi_above_1p5": float((cv_series > 1.5).mean()),
            "frac_transition_0p3_to_1p5": float(((cv_series >= 0.3) & (cv_series <= 1.5)).mean()),
            "frac_water_below_0p3": float((cv_series < 0.3).mean()),
            "wall_hits_above_2p95_count": int((cv_series > 2.95).sum()),
            "wall_hits_frac": float((cv_series > 2.95).mean()),
        },
        "comparison_paper2": {
            "MACE_MP0_saddle_eV": 0.80,
            "CHGNet_v030_saddle_eV": 0.32,
            "MLIP_bracket_paper2": [0.32, 0.80],
            "DFT_v1_LB_wall_corrected_eV": 0.40,
            "DFT_v2_basin_depth_LB_eV": float(basin_full),
            "NOTE": "DFT v2 basin depth LB is along COORDINATION CV; MLIP US used smooth-min(d_FeH) CV. Cross-CV comparison not directly bracketable per consilium s144; topology-disagreement framing (chemist): DFT basin depth > entire CHGNet PMF range conditional on CV faithfulness.",
        },
        "consilium_status": {
            "pass_1_completed_2026_05_15": True,
            "pass_1_verdict_chemist": "CONDITIONAL_PASS_9_fixes",
            "pass_1_verdict_physicist": "BLOCKED_2_blockers",
            "pass_2_required_before_paper_grade": True,
        },
    }
    with open(OUT_DIR / "summary_w2_metad_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary → {OUT_DIR / 'summary_w2_metad_v2.json'}")

    print()
    print("=" * 70)
    print("CONCLUSION (post-consilium, honest framing)")
    print("=" * 70)
    print(f"CP2K WT-MetaD v2, 15 ps, γ=5, WW=5 kJ/mol — along H38-Fe COORDINATION CV:")
    print(f"  basin-depth LB (Fe-chemi → forward explored edge):   {basin_full:.3f} eV ({basin_full*1000:.0f} meV)")
    print(f"  basin-depth LB (Fe-chemi → reverse explored edge):   {basin_depth_LB_high:.3f} eV")
    print(f"  hill placement range: CV ∈ [{cv_hills_min:.2f}, {cv_hills_max:.2f}]")
    print(f"  Bonomi-Parrinello h_last/h_first: {bp_ratio*100:.1f}% (criterion <5%) → "
          f"{'PASS' if bp_ratio < 0.05 else 'FAIL — not converged'}")
    print()
    print(f"  This is NOT a desorption saddle along a 2D PMF.")
    print(f"  MLIP bracket (s136): [0.32, 0.80] eV (paper-grade, MLIP US alone — keep until consilium pass-2).")
    print(f"  DFT basin depth LB ({basin_full:.2f} eV) > entire CHGNet PMF range (0.32 eV)")
    print(f"   → valid topology-disagreement claim conditional on CV faithfulness (chemist framing).")


if __name__ == "__main__":
    main()
