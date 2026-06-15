#!/usr/bin/env python3
"""analyze_dft_us.py -- per-window analysis of a CP2K DFT umbrella-sampling run.

Paper #2 (KineticTrap, mackinawite Fe-H trap). Processes ONE umbrella window:
a multi-frame CP2K position trajectory (.xyz, written `EACH 5` MD steps) plus an
optional CP2K restraint colvar log (PROJECT-1.restraint), and emits the
carrier-decisive observables + the Umbrella-Integration (UI) spot mean force.

The two decisions this window contributes to (per the s167 consilium):

  1. CARRIER of the excess proton (atom 38, 1-based) along the CV: S-H shuttle
     (CHGNet path) vs water-O (MACE path) vs Fe-bound hydride. Judged by
     GEOMETRY (reduced bond ratios), 3-component + in_flight, with a covalent
     S-H gate and a time-persistence filter -- NOT by barrier height (the +1
     charge + MP corpus bias the height; see consilium framing note).

  2. UI spot mean force F'(<xi>) for the eventual stitched (or per-window-only)
     PMF readout.

CRITICAL UNIT CONVENTION (consilium BLOCKER 1, verified vs the CP2K manual)
---------------------------------------------------------------------------
A CP2K `&RESTRAINT` adds energy  E = K * (X - TARGET)^2   -- WITHOUT the 1/2
factor that the MLIP US used (V = 1/2 * kappa * d^2). Therefore the effective
harmonic stiffness is  kappa_eff = 2 * K_cp2k, and the Kaestner-Thiel UI spot
mean force at the biased mean is

    F'(<xi>) = kappa_eff * (center - <xi>) = 2 * K_cp2k * (center - <xi>)   [kJ/mol/A]

If you (wrongly) use K_cp2k you get HALF the mean force. With the production
K_cp2k = 500 kJ/mol/A^2, kappa_eff = 1000 kJ/mol/A^2 = the MLIP kappa
(apples-to-apples). The sampled width is sigma = sqrt(kT / (2 K_cp2k)) ~ 0.0499 A
at 300 K, K=500 (the /(2K) again reflects the missing 1/2).

The carrier classification scheme is reused (for consistency) from
``prodromos.us_preflight_gate._classify_carrier``:
    carrier = argmin_X (d_X / L_X)   if that min < bond_cutoff_ratio else 'in_flight'
with L_Fe=1.60, L_S=1.36, L_O=0.98, bond_cutoff_ratio=1.25.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# --- physical constants (match prodromos.us_preflight_gate) -----------------
KB_KJ_MOL_K = 0.0083145          # Boltzmann constant in kJ/mol/K
EV_PER_KJMOL = 1.0 / 96.485      # 1 kJ/mol in eV

# --- carrier-classification scheme (REUSED from us_preflight_gate U5a) -------
L_FE_DEFAULT = 1.60
L_S_DEFAULT = 1.36
L_O_DEFAULT = 0.98
BOND_CUTOFF_RATIO_DEFAULT = 1.25

# covalent S-H gate window (consilium chemist: true S-H only in this range)
SH_COVALENT_LO = 1.30
SH_COVALENT_HI = 1.50

# 3-component carrier labels (water = O-bound) + in_flight
CARRIERS = ("Fe", "S-H", "water", "in_flight")

# default CP2K excess-proton atom (1-based 38 -> 0-based 37)
EXCESS_H_INDEX0_DEFAULT = 37

# default cell from w2_frame0.xyz (orthorhombic)
DEFAULT_CELL = (11.022, 11.022, 6.5)


def kT_kJ_mol(temperature_K: float) -> float:
    """kT in kJ/mol. kT(300 K) = 2.4943 kJ/mol."""
    return KB_KJ_MOL_K * float(temperature_K)


# ---------------------------------------------------------------------------
# trajectory parsing (CP2K multi-frame .xyz)
# ---------------------------------------------------------------------------
def parse_xyz_trajectory(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse a multi-frame CP2K .xyz trajectory.

    Returns (species, coords) where ``species`` is the per-atom element list
    (taken from the FIRST frame; the topology is fixed) and ``coords`` is an
    array of shape (n_frames, n_atoms, 3) in Angstrom.

    Robust to: blank trailing lines, a comment line of any content, and an
    incomplete final frame (silently dropped). Raises ValueError on a
    fundamentally malformed first header.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    i = 0
    n_lines = len(lines)
    species: list[str] | None = None
    frames: list[np.ndarray] = []

    while i < n_lines:
        # skip blank lines between frames
        if not lines[i].strip():
            i += 1
            continue
        # header: atom count
        try:
            nat = int(lines[i].strip().split()[0])
        except (ValueError, IndexError):
            raise ValueError(f"malformed .xyz header at line {i + 1}: {lines[i]!r}")
        # need comment line + nat atom lines after the count line
        if i + 1 + nat >= n_lines + 1 and (i + 2 + nat) > n_lines:
            # not enough lines for a complete frame -> drop the partial frame
            break
        if i + 2 + nat > n_lines:
            break
        sp_frame: list[str] = []
        xyz_frame = np.empty((nat, 3), dtype=float)
        ok = True
        for a in range(nat):
            row = lines[i + 2 + a].split()
            if len(row) < 4:
                ok = False
                break
            sp_frame.append(row[0])
            try:
                xyz_frame[a, 0] = float(row[1])
                xyz_frame[a, 1] = float(row[2])
                xyz_frame[a, 2] = float(row[3])
            except ValueError:
                ok = False
                break
        if not ok:
            # malformed/partial frame -> stop here (graceful)
            break
        if species is None:
            species = sp_frame
        frames.append(xyz_frame)
        i += 2 + nat

    if species is None or not frames:
        return (species or []), np.empty((0, 0, 3), dtype=float)
    coords = np.stack(frames, axis=0)
    return species, coords


# ---------------------------------------------------------------------------
# minimum-image distances (orthorhombic cell)
# ---------------------------------------------------------------------------
def mic_min_distance(h_pos: np.ndarray, target_pos: np.ndarray,
                     cell: tuple[float, float, float]) -> float:
    """Minimum over ``target_pos`` of the orthorhombic-MIC distance H<->target.

    ``h_pos`` shape (3,), ``target_pos`` shape (m, 3). Returns nan if no targets.
    """
    if target_pos.shape[0] == 0:
        return float("nan")
    L = np.asarray(cell, dtype=float)
    d = target_pos - h_pos[None, :]          # (m, 3)
    d -= L[None, :] * np.round(d / L[None, :])  # wrap each component to [-L/2, L/2)
    dist = np.sqrt(np.sum(d * d, axis=1))
    return float(np.min(dist))


def frame_carrier_distances(
    coords_frame: np.ndarray, species: list[str], h_index0: int,
    cell: tuple[float, float, float],
) -> tuple[float, float, float]:
    """For one frame return (d_FeH, d_SH, d_OH) = min MIC distance from the
    excess H (index ``h_index0``) to any Fe / S / O atom. Species read from the
    xyz, NOT hardcoded indices."""
    h_pos = coords_frame[h_index0]
    sp = np.asarray(species)
    fe_pos = coords_frame[sp == "Fe"]
    s_pos = coords_frame[sp == "S"]
    o_pos = coords_frame[sp == "O"]
    d_fe = mic_min_distance(h_pos, fe_pos, cell)
    d_s = mic_min_distance(h_pos, s_pos, cell)
    d_o = mic_min_distance(h_pos, o_pos, cell)
    return d_fe, d_s, d_o


# ---------------------------------------------------------------------------
# per-frame carrier classification (3-component + covalent S-H gate)
# ---------------------------------------------------------------------------
def classify_carrier_frame(
    d_fe: float, d_s: float, d_o: float,
    *, L_Fe: float = L_FE_DEFAULT, L_S: float = L_S_DEFAULT, L_O: float = L_O_DEFAULT,
    bond_cutoff_ratio: float = BOND_CUTOFF_RATIO_DEFAULT,
) -> str:
    """Per-frame H-carrier as a 3-component label (Fe / S-H / water) + in_flight.

    Step 1 (REUSED scheme from us_preflight_gate._classify_carrier): carrier =
    argmin of reduced bond ratio r_X = d_X / L_X, if min < bond_cutoff_ratio else
    'in_flight'. The O-bound minimum is relabelled 'water'.

    Step 2 (consilium chemist, covalent S-H gate): a frame is only counted as a
    *true* S-H bond if d_SH in [1.30, 1.50] AND d_SH < d_OH AND d_SH < d_FeH. If
    argmin says 'S' but this covalent gate fails, the H is not covalently on S ->
    fall back to whichever of {Fe, water} is closer, or in_flight.
    """
    ratios = {"Fe": d_fe / L_Fe, "S": d_s / L_S, "O": d_o / L_O}
    best = min(ratios, key=ratios.get)
    if ratios[best] >= bond_cutoff_ratio:
        return "in_flight"

    if best == "O":
        return "water"
    if best == "Fe":
        return "Fe"

    # best == "S": apply the covalent S-H gate
    is_true_sh = (
        (SH_COVALENT_LO <= d_s <= SH_COVALENT_HI)
        and (d_s < d_o)
        and (d_s < d_fe)
    )
    if is_true_sh:
        return "S-H"
    # argmin picked S but the covalent gate failed: re-decide between Fe/water
    fb = {"Fe": ratios["Fe"], "water": ratios["O"]}
    fb_best = min(fb, key=fb.get)
    return fb_best if fb[fb_best] < bond_cutoff_ratio else "in_flight"


def apply_persistence(labels: list[str], persistence_frames: int) -> list[str]:
    """Time-persistence filter: a carrier label is only KEPT if it is part of a
    run of >= ``persistence_frames`` consecutive identical labels; otherwise the
    frame is demoted to 'in_flight' (transient, sub-threshold residence).

    This is what blocks a single S-H flicker among water frames from being
    counted as an S-H shuttle. persistence_frames <= 1 disables the filter.
    """
    n = len(labels)
    if persistence_frames <= 1 or n == 0:
        return list(labels)
    out = ["in_flight"] * n
    start = 0
    while start < n:
        end = start
        while end < n and labels[end] == labels[start]:
            end += 1
        run_len = end - start
        if run_len >= persistence_frames:
            for k in range(start, end):
                out[k] = labels[start]
        start = end
    return out


# ---------------------------------------------------------------------------
# CP2K restraint colvar log parsing (PROJECT-1.restraint)
# ---------------------------------------------------------------------------
def parse_restraint_colvar(path: Path) -> np.ndarray | None:
    """Parse a CP2K ``PROJECT-1.restraint`` colvar log into a 1-D array of the
    collective-variable value xi per logged step.

    The CP2K restraint log lines look like (whitespace-separated numeric columns)::

        <step>  <time>  <target>  <actual_cv>  <restraint_energy> ...

    Column conventions vary between CP2K versions; we read all numeric rows and
    take the column whose values are closest (median) to a plausible CV distance
    (1-4 Angstrom) -- the *actual* CV, not the target. If the log is absent or
    has no numeric rows, returns None (the caller falls back to the trajectory).
    """
    p = Path(path)
    if not p.exists():
        return None
    rows: list[list[float]] = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        try:
            rows.append([float(x) for x in parts])
        except ValueError:
            continue
    if not rows:
        return None
    width = min(len(r) for r in rows)
    arr = np.array([r[:width] for r in rows], dtype=float)
    if arr.shape[1] < 2 or arr.shape[0] == 0:
        return None
    # The *actual* CV column: median in a bond-distance band AND oscillating
    # tightly around the (constant) target -- so 0 < std < a small bound. This
    # distinguishes it from (a) the constant target column (std == 0), (b) the
    # restraint-energy column ~K*(x-target)^2 which is in-band but one-sided with
    # a larger relative spread, and (c) step/time columns (median out of band).
    # Among qualifying columns pick the one with the SMALLEST non-zero std (the
    # harmonically-restrained CV is the tightest in-band oscillator).
    CV_STD_MAX = 0.6     # restrained CV width << 0.6 A; energy spread is larger
    CV_STD_MIN = 1e-6    # exclude the constant target column (np.std roundoff ~1e-16)
    candidates: list[tuple[float, int]] = []
    for c in range(arr.shape[1]):
        col = arr[:, c]
        med = float(np.median(col))
        std = float(np.std(col))
        if 1.0 <= med <= 5.0 and CV_STD_MIN < std < CV_STD_MAX:
            candidates.append((std, c))
    if candidates:
        candidates.sort(key=lambda t: t[0])  # smallest std first
        return arr[:, candidates[0][1]]
    # no qualifying oscillating column: fall back to the last in-band column
    for c in range(arr.shape[1] - 1, -1, -1):
        if 1.0 <= float(np.median(arr[:, c])) <= 5.0:
            return arr[:, c]
    return None


# ---------------------------------------------------------------------------
# main per-window analysis
# ---------------------------------------------------------------------------
def analyze_window(
    window_dir: str | Path,
    *,
    center_A: float,
    K_cp2k: float,
    n_eq_frames: int,
    temperature_K: float = 300.0,
    cell: tuple[float, float, float] | None = None,
    h_index0: int = EXCESS_H_INDEX0_DEFAULT,
    persistence_frames: int = 20,
    fe_atom_index0: int | None = 12,   # Fe 13 (1-based) for the CV cross-check
    L_Fe: float = L_FE_DEFAULT,
    L_S: float = L_S_DEFAULT,
    L_O: float = L_O_DEFAULT,
    bond_cutoff_ratio: float = BOND_CUTOFF_RATIO_DEFAULT,
    sigma_tol: float = 0.20,
    traj_glob: tuple[str, ...] = ("*pos*.xyz", "*-pos-1.xyz", "*.xyz"),
    restraint_glob: tuple[str, ...] = ("*-1.restraint", "*.restraint"),
) -> dict:
    """Analyze one umbrella window. Returns the result dict (see module doc).

    center_A    : restraint TARGET [angstrom] for this window.
    K_cp2k      : the CP2K RESTRAINT K in kJ/mol/A^2 (NOT the half-stiffness;
                  kappa_eff = 2*K_cp2k is used for the mean force / sigma).
    n_eq_frames : number of leading (equilibration) frames to discard.
    """
    wd = Path(window_dir)
    if cell is None:
        cell = DEFAULT_CELL

    # locate the trajectory file
    traj_path = None
    for pat in traj_glob:
        cands = sorted(wd.glob(pat))
        # avoid picking the input/seed frame (single-frame) if a real traj exists
        cands = [c for c in cands if not c.name.lower().endswith("frame0.xyz")]
        if cands:
            traj_path = cands[0]
            break
    if traj_path is None:
        return _error_result(window_dir, center_A, K_cp2k,
                             f"no trajectory .xyz found under {wd} (globs {traj_glob})")

    species, coords = parse_xyz_trajectory(traj_path)
    n_total = coords.shape[0]
    if n_total == 0:
        return _error_result(window_dir, center_A, K_cp2k,
                             f"trajectory {traj_path.name} has 0 parseable frames")
    n_atoms = coords.shape[1]
    if h_index0 >= n_atoms:
        return _error_result(window_dir, center_A, K_cp2k,
                             f"h_index0={h_index0} out of range (n_atoms={n_atoms})")

    eq_dropped = min(max(int(n_eq_frames), 0), n_total)
    prod = coords[eq_dropped:]
    n_frames = prod.shape[0]
    if n_frames < 2:
        return _error_result(
            window_dir, center_A, K_cp2k,
            f"too few production frames ({n_frames}) after dropping {eq_dropped} "
            f"of {n_total}", n_frames=n_frames, eq_dropped=eq_dropped,
        )

    # per-frame carrier distances + raw labels
    d_fe_arr = np.empty(n_frames)
    d_s_arr = np.empty(n_frames)
    d_o_arr = np.empty(n_frames)
    raw_labels: list[str] = []
    for k in range(n_frames):
        d_fe, d_s, d_o = frame_carrier_distances(prod[k], species, h_index0, cell)
        d_fe_arr[k] = d_fe
        d_s_arr[k] = d_s
        d_o_arr[k] = d_o
        raw_labels.append(classify_carrier_frame(
            d_fe, d_s, d_o, L_Fe=L_Fe, L_S=L_S, L_O=L_O,
            bond_cutoff_ratio=bond_cutoff_ratio))

    # time-persistence filter
    labels = apply_persistence(raw_labels, persistence_frames)
    occupancy = {c: labels.count(c) / n_frames for c in CARRIERS}
    dominant_carrier = max(CARRIERS, key=lambda c: occupancy[c])
    sh_shuttle_present = occupancy["S-H"] > 0.0

    # decisive verdict text
    if dominant_carrier == "S-H":
        verdict = "S-H-shuttle"
    elif dominant_carrier == "water":
        verdict = "water-carrier"
    elif dominant_carrier == "Fe":
        verdict = "Fe-bound-hydride"
    else:
        verdict = "in-flight/no-clear-carrier"

    # --- CV (xi) for mean-force + sigma ---------------------------------------
    # Prefer the CP2K restraint colvar log; fall back to d(H, Fe-fe_atom_index0)
    # computed directly from the trajectory (the input CV = DISTANCE(38,13)).
    restraint_path = None
    for pat in restraint_glob:
        cands = sorted(wd.glob(pat))
        if cands:
            restraint_path = cands[0]
            break
    xi_source = "restraint_log"
    xi_full = parse_restraint_colvar(restraint_path) if restraint_path else None
    if xi_full is not None and len(xi_full) >= n_total:
        xi = xi_full[eq_dropped:eq_dropped + n_frames]
    elif xi_full is not None and len(xi_full) >= 2:
        # length mismatch (different logging stride): drop a proportional eq head
        drop = min(eq_dropped, max(len(xi_full) - 2, 0))
        xi = xi_full[drop:]
        xi_source = "restraint_log(length-mismatch)"
    else:
        # fall back: CV = MIC distance H <-> the named Fe atom
        xi_source = "trajectory_d(H,Fe%d)" % ((fe_atom_index0 or 0) + 1)
        if fe_atom_index0 is None or fe_atom_index0 >= n_atoms:
            xi = d_fe_arr.copy()  # last resort: nearest-Fe distance
            xi_source = "trajectory_d(H,nearestFe)"
        else:
            xi = np.empty(n_frames)
            fe_pos_series = prod[:, fe_atom_index0, :]
            for k in range(n_frames):
                xi[k] = mic_min_distance(prod[k, h_index0],
                                         fe_pos_series[k][None, :], cell)

    xi = np.asarray(xi, dtype=float)
    mean_xi = float(np.mean(xi))
    sigma_xi = float(np.std(xi, ddof=1)) if len(xi) > 1 else 0.0

    # --- UI mean force (CRITICAL factor-2) ------------------------------------
    kappa_eff = 2.0 * float(K_cp2k)                     # NO 1/2 in CP2K E=K*d^2
    F_mean_kJ = kappa_eff * (float(center_A) - mean_xi)  # kJ/mol/A
    F_mean_eV = F_mean_kJ * EV_PER_KJMOL

    # --- sigma check ----------------------------------------------------------
    kT = kT_kJ_mol(temperature_K)
    sigma_expected = math.sqrt(kT / (2.0 * float(K_cp2k))) if K_cp2k > 0 else float("nan")
    sigma_rel_dev = (abs(sigma_xi - sigma_expected) / sigma_expected
                     if sigma_expected > 0 else float("nan"))
    sigma_flag = bool(sigma_rel_dev > sigma_tol) if sigma_rel_dev == sigma_rel_dev else True

    result = {
        "window_dir": str(wd),
        "trajectory_file": traj_path.name,
        "center_A": float(center_A),
        "K_cp2k_kJ_mol_A2": float(K_cp2k),
        "kappa_eff_kJ_mol_A2": kappa_eff,
        "temperature_K": float(temperature_K),
        "cell_A": list(cell),
        "h_index_1based": int(h_index0) + 1,
        "n_frames_total": int(n_total),
        "eq_dropped": int(eq_dropped),
        "n_frames": int(n_frames),
        "persistence_frames": int(persistence_frames),
        "xi_source": xi_source,
        "mean_xi_A": mean_xi,
        "sigma_xi_A": sigma_xi,
        "sigma_expected_A": sigma_expected,
        "sigma_rel_dev": sigma_rel_dev,
        "sigma_flag_bad_K": sigma_flag,
        "F_prime_kJ_mol_A": F_mean_kJ,
        "F_prime_eV_A": F_mean_eV,
        "carrier_occupancy": occupancy,
        "dominant_carrier": dominant_carrier,
        "sh_shuttle_present": bool(sh_shuttle_present),
        "carrier_verdict": verdict,
        "carrier_distance_means_A": {
            "d_FeH": float(np.mean(d_fe_arr)),
            "d_SH": float(np.mean(d_s_arr)),
            "d_OH": float(np.mean(d_o_arr)),
        },
        "carrier_distance_min_A": {
            "d_FeH": float(np.min(d_fe_arr)),
            "d_SH": float(np.min(d_s_arr)),
            "d_OH": float(np.min(d_o_arr)),
        },
        "error": None,
    }
    result["summary_text"] = _summary_text(result)
    return result


def _error_result(window_dir, center_A, K_cp2k, msg, *, n_frames=0, eq_dropped=0) -> dict:
    return {
        "window_dir": str(window_dir),
        "center_A": float(center_A),
        "K_cp2k_kJ_mol_A2": float(K_cp2k),
        "kappa_eff_kJ_mol_A2": 2.0 * float(K_cp2k),
        "n_frames": int(n_frames),
        "eq_dropped": int(eq_dropped),
        "mean_xi_A": None,
        "sigma_xi_A": None,
        "F_prime_kJ_mol_A": None,
        "carrier_occupancy": None,
        "dominant_carrier": None,
        "sh_shuttle_present": None,
        "error": msg,
        "summary_text": f"[ERROR] window {window_dir}: {msg}",
    }


def _summary_text(r: dict) -> str:
    occ = r["carrier_occupancy"]
    occ_str = ", ".join(f"{c}={occ[c]:.2f}" for c in CARRIERS)
    flag = " [WARN: sigma mismatch -> K may be wrong]" if r["sigma_flag_bad_K"] else ""
    return (
        f"window center {r['center_A']:.3f} A | n_frames={r['n_frames']} "
        f"(dropped {r['eq_dropped']}) | <xi>={r['mean_xi_A']:.4f} A, "
        f"sigma={r['sigma_xi_A']:.4f} A (expected {r['sigma_expected_A']:.4f}){flag}\n"
        f"  UI mean force F' = 2*K*(center-<xi>) = {r['F_prime_kJ_mol_A']:.2f} kJ/mol/A "
        f"({r['F_prime_eV_A']:.4f} eV/A), kappa_eff = {r['kappa_eff_kJ_mol_A2']:.1f} kJ/mol/A^2\n"
        f"  carrier occupancy: {occ_str} -> dominant = {r['dominant_carrier']} "
        f"({r['carrier_verdict']}); S-H shuttle present = {r['sh_shuttle_present']}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Per-window analysis of a CP2K DFT umbrella-sampling run "
                    "(KineticTrap): proton carrier (S vs water vs Fe) + UI mean force.")
    p.add_argument("window_dir", help="window directory (CP2K .xyz traj + .restraint)")
    p.add_argument("--center", type=float, required=True, help="restraint TARGET [A]")
    p.add_argument("--k-cp2k", type=float, required=True,
                   help="CP2K RESTRAINT K [kJ/mol/A^2] (NOT half; kappa_eff=2*K)")
    p.add_argument("--n-eq-frames", type=int, default=0,
                   help="leading equilibration frames to discard")
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--persistence-frames", type=int, default=20,
                   help="min consecutive frames for a carrier to count (~50-100 fs)")
    p.add_argument("--h-index", type=int, default=EXCESS_H_INDEX0_DEFAULT + 1,
                   help="excess-H atom index (1-based; default 38)")
    p.add_argument("--fe-cv-index", type=int, default=13,
                   help="Fe atom for the CV cross-check (1-based; default 13)")
    p.add_argument("--cell", type=float, nargs=3, default=None,
                   metavar=("A", "B", "C"), help="orthorhombic cell (A); default w2_frame0")
    p.add_argument("--output", type=Path, default=None, help="write result JSON here")
    p.add_argument("--json", action="store_true", help="print result JSON to stdout")
    args = p.parse_args(argv)

    res = analyze_window(
        args.window_dir,
        center_A=args.center,
        K_cp2k=args.k_cp2k,
        n_eq_frames=args.n_eq_frames,
        temperature_K=args.temperature,
        cell=tuple(args.cell) if args.cell else None,
        h_index0=args.h_index - 1,
        fe_atom_index0=args.fe_cv_index - 1,
        persistence_frames=args.persistence_frames,
    )
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(res["summary_text"])
    return 1 if res.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
