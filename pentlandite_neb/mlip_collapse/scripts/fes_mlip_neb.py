"""
Fe-S V_Fe+H proton-hop MLIP-vs-DFT NEB benchmark harness.

Generalized from tmp/peer_tests/test_eqv2_omat.py (W-specific) to load endpoints
from curated extxyz files for 5 Fe-S minerals and run, per the LOCKED protocol:
  1. Load endA / endB (from separate files, or band frames 0 / -1).
  2. Built-in structure-identity self-check: formula vs expected, N atoms, min-dist,
     endA vs endB must DIFFER (a proton hop moved something).
  3. MLIP-relax BOTH endpoints (BFGS, fmax=0.03, maxstep=0.1, cell fixed).
  4. IDPP interpolation (mic=True), 7 images (5 interior), CI-NEB (climb, k=0.1).
  5. CI-NEB optimize (FIRE, fmax=0.05).
  6. barrier_eV = max(E_images) - E(endA_relaxed).
  7. Energy-force consistency on relaxed endA: FD vs predicted force,
     delta=0.01, 15 random (atom,axis) samples rng(42) -> max & RMS meV/A.

Models (--model): eqv2 (OMat24 hero, non-conservative, local .pt),
                  orb (orb-v2, non-conservative),
                  mace-mp (MACE-MP-0 medium, conservative CONTROL),
                  mace-omat (MACE OMat24, gomer GPU only).

Output: results/dft_datasets/2026-06-25_mlip_neb_sweep/<model>_<mineral>.json

Usage:
  python -u fes_mlip_neb.py --model mace-mp --mineral pyrite_VFe [--device cpu] [--skip-neb]
"""
# safejson-override

import argparse
import json
import sys
import time
import traceback
import warnings
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np
from ase.io import read
from ase.optimize import BFGS, FIRE

warnings.filterwarnings("ignore")

EV_PER_AA_TO_MEV_PER_AA = 1000.0

STRUCT_DIR = Path(
    "D:/home/ignat/project-third-matter/git/mlip-vs-dft-iron-sulfides/data/structures"
)
DEFAULT_OUT_DIR = Path(
    "D:/home/ignat/project-third-matter/results/dft_datasets/2026-06-25_mlip_neb_sweep"
)
EQV2_CKPT = Path("D:/home/ignat/project-third-matter/data/eqV2_31M_omat_mp_salex.pt")

# Gate-verified (tmp/verify_fes_endpoints.py). formula = order-independent multiset.
# band: load endpoints from a single multiframe band file (idx 0 and -1).
# else: separate endA/endB files (each read with index=-1 = final relaxed frame).
MINERALS = {
    "pyrite_VFe": {
        "band": "pyrite_VFe_band_MACE-MP-0.extxyz",
        "endA_idx": 0, "endB_idx": -1,
        "formula": "Fe31S64H", "N": 96, "dft_meV": 268, "nspin": 1,
        "dft_method": "dimer", "magnetic": False, "symmetric": True,
        "endpoint_origin": "MACE-MP-0 band (DFT NEB did not converge for pyrite)",
    },
    "mackinawite": {
        "endA": "mackinawite_endA.extxyz", "endB": "mackinawite_endB.extxyz",
        "formula": "Fe35S36H", "N": 72, "dft_meV": 43, "nspin": 1,
        "dft_method": "NEB", "magnetic": False, "symmetric": True,
        "endpoint_origin": "DFT-relaxed",
    },
    "marcasite": {
        "endA": "marcasite_endA.extxyz", "endB": "marcasite_endB.extxyz",
        "formula": "Fe31S64H", "N": 96, "dft_meV": 208, "nspin": 2,
        "dft_method": "NEB", "magnetic": True, "symmetric": False,
        "endpoint_origin": "DFT-relaxed",
    },
    "pentlandite": {
        "endA": "pentlandite_endA.extxyz", "endB": "pentlandite_endB.extxyz",
        "formula": "Fe71S64H", "N": 136, "dft_meV": None, "nspin": 1,
        "dft_method": "prediction-only", "magnetic": False, "symmetric": False,
        "endpoint_origin": "DFT-relaxed (EXCLUDED: endpoints unrelated, rmsd 6.6 A)",
    },
    "greigite": {
        "endA": "greigite_endA.extxyz", "endB": "greigite_endB.extxyz",
        "formula": "Fe23S32H", "N": 56, "dft_meV": 1861, "nspin": 2,
        "dft_method": "NEB", "magnetic": True, "symmetric": True,
        "endpoint_origin": "DFT-relaxed",
    },
}


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# ============================================================
# Structure loading + identity self-check
# ============================================================
def canonical_formula(atoms):
    """Order-independent Hill-like multiset string, e.g. -> 'Fe31HS64' (alphabetical)."""
    c = Counter(atoms.get_chemical_symbols())
    parts = []
    for sym in sorted(c):
        n = c[sym]
        parts.append(f"{sym}{n}" if n > 1 else sym)
    return "".join(parts)


def parse_formula(s):
    """'Fe31S64H' -> Counter({'Fe':31,'S':64,'H':1}). Order-independent."""
    import re
    out = Counter()
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", s):
        if sym:
            out[sym] += int(num) if num else 1
    return out


def load_endpoints(mineral):
    spec = MINERALS[mineral]
    if "band" in spec:
        path = STRUCT_DIR / spec["band"]
        endA = read(str(path), index=spec["endA_idx"])
        endB = read(str(path), index=spec["endB_idx"])
        srcA = f"{spec['band']}[{spec['endA_idx']}]"
        srcB = f"{spec['band']}[{spec['endB_idx']}]"
    else:
        # Files may be multiframe relaxation trajs (e.g. pentlandite_endA) -> take final.
        endA = read(str(STRUCT_DIR / spec["endA"]), index=-1)
        endB = read(str(STRUCT_DIR / spec["endB"]), index=-1)
        srcA, srcB = spec["endA"], spec["endB"]
    return endA, endB, srcA, srcB


def min_distance(atoms):
    d = atoms.get_all_distances(mic=True)
    n = len(atoms)
    iu = np.triu_indices(n, k=1)
    return float(d[iu].min())


def h_host_check(atoms, label=""):
    """BLOCKER-2 gate (chemist): did the H stay on an S-H covalent site, or did a
    spin-blind MLIP drag it into an Fe-hydride/Fe-bridge site? If H moved off S,
    the barrier is NOT a S-H<->S-H hop and must not be compared to the DFT S-H ref."""
    sym = atoms.get_chemical_symbols()
    h_idx = [i for i, s in enumerate(sym) if s == "H"]
    if not h_idx:
        return {"host": None, "d_nn_A": None, "s_h_preserved": None, "note": "no H"}
    h = h_idx[0]
    d = atoms.get_distances(h, range(len(atoms)), mic=True)
    order = np.argsort(d)
    nbr = int(order[1])  # order[0] is H itself (d=0)
    host, dnn = sym[nbr], float(d[nbr])
    ok = bool((host == "S") and (1.25 <= dnn <= 1.60))
    if label:
        print(f"    [{label}] H nearest = {host} at {dnn:.3f} A  s_h_preserved={ok}", flush=True)
    return {"host": host, "d_nn_A": dnn, "s_h_preserved": ok}


def endpoint_rmsd(a, b):
    """RMSD of positions (mic) between two same-N configs -- must be > 0 (a hop occurred)."""
    if len(a) != len(b):
        return float("nan")
    # use raw position difference wrapped by cell -> displacement magnitude
    diff = a.get_positions() - b.get_positions()
    # minimum image of the displacement
    cell = a.get_cell()
    inv = np.linalg.inv(cell.array)
    frac = diff @ inv
    frac -= np.round(frac)
    cart = frac @ cell.array
    return float(np.sqrt((cart ** 2).sum(axis=1).mean()))


def verify_endpoints(mineral, endA, endB):
    """Built-in structure-identity gate. Returns dict; raises on hard failure."""
    spec = MINERALS[mineral]
    fA, fB = canonical_formula(endA), canonical_formula(endB)
    # normalize expected (drop trailing '1')
    exp = spec["formula"]
    mdA, mdB = min_distance(endA), min_distance(endB)
    rmsd = endpoint_rmsd(endA, endB)
    info = {
        "expected_formula": exp,
        "formula_endA": fA, "formula_endB": fB,
        "N_endA": len(endA), "N_endB": len(endB), "N_expected": spec["N"],
        "min_dist_endA_A": mdA, "min_dist_endB_A": mdB,
        "endpoint_rmsd_A": rmsd,
    }
    # Order-independent multiset comparison (parse both to Counters)
    cA = Counter(endA.get_chemical_symbols())
    cB = Counter(endB.get_chemical_symbols())
    cExp = parse_formula(exp)
    if cA != cB:
        raise ValueError(f"endA formula {fA} != endB formula {fB} (composition changed!)")
    if cA != cExp:
        raise ValueError(f"formula {fA} (counts {dict(cA)}) != expected {exp} "
                         f"(counts {dict(cExp)}) for {mineral}")
    if len(endA) != spec["N"] or len(endB) != spec["N"]:
        raise ValueError(f"N mismatch: endA {len(endA)}, endB {len(endB)}, expected {spec['N']}")
    if min(mdA, mdB) < 1.0:
        raise ValueError(f"min-dist too small ({min(mdA,mdB):.3f} A) -- bad geometry "
                         f"(Fe-S-H physical min: S-H~1.34, Fe-H~1.6, Fe-S~2.2)")
    if not (rmsd > 1e-3):
        raise ValueError(f"endA == endB (rmsd {rmsd:.5f} A) -- no proton hop; NEB meaningless")
    if rmsd > 1.5:
        raise ValueError(
            f"endpoint rmsd {rmsd:.3f} A too large for a proton hop "
            f"(clean hops are ~0.1-0.4 A over all atoms) -- endpoints look unrelated; "
            f"NEB interpolation would be meaningless")
    return info


# ============================================================
# Calculator factories
# ============================================================
def make_calc_factory(model, device):
    if model == "eqv2":
        if not EQV2_CKPT.exists():
            raise FileNotFoundError(f"eqV2 checkpoint not found: {EQV2_CKPT}")
        from fairchem.core import OCPCalculator
        cpu = (device == "cpu")
        def factory():
            calc = OCPCalculator(checkpoint_path=str(EQV2_CKPT), cpu=cpu)
            return calc, "eqV2_31M_omat_mp_salex"
        return factory, "non-conservative"
    if model == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator
        def factory():
            m, adapter = pretrained.orb_v2(device=device)
            calc = ORBCalculator(m, adapter, device=device)
            return calc, "orb-v2 (orb-models 0.7.0)"
        return factory, "non-conservative"
    if model == "mace-mp":
        from mace.calculators import mace_mp
        def factory():
            calc = mace_mp(model="medium", device=device, default_dtype="float64")
            return calc, "mace-mp-0-medium"
        return factory, "conservative"
    if model == "mace-omat":
        # mace-torch 0.3.15: keyword "medium-omat-0" (OMat24-trained MACE, ASL license).
        from mace.calculators import mace_mp
        def factory():
            calc = mace_mp(model="medium-omat-0", device=device, default_dtype="float64")
            return calc, "mace-omat-0 (medium-omat-0)"
        return factory, "conservative"
    raise ValueError(f"unknown model {model}")


# ============================================================
# Consistency (energy-force) test
# ============================================================
def fd_force_component(calc, atoms_in, atom_idx, axis, delta=0.01):
    atoms_p = deepcopy(atoms_in)
    atoms_p.positions[atom_idx, axis] += delta
    atoms_p.calc = calc
    e_plus = atoms_p.get_potential_energy()
    atoms_m = deepcopy(atoms_in)
    atoms_m.positions[atom_idx, axis] -= delta
    atoms_m.calc = calc
    e_minus = atoms_m.get_potential_energy()
    return -(e_plus - e_minus) / (2.0 * delta)


def run_consistency(calc_factory, atoms_ref, n_samples=15, delta=0.01, seed=42):
    # Reuse a single calc (NICE-fix): models are deterministic on a fixed geometry,
    # and ASE recomputes when positions change -> correct AND ~Nx faster for heavy
    # loaders (eqV2). f_pred and all FD energies come from the SAME model instance.
    calc, _ = calc_factory()
    a = deepcopy(atoms_ref)
    a.calc = calc
    e_ref = a.get_potential_energy()
    f_pred = a.get_forces().copy()
    N = len(atoms_ref)
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < n_samples:
        pairs.add((int(rng.integers(0, N)), int(rng.integers(0, 3))))
    pairs = sorted(pairs)
    diffs = []
    triples = []
    for i, (ai, ax) in enumerate(pairs):
        f_fd = fd_force_component(calc, atoms_ref, ai, ax, delta=delta)
        d = float(f_pred[ai, ax]) - float(f_fd)
        diffs.append(d * 1000.0)
        triples.append({"atom": ai, "axis": ax,
                        "f_pred_eV_A": float(f_pred[ai, ax]),
                        "f_fd_eV_A": float(f_fd), "diff_meV_A": float(d * 1000.0)})
        if (i + 1) % 5 == 0:
            print(f"    consistency {i+1}/{n_samples}", flush=True)
    diffs = np.array(diffs)
    return {
        "e_ref_eV": float(e_ref),
        "max_abs_diff_meV_A": float(np.max(np.abs(diffs))),
        "rms_diff_meV_A": float(np.sqrt(np.mean(diffs ** 2))),
        "n_samples": n_samples, "delta_A": delta, "seed": seed,
        "fd_noise_floor_note": (
            "float32 models (eqV2/orb): FD noise floor ~18 meV/A at delta=0.01 "
            "(machine eps x |E| / 2delta); float64 (MACE): <0.1 meV/A. "
            "Non-conservative signal must exceed this floor to be meaningful."
        ),
        "raw_triples": triples,
    }


# ============================================================
# Relax + NEB
# ============================================================
def relax(atoms_in, calc_factory, label, fmax=0.03, max_steps=200, maxstep=0.1):
    atoms = deepcopy(atoms_in)
    calc, _ = calc_factory()
    atoms.calc = calc
    sc = [0]
    t0 = time.time()
    opt = BFGS(atoms, logfile=None, maxstep=maxstep)

    def cnt():
        sc[0] += 1
        if sc[0] % 25 == 0:
            fm = float(np.max(np.abs(atoms.get_forces())))
            print(f"    [{label}] step {sc[0]}, fmax={fm:.4f}, t={time.time()-t0:.0f}s", flush=True)
    opt.attach(cnt)
    conv = False
    try:
        conv = bool(opt.run(fmax=fmax, steps=max_steps))
    except Exception as e:
        print(f"    [{label}] BFGS exception: {e}", flush=True)
    fm = float(np.max(np.abs(atoms.get_forces())))
    e = float(atoms.get_potential_energy())
    print(f"    [{label}] done: {sc[0]} steps, fmax={fm:.4f}, E={e:.6f}, conv={conv}", flush=True)
    return atoms, conv, sc[0], e, fm


def run_neb(calc_factory, ep_a, ep_b, n_interior=5, fmax=0.05, max_steps=300):
    """Two-phase CI-NEB (physicist SHOULD-FIX): phase 1 plain NEB (climb=False) to a
    loose fmax to find the rough MEP, then phase 2 enable climbing image to fmax.
    Avoids climbing-image divergence on flat PES (e.g. mackinawite ~43 meV)."""
    from ase.mep.neb import NEB
    images = [deepcopy(ep_a)]
    for _ in range(n_interior):
        img = deepcopy(ep_a)
        c, _ = calc_factory()
        img.calc = c
        images.append(img)
    images.append(deepcopy(ep_b))
    ca, _ = calc_factory()
    cb, _ = calc_factory()
    images[0].calc = ca
    images[-1].calc = cb
    neb = NEB(images, climb=False, k=0.1)
    neb.interpolate(method="idpp", mic=True)
    mds_interp = [min_distance(im) for im in images]
    print(f"    interpolated min-dists: {[round(x,3) for x in mds_interp]}", flush=True)
    sc = [0]
    t0 = time.time()

    def make_cnt(tag):
        def cnt():
            sc[0] += 1
            if sc[0] % 10 == 0:
                try:
                    fm = float(np.max(np.abs(neb.get_forces())))
                except Exception:
                    fm = float("nan")
                print(f"    [NEB-{tag}] step {sc[0]}, fmax={fm:.4f}, t={time.time()-t0:.0f}s",
                      flush=True)
        return cnt

    half = max(max_steps // 2, 50)
    conv = False
    # Phase 1: plain NEB to loose fmax (rough MEP)
    try:
        opt1 = FIRE(neb, logfile=None, maxstep=0.1, dt=0.05)
        opt1.attach(make_cnt("plain"))
        opt1.run(fmax=max(fmax, 0.10), steps=half)
    except Exception as e:
        print(f"    [NEB-plain] exception: {e}", flush=True)
    # Phase 2: enable climbing image, tighten to target fmax
    try:
        neb.climb = True
        opt2 = FIRE(neb, logfile=None, maxstep=0.1, dt=0.05)
        opt2.attach(make_cnt("climb"))
        conv = bool(opt2.run(fmax=fmax, steps=max_steps - sc[0]))
    except Exception as e:
        print(f"    [NEB-climb] exception: {e}", flush=True)
    energies = []
    for im in images:
        try:
            energies.append(float(im.get_potential_energy()))
        except Exception:
            energies.append(float("nan"))
    try:
        fm = float(np.max(np.abs(neb.get_forces())))
    except Exception:
        fm = float("nan")
    mds_final = [min_distance(im) for im in images]
    return conv, energies, sc[0], fm, mds_interp, mds_final


# ============================================================
# Main
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["eqv2", "orb", "mace-mp", "mace-omat"])
    p.add_argument("--mineral", required=True, choices=list(MINERALS))
    p.add_argument("--device", default="cpu")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--skip-neb", action="store_true")
    p.add_argument("--skip-consistency", action="store_true")
    p.add_argument("--neb-max-steps", type=int, default=300)
    p.add_argument("--relax-max-steps", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=15)
    p.add_argument("--delta", type=float, default=0.01)
    args = p.parse_args()

    spec = MINERALS[args.mineral]
    print("=" * 60, flush=True)
    print(f"FES MLIP NEB: model={args.model} mineral={args.mineral} device={args.device}", flush=True)
    print("=" * 60, flush=True)

    results = {
        "model": args.model,
        "mineral": args.mineral,
        "device": args.device,
        "dft_ref_meV": spec["dft_meV"],
        "dft_ref_basis": "electronic_no_ZPE",
        "dft_method": spec["dft_method"],
        "nspin_dft": spec["nspin"],
        "magnetic": spec["magnetic"],
        "symmetric_dft_endpoints": spec.get("symmetric"),
        "endpoint_origin": spec.get("endpoint_origin"),
        "had_initial_magmoms": None,
        "protocol": {
            "relax": "BFGS fmax=0.03 maxstep=0.1 cell-fixed",
            "interp": "IDPP mic=True 7 images (5 interior)",
            "neb": "two-phase CI-NEB FIRE: climb=False to fmax~0.10 then climb=True to 0.05, k=0.1",
            "barrier_def": "max(E_images) - E_images[0] (NEB endpoint A, self-consistent zero)",
            "consistency": f"FD delta={args.delta} {args.n_samples} samples rng(42)",
        },
        "structure_check": None,
        "model_type": None,
        "endpoint_a": None, "endpoint_b": None,
        "neb": None, "consistency": None, "barrier_eV": None,
    }

    # --- Load + structure-identity gate ---
    try:
        endA, endB, srcA, srcB = load_endpoints(args.mineral)
        chk = verify_endpoints(args.mineral, endA, endB)
        chk["srcA"], chk["srcB"] = srcA, srcB
        results["structure_check"] = chk
        results["had_initial_magmoms"] = bool(
            endA.has("initial_magmoms") or "magmoms" in endA.arrays)
        print(f"  STRUCT OK: {chk['formula_endA']} N={chk['N_endA']} "
              f"min-d {chk['min_dist_endA_A']:.3f}/{chk['min_dist_endB_A']:.3f} "
              f"rmsd {chk['endpoint_rmsd_A']:.3f}", flush=True)
    except Exception as e:
        print(f"  STRUCT GATE FAILED: {e}", flush=True)
        results["structure_check"] = {"status": "FAILED", "error": str(e)}
        _save(args, results)
        sys.exit(2)

    # --- Calculator ---
    try:
        calc_factory, mtype = make_calc_factory(args.model, args.device)
        results["model_type"] = mtype
    except Exception as e:
        print(f"  CALC LOAD FAILED: {e}", flush=True)
        traceback.print_exc()
        results["model_type"] = {"status": "FAILED", "error": str(e),
                                 "traceback": traceback.format_exc()}
        _save(args, results)
        sys.exit(3)

    # --- Relax endpoints + NEB ---
    if not args.skip_neb:
        try:
            t0 = time.time()
            print("  Relaxing endpoint A...", flush=True)
            a_rel, ca, sa, ea, fma = relax(endA, calc_factory, "EP_A",
                                           max_steps=args.relax_max_steps)
            print("  Relaxing endpoint B...", flush=True)
            b_rel, cb, sb, eb, fmb = relax(endB, calc_factory, "EP_B",
                                           max_steps=args.relax_max_steps)
            rmsd_rel = endpoint_rmsd(a_rel, b_rel)
            md_a, md_b = min_distance(a_rel), min_distance(b_rel)
            print(f"  relaxed endpoint rmsd: {rmsd_rel:.4f} A  "
                  f"min-dist {md_a:.3f}/{md_b:.3f}", flush=True)
            # BLOCKER-2 (chemist): did H stay on its S-H site after MLIP relax?
            hA = h_host_check(a_rel, "EP_A")
            hB = h_host_check(b_rel, "EP_B")
            s_h_ok = bool(hA.get("s_h_preserved") and hB.get("s_h_preserved"))
            relaxed_geom_ok = bool(md_a >= 1.0 and md_b >= 1.0)
            print("  Running two-phase CI-NEB...", flush=True)
            conv, energies, nsteps, nfmax, mds_interp, mds_final = run_neb(
                calc_factory, a_rel, b_rel, fmax=0.05, max_steps=args.neb_max_steps)
            valid = [e for e in energies if not np.isnan(e)]
            # BLOCKER-1 (both): self-consistent zero = NEB endpoint A (energies[0]).
            e_neb_a, e_neb_b = energies[0], energies[-1]
            if abs(e_neb_a - ea) > 0.005:
                print(f"  WARNING: E(NEB img0)={e_neb_a:.6f} vs E(relax A)={ea:.6f} "
                      f"-> delta {abs(e_neb_a-ea)*1000:.2f} meV", flush=True)
            e_max = max(valid) if valid else float("nan")
            saddle_idx = int(np.nanargmax(energies)) if valid else -1
            barrier_fwd = float(e_max - e_neb_a)
            barrier_rev = float(e_max - e_neb_b)
            dE_endpoints_meV = float((e_neb_b - e_neb_a) * 1000.0)
            rel = [float(e - e_neb_a) for e in energies]
            barrier = barrier_fwd
            # reliability flags
            min_img = float(min(mds_final)) if mds_final else float("nan")
            saddle_on_endpoint = saddle_idx in (0, len(energies) - 1)
            sym_violated = bool(spec.get("symmetric") and abs(dE_endpoints_meV) > 50)
            reliability = []
            if not conv:
                reliability.append("NEB_UNCONVERGED")
            if saddle_on_endpoint:
                reliability.append("SADDLE_ON_ENDPOINT_no_barrier")
            if sym_violated:
                reliability.append("MLIP_BROKE_DFT_SYMMETRY")
            if not s_h_ok:
                reliability.append("H_LEFT_S-H_SITE_mechanism_changed")
            if min_img < 1.0:
                reliability.append("IMAGE_MIN_DIST_lt_1.0_possible_collapse")
            reliable = (len(reliability) == 0)
            results["endpoint_a"] = {"converged": ca, "n_steps": sa, "energy_eV": ea,
                                     "final_fmax": fma, "min_dist_A": md_a,
                                     "h_host": hA, "src": srcA}
            results["endpoint_b"] = {"converged": cb, "n_steps": sb, "energy_eV": eb,
                                     "final_fmax": fmb, "min_dist_A": md_b,
                                     "h_host": hB, "src": srcB,
                                     "rmsd_vs_a_relaxed_A": rmsd_rel}
            results["neb"] = {
                "converged": conv, "n_steps": nsteps, "final_fmax_eV_A": nfmax,
                "image_energies_eV": energies,
                "image_energies_relative_to_A_eV": rel,
                "interp_min_dists_A": mds_interp,
                "final_min_dists_A": mds_final,
                "saddle_image_idx": saddle_idx,
                "barrier_fwd_eV": barrier_fwd, "barrier_fwd_meV": barrier_fwd * 1000.0,
                "barrier_rev_eV": barrier_rev, "barrier_rev_meV": barrier_rev * 1000.0,
                "dE_endpoints_meV": dE_endpoints_meV,
                "barrier_eV": barrier, "barrier_meV": barrier * 1000.0,
                "s_h_preserved_both": s_h_ok,
                "relaxed_geometry_ok": relaxed_geom_ok,
                "barrier_reliable": reliable,
                "reliability_flags": reliability,
                "elapsed_s": time.time() - t0,
            }
            results["barrier_eV"] = barrier
            dref = spec["dft_meV"]
            dstr = f"{dref} meV" if dref is not None else "n/a (prediction)"
            flagstr = "RELIABLE" if reliable else ("FLAGS: " + ", ".join(reliability))
            print(f"  BARRIER fwd={barrier_fwd*1000:.1f} rev={barrier_rev*1000:.1f} meV "
                  f"(DFT {dstr}), dE_endpoints={dE_endpoints_meV:.1f} meV, "
                  f"saddle_idx={saddle_idx}, conv={conv}  [{flagstr}]", flush=True)
            print(f"  MEP relative (meV): {[round(x*1000,1) for x in rel]}", flush=True)
        except Exception as e:
            print(f"  NEB FAILED: {e}", flush=True)
            traceback.print_exc()
            results["neb"] = {"status": "FAILED", "error": str(e),
                              "traceback": traceback.format_exc()}

    # --- Consistency on relaxed endA (fallback: raw endA if NEB skipped/failed) ---
    if not args.skip_consistency:
        if (not args.skip_neb) and isinstance(results.get("endpoint_a"), dict) \
                and "energy_eV" in results["endpoint_a"]:
            ref_atoms = a_rel
        else:
            ref_atoms = endA
        try:
            print("  Energy-force consistency...", flush=True)
            cons = run_consistency(calc_factory, ref_atoms,
                                   n_samples=args.n_samples, delta=args.delta)
            results["consistency"] = cons
            print(f"  consistency: max={cons['max_abs_diff_meV_A']:.2f} "
                  f"RMS={cons['rms_diff_meV_A']:.2f} meV/A", flush=True)
        except Exception as e:
            print(f"  CONSISTENCY FAILED: {e}", flush=True)
            traceback.print_exc()
            results["consistency"] = {"status": "FAILED", "error": str(e)}

    _save(args, results)
    print("\n=== SUMMARY ===", flush=True)
    print(f"  {args.model} / {args.mineral}: "
          f"barrier={results['barrier_eV']} eV  DFT={spec['dft_meV']} meV", flush=True)


def _save(args, results):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.model}_{args.mineral}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, cls=SafeJSONEncoder)
    print(f"  saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
