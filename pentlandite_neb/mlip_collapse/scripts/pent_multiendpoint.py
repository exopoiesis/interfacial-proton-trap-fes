"""Pentlandite multi-endpoint H-landscape with a foundation MLIP (v3 approach).

Pentlandite is multi-state (cubane Fe4S4; S-H sites vs Fe-hydride wells), so a blind
symmetric S-H<->S-H NEB is ill-posed. Here we MAP the landscape:
  1. Relax the donor (H on its S) with the model.
  2. Find the Fe-vacancy centre as the largest void near H (grid max-clearance).
  3. Identify the cage S atoms coordinating the vacancy = candidate H sites.
  4. For each cage S: place H at 1.4 A toward the vacancy, fully relax, record where H
     ends up (S-bound minimum vs collapse into an Fe-hydride well), energy, d_H-S, d_H-Fe.

This directly tests Paper #2's claim (foundation MLIPs collapse the proton into the
deeper Fe-hydride well) for a SOTA OMat24 model, maps the multi-state landscape (v3),
and locates a valid S-bound hop pair if one exists (for a clean symmetric NEB).

Usage: python pent_multiendpoint.py --model mace-omat
"""
import argparse, json, sys, warnings
from copy import deepcopy
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, Path):        return str(obj)
        return super().default(obj)


sys.path.insert(0, "D:/home/ignat/project-third-matter/tmp")
import fes_mlip_neb as H
from ase.io import read, write

DONOR = "D:/home/ignat/project-third-matter/dft-neb/ph-diagnostic/pent_selective_harvest_2026-06-06/pent_endA/endA.traj"
OUT = Path("D:/home/ignat/project-third-matter/results/dft_datasets/2026-06-25_mlip_neb_sweep")


def mic_fn(atoms):
    cell = atoms.get_cell(); inv = np.linalg.inv(cell.array)
    def mic(p, q):
        d = p - q; f = d @ inv; f -= np.round(f); return f @ cell.array
    return mic


def find_vacancy_center(atoms, h_idx):
    """Largest-clearance void near H (excludes H). Grid search +/-2.0 A around H."""
    mic = mic_fn(atoms)
    pos = atoms.get_positions()
    others = [i for i in range(len(atoms)) if i != h_idx]
    best, bestclear = None, -1.0
    for dx in np.linspace(-2.0, 2.0, 9):
        for dy in np.linspace(-2.0, 2.0, 9):
            for dz in np.linspace(-2.0, 2.0, 9):
                pt = pos[h_idx] + np.array([dx, dy, dz])
                clr = min(np.linalg.norm(mic(pt, pos[i])) for i in others)
                if clr > bestclear:
                    bestclear, best = clr, pt
    return best, bestclear


def cage_sulfurs(atoms, vac, rcut=2.9):
    mic = mic_fn(atoms)
    sym = np.array(atoms.get_chemical_symbols()); pos = atoms.get_positions()
    S = [i for i in np.where(sym == "S")[0]
         if np.linalg.norm(mic(vac, pos[i])) <= rcut]
    return sorted(S, key=lambda i: np.linalg.norm(mic(vac, pos[i])))


def place_h_on_S(base, h_idx, s_idx, vac):
    mic = mic_fn(base)
    a = deepcopy(base); pos = a.get_positions()
    d = mic(pos[s_idx], vac); d = d / np.linalg.norm(d)
    pos[h_idx] = pos[s_idx] + 1.42 * d
    a.set_positions(pos)
    return a


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mace-omat")
    args = p.parse_args()
    print(f"=== PENT MULTI-ENDPOINT: {args.model} ===", flush=True)

    donor0 = read(DONOR, index=-1)
    factory, mtype = H.make_calc_factory(args.model, "cpu")
    print("relax donor...", flush=True)
    donor, *_ = H.relax(donor0, factory, "DONOR", fmax=0.03, max_steps=300)
    e_donor = float(donor.get_potential_energy())
    h = int(np.where(np.array(donor.get_chemical_symbols()) == "H")[0][0])
    hostD = H.h_host_check(donor, "donor")
    donor_S = sorted(np.where(np.array(donor.get_chemical_symbols()) == "S")[0],
                     key=lambda i: donor.get_distance(h, i, mic=True))[0]
    print(f"  donor: E={e_donor:.4f} H-host={hostD['host']}@{hostD['d_nn_A']:.3f} donor_S={donor_S}", flush=True)

    vac, clr = find_vacancy_center(donor, h)
    print(f"  vacancy void clearance={clr:.2f}A", flush=True)
    cage = cage_sulfurs(donor, vac)
    print(f"  cage S sites: {list(cage)}", flush=True)

    sites = []
    for s in cage:
        cfg = place_h_on_S(donor, h, s, vac)
        rel, conv, steps, e, fm = H.relax(cfg, factory, f"S{s}", fmax=0.03, max_steps=250)
        host = H.h_host_check(rel, f"S{s}")
        mic = mic_fn(rel); pos = rel.get_positions()
        dE = (e - e_donor) * 1000.0
        sites.append({
            "placed_on_S": int(s), "final_host": host["host"],
            "d_nn_A": host["d_nn_A"], "s_h_preserved": host["s_h_preserved"],
            "E_eV": e, "dE_vs_donor_meV": dE, "converged": conv, "min_dist_A": float(H.min_distance(rel)),
        })
        print(f"  placed S{s}: -> {host['host']}@{host['d_nn_A']:.3f}A  "
              f"dE={dE:+.1f}meV  s_bound={host['s_h_preserved']}", flush=True)
        rel._site = s
        write(f"D:/home/ignat/project-third-matter/tmp/pent_neb_inputs/pent_site_S{s}_{args.model}.extxyz", rel)

    s_bound = [x for x in sites if x["s_h_preserved"]]
    fe_collapsed = [x for x in sites if x["final_host"] == "Fe"]
    print(f"\n  S-bound minima: {len(s_bound)}/{len(sites)}  Fe-collapsed: {len(fe_collapsed)}", flush=True)
    print(f"  => Paper#2 'MLIP collapses H to Fe-hydride' for {args.model}: "
          f"{'CONFIRMED (some sites collapse)' if fe_collapsed else 'NOT reproduced (all S-bound)'}", flush=True)

    res = {
        "model": args.model, "model_type": mtype, "system": "pentlandite V_Fe H-landscape",
        "donor_E_eV": e_donor, "donor_S": int(donor_S), "donor_host": hostD,
        "vacancy_void_clearance_A": clr, "cage_S": [int(x) for x in cage],
        "sites": sites,
        "n_s_bound": len(s_bound), "n_fe_collapsed": len(fe_collapsed),
        "paper2_fe_collapse": ("confirmed" if fe_collapsed else "not_reproduced"),
        "cp2k_ref_meV": 385, "abacus_ref_meV": 442,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}_pentlandite_multiendpoint.json").write_text(
        json.dumps(res, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
    print(f"  saved -> {OUT / (args.model + '_pentlandite_multiendpoint.json')}", flush=True)


if __name__ == "__main__":
    main()
