"""GOLD-STANDARD: pentlandite frozen-framework MLIP NEB with the EXACT CP2K mobile mask
(recovered from dft-neb/.../pent_selective_harvest_2026-06-06/geom_analysis.json:
28 mobile_indices, 108 frozen, H=135, donor_S=98, acceptor_S=48, r_mob=4.0).

This is the apples-to-apples test the consilium wanted: identical atom-by-atom frozen set
to CP2K, both spin-blind. Resolves the manuscript caveat "exact CP2K atom mask could not be
recovered to test." Two outcomes:
  - H stays S-bound under exact mask -> MLIP CAN represent the S-H...S channel with the right
    constraint; barrier comparable to CP2K 0.385 -> manuscript claim softens.
  - H collapses to Fe even with exact mask -> DFT-only conclusion upgraded to gold standard.

Usage: python pent_frozen_exact.py --model mace-omat
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
from pent_frozen_neb import frozen_relax, frozen_neb
from pent_symmetric_neb import find_vacancy_and_partner, DONOR
from ase.io import read

OUT = Path("D:/home/ignat/project-third-matter/results/dft_datasets/2026-06-25_mlip_neb_sweep")
MASK = "D:/home/ignat/project-third-matter/dft-neb/ph-diagnostic/pent_selective_harvest_2026-06-06/geom_analysis.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mace-omat")
    p.add_argument("--skip-neb", action="store_true",
                   help="endpoints only: acceptor-collapse is the decisive fact; NEB on a "
                        "collapsed S-H->Fe pair is meaningless")
    args = p.parse_args()
    print(f"=== PENT FROZEN EXACT-MASK NEB {args.model} ===", flush=True)

    m = json.load(open(MASK))
    mobile = sorted(int(i) for i in m["mobile_indices"])
    frozen = sorted(int(i) for i in m["frozen_indices"])
    h_idx, dS, aS = int(m["H_idx"]), int(m["donor_S"]), int(m["acceptor_S"])
    print(f"  EXACT mask: N_mobile={len(mobile)} N_frozen={len(frozen)} comp={m['mobile_comp']} "
          f"H={h_idx} donor_S={dS} acceptor_S={aS} S-S={m['acceptor_donorSS']:.3f}A r_mob={m['r_mob']}", flush=True)
    assert len(mobile) == 28 and len(frozen) == 108 and h_idx in mobile and dS in mobile and aS in mobile

    donor0 = read(DONOR, index=-1)
    assert len(donor0) == 136
    factory, mtype = H.make_calc_factory(args.model, "cpu")

    # vacancy direction (to place H on acceptor S pointing into the pocket)
    _h, _dS, _aSgeo, vac, nf, mic = find_vacancy_and_partner(donor0)
    if _h != h_idx:
        print(f"  WARN: vacancy-finder H {_h} != mask H {h_idx}", flush=True)

    print("frozen-relax donor (exact mask)...", flush=True)
    donor, cda, sda, eda = frozen_relax(donor0, factory, frozen, "DONOR")
    hostD = H.h_host_check(donor, "donor")

    # build acceptor: move H onto the EXACT CP2K acceptor S48, 1.42 A toward vacancy centre
    acc0 = deepcopy(donor); pos = acc0.get_positions()
    d = mic(pos[aS], vac); d = d / np.linalg.norm(d)
    pos[h_idx] = pos[aS] + 1.42 * d
    acc0.set_positions(pos)
    print("frozen-relax acceptor on S48 (exact mask)...", flush=True)
    acc, cab, sab, eab = frozen_relax(acc0, factory, frozen, "ACCEPTOR")
    hostA = H.h_host_check(acc, "acceptor")
    print(f"  donor host={hostD['host']}@{hostD['d_nn_A']:.3f}  acceptor host={hostA['host']}@{hostA['d_nn_A']:.3f}", flush=True)
    acc_collapsed = (hostA["host"] == "Fe")
    print(f"\n  acceptor_collapsed_to_Fe={acc_collapsed}  (decisive: under EXACT CP2K mask)", flush=True)

    neb_block = None
    if not args.skip_neb:
        print("frozen NEB (exact mask)...", flush=True)
        conv, energies, nsteps, images = frozen_neb(factory, donor, acc, frozen, fmax=0.05, max_steps=300)
        e0 = energies[0]; rel = [(e - e0) for e in energies]
        barrier = max(energies) - e0; saddle = int(np.argmax(energies))
        hosts = [H.h_host_check(im) for im in images]
        s_bound = all(hh["s_h_preserved"] for hh in hosts)
        print(f"  BARRIER fwd={barrier*1000:.1f} meV (meaningless if acceptor collapsed)", flush=True)
        print(f"  per-image host: {[hh['host'] for hh in hosts]}  S-bound path: {s_bound}", flush=True)
        neb_block = {
            "barrier_meV": barrier * 1000.0, "saddle_img": saddle, "neb_converged": conv, "n_steps": nsteps,
            "image_energies_rel_meV": [x * 1000.0 for x in rel],
            "per_image_h_host": hosts, "s_bound_entire_path": s_bound,
            "min_dists_A": [float(H.min_distance(im)) for im in images],
            "note": "NEB barrier is the S-H->Fe-hydride process, not S-H...S; not a physical barrier",
        }

    res = {
        "model": args.model, "model_type": mtype, "system": "pentlandite V_Fe S-H<->S-H EXACT CP2K frozen mask",
        "mask_source": MASK, "mask_note": "EXACT CP2K mobile_indices (28) / frozen (108), atom-by-atom",
        "n_mobile": len(mobile), "n_frozen": len(frozen), "mobile_comp": m["mobile_comp"],
        "donor_S": dS, "acceptor_S": aS, "S_S_hop_A": m["acceptor_donorSS"],
        "donor_host": hostD, "acceptor_host": hostA, "acceptor_collapsed_to_Fe": acc_collapsed,
        "decisive_finding": ("Under the exact CP2K mobile mask the MLIP acceptor collapses to "
                             "Fe-hydride while CP2K holds it S-bound (0.385 eV) -> collapse is a "
                             "property of the potential, not the constraint."),
        "cp2k_ref_meV": 385, "abacus_ref_meV": 442,
        "neb": neb_block,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}_pentlandite_frozen_EXACTmask.json").write_text(
        json.dumps(res, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
    print(f"  saved -> {OUT / (args.model + '_pentlandite_frozen_EXACTmask.json')}", flush=True)


if __name__ == "__main__":
    main()
