"""Pentlandite frozen-framework S-H<->S-H NEB with MLIP -- apples-to-apples with the
CP2K frozen-framework reference (0.385 eV; 28 mobile = H + 1st coordination sphere,
108 frozen). The exact CP2K mobile index-set is not recoverable (it lived in a tmp/
JSON, now gone), so we RECONSTRUCT the documented rule: mobile = H + all atoms within
radius r of the vacancy centre, with r tuned so N_mobile ~= 28 (and donor-S, acceptor-S,
H guaranteed mobile). Honest caveat: reproduces CP2K's STATED mobile set (size + rule),
not necessarily the identical indices.

Both CP2K (RKS nspin=1) and MACE (foundation, spin-blind) are spin-free -> matched.
Freezing the framework prevents Fe from relaxing inward to capture H, so the S-H...S
metastable channel is preserved for BOTH methods identically. Per-image h_host verifies
the proton stays S-bound even under the frozen constraint.

Usage: python pent_frozen_neb.py --model mace-omat [--r 3.7] [--target-mobile 28]
"""
import argparse, json, sys, time, warnings
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
from pent_symmetric_neb import find_vacancy_and_partner, build_acceptor, DONOR
from ase.io import read
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE
from ase.mep.neb import NEB

OUT = Path("D:/home/ignat/project-third-matter/results/dft_datasets/2026-06-25_mlip_neb_sweep")


def select_mobile(atoms, vac, mic, h, dS, aS, r, target=28):
    """Mobile = atoms within r of vacancy + guaranteed {H, donor-S, acceptor-S}. Tune r to ~target."""
    pos = atoms.get_positions()
    best_r, best_mob = r, None
    for rr in [r] + list(np.arange(2.8, 5.2, 0.1)):
        mob = {h, dS, aS}
        for i in range(len(atoms)):
            if np.linalg.norm(mic(pos[i], vac)) <= rr:
                mob.add(i)
        if best_mob is None:
            best_mob, best_r = set(mob), rr
        if abs(len(mob) - target) < abs(len(best_mob) - target):
            best_mob, best_r = set(mob), rr
    return sorted(best_mob), best_r


def frozen_relax(atoms_in, calc_factory, frozen, label, fmax=0.03, steps=300):
    a = deepcopy(atoms_in)
    a.set_constraint(FixAtoms(indices=frozen))
    c, _ = calc_factory(); a.calc = c
    sc = [0]; t0 = time.time()
    opt = BFGS(a, logfile=None, maxstep=0.1)
    def cnt():
        sc[0] += 1
        if sc[0] % 25 == 0:
            print(f"    [{label}] step {sc[0]} fmax={float(np.max(np.abs(a.get_forces()))):.4f} t={time.time()-t0:.0f}s", flush=True)
    opt.attach(cnt)
    try: conv = bool(opt.run(fmax=fmax, steps=steps))
    except Exception as e: print(f"    [{label}] exc {e}", flush=True); conv = False
    return a, conv, sc[0], float(a.get_potential_energy())


def frozen_neb(calc_factory, ep_a, ep_b, frozen, n_int=5, fmax=0.05, max_steps=300):
    images = [deepcopy(ep_a)]
    for _ in range(n_int):
        im = deepcopy(ep_a); c, _ = calc_factory(); im.calc = c; images.append(im)
    images.append(deepcopy(ep_b))
    ca, _ = calc_factory(); cb, _ = calc_factory()
    images[0].calc = ca; images[-1].calc = cb
    for im in images:
        im.set_constraint(FixAtoms(indices=frozen))
    neb = NEB(images, climb=False, k=0.1)
    neb.interpolate(method="idpp", mic=True)
    sc = [0]; t0 = time.time()
    def cnt():
        sc[0] += 1
        if sc[0] % 10 == 0:
            try: fm = float(np.max(np.abs(neb.get_forces())))
            except Exception: fm = float("nan")
            print(f"    [fNEB] step {sc[0]} fmax={fm:.4f} t={time.time()-t0:.0f}s", flush=True)
    half = max(max_steps // 2, 50)
    try:
        o1 = FIRE(neb, logfile=None, maxstep=0.1, dt=0.05); o1.attach(cnt)
        o1.run(fmax=max(fmax, 0.10), steps=half)
    except Exception as e: print("  plain exc", e, flush=True)
    conv = False
    try:
        neb.climb = True
        o2 = FIRE(neb, logfile=None, maxstep=0.1, dt=0.05); o2.attach(cnt)
        conv = bool(o2.run(fmax=fmax, steps=max_steps - sc[0]))
    except Exception as e: print("  climb exc", e, flush=True)
    energies = [float(im.get_potential_energy()) for im in images]
    return conv, energies, sc[0], images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mace-omat")
    p.add_argument("--r", type=float, default=3.7)
    p.add_argument("--target-mobile", type=int, default=28)
    args = p.parse_args()
    print(f"=== PENT FROZEN-FRAMEWORK NEB {args.model} ===", flush=True)

    donor0 = read(DONOR, index=-1)
    factory, mtype = H.make_calc_factory(args.model, "cpu")

    # locate vacancy + hop S's from the (unrelaxed-frozen-target) donor geometry
    h, dS, aS, vac, nf, mic = find_vacancy_and_partner(donor0)
    mobile, r_used = select_mobile(donor0, vac, mic, h, dS, aS, args.r, args.target_mobile)
    frozen = sorted(set(range(len(donor0))) - set(mobile))
    dpos = donor0.get_positions()
    print(f"  H={h} donor_S={dS} acceptor_S={aS} S-S={np.linalg.norm(mic(dpos[dS],dpos[aS])):.2f}A "
          f"vac-nearestFe={nf:.2f}A", flush=True)
    print(f"  r_used={r_used:.2f}A  N_mobile={len(mobile)} (target {args.target_mobile})  N_frozen={len(frozen)}", flush=True)

    print("frozen-relax donor...", flush=True)
    donor, cda, sda, eda = frozen_relax(donor0, factory, frozen, "DONOR")
    hostD = H.h_host_check(donor, "donor")
    acc0 = build_acceptor(donor, h, dS, aS, vac, mic)
    print("frozen-relax acceptor (H on partner S, framework fixed)...", flush=True)
    acc, cab, sab, eab = frozen_relax(acc0, factory, frozen, "ACCEPTOR")
    hostA = H.h_host_check(acc, "acceptor")
    print(f"  donor host={hostD['host']}@{hostD['d_nn_A']:.3f}  acceptor host={hostA['host']}@{hostA['d_nn_A']:.3f}", flush=True)

    acc_collapsed = (hostA["host"] == "Fe")
    if acc_collapsed:
        print("  NOTE: acceptor collapsed to Fe EVEN under frozen framework -> "
              "the S-H...S channel is not preserved by this mobile set; barrier will be S-H->Fe.", flush=True)

    print("frozen NEB...", flush=True)
    conv, energies, nsteps, images = frozen_neb(factory, donor, acc, frozen, fmax=0.05, max_steps=300)
    e0 = energies[0]; rel = [(e - e0) for e in energies]
    barrier = max(energies) - e0; saddle = int(np.argmax(energies))
    hosts = [H.h_host_check(im) for im in images]
    s_bound = all(hh["s_h_preserved"] for hh in hosts)
    print(f"\n  BARRIER fwd={barrier*1000:.1f} meV  (CP2K frozen-framework ref 385, ABACUS 442)", flush=True)
    print(f"  MEP rel meV: {[round(x*1000,1) for x in rel]}  saddle={saddle} conv={conv}", flush=True)
    print(f"  per-image host: {[hh['host'] for hh in hosts]}  S-bound path: {s_bound}", flush=True)

    res = {
        "model": args.model, "model_type": mtype, "system": "pentlandite V_Fe S-H<->S-H frozen-framework",
        "mask_note": "RECONSTRUCTED CP2K rule (H+1st shell, ~28 mobile), NOT identical indices",
        "r_used_A": r_used, "n_mobile": len(mobile), "n_frozen": len(frozen),
        "mobile_indices": mobile, "donor_S": int(dS), "acceptor_S": int(aS),
        "S_S_hop_A": float(np.linalg.norm(mic(dpos[dS], dpos[aS]))),
        "donor_host": hostD, "acceptor_host": hostA, "acceptor_collapsed_to_Fe": acc_collapsed,
        "barrier_meV": barrier * 1000.0, "saddle_img": saddle, "neb_converged": conv, "n_steps": nsteps,
        "image_energies_rel_meV": [x * 1000.0 for x in rel],
        "per_image_h_host": hosts, "s_bound_entire_path": s_bound,
        "cp2k_ref_meV": 385, "abacus_ref_meV": 442,
        "min_dists_A": [float(H.min_distance(im)) for im in images],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}_pentlandite_frozen.json").write_text(
        json.dumps(res, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
    print(f"  saved -> {OUT / (args.model + '_pentlandite_frozen.json')}", flush=True)


if __name__ == "__main__":
    main()
