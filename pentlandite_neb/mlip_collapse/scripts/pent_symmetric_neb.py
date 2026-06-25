"""Pentlandite SYMMETRIC S-H<->S-H NEB with foundation MLIPs (full relaxation, all 136
atoms), to compare against the CP2K frozen-framework result (0.385 eV) / ABACUS (0.44).

Donor = CP2K NEB donor endpoint (endA.traj, HFe71S64). Acceptor CONSTRUCTED by the
documented S<->S hop across the Fe vacancy: reflect the donor S through the vacancy
centre to find the partner S, place H on it, relax. Both endpoints relaxed with the
SAME model -> consistent atom ordering -> valid NEB.

Per-image h_host check along the final band detects whether the proton stays S-bound
(S-H...S transfer, validates CP2K) or collapses into the deeper Fe-hydride well
(Paper #2 claim for MACE-MP-0/CHGNet) at any point along the PATH.

Usage: python pent_symmetric_neb.py --model mace-omat
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
from ase.io import read, write
from ase.optimize import FIRE

DONOR = "D:/home/ignat/project-third-matter/dft-neb/ph-diagnostic/pent_selective_harvest_2026-06-06/pent_endA/endA.traj"
OUT = Path("D:/home/ignat/project-third-matter/results/dft_datasets/2026-06-25_mlip_neb_sweep")
CP2K_REF_meV = 385
ABACUS_REF_meV = 442


def find_vacancy_and_partner(atoms):
    """Return (h_idx, donor_S_idx, acceptor_S_idx, vacancy_center, nearest_Fe, mic)."""
    sym = np.array(atoms.get_chemical_symbols())
    pos = atoms.get_positions()
    cell = atoms.get_cell(); inv = np.linalg.inv(cell.array)

    def mic(p, q):
        d = p - q; f = d @ inv; f -= np.round(f); return f @ cell.array

    h = int(np.where(sym == "H")[0][0])
    S_idx = np.where(sym == "S")[0]
    Fe_idx = np.where(sym == "Fe")[0]
    dS = sorted(S_idx, key=lambda i: np.linalg.norm(mic(pos[h], pos[i])))[0]
    near_dS = sorted(S_idx, key=lambda i: np.linalg.norm(mic(pos[dS], pos[i])))[:4]
    vac = pos[dS] + np.mean([mic(pos[dS], pos[i]) for i in near_dS], axis=0)
    nf = min(np.linalg.norm(mic(vac, pos[i])) for i in Fe_idx)
    refl = vac + (vac - pos[dS])
    aS = sorted([i for i in S_idx if i != dS],
                key=lambda i: np.linalg.norm(mic(refl, pos[i])))[0]
    return h, int(dS), int(aS), vac, float(nf), mic


def build_acceptor(donor, h, dS, aS, vac, mic):
    acc = deepcopy(donor)
    pos = acc.get_positions()
    direction = mic(pos[aS], vac)
    direction = direction / np.linalg.norm(direction)
    pos[h] = pos[aS] + 1.42 * direction
    acc.set_positions(pos)
    return acc


def per_image_h_host(images):
    return [H.h_host_check(im) for im in images]


def neb_keep_images(calc_factory, ep_a, ep_b, n_interior=5, fmax=0.05, max_steps=300):
    from ase.mep.neb import NEB
    images = [deepcopy(ep_a)]
    for _ in range(n_interior):
        img = deepcopy(ep_a); c, _ = calc_factory(); img.calc = c; images.append(img)
    images.append(deepcopy(ep_b))
    ca, _ = calc_factory(); cb, _ = calc_factory()
    images[0].calc = ca; images[-1].calc = cb
    neb = NEB(images, climb=False, k=0.1)
    neb.interpolate(method="idpp", mic=True)
    sc = [0]; t0 = time.time()

    def cnt():
        sc[0] += 1
        if sc[0] % 10 == 0:
            try: fm = float(np.max(np.abs(neb.get_forces())))
            except Exception: fm = float("nan")
            print(f"    [NEB] step {sc[0]} fmax={fm:.4f} t={time.time()-t0:.0f}s", flush=True)

    half = max(max_steps // 2, 50)
    try:
        o1 = FIRE(neb, logfile=None, maxstep=0.1, dt=0.05); o1.attach(cnt)
        o1.run(fmax=max(fmax, 0.10), steps=half)
    except Exception as e:
        print("   plain exc", e, flush=True)
    conv = False
    try:
        neb.climb = True
        o2 = FIRE(neb, logfile=None, maxstep=0.1, dt=0.05); o2.attach(cnt)
        conv = bool(o2.run(fmax=fmax, steps=max_steps - sc[0]))
    except Exception as e:
        print("   climb exc", e, flush=True)
    energies = [float(im.get_potential_energy()) for im in images]
    return conv, energies, sc[0], images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mace-omat")
    args = p.parse_args()
    print(f"=== PENT SYMMETRIC NEB: {args.model} ===", flush=True)

    donor0 = read(DONOR, index=-1)
    print(f"donor loaded: {donor0.get_chemical_formula()} N={len(donor0)}", flush=True)
    factory, mtype = H.make_calc_factory(args.model, "cpu")

    print("relax donor (full, all atoms)...", flush=True)
    donor, cda, sda, eda, fda = H.relax(donor0, factory, "DONOR", fmax=0.03, max_steps=300)
    hostD = H.h_host_check(donor, "donor")
    if not hostD["s_h_preserved"]:
        print(f"  WARNING: donor H not on S after relax: {hostD}", flush=True)

    h, dS, aS, vac, nf, mic = find_vacancy_and_partner(donor)
    dpos = donor.get_positions()
    dS_aS = float(np.linalg.norm(mic(dpos[dS], dpos[aS])))
    print(f"  H={h} donor_S={dS} acceptor_S={aS}  S-S hop={dS_aS:.2f}A  "
          f"vacancy nearest-Fe={nf:.2f}A (want >2.3)", flush=True)
    acc0 = build_acceptor(donor, h, dS, aS, vac, mic)
    print("relax acceptor (full)...", flush=True)
    acc, cab, sab, eab, fab = H.relax(acc0, factory, "ACCEPTOR", fmax=0.03, max_steps=300)
    hostA = H.h_host_check(acc, "acceptor")
    rmsd = H.endpoint_rmsd(donor, acc)
    print(f"  donor<->acceptor rmsd={rmsd:.3f}A  acceptor H host={hostA}", flush=True)

    # ABORT-GATE (chemist consilium 2026-06-25): if the acceptor collapsed to Fe-hydride,
    # the S-H<->S-H minimum-to-minimum path does NOT exist on this PES. A free-relax NEB
    # would silently report a plausible-looking S-H -> Fe-hydride barrier (wrong channel).
    # Do not produce that number; report the collapse instead.
    if not hostA.get("s_h_preserved"):
        print("  ABORT: acceptor collapsed to Fe-hydride -> S-H...S endpoint does not exist "
              "on this PES. Free-relax symmetric NEB is ill-posed. Use frozen-framework "
              "(exact CP2K mobile mask) or report the metastability finding instead.", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.model}_pentlandite_symmetric.json").write_text(json.dumps({
            "model": args.model, "status": "ABORTED_acceptor_collapsed_to_Fe",
            "donor_h_host": hostD, "acceptor_h_host": hostA,
            "note": "no S-H acceptor minimum; free-relax NEB ill-posed; see multiendpoint json",
        }, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
        return

    print("two-phase NEB...", flush=True)
    conv, energies, nsteps, images = neb_keep_images(factory, donor, acc, fmax=0.05, max_steps=300)
    e0 = energies[0]
    rel = [(e - e0) for e in energies]
    barrier = max(energies) - e0
    saddle = int(np.argmax(energies))
    hosts = per_image_h_host(images)
    s_bound_path = all(hh["s_h_preserved"] for hh in hosts)
    mds = [float(H.min_distance(im)) for im in images]
    collapsed = [i for i, hh in enumerate(hosts) if hh["host"] == "Fe"]

    print(f"\n  BARRIER = {barrier*1000:.1f} meV (CP2K {CP2K_REF_meV}, ABACUS {ABACUS_REF_meV})", flush=True)
    print(f"  MEP rel meV: {[round(x*1000,1) for x in rel]}", flush=True)
    print(f"  saddle img={saddle} conv={conv} nsteps={nsteps}", flush=True)
    print(f"  per-image H host: {[hh['host'] for hh in hosts]}", flush=True)
    print(f"  S-bound ENTIRE path: {s_bound_path}   collapsed-to-Fe images: {collapsed}", flush=True)
    print(f"  min-dists: {[round(x,3) for x in mds]}", flush=True)

    res = {
        "model": args.model, "model_type": mtype, "system": "pentlandite V_Fe S-H<->S-H",
        "donor_src": "CP2K NEB donor endA.traj (full MLIP re-relax)",
        "acceptor_src": "constructed (S<->S hop across vacancy), MLIP-relaxed",
        "cp2k_ref_meV": CP2K_REF_meV, "abacus_ref_meV": ABACUS_REF_meV,
        "S_S_hop_A": dS_aS, "vacancy_nearest_Fe_A": nf,
        "donor_h_host": hostD, "acceptor_h_host": hostA,
        "donor_acceptor_rmsd_A": rmsd,
        "barrier_meV": barrier * 1000.0, "saddle_img": saddle, "neb_converged": conv,
        "n_steps": nsteps,
        "image_energies_rel_meV": [x * 1000.0 for x in rel],
        "per_image_h_host": hosts,
        "s_bound_entire_path": s_bound_path,
        "collapsed_to_Fe_images": collapsed,
        "min_dists_A": mds,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.model}_pentlandite_symmetric.json").write_text(
        json.dumps(res, indent=2, cls=SafeJSONEncoder), encoding="utf-8")
    ind = Path("D:/home/ignat/project-third-matter/tmp/pent_neb_inputs"); ind.mkdir(exist_ok=True)
    write(str(ind / f"pent_donor_{args.model}.extxyz"), donor)
    write(str(ind / f"pent_acceptor_{args.model}.extxyz"), acc)
    print(f"  saved -> {OUT / (args.model + '_pentlandite_symmetric.json')}", flush=True)


if __name__ == "__main__":
    main()
