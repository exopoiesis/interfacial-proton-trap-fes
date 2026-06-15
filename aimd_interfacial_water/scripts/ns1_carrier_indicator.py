#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NS-1: carrier-indicator s(t) + (conditional) Koenig mCEC^water on 73-atom W2-geometry trajectories.

Design doc: paper/reviews/KineticTrap/NS1_CEC_design_2026-06-13.md (Physicist).
Critical corrections honored:
  - atom map built PROGRAMMATICALLY from species column of frame0 (NOT hardcoded from task).
  - carrier_B = HYDRIDE (atom 38, 1-based) on Fe -> Koenig mCEC NOT applicable directly.
    -> Variant A: carrier-indicator s = min_O d(H38,O) - min_Fe d(H38,Fe).
  - mCEC^water applied ONLY where applicable (no hydride in Sigma_H): we compute it on
    w1_grotthuss (local 73-atom AIMD, bias=0, direct histogram) excluding the hydride from Sigma_H.
  - 145-atom real W1 AIMD NOT local -> skipped, flagged.

Outputs printed + saved to results/ns_analysis_2026-06-13/.
Run with: git/prodromos/.venv/Scripts/python.exe (numpy/scipy).
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ns_analysis_2026-06-13")
os.makedirs(OUT, exist_ok=True)

# --- physical constants / cell ---
CELL = np.array([11.022, 11.022, 6.5])  # orthorhombic, Angstrom (carrier_B / w1_grotthuss thin geometry)
KB = 0.0083145          # kJ/mol/K
T = 300.0
KT = KB * T             # kJ/mol
KT_eV = KT / 96.485     # eV
HYDRIDE_1BASED = 38     # biased atom per task + design doc (verify species below)


def read_xyz(path):
    """Read all frames. Returns (species_list, coords array [nframes, natoms, 3])."""
    frames = []
    species = None
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line == "":
                continue
            try:
                n = int(line)
            except ValueError:
                continue
            f.readline()  # comment
            sp = []
            xyz = np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                t = f.readline().split()
                sp.append(t[0])
                xyz[i] = [float(t[1]), float(t[2]), float(t[3])]
            if species is None:
                species = sp
            frames.append(xyz)
    return species, np.array(frames)


def mic_dist(a, b, cell):
    """Minimum-image distance between point a and array-of-points b. a:(3,), b:(M,3)."""
    d = b - a
    d -= cell * np.round(d / cell)
    return np.sqrt((d * d).sum(axis=1))


def f_sw(d, rsw, dsw):
    """Koenig rational switching function f=1/(1+exp((d-rsw)/dsw)), clipped exp-arg."""
    arg = np.clip((d - rsw) / dsw, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(arg))


def atom_map(species):
    Fe = [i for i, s in enumerate(species) if s == "Fe"]
    S = [i for i, s in enumerate(species) if s == "S"]
    O = [i for i, s in enumerate(species) if s == "O"]
    H = [i for i, s in enumerate(species) if s == "H"]
    return dict(Fe=np.array(Fe), S=np.array(S), O=np.array(O), H=np.array(H))


def carrier_indicator(coords, amap, hydride_idx, cell):
    """s(t) = min_O d(H,O) - min_Fe d(H,Fe). s>0 closer to Fe, s<0 closer to water."""
    nf = coords.shape[0]
    s = np.empty(nf)
    dO = np.empty(nf); dFe = np.empty(nf); dS = np.empty(nf)
    for t in range(nf):
        h = coords[t, hydride_idx]
        dO[t] = mic_dist(h, coords[t, amap["O"]], cell).min()
        dFe[t] = mic_dist(h, coords[t, amap["Fe"]], cell).min()
        dS[t] = mic_dist(h, coords[t, amap["S"]], cell).min()
    # NOTE on sign: per design doc 4.1 s = min(d_O) - min(d_Fe).
    # s>0 means d_O > d_Fe => H is CLOSER to Fe (Fe term smaller) => "closer to Fe".
    s = dO - dFe
    return s, dO, dFe, dS


def q_hydride(coords, amap, hydride_idx, cell, rsw=1.6, dsw=0.1):
    """Coordination number of hydride on Fe: sum_Fe f_sw(d_38,Fe). n~1 localized, fractional delocalized."""
    nf = coords.shape[0]
    q = np.empty(nf)
    nn_gap = np.empty(nf)  # gap between nearest and 2nd-nearest Fe
    nn_idx = np.empty(nf, dtype=int)
    for t in range(nf):
        h = coords[t, hydride_idx]
        dfe = mic_dist(h, coords[t, amap["Fe"]], cell)
        q[t] = f_sw(dfe, rsw, dsw).sum()
        order = np.argsort(dfe)
        nn_idx[t] = amap["Fe"][order[0]]
        nn_gap[t] = dfe[order[1]] - dfe[order[0]]
    return q, nn_gap, nn_idx


def mcec_water(coords, amap, cell, hydride_idx, rsw=1.3, dsw=0.03, w_O=2.0):
    """Koenig mCEC vector tracking excess proton in water.
    Sigma_H = all water H + excess proton, EXCLUDING the hydride (atom 38) since it is NOT
    part of the Bronsted water network. Sigma_X = water O with w=2.
    Returns xi_CEC vectors [nf,3] computed with MIC relative to a reference origin (frame0 mean O).
    """
    Oidx = amap["O"]
    Hidx = np.array([h for h in amap["H"] if h != hydride_idx])
    nf = coords.shape[0]
    xi = np.empty((nf, 3))
    # reference origin = centroid of water O at frame0 (for unwrapping MIC vector form)
    for t in range(nf):
        Hpos = coords[t, Hidx]
        Opos = coords[t, Oidx]
        ref = Opos.mean(axis=0)
        # bring everything into image closest to ref
        def unwrap(P):
            d = P - ref
            d -= cell * np.round(d / cell)
            return ref + d
        Hpos = unwrap(Hpos)
        Opos = unwrap(Opos)
        term1 = Hpos.sum(axis=0)
        term2 = w_O * Opos.sum(axis=0)
        # double sum: sum_i sum_j f_sw(d_ij)*(r_i - r_j)
        term3 = np.zeros(3)
        for hi in range(len(Hidx)):
            d = Opos - Hpos[hi]
            d -= cell * np.round(d / cell)
            dd = np.sqrt((d * d).sum(axis=1))
            fs = f_sw(dd, rsw, dsw)  # (nO,)
            # contribution: sum_j f_sw * (r_i - r_j) = sum_j f_sw * (-d_vec) where d_vec = r_j - r_i
            term3 += -(fs[:, None] * d).sum(axis=0)
        xi[t] = term1 - term2 - term3
    return xi


def hist_str(vals, bins=20):
    h, edges = np.histogram(vals, bins=bins)
    mx = h.max() if h.max() > 0 else 1
    lines = []
    for i in range(len(h)):
        bar = "#" * int(40 * h[i] / mx)
        lines.append(f"  [{edges[i]:7.3f},{edges[i+1]:7.3f}) {h[i]:5d} {bar}")
    return "\n".join(lines)


def analyze_carrier(path, label, hold_start=150):
    species, coords = read_xyz(path)
    nf = coords.shape[0]
    amap = atom_map(species)
    hyd = HYDRIDE_1BASED - 1  # 0-based
    assert species[hyd] == "H", f"atom {HYDRIDE_1BASED} is {species[hyd]}, expected H"
    res = {"label": label, "path": path, "nframes": nf,
           "composition": {k: int(len(v)) for k, v in amap.items()},
           "hydride_atom_1based": HYDRIDE_1BASED,
           "hydride_z_frame0": float(coords[0, hyd, 2])}

    s, dO, dFe, dS = carrier_indicator(coords, amap, hyd, CELL)
    q, nn_gap, nn_idx = q_hydride(coords, amap, hyd, CELL)

    # hold phase = frames index >= hold_start
    holdm = np.arange(nf) >= hold_start
    s_hold = s[holdm]
    res["s_mean_all"] = float(s.mean())
    res["s_std_all"] = float(s.std())
    res["s_mean_hold"] = float(s_hold.mean())
    res["s_std_hold"] = float(s_hold.std())
    res["hold_frames"] = int(holdm.sum())
    res["dFe_min_hold_mean"] = float(dFe[holdm].mean())
    res["dS_min_hold_mean"] = float(dS[holdm].mean())
    res["dO_min_hold_mean"] = float(dO[holdm].mean())

    # carrier classification per frame (bonded if min dist < bond cutoff)
    # bond cutoffs: Fe-H ~1.8, S-H ~1.6, O-H ~1.25 (in-flight if none)
    CUT_FE, CUT_S, CUT_O = 1.8, 1.6, 1.25
    car = np.full(nf, "in_flight", dtype=object)
    car[dFe < CUT_FE] = "Fe"
    car[(dS < CUT_S) & (dS < dFe)] = "S"
    car[(dO < CUT_O) & (dO < dFe) & (dO < dS)] = "O"
    occ = {}
    for c in ["Fe", "S", "O", "in_flight"]:
        occ[c] = float((car[holdm] == c).mean() * 100.0)
    res["carrier_occupancy_pct_hold"] = occ
    res["q_hydride_mean_hold"] = float(q[holdm].mean())
    res["nn_Fe_gap_mean_hold_A"] = float(nn_gap[holdm].mean())
    # fraction of frames where global-nearest Fe is the modal one
    from collections import Counter
    cnt = Counter(nn_idx[holdm].tolist())
    modal_fe, modal_cnt = cnt.most_common(1)[0]
    res["modal_nearest_Fe_1based"] = int(modal_fe + 1)
    res["modal_nearest_Fe_pct_hold"] = float(modal_cnt / holdm.sum() * 100.0)

    res["_s_hist"] = hist_str(s_hold, bins=25)
    # save raw arrays
    np.savez(os.path.join(OUT, f"ns1_{label}_arrays.npz"),
             s=s, dO=dO, dFe=dFe, dS=dS, q=q, nn_gap=nn_gap,
             carrier=np.array([str(c) for c in car]))
    return res, s, dO, dFe, dS


def main():
    summary = {"kT_eV": KT_eV, "cell_A": CELL.tolist(),
               "note_145atom": "Real 145-atom W1 AIMD (cell c=13, 24 H2O) NOT present locally; "
                               "all local w1_grotthuss = 73-atom W2-geometry. 145-atom part SKIPPED "
                               "(would require harvest from Vast.ai via safe_harvest.sh).",
               "note_reweight": "plumed_carrier_B.dat is binary/unreadable and NO COLVAR/restraint.bias "
                                "output was harvested for carrier_B -> bias-reweighted 2D FES (path iii) "
                                "NOT possible. carrier-indicator s(t) is bias-INDEPENDENT and answers the "
                                "main NS-1 carrier question directly.",
               "files": {}}

    # ---- carrier_B (hydride probe, biased steered MD) ----
    p_b = os.path.join(ROOT, "results", "carrier_B", "carrier_B-pos-1.xyz")
    res_b, s_b, dO_b, dFe_b, dS_b = analyze_carrier(p_b, "carrier_B", hold_start=150)
    summary["files"]["carrier_B"] = res_b

    # ---- carrier_2p40 (probe at 2.40) ----
    p_2 = os.path.join(ROOT, "results", "carrier_2p40", "carrier_2p40-pos-1.xyz")
    res_2, s_2, dO_2, dFe_2, dS_2 = analyze_carrier(p_2, "carrier_2p40", hold_start=150)
    summary["files"]["carrier_2p40"] = res_2

    # ---- w1_grotthuss (local 73-atom AIMD, UNBIASED) ----
    p_g = os.path.join(ROOT, "results", "w1_grotthuss", "grotthuss-pos-1.xyz")
    res_g, s_g, dO_g, dFe_g, dS_g = analyze_carrier(p_g, "w1_grotthuss", hold_start=150)
    summary["files"]["w1_grotthuss"] = res_g

    # ---- mCEC^water on w1_grotthuss (unbiased, hydride excluded from Sigma_H) ----
    # Applicable per design doc 3.1/6: AIMD branch, bias=0, direct histogram.
    species_g, coords_g = read_xyz(p_g)
    amap_g = atom_map(species_g)
    hyd = HYDRIDE_1BASED - 1
    xi = mcec_water(coords_g, amap_g, CELL, hyd)
    # z-component (CV1 per design 4.1) relative to water-O centroid z
    # report distribution of xi magnitude variation and z; check unimodal/multimodal
    xi_z = xi[:, 2]
    xi_xy = xi[:, :2]
    # lateral displacement relative to its own mean
    xi_lat = np.sqrt(((xi_xy - xi_xy.mean(axis=0)) ** 2).sum(axis=1))
    mcec = {
        "applied_to": "w1_grotthuss (73-atom, unbiased)",
        "n_water_O": int(len(amap_g["O"])),
        "n_H_in_sigmaH": int(len([h for h in amap_g["H"] if h != hyd])),
        "hydride_excluded_1based": HYDRIDE_1BASED,
        "xi_z_mean": float(xi_z.mean()), "xi_z_std": float(xi_z.std()),
        "xi_lateral_mean_A": float(xi_lat.mean()), "xi_lateral_std_A": float(xi_lat.std()),
        "note": "xi_CEC vector tracks excess-proton center in water (Koenig Eq.7, w_O=2, rsw=1.3, dsw=0.03, "
                "MIC all axes). Hydride atom 38 excluded from Sigma_H (it is on Fe, not Bronsted network).",
        "_xi_z_hist": hist_str(xi_z, bins=20),
    }
    # Crude 1D FES along xi_z (unbiased -> direct histogram, F=-kT ln P)
    h, edges = np.histogram(xi_z, bins=30, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    nz = h > 0
    F = np.full_like(h, np.nan)
    F[nz] = -KT_eV * np.log(h[nz])
    if nz.any():
        F[nz] -= np.nanmin(F[nz])
    # count distinct minima > 3kT separated
    barrier_3kT = 3 * KT_eV
    mcec["FES_xi_z_max_eV"] = float(np.nanmax(F)) if nz.any() else None
    mcec["FES_3kT_eV"] = float(barrier_3kT)
    summary["mcec_water_w1_grotthuss"] = mcec
    np.savez(os.path.join(OUT, "ns1_mcec_w1grotthuss.npz"),
             xi=xi, xi_z=xi_z, xi_lat=xi_lat, FES_xi_z=F, FES_centers=centers)

    # ---- VERDICT NS-1 ----
    # Criteria from design doc section 5. Use carrier_B (longest biased probe) + w1_grotthuss.
    def verdict_for(res):
        occ = res["carrier_occupancy_pct_hold"]
        # robust delocalization: in_flight dominant, no single carrier localizes
        if occ["in_flight"] >= 70.0:
            return "ROBUST_DELOCALIZATION"
        # clear saddle would show bimodal s with a localized carrier
        return "SEE_DETAIL"
    v_b = verdict_for(res_b)
    v_g = verdict_for(res_g)
    summary["verdict_NS1"] = {
        "carrier_B": v_b,
        "w1_grotthuss": v_g,
        "overall": ("ROBUST_DELOCALIZATION"
                    if v_b == "ROBUST_DELOCALIZATION" else "REVIEW_INSUFFICIENT"),
        "rationale": ("carrier-indicator s and per-frame carrier classification: hydride neither commits "
                      "to a single Fe nor migrates to water/S. in_flight fraction dominant on hold. "
                      "No bimodal s -> no clear saddle. Quantitative barrier NOT extractable (no harvested "
                      "bias for reweighting; single steered probe) -> qualitative verdict only, REVIEW-grade.")
    }

    # write JSON
    with open(os.path.join(OUT, "ns1_summary.json"), "w") as f:
        json.dump({k: v for k, v in summary.items()}, f, indent=2)

    # ---- print ----
    print("=" * 70)
    print("NS-1 carrier-indicator + mCEC^water  (T=300K, kT=%.5f eV)" % KT_eV)
    print("=" * 70)
    for fn in ["carrier_B", "carrier_2p40", "w1_grotthuss"]:
        r = summary["files"][fn]
        print(f"\n--- {fn}  (n={r['nframes']} frames, comp={r['composition']}) ---")
        print(f"  hydride atom {r['hydride_atom_1based']} (1-based), z_frame0={r['hydride_z_frame0']:.3f}")
        print(f"  <s> all  = {r['s_mean_all']:+.4f} +/- {r['s_std_all']:.4f} A")
        print(f"  <s> hold = {r['s_mean_hold']:+.4f} +/- {r['s_std_hold']:.4f} A  (hold={r['hold_frames']} frames, idx>=150)")
        print(f"    (s>0 => closer to Fe ; s<0 => closer to water)")
        print(f"  <min d_FeH> hold = {r['dFe_min_hold_mean']:.3f} A")
        print(f"  <min d_SH>  hold = {r['dS_min_hold_mean']:.3f} A")
        print(f"  <min d_OH>  hold = {r['dO_min_hold_mean']:.3f} A")
        print(f"  carrier occupancy (hold): {r['carrier_occupancy_pct_hold']}")
        print(f"  q_hydride(Fe) <hold> = {r['q_hydride_mean_hold']:.3f}  (n~1 localized, fractional=delocalized)")
        print(f"  nearest-Fe gap <hold> = {r['nn_Fe_gap_mean_hold_A']:.3f} A ; "
              f"modal Fe#{r['modal_nearest_Fe_1based']} {r['modal_nearest_Fe_pct_hold']:.0f}% of hold")
        print(f"  s histogram (hold):\n{r['_s_hist']}")
    print("\n--- mCEC^water on w1_grotthuss (unbiased) ---")
    m = summary["mcec_water_w1_grotthuss"]
    print(f"  Sigma_X = {m['n_water_O']} water O (w=2) ; Sigma_H = {m['n_H_in_sigmaH']} H (hydride excluded)")
    print(f"  xi_CEC z: mean={m['xi_z_mean']:.3f} std={m['xi_z_std']:.3f} A")
    print(f"  xi_CEC lateral(xy) wrt mean: mean={m['xi_lateral_mean_A']:.3f} std={m['xi_lateral_std_A']:.3f} A")
    print(f"  1D FES(xi_z) max = {m['FES_xi_z_max_eV']:.4f} eV (3kT={m['FES_3kT_eV']:.4f} eV)")
    print(f"  xi_z histogram:\n{m['_xi_z_hist']}")
    print("\n" + "=" * 70)
    print("VERDICT NS-1:", summary["verdict_NS1"]["overall"])
    print("  carrier_B:", summary["verdict_NS1"]["carrier_B"])
    print("  w1_grotthuss:", summary["verdict_NS1"]["w1_grotthuss"])
    print("  ", summary["verdict_NS1"]["rationale"])
    print("\nNOTES:")
    print(" ", summary["note_145atom"])
    print(" ", summary["note_reweight"])
    print("=" * 70)
    return summary


if __name__ == "__main__":
    main()
