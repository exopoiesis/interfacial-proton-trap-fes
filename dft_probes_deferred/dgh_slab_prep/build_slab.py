#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build structures for Delta-G_H* (hydrogen adsorption) on the mackinawite (001) surface.
Pure ASE mechanics. No DFT.

Mackinawite: tetragonal P4/nmm (#129), FeS, layered (S-Fe-S sandwiches stacked along c
with a van der Waals gap between sandwiches).

Bulk source: relaxed Paper #1 / MLIPvsDFT structure
  results/dft_datasets/2026-05-02/mack_smoke_W3/relaxed_pristine.xyz
  (72-atom 3x3x2 supercell; conventional a=b=3.674 A, c=5.033 A).
We extract the conventional 1x1x1 cell from that relaxed file (consistency with Paper #1),
then build a symmetric, S-terminated (001) slab: 4 FeS layers, 2x2 lateral -> Fe32 S32,
vacuum >=15 A along c. H placed on ONE (top) side only (intentional asymmetry for dipole
correction).

Outputs (in this directory):
  slab_clean.xyz, slab_H_Stop.xyz, slab_H_Fetop.xyz, h2_box.xyz, STRUCTURE_VERIFY.txt
"""
import os
import sys
import numpy as np
from ase import Atoms
from ase.io import read, write
from ase.build import surface

HERE = os.path.dirname(os.path.abspath(__file__))
# repo root = .../project-third-matter ; HERE = .../dft-neb/u_gate/dgh_inputs
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RELAXED_BULK = os.path.join(
    REPO, "results", "dft_datasets", "2026-05-02", "mack_smoke_W3",
    "relaxed_pristine.xyz",
)

# Fallback P4/nmm parameters (consilium chemist+physicist 2026-06-13)
A_LAT = 3.674
C_LAT = 5.032
Z_S = 0.2602  # S fractional z within conventional cell

VAC = 18.0           # target vacuum along c (>=15 A spec; gap = c_box - slab_thickness)
N_LAYERS = 4         # FeS layers
LATERAL = (2, 2)     # in-plane supercell -> 2x2

# electron counts per neutral atom (pseudopotential valence not used; full-Z bookkeeping
# requested in spec: Fe q16, S q6, H q1)
Z_ELEC = {"Fe": 16, "S": 6, "H": 1}


def get_conventional_bulk():
    """Return (atoms, source_str). Build a CLEAN P4/nmm conventional cell (Fe2S2),
    using relaxed Paper #1 lattice constants + relaxed S z-fraction when available.

    Direct carving of the 72-atom relaxed supercell is fragile (boundary-image
    duplicates -> wrong stoichiometry), so instead we extract the relaxed lattice
    parameters (a, c) and the relaxed S z-fraction from the Paper #1 file and rebuild
    an ideal P4/nmm cell. This keeps consistency with Paper #1 (same a, c, z_S) while
    guaranteeing exact Fe2S2 stoichiometry and clean planes.
    """
    from ase.spacegroup import crystal
    a_use, c_use, z_use = A_LAT, C_LAT, Z_S
    src = "built from P4/nmm (#129) spec parameters (relaxed bulk not found)"
    if os.path.isfile(RELAXED_BULK):
        big = read(RELAXED_BULK)
        L = big.get_cell().lengths()
        nx = int(round(L[0] / A_LAT))
        ny = int(round(L[1] / A_LAT))
        nz = int(round(L[2] / C_LAT))
        a_use = L[0] / nx
        c_use = L[2] / nz
        # relaxed S z-fraction: take an S atom, fold its cartesian z into [0,c_use)
        sym = np.array(big.get_chemical_symbols())
        zpos = big.get_positions()[:, 2]
        zs = zpos[sym == "S"]
        # fold into one conventional period
        z_fold = np.mod(zs, c_use)
        # the lower S sublattice (z ~ z_S*c); take the smallest cluster mean
        z_low = z_fold[z_fold < c_use / 2.0]
        z_use = float(np.mean(z_low)) / c_use if len(z_low) else Z_S
        src = (f"relaxed Paper#1 lattice (a={a_use:.4f}, c={c_use:.4f}, "
               f"z_S={z_use:.4f}) rebuilt as ideal P4/nmm Fe2S2")
    conv = crystal(
        symbols=["Fe", "S"],
        basis=[(0.0, 0.0, 0.0), (0.0, 0.5, z_use)],
        spacegroup=129,
        cellpar=[a_use, a_use, c_use, 90, 90, 90],
    )
    return conv, src


def build_symmetric_Sterm_slab(conv, n_layers, lateral, vac):
    """Build a symmetric S-terminated (001) slab with n_layers FeS sandwiches.

    Mackinawite (001): one FeS 'layer' = one S-Fe-S sandwich (a Fe plane capped by
    S on both sides), sandwiches separated by a van der Waals gap. We stack explicitly
    so that both outer faces are S planes (symmetric, S-terminated) and the slab
    contains exactly n_layers Fe planes.

    Per 1x1 column: n_layers Fe planes (2 Fe each in conv 1x1?) -> conventional 1x1 has
    Fe2 S2, so n_layers sandwiches = Fe(2*n) S(2*n). With 2x2 lateral, n_layers=4 ->
    Fe32 S32.
    """
    a = conv.get_cell()[0, 0]
    c = conv.get_cell()[2, 2]
    sym = np.array(conv.get_chemical_symbols())
    frac = conv.get_scaled_positions(wrap=True)

    # in-plane (x,y) cartesian for each Fe and each S sublattice site, plus z-fraction
    fe_sites = [(frac[i, 0] * a, frac[i, 1] * a) for i in range(len(sym)) if sym[i] == "Fe"]
    # S sites: each has a z-fraction; lower S near z_S, upper near 1-z_S
    s_sites = []  # (x, y, zfrac)
    for i in range(len(sym)):
        if sym[i] == "S":
            s_sites.append((frac[i, 0] * a, frac[i, 1] * a, frac[i, 2]))
    z_s = min(s[2] for s in s_sites)  # ~0.23

    # Build sandwiches k=0..n_layers-1: Fe plane at z=k*c; S at (k*c - z_s*c) and (k*c + z_s*c)
    atoms_sym = []
    atoms_pos = []
    for k in range(n_layers):
        zfe = k * c
        # Fe plane
        for (x, y) in fe_sites:
            atoms_sym.append("Fe")
            atoms_pos.append([x, y, zfe])
        # S below and S above this Fe plane.
        # S sublattice in-plane positions: use the two S xy sites from conv.
        s_xy = [(s[0], s[1]) for s in s_sites]
        # lower S of this sandwich at zfe - z_s*c ; upper S at zfe + z_s*c
        # assign the two distinct in-plane S sites to lower/upper (P4/nmm: they differ)
        # In P4/nmm the two S in the cell are the lower (z_s) and upper (1-z_s) with
        # swapped (x,y). Reconstruct: lower S uses the site whose zfrac==z_s.
        lower_xy = [(s[0], s[1]) for s in s_sites if abs(s[2] - z_s) < 1e-6]
        upper_xy = [(s[0], s[1]) for s in s_sites if abs(s[2] - z_s) >= 1e-6]
        for (x, y) in lower_xy:
            atoms_sym.append("S")
            atoms_pos.append([x, y, zfe - z_s * c])
        for (x, y) in upper_xy:
            atoms_sym.append("S")
            atoms_pos.append([x, y, zfe + z_s * c])

    atoms_pos = np.array(atoms_pos)
    # shift so min z = 0
    atoms_pos[:, 2] -= atoms_pos[:, 2].min()
    thickness = atoms_pos[:, 2].max() - atoms_pos[:, 2].min()
    c_box = thickness + vac
    cell = np.diag([a, a, c_box])
    slab = Atoms(symbols=atoms_sym, positions=atoms_pos, cell=cell, pbc=True)
    # center along z
    p = slab.get_positions()
    p[:, 2] += (c_box - thickness) / 2.0 - p[:, 2].min()
    slab.set_positions(p)
    # lateral supercell
    slab = slab * (lateral[0], lateral[1], 1)
    slab.wrap()
    return slab


def add_H_on_top(slab, mode, d_bond):
    """Add 1 H above the highest surface atom of given species, on top (+z) side.
    mode: 'S' -> over surface S at d(S-H)=d_bond; 'Fe' -> over surface Fe at d(Fe-H)=d_bond.
    """
    sym = np.array(slab.get_chemical_symbols())
    pos = slab.get_positions()
    zmax_all = pos[:, 2].max()
    # surface region = top ~2.5 A
    surf_mask = pos[:, 2] > zmax_all - 2.5
    target_idx = None
    if mode == "S":
        cands = [i for i in range(len(sym)) if sym[i] == "S" and surf_mask[i]]
        # highest S
        target_idx = max(cands, key=lambda i: pos[i, 2])
        h_pos = pos[target_idx].copy()
        h_pos[2] += d_bond
    elif mode == "Fe":
        cands = [i for i in range(len(sym)) if sym[i] == "Fe" and surf_mask[i]]
        if not cands:
            # Fe not in topmost plane (S-terminated); pick highest Fe overall and place
            # H above it at d_bond, laterally at that Fe (x,y)
            cands = [i for i in range(len(sym)) if sym[i] == "Fe"]
        target_idx = max(cands, key=lambda i: pos[i, 2])
        h_pos = pos[target_idx].copy()
        h_pos[2] += d_bond
    else:
        raise ValueError(mode)

    new = slab.copy()
    new += Atoms("H", positions=[h_pos])
    return new, target_idx


def min_distance(atoms, exclude_h=False):
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, np.inf)
    if exclude_h:
        sym = np.array(atoms.get_chemical_symbols())
        keep = np.where(sym != "H")[0]
        if len(keep) < 2:
            return float("inf")
        d = d[np.ix_(keep, keep)]
    return d.min()


def count_layers_fe(atoms):
    pos = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols())
    zfe = np.sort(pos[sym == "Fe", 2])
    if len(zfe) == 0:
        return 0, []
    planes = [zfe[0]]
    groups = [[zfe[0]]]
    for z in zfe[1:]:
        if z - groups[-1][-1] < 0.6:
            groups[-1].append(z)
        else:
            groups.append([z])
    centers = [float(np.mean(g)) for g in groups]
    return len(groups), centers


def vacuum_gap(atoms):
    pos = atoms.get_positions()
    c = atoms.get_cell()[2, 2]
    zmin, zmax = pos[:, 2].min(), pos[:, 2].max()
    thickness = zmax - zmin
    gap = c - thickness
    return c, thickness, gap


def n_electrons(atoms):
    return sum(Z_ELEC[s] for s in atoms.get_chemical_symbols())


def formula_counts(atoms):
    sym = atoms.get_chemical_symbols()
    from collections import Counter
    c = Counter(sym)
    return c


def termination_check(atoms):
    """Return species of bottom and top atomic plane."""
    pos = atoms.get_positions()
    sym = np.array(atoms.get_chemical_symbols())
    # exclude any H adsorbate for termination check (H is adsorbate, not lattice term)
    lat_mask = sym != "H"
    p = pos[lat_mask]
    s = sym[lat_mask]
    order = np.argsort(p[:, 2])
    p = p[order]
    s = s[order]
    # bottom plane
    bot = [s[0]]
    i = 1
    while i < len(s) and p[i, 2] - p[0, 2] < 0.6:
        bot.append(s[i]); i += 1
    # top plane
    top = [s[-1]]
    j = len(s) - 2
    while j >= 0 and p[-1, 2] - p[j, 2] < 0.6:
        top.append(s[j]); j -= 1
    return "".join(sorted(set(bot))), "".join(sorted(set(top)))


def adsorbate_distances(atoms):
    """For a slab with one H, return (nearest S dist, nearest Fe dist, nearest overall)."""
    sym = np.array(atoms.get_chemical_symbols())
    h_idx = [i for i in range(len(sym)) if sym[i] == "H"]
    if not h_idx:
        return None
    hi = h_idx[0]
    d = atoms.get_all_distances(mic=True)[hi]
    dS = min((d[i] for i in range(len(sym)) if sym[i] == "S"), default=np.inf)
    dFe = min((d[i] for i in range(len(sym)) if sym[i] == "Fe"), default=np.inf)
    dmin = min(dS, dFe)
    return dS, dFe, dmin


def report(atoms, name, lines, extra=None):
    c = formula_counts(atoms)
    cell = atoms.get_cell()
    ne = n_electrons(atoms)
    has_h = c.get("H", 0) > 0
    dmin = min_distance(atoms)
    dmin_lat = min_distance(atoms, exclude_h=True)
    cbox, thick, gap = vacuum_gap(atoms)
    nlay, centers = count_layers_fe(atoms)
    bot, top = termination_check(atoms)
    lines.append(f"### {name}")
    lines.append(f"  formula        : {atoms.get_chemical_formula()}")
    lines.append(f"  counts         : Fe={c.get('Fe',0)} S={c.get('S',0)} H={c.get('H',0)}")
    lines.append(f"  N_electrons    : {ne}  (Fe*16 + S*6 + H*1)")
    lines.append(f"  cell a,b,c     : {cell[0,0]:.4f}, {cell[1,1]:.4f}, {cbox:.4f} A")
    # lattice min-dist must be >1.4; H-adsorbate bond (S-H~1.35, Fe-H~1.55) is intentional
    lat_ok = "OK >1.4" if dmin_lat > 1.4 else "FAIL <=1.4"
    if has_h:
        lines.append(f"  min dist (lat) : {dmin_lat:.4f} A  ({lat_ok})  [lattice only]")
        lines.append(f"  min dist (all) : {dmin:.4f} A  [incl. intended H bond]")
    else:
        lines.append(f"  min dist (MIC) : {dmin:.4f} A  ({lat_ok})")
    lines.append(f"  slab thickness : {thick:.4f} A")
    lines.append(f"  vacuum gap     : {gap:.4f} A  ({'OK >=15' if gap>=15 else 'FAIL <15'})")
    lines.append(f"  Fe layers      : {nlay}  z-centers={['%.2f'%z for z in centers]}")
    lines.append(f"  termination    : bottom-plane={bot}  top-plane={top}")
    if extra:
        for k, v in extra.items():
            lines.append(f"  {k:14s} : {v}")
    lines.append("")


def main():
    lines = []
    lines.append("STRUCTURE VERIFICATION -- mackinawite (001) Delta-G_H* inputs")
    lines.append("=" * 70)

    conv, src = get_conventional_bulk()
    lines.append(f"bulk source     : {src}")
    cc = formula_counts(conv)
    lines.append(f"conventional    : {conv.get_chemical_formula()}  Fe={cc.get('Fe',0)} S={cc.get('S',0)}")
    ccell = conv.get_cell().lengths()
    lines.append(f"conv cell a,b,c : {ccell[0]:.4f}, {ccell[1]:.4f}, {ccell[2]:.4f} A")
    lines.append(f"conv min dist   : {min_distance(conv):.4f} A")
    lines.append("")

    # ---- clean slab ----
    slab = build_symmetric_Sterm_slab(conv, N_LAYERS, LATERAL, VAC)
    write(os.path.join(HERE, "slab_clean.xyz"), slab)
    report(slab, "slab_clean.xyz", lines)

    # ---- H on S-top ----
    slab_Sh, sidx = add_H_on_top(slab, "S", 1.35)
    dS, dFe, dmin = adsorbate_distances(slab_Sh)
    write(os.path.join(HERE, "slab_H_Stop.xyz"), slab_Sh)
    report(slab_Sh, "slab_H_Stop.xyz", lines,
           extra={"H->nearest S": f"{dS:.4f} A", "H->nearest Fe": f"{dFe:.4f} A",
                  "H->nearest any": f"{dmin:.4f} A"})

    # ---- H on Fe-top ----
    slab_Feh, fidx = add_H_on_top(slab, "Fe", 1.55)
    dS, dFe, dmin = adsorbate_distances(slab_Feh)
    write(os.path.join(HERE, "slab_H_Fetop.xyz"), slab_Feh)
    report(slab_Feh, "slab_H_Fetop.xyz", lines,
           extra={"H->nearest S": f"{dS:.4f} A", "H->nearest Fe": f"{dFe:.4f} A",
                  "H->nearest any": f"{dmin:.4f} A"})

    # ---- H2 box ----
    L = 13.0
    h2 = Atoms("H2", positions=[[L/2, L/2, L/2 - 0.37], [L/2, L/2, L/2 + 0.37]],
               cell=[L, L, L], pbc=True)
    write(os.path.join(HERE, "h2_box.xyz"), h2)
    dHH = h2.get_distance(0, 1)
    lines.append("### h2_box.xyz")
    lines.append(f"  formula        : {h2.get_chemical_formula()}")
    lines.append(f"  counts         : H={len(h2)}")
    lines.append(f"  N_electrons    : {n_electrons(h2)}")
    lines.append(f"  cell (cube)    : {L:.1f} A")
    lines.append(f"  d(H-H)         : {dHH:.4f} A  ({'OK ~0.74' if abs(dHH-0.74)<0.05 else 'check'})")
    lines.append(f"  min dist       : {dHH:.4f} A")
    lines.append("")

    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(HERE, "STRUCTURE_VERIFY.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()
