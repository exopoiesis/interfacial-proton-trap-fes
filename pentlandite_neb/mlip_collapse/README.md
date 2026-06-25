# pentlandite_neb/mlip_collapse — foundation MLIPs do not reproduce the S–H···S channel

Supports manuscript §3.6 / §4.3 and `SI/SI_pentlandite_mlip_collapse_2026-06-25.md`: the
pentlandite vacancy-anchored proton-migration barrier (0.385 eV CP2K / 0.44 eV ABACUS) is
necessarily a DFT result because foundation MLIPs relax the proton into the global Fe-hydride
sink rather than holding the metastable S-bound state — even under the exact CP2K frozen mask.

## Contents

- `structures/geom_analysis.json` — the **exact** CP2K frozen-framework mask (28 mobile atoms =
  H + first coordination shell, 108 frozen; donor S98, acceptor S48), recovered from the original
  CP2K NEB setup and confirmed atom-for-atom against the CP2K `&FIXED_ATOMS` input.
- `structures/pent_donor_endA.extxyz` — DFT-relaxed donor endpoint (HFe₇₁S₆₄, H on S98).
- `scripts/` — NEB/relaxation harness and the two tests:
  - `fes_mlip_neb.py` — harness (structure-identity + H-host gates, two-phase CI-NEB).
  - `pent_multiendpoint.py` — places H on each cage sulfur and fully relaxes (where does it land?).
  - `pent_frozen_exact.py` — exact-CP2K-mask frozen-framework test (`pent_frozen_neb.py`,
    `pent_symmetric_neb.py` are imported helpers).
  - Paths near the top of each script are project-local; adjust `STRUCT_DIR` / mask path to reuse.
- `results/` — machine-readable outputs:
  - `mace-omat_pentlandite_multiendpoint.json`, `mace-mp_pentlandite_multiendpoint.json` — cage-site
    enumeration (final H host per site).
  - `mace-omat_pentlandite_frozen_EXACTmask.json` — exact-mask result: donor stays S-bound,
    acceptor (S48) collapses to Fe-hydride (d_{H–Fe} = 1.70 Å).

## Key result

| | MACE-omat-0 (exact CP2K mask) | CP2K (same mask) |
|---|---|---|
| Donor (H on S98) | S-bound (1.55 Å) | S-bound |
| Acceptor (H on S48) | Fe-hydride (1.70 Å) | S-bound (0.385 eV) |

Identical frozen constraint, only the potential differs → the collapse is a property of the MLIP
potential-energy surface, not of the frozen-framework approximation.

## Models / scope

MACE-MP-0 and the OMat24-trained MACE-omat-0 tested here. eqV2-OMat24 excluded (non-conservative
head + OCPCalculator dtype incompatibility). CHGNet is used for the mackinawite *surface* trap
(`mlip_umbrella_sampling/`), not for this pentlandite bulk hop. Cross-mineral foundation-MLIP
benchmark: companion study (Morozov 2026).

## Environment

mace-torch 0.3.15 (`medium-omat-0` weights, ASL license), ASE 3.27, Python 3.13. Foundation-model
weights are downloaded by the libraries (not redistributed here).
