# ΔG_H* anchor inputs (Paper #2 B-path) — DRAFT, pending /test

Spec source: `paper/reviews/KineticTrap/CONSILIUM_DELTAGH_{chemist,physicist}_2026-06-13.md`. Decision: РЕШЕНИЕ-116.
Structures: `build_slab.py` → slab_clean / slab_H_Stop / slab_H_Fetop / h2_box (verified `STRUCTURE_VERIFY.txt`).

## Quantity
ΔG_H* = [E(slab+H*) − E(slab) − ½E(H₂)] + ΔZPE.  ΔZPE explicit (physicist: S–H +0.114, Fe–H +0.064 eV; generic +0.24 invalid for sulfide). Per site (S-top, Fe-top). EQUILIBRIUM binding, NOT the [0.32,0.80] eV barrier (present complementary).

## Calc set (this batch = GEO_OPT; ZPE = follow-up VIBRATIONAL_ANALYSIS on relaxed H)
| input | structure | nspin | N_e | Poisson |
|---|---|---|---|---|
| slab_clean_geoopt.inp | Fe32S32 | 1 (704 even) | 704 | XYZ + surf-dipole |
| slab_H_Stop_geoopt.inp | Fe32S32H @S | 2 (UKS, 705 odd) | 705 | XYZ + surf-dipole |
| slab_H_Fetop_geoopt.inp | Fe32S32H @Fe | 2 (UKS, 705 odd) | 705 | XYZ + surf-dipole |
| h2_geoopt.inp | H₂ | 1 | 2 | NONE + MT |

Common: PBE-D3(BJ), GTH q16/q6/q1, DZVP-MOLOPT-SR, CUTOFF 600/REL 60, &FM SCALAPACK (F-053/057), FERMI 300K (slabs).

## OPEN QUESTIONS for /test (verify before deploy)
1. **spin×smearing:** UKS + MULTIPLICITY 2 + FERMI_DIRAC — does CP2K honor multiplicity with smearing, or fractional-occupy? Is MULTIPLICITY the right seed vs free spin? (load-bearing for odd-e slab+H).
2. **SURFACE_DIPOLE_CORRECTION + PERIODIC XYZ + SURF_DIP_DIR Z** — correct keyword/usage in our CP2K build; needed only for asymmetric (+H) but applied to all (harmless on symmetric clean?).
3. **ADDED_MOS 120** — sufficient for 704/705 e⁻ + Fermi tail (per spin for UKS)?
4. **ASPC ORDER 3 on fresh GEO_OPT** — s138 segfault history (wfi_extrapolate); fallback ladder ORDER 1 → USE_GUESS if it crashes on step 0.
5. **fixed atoms:** v0 relaxes ALL (symmetric-slab relaxation cancels in ΔG_H* difference) — confirm acceptable vs freeze bottom 2 layers.
6. **H₂ MT box 13 Å** adequacy; ½E(H₂) consistency.
7. **ZPE step design** — separate RUN_TYPE VIBRATIONAL_ANALYSIS on adsorbed H (after GEO_OPT), 3 modes finite-diff.
8. **structure_identity** (prodromos) on the 3 slab xyz before deploy.

## Deploy — LIVE (s169, 2026-06-13)
**Deployed to W1** (A100, ssh4.vast.ai:21836, cu128, /workspace/dgh) via `tmp/dgh_deploy.sh`. Sequential GEO_OPT: h2 -> slab_clean -> slab_H_Stop -> slab_H_Fetop (single A100, mpirun -np 16 OMP 1, F-047). nspin=1 nonmag + FERMI smear (consilium alpha). First job (h2) confirmed running clean: E(H2)=-1.162 Ha, SCF converges, real CP2K parser accepted inputs.
- **Monitor:** `bash tmp/dgh_monitor.sh` (flags `*.DONE`/`*.FAILED`/`ALLDONE`, run.log, newest .out, GPU).
- **Lib-env / launch:** in `tmp/run_dgh_remote.sh` (CP2K_DATA_DIR=/opt/cp2k/data resolves BASIS_MOLOPT/GTH_POTENTIALS; LD_LIBRARY_PATH = devops cu128 libs).

## Analysis (when ALLDONE) -> DeltaG_H*
Harvest final total energy from each `*.out` (last "ENERGY| Total FORCE_EVAL ... " after GEO_OPT converged):
  DeltaE_H(site) = E(slab_H_<site>) - E(slab_clean) - 0.5*E(h2)
  DeltaG_H*(site) = DeltaE_H + DeltaZPE(site)   [DeltaZPE: S-H ~+0.114, Fe-H ~+0.064 eV per consilium; compute explicit via VIBRATIONAL_ANALYSIS on the adsorbed H of the relaxed structure -- FOLLOW-UP step, not in this batch]
Present complementary to MLIP bracket [0.32,0.80] eV (equilibrium binding, NOT the kinetic barrier). Compare magnitude to Dzade 2016 (water dissoc. 0.62-0.83 eV CI-NEB on mackinawite) + HER ΔG_H* literature.

## Resurrection (if W1 dies)
Structures + inputs all in this dir (build_slab.py rebuilds from Paper#1 relaxed lattice). Re-deploy: `bash tmp/dgh_deploy.sh` (edit PORT/NODE for new A100). A100/H100 only (FP64).
