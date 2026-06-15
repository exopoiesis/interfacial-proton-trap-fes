# Variant-A carrier-probe result (2026-06-12, s167) — ξ≈2.12: proton DELOCALIZED, neither S nor water

W1 (A100 PCIE), CP2K cu128, PLUMED smooth-min(d_FeH over 18 Fe, β=10/Å, log-sum-exp) + MOVINGRESTRAINT
ramp 1.55→2.40, KAPPA=1000, charge+1, nspin=1, FM+PREFERRED_DIAG SCALAPACK, gentle seed + TIMECON 100fs.
600 steps (300 fs), 601/601 SCF converged, DONE clean. Gentle seed worked: T held ~270-330K (vs 590K for the
violent DISTANCE smoke).

## Result (firm, 151 hold-phase frames, steps 300-600)
- **carrier occupancy: Fe 0%, S 0%, O 0%, in_flight 100%.** Proton bonded to NOTHING.
- ⟨d_FeH_min⟩ = 2.19 Å, ⟨d_SH_min⟩ = 2.53 Å, ⟨d_OH_min⟩ = 2.46 Å — proton ~2.5 from both S and O.
- PLUMED smin held at **~2.12 Å** (NOT the 2.40 target — residual bias ~38 kJ/mol): the proton RESISTS
  being pulled further off the Fe sheet. Effective ξ sampled ≈ 2.1-2.2.

## Interpretation (PRELIMINARY — single 0.3ps probe, thin 3×3×1 cell, charge+1)
- ✅ **Variant A (smooth-min) works**: detaches the proton from the WHOLE Fe sheet (Fe 0% carrier here vs
  91% Fe for the DISTANCE-CV smoke). The B-vs-A adjudication is settled — A is correct.
- 🔑 **At ξ≈2.1, DFT shows a DELOCALIZED/in-flight proton — NOT the S-H intermediate CHGNet predicts,
  NOR the water-bound state MACE predicts.** DFT disagrees with BOTH foundation MLIPs at this distance.
- **Hypothesis:** the MACE-vs-CHGNet factor-2.5 disagreement = two DIFFERENT artificial localizations
  (MACE→water, CHGNet→S) of a state that DFT shows is actually delocalized/shared at the interface. Both
  MLIPs over-localize the detaching proton (each its own way); DFT keeps it in-flight.
- The proton resists full detachment past ~2.1 Å (restraint couldn't reach 2.40) → a shallow delocalized
  interfacial state, strong PES pull-back toward Fe. No easy S-H or water-bound site at this confined interface.

## Caveats / open
- Single short probe (0.3 ps hold), non-fully-eq; charge+1 (Coulombic ~0.5-0.7 eV, "complementary"); thin
  cell (confined water, not bulk). NOT a converged free-energy or a paper-claim — a mechanism scout.
- smin reached only ~2.12 (not 2.40): to probe further detachment need stronger KAPPA or accept the PES resists.
- Next options (Igor): (a) ξ=2.55 probe with stronger KAPPA (does it ever commit to S/water if forced further?);
  (b) longer eq at 2.12 to confirm delocalization isn't a sampling artifact; (c) accept the delocalized-
  interface finding as the mechanism answer (both MLIPs localize-artifact) and write it up.

## carrier_B longer-eq verdict (2026-06-13, s169) — DELOCALIZATION ROBUST (non-eq EXCLUDED)
W1 (A100 PCIE), same setup, **hold ×3.3 longer** (steps 300→608, 155 hold frames, ~0.5 ps hold vs 0.15 ps preliminary). MOVINGRESTRAINT ramp 1.55→2.40 over 300 steps then hold to 1300; harvested at step 608.

- **carrier occupancy: Fe 0%, S 0%, O 0%, in_flight 100%** — IDENTICAL to preliminary. Proton bonded to nothing.
- ⟨d_FeH_min⟩ = 2.185 Å, ⟨d_SH_min⟩ = 2.526 Å, ⟨d_OH_min⟩ = 2.463 Å — proton ~2.5 Å from both S and water, same as preliminary.
- Fe13 global-nearest only **50%** of frames, gap to 2nd-nearest Fe = **0.058 Å** → proton NOT committed to a single Fe, floats between several at similar distance (reinforces delocalized picture).
- smin held ~2.21 (target 2.40) → **same pull-back toward Fe sheet** as preliminary (~2.12). KAPPA=1000 cannot push past ~2.1-2.2 Å.
- Temp stable 315-319 K through hold (NOT cooling; analyze-script "cooling" NOTE is hardcoded boilerplate).

**VERDICT (firm for the qualitative carrier question):** delocalization is ROBUST to 3.3× longer equilibration → **NOT a non-eq sampling artifact**. At the transfer CV the proton is a delocalized in-flight interfacial state — neither the CHGNet S-H intermediate nor the MACE water-bound state. Both foundation MLIPs over-localize (each its own way) a state DFT keeps shared. → **Option C confirmed** (consolidate finding into KineticTrap §3.2-3.3 + Paper #5). Still single-CV-window scout, not a converged PMF — quantitative barrier needs the full US.

**⚠️ Production-US design constraint (NEW):** the strong PES pull-back means the s136 window scheme (centers 1.5→4.0 Å, K=1000 kJ/mol/Å², from `generate_windows.py`) will NOT sample upper windows (≥2.55) at their nominal CV under DFT — proton sits short of target → poor overlap / WHAM gaps above ~2.2 Å. Production US must address this (stronger KAPPA in detachment region, or accept PMF truncation at ~2.5 Å, or finer windows where the basin is). → consilium (chemist+physicist) before deploy.

### carrier_B FULL-run confirmation (step 1300 DONE, 501 hold-frames steps 300-1300 ≈ 2.5 ps hold — MAXIMAL eq)
- **carrier occupancy: Fe 0% / S 0% / O 0% / in_flight 100%** — UNCHANGED over the full 2.5 ps. Delocalization is rock-solid, not an eq artifact.
- ⟨d_FeH_min⟩=2.197, ⟨d_SH_min⟩=2.519, ⟨d_OH_min⟩=**2.246** Å; ⟨d(Fe13)⟩=2.222±0.052 (tighter σ than 608-cut); Fe13 global-nearest 50%, gap 0.046 Å.
- **Nuance:** ⟨d_OH_min⟩ drifted 2.46→2.25 Å over the long run → proton leans FAINTLY toward water, but **NO O-H bond forms** (water O-H bond ≈1.0 Å, sampled 2.25 Å) → still in-flight, NOT localized. A weak tendency, not the MACE water-bound state.
- Tail temp ~245 K (thermostat hold-phase undershoot from 300 K target); does not affect the occupancy-based carrier verdict. Run DONE clean, 0 SCF-NOT-converged across 1301 SCF.
- **Bottom line:** Option C confirmed at maximal eq. Carrier = delocalized in-flight interfacial proton, neither MLIP's localization. carrier_B compute purpose COMPLETE.

## NS-1/2/3 "small attempt" (2026-06-13, s169) — cross-project ideas (CEC / artifact / MDL-BIC). $0 + 2 cheap SP.
Igor's redirect (away from the ΔG_H* slab SCF rabbit hole) — analyze EXISTING data with better tools.

**NS-1 (carrier coordinate, $0):** carrier-indicator s = min_O d(H38,O) − min_Fe d(H38,Fe) on carrier_B (651 fr) + carrier_2p40.
- carrier_B hold ⟨s⟩=+0.05±0.21 Å, **100% in-flight (Fe/S/O = 0%)**, unimodal → **robust delocalization (REVIEW-grade)**, reproduces this doc.
- König mCEC honestly **inapplicable** (carrier_B = Fe-hydride, not a Brønsted proton); real 145-atom AIMD (the delocalization run) is NOT local — on Vast (needs harvest). w1_grotthuss local AIMD = equilibrium Fe-bound hydride (different state).
- `results/ns_analysis_2026-06-13/NS1_RESULT.md`.

**NS-3 (MDL/BIC + CRB, $0, all data in `us_preflight_s136.json`):** model-select {flat/single/double-well} on MLIP mean-force.
- **flat OVERWHELMINGLY rejected** both MLIPs (ΔBIC M0−M1 = 720 MACE / 1296 CHGNet ≫6) → **barrier identifiable along the CV**. M2(double-well) beats M1 (multimodal).
- P(no-barrier): MACE 0.000, CHGNet 0.014. MACE needs +1.7× time at saddle for σ<0.1 eV (matches U4); CHGNet already there.
- ⚠️ χ²/dof≫1 → polynomial HEIGHTS unreliable (only bracket [0.32,0.80]/trapezoid trustworthy); MDL robust for SHAPE only.
- **Reconciles s168 paradox:** "delocalized ⇒ no barrier" was a CONFLATION — barrier exists along the d_FeH free-energy coordinate; carrier delocalization is ORTHOGONAL (different coordinate). Both true; together a STRONGER story (explains MACE↔CHGNet mechanism split). `NS3_RESULT.md`.

**NS-2 (charge-artifact control, 2 SP on confined cell):** is "delocalization" a charge+1 artifact?
- **q+1 control (nspin1, converged):** H38 Mulliken **+0.155** / Löwdin +0.44, pop 0.84 → modestly protonic, in-flight (NOT hydride H⁻, NOT covalent S-H, NOT bare H⁺) → **weakly AGAINST the +1-artifact** (H is positive, not a charge-induced hydride).
- **q0 test (neutral, nspin2 ⟸ N_e=493 odd):** hit nspin=2 Fe-S SCF sloshing; tamed (narrow MV 1000K + Broyden α0.1/nb12, no OUTER) but only floors ~2e-3 (not 1e-5). Salvaged with EPS_SCF 5e-3 → converged → H38 Mulliken net charge **positive (~+0.5), spin ≈0**. ⚠️ **Consilium (chem+phys): floored-sloshing Mulliken NUMBER not quotable** (limit-cycle moves the measured charge ±0.1-0.3e in the active S–H–Fe pocket; s163 stop-at-floor lesson) — but the **SIGN (positive, not hydride H⁻) IS robust**.
- **NS-2 VERDICT (consilium-corrected):** q+1 control (converged, H +0.16) AND q0 (floored, H ~+0.5, both POSITIVE, spin~0) → carrier is a **protonic in-flight species at BOTH charge states** → **delocalization is NOT a charge+1 artifact** (the "+1→hydride" hypothesis is directly falsified — H is positive, not H⁻, in both). Load-bearing arg = q+1 (converged) + the robust SIGN. **Honest open limitation:** these are FIXED in-flight-geometry SPs; the weaker objection "if RELAXED at neutral, would H fall onto covalent S-H?" needs a q0 nspin=2 GEO_OPT = a WALL (deferred). Best-closed-by-geometry (consilium) but our SPs are fixed-geom so trivial. State as: "charge-control rules out a hydride artifact; neutral relaxation deferred."

**Net:** NS-1 (delocalization robust) + NS-3 (barrier identifiable, flat rejected) = strong, paper-grade, $0. NS-2 q1 weakly anti-artifact. **ΔG_H* slab SCF NOT needed.** Paper #2 stands on bracket [0.32,0.80] + identifiable-barrier + orthogonal-delocalization. Paper #5 gains U6 (MDL/BIC model-selection layer).

## Benchmark (MPI/OMP, same W1, bench_md 10-step)
| MPI×OMP | s/step | vs 8×1 |
|--|--|--|
| 8×1 | 32.1 | 1.00 |
| **16×1** | **30.8** | **1.04 (best)** |
| 32×1 | 48.2 | 0.67 (anti-scale) |
| 8×4 | 30.8 | 1.04 |
| 16×4 | 32.0 | 1.00 |
| 16×8 | 34.6 | 0.93 |
→ **GPU-bound (F-047 confirmed on A100-PCIE/73at): 128 cores don't help; use MPI=16/OMP=1 (+4%), >16 anti-scales.**
