# Deferred / pivoted DFT probes of the confined system

This directory is the **honest record of the direct-DFT work around the confined-system
detachment** (§3.2–3.3) that was *prepared*, *partially run*, and whose **full umbrella-sampling
PMF reconstruction was deferred** — together with the cheap analyses it pivoted into. It is kept
as an anti-example + provenance trail, not as a paper-quotable result on its own.

## What happened (the smoke-run story — "not a failure, a pivot")

A full **direct-DFT umbrella-sampling PMF** of the Fe–H detachment was intended as the rigorous
disambiguation of the MACE-vs-CHGNet bracket. Instead of running it blind, two things happened:

1. **`$0` preflight gate** (`us_preflight_s136.json`). The prodromos *US-preflight gate* analysed
   the existing MLIP umbrella data and concluded that a full 18-window DFT PMF is **not** the
   cost-effective move: the discriminating information lives on the approach/departure flanks
   (recommended windows ≈ 1.71, 1.90, 3.17 Å), 3 DFT spot windows at ≥ 2 ps each would *select*
   the correct MLIP (discrimination distance d² = 409 at ρ=0, still 82 at the worst-case
   co-trained error correlation ρ=0.8), and — crucially — its **carrier-router (U5a)** showed the
   two MLIPs **disagree on the H-carrier topology** (MACE: Fe→water-O; CHGNet: Fe→S→water-O;
   max Jensen–Shannon = 0.71 at the barrier top), so the bracket compares *different mechanisms*.
   This directly underpins the §3.3 mechanistic reading and the decision to **defer** the full DFT
   PMF. (Gate engine: prodromos `us_preflight_gate.py` — see cross-references; the gate itself is
   the subject of a separate methods paper.)

2. **Steered-detachment DFT smoke runs** (the inputs in `dft_us_inputs/`). These went through
   progressive fixes:
   - **DISTANCE-CV smoke → failed** (the anti-example): the temperature spiked to ~590 K and the
     proton stayed ~91 % Fe-bound — the naive distance CV did not detach it.
   - **smooth-min CV + MOVINGRESTRAINT + gentle seed + TIMECON 100 fs** (`carrier_2p40`, s167) →
     clean (601/601 SCF, T ≈ 270–330 K); the proton detaches from the *whole* Fe sheet and is
     **100 % in-flight at ξ ≈ 2.12 Å** (bonded to neither S nor water).
   - **×3.3 longer equilibration, then full 2.5 ps** (`carrier_B`, s169) → the delocalization is
     **robust** (not a non-equilibrium artifact).
   This pivoted into the paper's **§3.3 steered-detachment finding** (the in-flight carrier) and the
   NS-1/NS-2/NS-3 analyses (see the paper §3.3 and its SI). So the smoke programme
   *succeeded at the mechanistic question*; only the quantitative full PMF reconstruction is deferred.

## Status of each artifact

| Artifact | Status | In the paper? |
|---|---|---|
| `dft_us_inputs/` (us_window_*, plumed_*, smoke, probe) | prepared; DISTANCE-CV smoke FAILED → smooth-min smoke OK | protocol; deferred-PMF noted §3.3 |
| `us_preflight_s136.json` + `analyze_dft_us.py` | gate ran, verdict PASS; full DFT PMF DEFERRED by gate decision | decision underlies §3.3 deferral + carrier split |
| `carrier_probe_results/` (carrier_2p40, carrier_B) | SUCCEEDED (mechanism scout) | **yes** — §3.3 in-flight carrier (NS-1/2/3) |
| `ns2_charge_control/` (q0/q1 SP) | ran; q+1 converged, q0 floored (sign robust) | **yes** — NS-2, §4.3 charge-control |
| `dgh_slab_prep/` (ΔG_H* slab) | prepared; SCF wall, DEFERRED | no — out of scope (future / Paper #5) |

## Honest caveats
- The steered-detachment probes are **single short DFT trajectories** at charge +1 in the confined
  cell — mechanism scouts, **not** converged free energies. The quantitative bracket comes from the
  MLIP umbrella sampling (`../mlip_umbrella_sampling/`); the full DFT PMF remains deferred (§3.3).
- The preflight gate's UI barriers (MACE 0.67 / CHGNet 0.30 eV) are umbrella-integration values on
  disjoint windows and read slightly below the WHAM 0.80 / 0.32 eV; the qualitative ordering
  (MACE ≫ CHGNet) is robust. The gate **refuses WHAM** on these disjoint windows by design.

## Cross-references
- **This paper:** §3.2–3.3 (data repo root `README.md`, `tm-spec/`).
- **Gate engine (separate methods paper):** prodromos US-preflight gate — the carrier-router /
  discrimination-optimal-window methodology. See `us_preflight_gate.tm.yaml` and the TM-Spec
  `cross_ref` block.
- **Companion:** *MLIPvsDFT* (<https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides>) — the
  foundation-MLIP failure-mode taxonomy.
