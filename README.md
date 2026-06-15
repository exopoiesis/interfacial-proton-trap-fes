# Confinement-Stabilized Interfacial Hydrogen Trap in a Pentlandite–Mackinawite Bilayer Membrane

![tests](https://github.com/exopoiesis/interfacial-proton-trap-fes/actions/workflows/ci.yml/badge.svg)

Data and code supporting the manuscript:

> **Confinement-Stabilized Interfacial Hydrogen Trap and Proton Retention in a Pentlandite–Mackinawite Bilayer Membrane**
> Igor N. Morozov (Independent Researcher, Ukraine). ORCID [0009-0007-3863-1747](https://orcid.org/0009-0007-3863-1747).

A first-principles + machine-learning-potential study quantifying the **free-energy landscape of an excess proton at the mackinawite (FeS)/water interface**, and the implications for an asymmetric, gradient-retaining (Fe,Ni)S membrane.

## Key result

A hydrogen dissociatively chemisorbed at the laterally confined mackinawite/water interface is held in an **Fe-chemisorption kinetic trap of ΔF# ∈ [0.32, 0.80] eV (≥ 12 k_BT)** — quantified three ways:

| Method | System | Result |
|---|---|---|
| Direct DFT well-tempered metadynamics (CP2K, PBE-D3) | 73-atom mackinawite + 12 H₂O + H⁺ (charge +1) | basin-depth lower bound **ΔF_basin ≥ 0.32 eV** |
| Apples-to-apples MLIP umbrella sampling (MACE-MP-0 vs CHGNet, identical protocol) | same 73-atom neutral system | **MACE 0.80 [0.74, 0.84] eV; CHGNet ≈ 0.32–0.40 [0.37, 0.43] eV** (95 % block-bootstrap CIs) |
| AIMD, thick interfacial water | 145-atom, 11.6 ps | excess proton stays **water-bound** (fluctuating-Eigen), never reaches Fe — a geometry-dependent residence contrast |

The factor-≈ 2.5 MACE–CHGNet disagreement is **statistically real** (CIs ≪ gap, non-overlapping) and **mechanistic** — a 2-D carrier reanalysis shows CHGNet diverts the hydrogen onto a surface sulfur (S-bound, ⟨d_SH⟩ ≈ 1.4 Å) while MACE keeps it in-flight: a foundation-MLIP failure mode for interfacial proton chemistry, not sampling noise (spin treatment is a second contributor).

Complementary design results: a DFT CI-NEB **pentlandite** proton-migration barrier of **0.39 eV (CP2K) / 0.44 eV (ABACUS)** ⇒ µs-scale proton residence and an electron-to-proton conductivity selectivity ≈ 10⁷ (proton-blocking inner layer); an electrochemical energy balance giving a **+61 % margin** for ΔpH-driven CO₂-to-formate reduction at ΔpH = 6; and a minimal transport-stability model whose **bistable** steady state has its fixed-point and absorbing-set structure **machine-checked in Lean 4**.

> **Note on methodology.** An earlier, preliminary exploration of these systems used foundation-MLIP NEB alone (e.g. a pentlandite barrier of ~1.43 eV from MACE-MP-0), which we have since shown is unreliable for these iron-sulfide proton paths. **All barriers here are computed with DFT** (CP2K / ABACUS); foundation MLIPs appear only in the controlled, apples-to-apples benchmark.

## Repository map (directory ↔ manuscript section)

The manuscript and its Supporting Information are published separately (preprint/journal; DOI added on acceptance) — this repository is **data + code only** and links to the paper by section/DOI.

| Directory | Paper § | Contents |
|---|---|---|
| `aimd_interfacial_water/` | §3.1 | Thick-water proton-localization analysis: carrier-indicator and information-criterion (MDL/BIC) scripts + result summaries (NS-1/NS-3) |
| `dft_metadynamics/` | §3.2 | Direct DFT WT-MetaD trap: CP2K inputs, free-energy surfaces (full / split-halves), convergence and CV-completeness diagnostics, FES analysis script |
| `mlip_umbrella_sampling/` | §3.3 | **Centerpiece.** Per-window collective-variable data (`data/{mace,chgnet}/window_00..17/colvar.dat`), WHAM reconstruction (`wham_native.py`), diagnostics, and the revision uncertainty-quantification + 2-D carrier reanalysis (`revision_uq_2d.py` → `results/uq_revision/`) |
| `pentlandite_neb/` | §3.6 | DFT climbing-image NEB of the pentlandite S–H···S migration (CP2K + ABACUS cross-check): scripts, endpoints, converged results |
| `transport_model/` | §3.4 | TM6v3-min three-variable ODE model + Sobol sensitivity + non-equilibrium steady-state + Gillespie analysis, with result JSON/figures |
| `energy_balance/` | §3.5 | pH-gradient electrochemical energy balance for CO₂-to-formate |
| `proofs/` | §3.4 | `KineticTrap.lean` — machine-checked (sorry-free) Lean 4 formalization of the model's fixed-point / Hurwitz-Jacobian / IVT-bistability / absorbing-set structure |
| `figures/` | — | Paper figure generators (where archived) + rendered PNG/PDF |
| `tm-spec/` | all | Machine-readable **TM-Spec** YAML for every computation (structure/method/sanity/provenance + `cross_ref` to the companion *MLIPvsDFT* study and the TM-Spec standard). Validates against schema; exports to NOMAD. See `tm-spec/README.md` |
| `dft_probes_deferred/` | §3.2–3.3 | Honest record of the direct-DFT probing that was prepared/partially run and whose full umbrella-sampling PMF was **deferred**: the `$0` US-preflight gate (`us_preflight_s136.json`), the prepared DFT-US inputs, the steered-detachment smoke runs that *pivoted* into the §3.3 in-flight carrier finding, NS-2 charge-control, and the (deferred) ΔG_H* slab prep. Includes a `us_preflight_gate.tm.yaml` anti-example spec cross-linked to the prodromos gate engine. See its `README.md` |

## Reproducibility notes (honest scope)

- **Raw trajectories are not in git.** The 145-atom thick-film AIMD trajectory, the confined-system WT-MetaD position/velocity trajectories, and the MLIP umbrella-sampling production trajectories (multi-GB) will be deposited on **Zenodo with an archival DOI on acceptance**. This repository contains the **scripts, inputs, processed/collective-variable data, free-energy surfaces, and analysis** needed to reproduce every reported number from those.
- **Reproduce the headline bracket from the bundled data:** `python mlip_umbrella_sampling/reproduce_bracket.py` runs WHAM on the included per-window `colvar.dat` and prints MACE ≈ 0.80 eV and CHGNet ≈ 0.32–0.40 eV — the §3.3 bracket — CPU-only, no GPU/DFT/MLIP (this is exercised by the test suite). The full MLIP umbrella-sampling protocol is also included: the production MD driver with the native minimum-distance restraint (`run_mlip_us_native.py`, finite-difference-validated — see `test_restraint_consistency.py`), the window generator (`generate_windows.py`), the system builder, `wham_native.py`/`wham_analysis.py`, and the fuller revision UQ + 2-D carrier reanalysis (`revision_uq_2d.py`; set its data path). A prepared (deferred) DFT umbrella-sampling input template is included for completeness.
- **Figure scripts** for `fig_w1_localization`, `fig_w2_pmf_bracket` and `fig5_transport_stability` are not archived (one-off); the rendered PDF/PNG are included. `fig_basin_portrait.py`, `fig6_energy_balance.py` and `generate_all.py` are included.

## Requirements

See `requirements.txt`. Core: Python ≥ 3.10, NumPy, SciPy, Matplotlib, ASE; MACE-torch and CHGNet (GPU) for MLIP sampling; CP2K and ABACUS for the DFT work; SALib for Sobol analysis. The Lean proof uses Lean 4 + mathlib (`proofs/`).

## Tests

A CPU-only test suite (`tests/`, run with `pytest`; CI: `.github/workflows/ci.yml`) guards the
parts that can run without GPU/DFT/MLIP:

- **`test_tmspec_valid.py`** — every TM-Spec YAML is well-formed, carries `spec/kind/id`, is
  cross-linked (`cross_ref.paper` + the TM-Spec standard), has a unique id, and leaks no internal
  instance labels.
- **`test_reproduce_bracket.py`** — WHAM on the bundled `colvar.dat` reconstructs the §3.3 bracket
  (MACE ≈ 0.80 eV, CHGNet 0.32–0.40 eV, factor-~2 gap) — a genuine data→result reproducibility check.
- **`test_energy_balance.py`** — the §3.5 energy balance runs and yields the 0.355 V available EMF
  at ΔpH = 6 with a feasible verdict.

GPU/DFT/MLIP runs and the (slower) full transport-model sweep are documented and scripted but not
CI-tested.

## Author

**Igor N. Morozov** — Independent Researcher, Ukraine — [exopoiesis.space](https://exopoiesis.space) · ORCID [0009-0007-3863-1747](https://orcid.org/0009-0007-3863-1747)

## AI-assisted research disclosure

This work was carried out with **Claude (Anthropic)** as a computational and analytical assistant under the author's direction and scientific oversight. The author conceived the study, made all scientific decisions, and bears full responsibility for the claims. Per current authorship standards (COPE/Nature/Science), the AI is not an author. Commits in this repository may carry a `Co-Authored-By: Claude` trailer to record the collaboration transparently.

## License

- **Code** (`.py`, `.lean`, `.inp`): [MIT](LICENSE)
- **Data, figures, and specifications** (`.json`, `.dat`, `.xyz`, `.npz`, `.png`, `.tm.yaml`): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)

## Citation

```bibtex
@misc{morozov2026_interfacial_proton_trap,
  author       = {Morozov, Igor N.},
  title        = {Confinement-Stabilized Interfacial Hydrogen Trap and Proton
                  Retention in a Pentlandite--Mackinawite Bilayer Membrane},
  year         = {2026},
  howpublished = {\url{https://github.com/exopoiesis/interfacial-proton-trap-fes}},
  note         = {Data and code repository; Zenodo DOI on acceptance}
}
```
