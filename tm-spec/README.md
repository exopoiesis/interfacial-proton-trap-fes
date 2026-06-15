# Machine-readable specifications (TM-Spec)

Every computation behind this paper is described by a **TM-Spec** YAML file — a machine-readable
record of the structure, defects, magnetic treatment, calculation level, workflow, sanity gates,
provenance (a `parents` DAG) and results. The format is the open **TM-Spec** standard
(`spec: tm-spec/0.2`, schema at <https://exopoiesis.space/tm-spec>); files validate with
`tm_spec_validator.py` and export to NOMAD with `tm_spec_export_nomad.py` (TM-Spec tooling).

## Specs in this repository

| File | `id` | Manuscript § | What it pins down |
|---|---|---|---|
| `aimd_grotthuss.tm.yaml` | `tm.mack_3x3x2.h_proton.aimd_grotthuss_v1` | §3.1 | Thick-film interfacial-water proton localization (AIMD) |
| `dft_metad.tm.yaml` | `tm.mack_3x3x1.h_proton.metad_v1` | §3.2 | Confined-system direct-DFT WT-metadynamics trap lower bound |
| `us_pmf.tm.yaml` | `tm.mack_3x3x1.h_proton.us_pmf` | §3.3 | Confined-system MLIP umbrella-sampling PMF (kinetic-trap bracket [0.32, 0.80] eV) |
| `mlip_benchmark.tm.yaml` | `tm.mack_3x3x1.h_proton.mlip_bench_us` | §3.3 | Apples-to-apples MACE-vs-CHGNet benchmark on the detachment PMF |
| `pentlandite_neb.tm.yaml` | `tm.pentlandite_vfe.h_proton.neb_dft` | §3.6 | Inner-layer pentlandite S–H···S proton-migration barrier (direct DFT, CP2K + ABACUS) |

A sixth, **anti-example** spec lives in `../dft_probes_deferred/us_preflight_gate.tm.yaml`
(`kind: USPreflightGate`): it records the `$0` prodromos preflight gate that decided the full
direct-DFT umbrella-sampling PMF was **deferred** (§3.3) and surfaced the MACE-vs-CHGNet
carrier-topology split — a decision record cross-linked to the prodromos gate engine, kept for
honesty rather than as a paper-quotable barrier.

## Cross-references (why these files link the papers)

Each spec carries a `cross_ref` block linking it to (a) **this paper** (KineticTrap, with its repo
and forthcoming Zenodo DOI), (b) the **companion study** *MLIPvsDFT*
(<https://github.com/exopoiesis/mlip-vs-dft-iron-sulfides>) whose foundation-MLIP failure-mode
taxonomy this work builds on, and (c) the **TM-Spec standard** itself. Within the project the
`provenance.parents` field forms a DAG of upstream runs/specs. Together these let a reader who
finds any one artifact — a paper, a repo, or a single spec — discover the rest of the family.

> Note: specs are dated snapshots. A few embedded notes predate the split of the kinetic-trap
> work out of the MLIPvsDFT paper and still say "paper #1"; the `id:` headers carry a Paper #2
> mapping note. The DFT metadynamics lower bound evolved from an earlier wall-corrected ≥0.40 eV
> to the manuscript's ≥0.32 eV (coordination CV, §3.2) — see the manuscript for the canonical value.
