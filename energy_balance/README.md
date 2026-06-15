# Electrochemical energy balance (§3.5)

`energy_balance_R1_dpH.py` computes whether a transmembrane pH gradient supplies enough
electromotive force to drive CO₂-to-formate reduction (reaction R1) across the bilayer.

## On the two "margin" definitions (avoid confusion)

The manuscript headline is **+61 % margin at ΔpH = 6** — this is the margin **over the total
required EMF**:

```
available EMF (ΔpH=6)  = 0.0592 × 6        = 0.355 V
required  EMF          = ΔG⁰/nF + losses   = 171 + 23 + 20 + 6 mV = 0.220 V
margin (manuscript)    = (0.355 − 0.220) / 0.220 = +61 %
```

The script's per-row **`Margin`** column uses a **different base** — the margin of the *net* free
energy over the bare thermodynamic cost ΔG⁰ (≈ +78.9 % for the same ΔpH = 6 scenario). Both are
internally consistent; they answer different questions. The **base-independent, directly
comparable quantity is the available EMF, 0.355 V at ΔpH = 6**, which both the script and the
manuscript report identically, and the feasibility verdict (ΔpH = 6 is comfortably feasible;
ΔpH_min ≈ 3.72 for mackinawite, 10.09 for greigite).

The automated test (`tests/test_energy_balance.py`) checks the base-independent quantity
(0.355 V) and feasibility, not the percentage (whose base differs).
