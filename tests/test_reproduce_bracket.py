"""Reproducibility test: the bundled per-window colvar data must reconstruct the
kinetic-trap bracket of the manuscript (§3.3) via WHAM.

CPU-only (numpy). Runs the repo's reproduce_bracket.py on the included data and
asserts the MACE and CHGNet detachment-barrier saddles fall in the expected ranges.
"""
import importlib.util
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent
MOD_PATH = REPO / "mlip_umbrella_sampling" / "reproduce_bracket.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("reproduce_bracket", MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bracket_reproduces_manuscript():
    mod = _load_module()
    out = mod.main([])                       # runs WHAM on ./data
    mace_dF = out["mace"][1]
    chgnet_dF = out["chgnet"][1]
    # MACE: well-defined saddle ~0.80 eV (manuscript)
    assert 0.70 <= mace_dF <= 0.90, f"MACE dF# = {mace_dF:.3f} eV outside [0.70, 0.90]"
    # CHGNet: monotonic rise, read as 0.32 (transfer) .. 0.40 (plateau) eV
    assert 0.30 <= chgnet_dF <= 0.50, f"CHGNet dF# = {chgnet_dF:.3f} eV outside [0.30, 0.50]"
    # the factor-~2 separation (the paper's headline disagreement) must hold
    assert mace_dF - chgnet_dF >= 0.25, f"MACE-CHGNet gap {mace_dF - chgnet_dF:.3f} eV too small"


def test_data_present():
    for engine in ("mace", "chgnet"):
        windows = list((REPO / "mlip_umbrella_sampling" / "data" / engine).glob("window_*/colvar.dat"))
        assert len(windows) == 18, f"{engine}: expected 18 windows, found {len(windows)}"
