"""Validate every TM-Spec YAML: well-formed, required fields, cross-linked.

CPU-only, no external deps beyond PyYAML. Run: pytest tests/
"""
from pathlib import Path
import yaml
import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC_FILES = sorted(REPO.glob("tm-spec/*.tm.yaml")) + [
    REPO / "dft_probes_deferred" / "us_preflight_gate.tm.yaml"
]


def test_specs_exist():
    assert len(SPEC_FILES) == 7, f"expected 7 specs, found {len(SPEC_FILES)}"


@pytest.mark.parametrize("path", SPEC_FILES, ids=lambda p: p.name)
def test_spec_well_formed_and_cross_linked(path):
    d = yaml.safe_load(path.read_text(encoding="utf-8"))
    # core fields
    assert str(d.get("spec", "")).startswith("tm-spec/"), f"{path.name}: bad/missing spec"
    assert d.get("kind"), f"{path.name}: missing kind"
    assert d.get("id"), f"{path.name}: missing id"
    # schema url points to the canonical home (not github.io)
    assert "github.io" not in str(d.get("schema_url", "")), f"{path.name}: stale github.io schema_url"
    # cross-reference block links the paper family
    xref = d.get("cross_ref") or {}
    assert "paper" in xref, f"{path.name}: missing cross_ref.paper"
    assert xref["paper"].get("repo", "").endswith("interfacial-proton-trap-fes"), \
        f"{path.name}: cross_ref.paper.repo not this repo"
    related = {r.get("name") for r in (xref.get("related") or [])}
    assert "tm-spec" in related, f"{path.name}: cross_ref.related missing tm-spec standard"


def test_ids_unique():
    ids = [yaml.safe_load(p.read_text(encoding="utf-8"))["id"] for p in SPEC_FILES]
    assert len(ids) == len(set(ids)), f"duplicate spec ids: {ids}"


def test_no_internal_instance_labels_in_ids():
    # the public repo must not leak internal worker/instance labels (w1/w2/w3/w4) in ids
    import re
    for p in SPEC_FILES:
        sid = yaml.safe_load(p.read_text(encoding="utf-8"))["id"]
        assert not re.search(r"\bw[1-4]\b", sid), f"{p.name}: instance label in id {sid}"
