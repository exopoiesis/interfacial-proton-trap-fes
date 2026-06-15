"""Smoke + value test for the pH-gradient electrochemical energy balance (§3.5).

Runs the script and checks the robust, definition-independent quantity:
the available EMF at dpH=6 is 0.355 V and the dpH=6 scenario is feasible.
(The headline "+61% margin" is margin over the *required EMF* (135/220 mV);
the script's per-row "Margin" uses a different base — see energy_balance/README.md.)
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "energy_balance" / "energy_balance_R1_dpH.py"


def test_energy_balance_runs_and_emf():
    r = subprocess.run([sys.executable, str(SCRIPT)],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120, cwd=str(REPO))
    assert r.returncode == 0, f"script failed:\n{r.stderr[-2000:]}"
    out = r.stdout
    # available EMF at dpH=6 (matches manuscript §3.5: 0.355 V) — base-independent
    assert "0.355" in out, "available EMF 0.355 V at dpH=6 not found in output"
    # dpH=6 must be feasible (a 'YES' feasibility row present)
    assert "YES" in out, "no feasible (YES) scenario reported"
