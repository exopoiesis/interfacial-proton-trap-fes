#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NS-3: MDL/BIC model selection {flat M0, single-well M1, double-well M2} on mean-force F'(xi)
      + Cramer-Rao bound on Var(dF_barrier).

Design doc: paper/reviews/KineticTrap/NS3_MDL_BIC_design_2026-06-13.md (info-theorist).
Reuses dft-neb/u_gate/us_preflight_s136.json per-window aggregates (mean_xi_A, F_mean_kJ_mol_A,
sigma_F_kJ_mol_A, trust). NO re-blocking. WLS linear fit (closed form), per-MLIP (mace, chgnet) separate.
"""
import sys, os, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "ns_analysis_2026-06-13")
os.makedirs(OUT, exist_ok=True)
JSON = os.path.join(ROOT, "dft-neb", "u_gate", "us_preflight_s136.json")

EV_PER_KJMOL = 1.0 / 96.485   # kJ/mol -> eV
SIGMA_TARGET = 0.1            # eV
RNG = np.random.default_rng(12345)


def wls_fit(x, y, s, deg):
    """WLS fit of F'(xi) = sum a_j xi^j, deg = polynomial degree of mean-force.
    Returns theta (highest power first, numpy.vander default), chi2, Finv (cov = (Phi^T W Phi)^-1)."""
    Phi = np.vander(x, deg + 1)            # columns: x^deg ... x^0
    W = np.diag(1.0 / s ** 2)
    A = Phi.T @ W @ Phi
    b = Phi.T @ W @ y
    theta = np.linalg.solve(A, b)
    resid = y - Phi @ theta
    chi2 = float(np.sum((resid / s) ** 2))
    Finv = np.linalg.inv(A)
    return theta, chi2, Finv


def barrier_from_coeffs(theta_meanforce):
    """theta = coeffs of F'(xi) (highest power first). PMF A has F'=dA/dxi.
    Find extrema of A (roots of F'), classify by A''=dF'/dxi, compute dF = A(xi_max)-A(xi_min).
    Returns dF in kJ/mol, or None if no proper min<max pair (monotonic PMF => no barrier)."""
    # F'(xi) polynomial coeffs (highest first)
    fp = theta_meanforce
    roots = np.roots(fp) if len(fp) > 1 else np.array([])
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    if len(real_roots) < 2:
        return None
    real_roots = np.sort(real_roots)
    # A'' = derivative of F'
    dfp = np.polyder(fp)
    # PMF A = integral of F': coeffs (antiderivative), highest first
    A_coeffs = np.polyint(fp)  # numpy.polyint integrates, adds 0 constant
    def A(xi):
        return np.polyval(A_coeffs, xi)
    # classify roots: maximum of A where A''<0, minimum where A''>0
    maxima = [r for r in real_roots if np.polyval(dfp, r) < 0]
    minima = [r for r in real_roots if np.polyval(dfp, r) > 0]
    if not maxima or not minima:
        return None
    # barrier = highest max minus the adjacent (lower) min along path. Use global:
    # dF_barrier = A(xi_max) - A(xi_min) for the max/min pair giving the largest positive barrier
    best = None
    for xm in maxima:
        # nearest minimum below the max in A (reactant side)
        lower_mins = [mn for mn in minima if mn < xm] or minima
        for mn in lower_mins:
            dF = A(xm) - A(mn)
            if dF > 0 and (best is None or dF > best):
                best = dF
    return best


def is_monotonic_pmf(theta_meanforce):
    """True if PMF has no interior min<max => no barrier (F' doesn't cross with proper signs)."""
    return barrier_from_coeffs(theta_meanforce) is None


def fit_mlip(windows, mlip):
    W = [w for w in windows[mlip] if w["trust"]]
    x = np.array([w["mean_xi_A"] for w in W])
    y = np.array([w["F_mean_kJ_mol_A"] for w in W])
    s = np.array([w["sigma_F_kJ_mol_A"] for w in W])
    n = len(W)
    const_term = 0.5 * np.sum(np.log(2 * np.pi * s ** 2))

    fits = {}
    for name, deg in [("M0", None), ("M1", 2), ("M2", 4)]:
        if name == "M0":
            theta = np.array([])
            chi2 = float(np.sum((y / s) ** 2))   # F'(xi)=0 => resid=y
            Finv = None
            k = 0
        else:
            theta, chi2, Finv = wls_fit(x, y, s, deg)
            k = deg + 1
        lnL = -0.5 * chi2 - const_term
        BIC = k * np.log(n) - 2 * lnL
        AIC = 2 * k - 2 * lnL
        AICc = AIC + (2 * k * (k + 1) / (n - k - 1)) if (n - k - 1) > 0 else np.inf
        fits[name] = dict(theta=theta.tolist(), k=k, chi2=chi2, lnL=lnL,
                          BIC=BIC, AIC=AIC, AICc=AICc, Finv=Finv,
                          _theta=theta)
    fits["_x"] = x; fits["_y"] = y; fits["_s"] = s; fits["_n"] = n
    return fits


def crb_barrier(fit_entry, n_mc=20000):
    """MC over theta ~ N(theta_hat, Finv) -> recompute barrier -> mean, std (eV).
    Also P(no barrier) = fraction of MC samples with monotonic PMF."""
    theta = fit_entry["_theta"]
    Finv = fit_entry["Finv"]
    if Finv is None or len(theta) == 0:
        return None
    dF0 = barrier_from_coeffs(theta)
    samples = RNG.multivariate_normal(theta, Finv, size=n_mc)
    dFs = []
    n_nobar = 0
    for th in samples:
        b = barrier_from_coeffs(th)
        if b is None:
            n_nobar += 1
        else:
            dFs.append(b)
    dFs = np.array(dFs)
    p_nobar = n_nobar / n_mc
    res = {
        "dF_point_eV": (dF0 * EV_PER_KJMOL) if dF0 is not None else None,
        "dF_mc_mean_eV": float(dFs.mean() * EV_PER_KJMOL) if len(dFs) else None,
        "dF_mc_std_eV": float(dFs.std() * EV_PER_KJMOL) if len(dFs) else None,
        "P_no_barrier": float(p_nobar),
        "n_mc": n_mc,
    }
    return res


def main():
    data = json.load(open(JSON))
    R = data["result"]
    windows = R["windows"]
    pmf = R["pmf"]
    fisher = R["fisher_U3"]

    summary = {"json": os.path.relpath(JSON, ROOT), "sigma_target_eV": SIGMA_TARGET,
               "estimator": R.get("estimator"), "by_mlip": {}}

    print("=" * 72)
    print("NS-3  MDL/BIC {flat,single-well,double-well} + CRB on Var(dF_barrier)")
    print("  WLS on mean-force F'(xi), heteroscedastic 1/sigma_F^2 weights, per-MLIP.")
    print("=" * 72)

    for mlip in ["mace", "chgnet"]:
        fits = fit_mlip(windows, mlip)
        n = fits["_n"]
        BIC0, BIC1, BIC2 = fits["M0"]["BIC"], fits["M1"]["BIC"], fits["M2"]["BIC"]
        AICc0, AICc1, AICc2 = fits["M0"]["AICc"], fits["M1"]["AICc"], fits["M2"]["AICc"]
        dBIC_01 = BIC0 - BIC1   # >6 barrier identifiable (M1 over flat)
        dBIC_12 = BIC1 - BIC2   # >6 double-well identifiable

        # CRB for M1 and M2
        crb_M1 = crb_barrier(fits["M1"])
        crb_M2 = crb_barrier(fits["M2"])

        # model selection
        if dBIC_12 > 6:
            sel = "M2"
        elif dBIC_01 > 6:
            sel = "M1"
        elif dBIC_01 < 2:
            sel = "M0"
        else:
            sel = "M1?"  # gray
        crb_sel = crb_M2 if sel == "M2" else crb_M1

        # AICc agreement check
        bic_best = min([("M0", BIC0), ("M1", BIC1), ("M2", BIC2)], key=lambda t: t[1])[0]
        aicc_best = min([("M0", AICc0), ("M1", AICc1), ("M2", AICc2)], key=lambda t: t[1])[0]
        criteria_agree = (bic_best == aicc_best)

        # verdict: use TRUSTWORTHY barrier height (JSON trapezoid PMF) + JSON fisher sigma for the
        # height-identifiability test; polynomial CRB is ill-conditioned and only used for P(no-barrier).
        dF_poly = crb_sel["dF_point_eV"] if crb_sel and crb_sel["dF_point_eV"] else None
        sig_poly = crb_sel["dF_mc_std_eV"] if crb_sel else None
        p_nobar = crb_sel["P_no_barrier"] if crb_sel else None
        dF = pmf[mlip]["barrier_eV"]                         # trustworthy height
        sig = fisher["sigma_dF_eV"][mlip]                    # trustworthy sigma (trapezoid CRB)
        identifiable = (dBIC_01 > 6 and sel != "M0"
                        and sig < dF / 2 and (p_nobar is None or p_nobar < 0.05))
        if identifiable:
            verdict = "BARRIER_IDENTIFIABLE"
        elif dBIC_01 < 2:
            verdict = "FLAT_ONLY"
        else:
            verdict = "GRAY_NEED_MORE_DATA"

        # how much time/windows for sigma_target (use JSON trapezoid sigma = trustworthy CRB proxy)
        sig_for_plan = fisher["sigma_dF_eV"][mlip]
        mult = (sig_for_plan / SIGMA_TARGET) ** 2  # x window-length or x n-windows (CRB ~ 1/t ~ 1/N)

        # reduced chi2 exposes model mis-specification (chi2/dof; dof = n-k)
        rchi2 = {m: (fits[m]["chi2"] / max(n - fits[m]["k"], 1)) for m in ["M0", "M1", "M2"]}

        entry = {
            "n_trust_windows": n,
            "fits": {m: {kk: fits[m][kk] for kk in ["k", "chi2", "lnL", "BIC", "AIC", "AICc", "theta"]}
                     for m in ["M0", "M1", "M2"]},
            "reduced_chi2": rchi2,
            "dBIC_M0_minus_M1": dBIC_01,
            "dBIC_M1_minus_M2": dBIC_12,
            "selected_model": sel,
            "bic_best": bic_best, "aicc_best": aicc_best, "criteria_agree": criteria_agree,
            "crb_M1": crb_M1, "crb_M2": crb_M2,
            "barrier_polynomial_selected_eV": dF_poly, "sigma_barrier_poly_CRB_eV": sig_poly,
            "barrier_trustworthy_eV": dF, "sigma_barrier_trustworthy_eV": sig,
            "P_no_barrier_selected": p_nobar,
            "json_pmf_barrier_eV": pmf[mlip]["barrier_eV"],
            "json_pmf_sigma_eV": pmf[mlip]["sigma_barrier_eV"],
            "fisher_sigma_dF_eV": fisher["sigma_dF_eV"][mlip],
            "verdict": verdict,
            "plan_multiplier_for_sigma0.1": mult,
            "plan_note": ("x window-length (fixed windows) OR x n-windows in barrier zone (fixed t). "
                          "CRB ~ 1/t ~ 1/N. sigma scales as 1/sqrt(t). Multiplier computed from the "
                          "JSON trapezoid sigma_dF (NOT the polynomial CRB, which is ill-conditioned "
                          "for the cubic over the wide 1.5-3.6 A range)."),
            "barrier_height_caveat": ("Polynomial-integrated barrier (M1/M2) is NOT a reliable height: "
                                      "reduced chi2 >> 1 means the low-order polynomial mis-specifies the "
                                      "oscillating mean-force landscape; spurious edge-extrema inflate dF "
                                      "(esp. CHGNet M1 CRB blow-up). Trustworthy barrier height = JSON "
                                      "trapezoid PMF (mace 0.67, chgnet 0.30 eV). MDL/BIC answers ONLY the "
                                      "SHAPE/identifiability question (flat vs structured), which is robust."),
        }
        summary["by_mlip"][mlip] = entry

        # ---- print ----
        print(f"\n##### {mlip.upper()}  (n_trust = {n} windows) #####")
        print(f"  chi2:  M0={fits['M0']['chi2']:8.2f}  M1={fits['M1']['chi2']:8.2f}  M2={fits['M2']['chi2']:8.2f}")
        print(f"  chi2/dof: M0={rchi2['M0']:7.2f}  M1={rchi2['M1']:7.2f}  M2={rchi2['M2']:7.2f}   "
              f"(>>1 => model mis-specified or sigma_F under-estimated)")
        print(f"  BIC:   M0={BIC0:8.2f}  M1={BIC1:8.2f}  M2={BIC2:8.2f}")
        print(f"  AICc:  M0={AICc0:8.2f}  M1={AICc1:8.2f}  M2={AICc2:8.2f}")
        print(f"  dBIC(M0-M1) = {dBIC_01:7.2f}   ( >6 barrier identifiable ; <2 flat ; 2-6 gray )")
        print(f"  dBIC(M1-M2) = {dBIC_12:7.2f}   ( >6 double-well identifiable )")
        print(f"  BIC-best={bic_best}  AICc-best={aicc_best}  agree={criteria_agree}")
        print(f"  selected model = {sel}")
        if crb_M1:
            print(f"  M1 barrier: dF={crb_M1['dF_point_eV']:.4f} eV  CRB-std={crb_M1['dF_mc_std_eV']:.4f} eV  "
                  f"P(no-barrier)={crb_M1['P_no_barrier']:.3f}")
        if crb_M2:
            print(f"  M2 barrier: dF={crb_M2['dF_point_eV']}  CRB-std={crb_M2['dF_mc_std_eV']}  "
                  f"P(no-barrier)={crb_M2['P_no_barrier']:.3f}")
        print(f"  TRUSTWORTHY barrier (JSON trapezoid PMF) = {dF:.4f} eV  sigma = {sig:.4f} eV  "
              f"(sigma/dF = {sig/dF:.3f}; identifiable-by-height if <0.5)")
        print(f"  plan: to reach sigma<0.1 eV need ~{mult:.2f}x window-length (or x n-windows)")
        print(f"  >>> VERDICT ({mlip}): {verdict}")

    summary["dpi_caveat"] = ("MACE and CHGNet co-trained on MP/PBE -> errors correlated; per-MLIP MDL "
                             "reported separately. Agreement in model choice != validation (shared corpus). "
                             "Final truth on PMF shape = DFT US only.")
    with open(os.path.join(OUT, "ns3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("DPI CAVEAT:", summary["dpi_caveat"])
    print("=" * 72)
    return summary


if __name__ == "__main__":
    main()
