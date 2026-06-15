import Mathlib

/-!
# Machine-checked lemmas for the TM6v3-min kinetic-trap model (Paper #2 §3.4)

Model (manuscript Eqs 1-3), state `(A, M, Fe)`, all parameters positive, Hill `n = 2`:

  dA/dt = k1 * (f1*M/(KMf+M)) * (A^2/(Ka^2+A^2)) - kA*A
  dM/dt = km*Fe*A - kM*M
  dFe/dt = kg*A + S - km*Fe*A - kF*Fe

Theorems (cf. `paper/KineticTrap/THEOREM_SPEC_kinetic_trap.md`):
* `production_on_inactive_sheet`, `dA_on_inactive_sheet` (L4 / consilium fix):
  production vanishes on the inactive sheet `M = 0` for EVERY `A` and ANY Hill
  exponent (only `phi 0 = 0` is used) -- corrects the earlier "n>=2 load-bearing".
* `E0_isFixedPoint` (L3): `E0 = (0,0,S/kF)` is a stationary point.
* `J0_upper_zero`, `J0_diag_neg` (L1): Jacobian at `E0` is lower-triangular with
  strictly negative diagonal `(-kA,-kM,-kF)`.
* `two_roots_of_sign_pattern` (L2): a function continuous on `[a,c]` with sign
  pattern `(-,+,-)` has >= 2 distinct roots -- existence core of bistability.
* `model_bistable_roots`: the model's reduced map `g` has >= 2 interior fixed
  points under the ignition sign pattern.
* `continuousOn_g`: `g` is continuous on `[a,b]` for `0 < a` (discharges the
  continuity hypothesis of `model_bistable_roots`).
* `dM_km_zero`, `membrane_decay_unique`, `membrane_decay_tendsto_zero`,
  `production_le_membrane` (Thm D, s168): with `km = 0` the membrane equation
  decouples to `dM = -kM*M`; its unique solution `M(t)=M0*exp(-kM t)` (proved from
  first principles) tends to 0, and the gated production is dominated by the
  membrane mass -- self-assembly of the boundary is necessary for the active state.
  (The residual `A(t)->0` is pen-and-paper: mathlib Gronwall bounds growth, not
  decay; see `proofs/VERIFICATION_REPORT.md`.)
-/

namespace TTProofs
open Set

/-- Strictly-positive parameters of TM6v3-min. -/
structure Params where
  k1 : ℝ
  f1 : ℝ
  Ka : ℝ
  KMf : ℝ
  km : ℝ
  kg : ℝ
  S : ℝ
  kA : ℝ
  kM : ℝ
  kF : ℝ
  k1_pos : 0 < k1
  f1_pos : 0 < f1
  Ka_pos : 0 < Ka
  KMf_pos : 0 < KMf
  km_pos : 0 < km
  kg_pos : 0 < kg
  S_pos  : 0 < S
  kA_pos : 0 < kA
  kM_pos : 0 < kM
  kF_pos : 0 < kF

/-- Michaelis-Menten membrane transport `phi(M) = f1*M/(KMf+M)`, `phi 0 = 0`. -/
noncomputable def phi (p : Params) (M : ℝ) : ℝ := p.f1 * M / (p.KMf + M)

/-- Hill production factor with `n = 2`: `H(A) = A^2/(Ka^2+A^2)`, `H 0 = 0`. -/
noncomputable def Hill (p : Params) (A : ℝ) : ℝ := A ^ 2 / (p.Ka ^ 2 + A ^ 2)

/-- Autocatalytic production `R(A,M) = k1 * phi(M) * H(A)`. -/
noncomputable def production (p : Params) (A M : ℝ) : ℝ := p.k1 * phi p M * Hill p A

noncomputable def dA (p : Params) (A M _Fe : ℝ) : ℝ := production p A M - p.kA * A
noncomputable def dM (p : Params) (A M Fe : ℝ) : ℝ := p.km * Fe * A - p.kM * M
noncomputable def dFe (p : Params) (A _M Fe : ℝ) : ℝ :=
  p.kg * A + p.S - p.km * Fe * A - p.kF * Fe

@[simp] lemma phi_zero (p : Params) : phi p 0 = 0 := by simp [phi]
@[simp] lemma Hill_zero (p : Params) : Hill p 0 = 0 := by simp [Hill, pow_two]

/-! ## L4 -- inactive sheet, independent of the Hill exponent (consilium fix) -/

/-- On the inactive sheet `M = 0` the autocatalytic production vanishes for every
`A`, using only `phi 0 = 0`. Independent of the Hill exponent `n`. -/
lemma production_on_inactive_sheet (p : Params) (A : ℝ) : production p A 0 = 0 := by
  simp [production]

/-- Hence `dA = -kA*A` on the inactive sheet: a pure linear decay in `A`, which is
what makes `E0` an attractor regardless of cooperativity. -/
lemma dA_on_inactive_sheet (p : Params) (A Fe : ℝ) : dA p A 0 Fe = - p.kA * A := by
  simp only [dA, production_on_inactive_sheet]
  ring

/-! ## Absorbing hyperplane `A = 0` -- autocatalysis cannot restart from a zero seed
(statmech consilium, s-2026-06-12). The deterministic shadow of the stochastic
"point of no return": once the autocatalytic product `A` is exhausted, it can never
be regenerated, because production requires a nonzero seed (`H 0 = 0`). -/

/-- Production vanishes when the seed `A = 0`, using only `H 0 = 0` -- independent of
membrane mass `M` and of the Hill exponent. With no product seed there is no
autocatalysis whatever the membrane. -/
lemma production_on_zero_seed (p : Params) (M : ℝ) : production p 0 M = 0 := by
  simp [production]

/-- Hence `dA = 0` on the hyperplane `A = 0`, so `{A = 0}` is forward-invariant: a
trajectory that reaches `A = 0` stays there. Autocatalysis is structurally off at the
seedless state -- the inactive trap is a point of no return for the closed model. -/
lemma dA_on_zero_seed (p : Params) (M Fe : ℝ) : dA p 0 M Fe = 0 := by
  simp only [dA, production_on_zero_seed]
  ring

/-! ## L3 -- the inactive (quiescent) fixed point -/

/-- `E0 = (0, 0, S/kF)` is a stationary point of the vector field. -/
theorem E0_isFixedPoint (p : Params) :
    dA p 0 0 (p.S / p.kF) = 0 ∧ dM p 0 0 (p.S / p.kF) = 0
      ∧ dFe p 0 0 (p.S / p.kF) = 0 := by
  have hkF : p.kF ≠ 0 := ne_of_gt p.kF_pos
  refine ⟨?_, ?_, ?_⟩
  · simp [dA, production]
  · simp [dM]
  · simp only [dFe]
    field_simp
    ring

/-! ## L1 -- Jacobian at E0 is lower-triangular with negative diagonal -/

/-- Jacobian of `(dA,dM,dFe)` at `E0=(0,0,S/kF)` (rows `A,M,Fe`). The upper
off-diagonal entries vanish because at `E0` both `phi(0)=0` and `H(0)=0` kill the
autocatalytic couplings into the `A`-row. -/
noncomputable def J0 (p : Params) : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ -p.kA,                        0,      0;
      p.km * (p.S / p.kF),         -p.kM,   0;
      p.kg - p.km * (p.S / p.kF),   0,     -p.kF ]

/-- The strictly-upper entries of `J0` vanish (lower-triangular). -/
lemma J0_upper_zero (p : Params) :
    J0 p 0 1 = 0 ∧ J0 p 0 2 = 0 ∧ J0 p 1 2 = 0 := by
  refine ⟨?_, ?_, ?_⟩ <;>
    simp [J0, Matrix.cons_val_zero, Matrix.cons_val_one,
      Matrix.cons_val', Matrix.empty_val',
      Matrix.cons_val_fin_one, Matrix.of_apply]

/-- The diagonal of `J0` is strictly negative; with lower-triangularity this gives
spectrum `{-kA,-kM,-kF}` in the open left half-plane (Hurwitz ⇒ `E0` LAS). -/
lemma J0_diag_neg (p : Params) : J0 p 0 0 < 0 ∧ J0 p 1 1 < 0 ∧ J0 p 2 2 < 0 := by
  refine ⟨?_, ?_, ?_⟩
  · simpa [J0, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val', Matrix.head_fin_const, Matrix.empty_val',
      Matrix.cons_val_fin_one, Matrix.of_apply] using neg_lt_zero.mpr p.kA_pos
  · simpa [J0, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val', Matrix.head_fin_const, Matrix.empty_val',
      Matrix.cons_val_fin_one, Matrix.of_apply] using neg_lt_zero.mpr p.kM_pos
  · simpa [J0, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons,
      Matrix.cons_val', Matrix.head_fin_const, Matrix.empty_val',
      Matrix.cons_val_fin_one, Matrix.of_apply] using neg_lt_zero.mpr p.kF_pos

/-! ## L2 -- two distinct roots from a (-,+,-) sign pattern (existence core) -/

/-- **Existence core of bistability.** A function continuous on `[a,c]`, negative
at `a`, positive at `b`, negative at `c` (`a < b < c`), has at least two distinct
roots: one in `(a,b)` and one in `(b,c)`. -/
theorem two_roots_of_sign_pattern {g : ℝ → ℝ} {a b c : ℝ}
    (hab : a < b) (hbc : b < c) (hg : ContinuousOn g (Icc a c))
    (ha : g a < 0) (hb : 0 < g b) (hc : g c < 0) :
    ∃ x₁ x₂, x₁ ∈ Ioo a b ∧ x₂ ∈ Ioo b c ∧ g x₁ = 0 ∧ g x₂ = 0 ∧ x₁ ≠ x₂ := by
  have hsub1 : Icc a b ⊆ Icc a c := Icc_subset_Icc_right hbc.le
  have hsub2 : Icc b c ⊆ Icc a c := Icc_subset_Icc_left hab.le
  have h1 : (0 : ℝ) ∈ Ioo (g a) (g b) := ⟨ha, hb⟩
  obtain ⟨x₁, hx₁mem, hx₁⟩ := intermediate_value_Ioo hab.le (hg.mono hsub1) h1
  have h2 : (0 : ℝ) ∈ Ioo (g c) (g b) := ⟨hc, hb⟩
  obtain ⟨x₂, hx₂mem, hx₂⟩ := intermediate_value_Ioo' hbc.le (hg.mono hsub2) h2
  exact ⟨x₁, x₂, hx₁mem, hx₂mem, hx₁, hx₂, ne_of_lt (lt_trans hx₁mem.2 hx₂mem.1)⟩

/-! ## Reduced 1-D map of the model and bistability of interior fixed points -/

/-- Steady-state `Fe(A)` from `dFe = 0`. -/
noncomputable def Fe_ss (p : Params) (A : ℝ) : ℝ := (p.kg * A + p.S) / (p.km * A + p.kF)

/-- Steady-state `M(A)` from `dM = 0`. -/
noncomputable def M_ss (p : Params) (A : ℝ) : ℝ := (p.km / p.kM) * Fe_ss p A * A

/-- Reduced 1-D ignition map: `dA = 0` along the steady manifold of `M,Fe`. Its
positive roots are exactly the interior fixed points of the full system. -/
noncomputable def g (p : Params) (A : ℝ) : ℝ :=
  p.k1 * phi p (M_ss p A) * Hill p A - p.kA * A

/-- On `A ≥ a > 0` all denominators are strictly positive, so `g` is continuous on
`[a,b]`. Discharges the continuity hypothesis of `model_bistable_roots`. -/
lemma continuousOn_g (p : Params) {a b : ℝ} (ha : 0 < a) :
    ContinuousOn (g p) (Icc a b) := by
  have hFe : ContinuousOn (Fe_ss p) (Icc a b) := by
    unfold Fe_ss
    apply ContinuousOn.div
    · fun_prop
    · fun_prop
    · intro A hA
      exact ne_of_gt (add_pos (mul_pos p.km_pos (lt_of_lt_of_le ha hA.1)) p.kF_pos)
  have hM : ContinuousOn (M_ss p) (Icc a b) := by
    unfold M_ss
    exact (continuousOn_const.mul hFe).mul continuousOn_id
  have hMnn : ∀ A ∈ Icc a b, 0 ≤ M_ss p A := by
    intro A hA
    have hA0 : 0 ≤ A := (lt_of_lt_of_le ha hA.1).le
    have hFe0 : 0 ≤ Fe_ss p A := by
      apply div_nonneg
      · exact add_nonneg (mul_nonneg p.kg_pos.le hA0) p.S_pos.le
      · exact add_nonneg (mul_nonneg p.km_pos.le hA0) p.kF_pos.le
    have hc : 0 ≤ p.km / p.kM := (div_pos p.km_pos p.kM_pos).le
    exact mul_nonneg (mul_nonneg hc hFe0) hA0
  have hphi : ContinuousOn (fun A => phi p (M_ss p A)) (Icc a b) := by
    unfold phi
    apply ContinuousOn.div
    · exact continuousOn_const.mul hM
    · exact continuousOn_const.add hM
    · intro A hA
      exact ne_of_gt (add_pos_of_pos_of_nonneg p.KMf_pos (hMnn A hA))
  have hHill : ContinuousOn (Hill p) (Icc a b) := by
    unfold Hill
    apply ContinuousOn.div
    · fun_prop
    · fun_prop
    · intro A _
      exact ne_of_gt (add_pos_of_pos_of_nonneg (pow_pos p.Ka_pos 2) (sq_nonneg A))
  unfold g
  exact ((continuousOn_const.mul hphi).mul hHill).sub
    (continuousOn_const.mul continuousOn_id)

/-- **Conditional bistability of the model.** If the reduced ignition map `g`
realizes the sign pattern `(-,+,-)` at some `0 < a < b < c` (the concrete form of
the ignition condition `(⋆)`), the full TM6v3-min system has at least two distinct
interior fixed points `A₁ ∈ (a,b)`, `A₂ ∈ (b,c)`. With the always-stable inactive
fixed point `E0` (`E0_isFixedPoint` + `J0_diag_neg`) this is the existence content
of bistability (Thm C). Continuity is discharged by `continuousOn_g`. -/
theorem model_bistable_roots (p : Params) {a b c : ℝ}
    (ha0 : 0 < a) (hab : a < b) (hbc : b < c)
    (hga : g p a < 0) (hgb : 0 < g p b) (hgc : g p c < 0) :
    ∃ A₁ A₂, A₁ ∈ Ioo a b ∧ A₂ ∈ Ioo b c ∧ g p A₁ = 0 ∧ g p A₂ = 0 ∧ A₁ ≠ A₂ :=
  two_roots_of_sign_pattern hab hbc (continuousOn_g p ha0) hga hgb hgc

/-! ## Thm D -- membrane self-assembly off (`km = 0`) ⇒ exponential collapse

When the membrane-precipitation rate `km = 0`, the membrane equation decouples
(`dM_km_zero`: `dM = -kM*M`, independent of `A,Fe`); its every solution is the
exponential `M(t) = M(0)·e^{-kM t}` (`membrane_decay_unique`, proved from first
principles: `w(t)=M(t)·e^{kM t}` has zero derivative hence is constant -- no fragile
ODE-API names), which tends to `0` (`membrane_decay_tendsto_zero`). Moreover the
gated autocatalytic production is dominated by the membrane mass
(`production_le_membrane`), so as `M→0` the drive vanishes and only `-kA·A` remains.
The residual `A(t)→0` (Gronwall on the linear remainder) is pen-and-paper (mathlib's
Gronwall bounds growth, not decay); see `VERIFICATION_REPORT.md`. -/

/-- With `km = 0` the membrane equation decouples to pure linear decay `dM = -kM*M`. -/
lemma dM_km_zero (p : Params) (A M Fe : ℝ) (h : p.km = 0) :
    dM p A M Fe = -(p.kM) * M := by
  simp only [dM, h]; ring

/-- **Membrane-decay uniqueness.** Any solution of the decoupled membrane equation
`M' = -kM*M` is the exponential `M(t)=M(0)·e^{-kM t}`. Proof: `w(t)=M(t)·e^{kM t}`
has identically zero derivative, hence is constant. (Self-contained; uses only the
product rule and `is_const_of_deriv_eq_zero`.) -/
theorem membrane_decay_unique (p : Params) (M : ℝ → ℝ)
    (hM : ∀ t, HasDerivAt M (-(p.kM) * M t) t) :
    ∀ t, M t = M 0 * Real.exp (-(p.kM) * t) := by
  have hexp : ∀ t : ℝ,
      HasDerivAt (fun s => Real.exp (p.kM * s)) (Real.exp (p.kM * t) * p.kM) t := by
    intro t
    have h1 : HasDerivAt (fun s : ℝ => p.kM * s) p.kM t := by
      simpa using (hasDerivAt_id t).const_mul p.kM
    simpa using h1.exp
  have hw : ∀ t : ℝ, HasDerivAt (fun s => M s * Real.exp (p.kM * s)) 0 t := by
    intro t
    -- type-ascribe the product rule into its beta-reduced form (defeq), then
    -- rewrite the derivative value to 0.
    have hprod : HasDerivAt (fun s => M s * Real.exp (p.kM * s))
        (-(p.kM) * M t * Real.exp (p.kM * t)
          + M t * (Real.exp (p.kM * t) * p.kM)) t := (hM t).mul (hexp t)
    have hz : -(p.kM) * M t * Real.exp (p.kM * t)
        + M t * (Real.exp (p.kM * t) * p.kM) = 0 := by ring
    rwa [hz] at hprod
  have hdiff : Differentiable ℝ (fun s => M s * Real.exp (p.kM * s)) :=
    fun t => (hw t).differentiableAt
  intro t
  have hconst : (fun s => M s * Real.exp (p.kM * s)) t
      = (fun s => M s * Real.exp (p.kM * s)) 0 :=
    is_const_of_deriv_eq_zero hdiff (fun x => (hw x).deriv) t 0
  simp only [mul_zero, Real.exp_zero, mul_one] at hconst
  have hE : Real.exp (-(p.kM) * t) = (Real.exp (p.kM * t))⁻¹ := by
    rw [neg_mul, Real.exp_neg]
  rw [hE]
  have hexp_ne : Real.exp (p.kM * t) ≠ 0 := Real.exp_ne_zero _
  field_simp
  linarith [hconst]

/-- The membrane-decay profile tends to `0` (since `kM > 0`): the active membrane
cannot persist once self-assembly stops. -/
theorem membrane_decay_tendsto_zero (p : Params) (M0 : ℝ) :
    Filter.Tendsto (fun t => M0 * Real.exp (-(p.kM) * t)) Filter.atTop (nhds 0) := by
  have hkM : Filter.Tendsto (fun t : ℝ => p.kM * t) Filter.atTop Filter.atTop :=
    Filter.Tendsto.const_mul_atTop p.kM_pos Filter.tendsto_id
  have hneg : Filter.Tendsto (fun t : ℝ => -(p.kM * t)) Filter.atTop Filter.atBot :=
    Filter.tendsto_neg_atTop_atBot.comp hkM
  have hexp0 : Filter.Tendsto (fun t : ℝ => Real.exp (-(p.kM * t))) Filter.atTop (nhds 0) :=
    Real.tendsto_exp_atBot.comp hneg
  have := hexp0.const_mul M0
  simpa [neg_mul, mul_zero] using this

/-- The gated autocatalytic production is dominated by the membrane mass:
`production ≤ (k1*f1/KMf)·M` for `M ≥ 0` (using `phi(M) ≤ f1*M/KMf` and `0 ≤ H ≤ 1`).
Hence `M→0` drives production to `0`, the bridge from membrane collapse to `A`-decay. -/
lemma production_le_membrane (p : Params) {A M : ℝ} (hM : 0 ≤ M) :
    production p A M ≤ p.k1 * p.f1 / p.KMf * M := by
  have hden : 0 < p.KMf + M := add_pos_of_pos_of_nonneg p.KMf_pos hM
  have hphi_nonneg : 0 ≤ phi p M :=
    div_nonneg (mul_nonneg p.f1_pos.le hM) hden.le
  have hKMf_ne : p.KMf ≠ 0 := ne_of_gt p.KMf_pos
  have hden_ne : p.KMf + M ≠ 0 := ne_of_gt hden
  have hphi_le : phi p M ≤ p.f1 * M / p.KMf := by
    rw [← sub_nonneg, phi]
    have key : p.f1 * M / p.KMf - p.f1 * M / (p.KMf + M)
        = p.f1 * M * M / (p.KMf * (p.KMf + M)) := by
      field_simp; ring
    rw [key]
    exact div_nonneg (mul_nonneg (mul_nonneg p.f1_pos.le hM) hM)
      (mul_nonneg p.KMf_pos.le hden.le)
  have hKa2 : (0:ℝ) < p.Ka ^ 2 + A ^ 2 :=
    add_pos_of_pos_of_nonneg (pow_pos p.Ka_pos 2) (sq_nonneg A)
  have hKa2_ne : p.Ka ^ 2 + A ^ 2 ≠ 0 := ne_of_gt hKa2
  have hHill_nonneg : 0 ≤ Hill p A := div_nonneg (sq_nonneg A) hKa2.le
  have hHill_le_one : Hill p A ≤ 1 := by
    rw [← sub_nonneg, Hill]
    have key : (1:ℝ) - A ^ 2 / (p.Ka ^ 2 + A ^ 2)
        = p.Ka ^ 2 / (p.Ka ^ 2 + A ^ 2) := by
      field_simp; ring
    rw [key]
    exact div_nonneg (sq_nonneg p.Ka) hKa2.le
  have hb_nonneg : 0 ≤ p.k1 * (p.f1 * M / p.KMf) :=
    mul_nonneg p.k1_pos.le (div_nonneg (mul_nonneg p.f1_pos.le hM) p.KMf_pos.le)
  calc production p A M = p.k1 * phi p M * Hill p A := rfl
    _ ≤ p.k1 * (p.f1 * M / p.KMf) * 1 :=
        mul_le_mul (mul_le_mul_of_nonneg_left hphi_le p.k1_pos.le)
          hHill_le_one hHill_nonneg hb_nonneg
    _ = p.k1 * p.f1 / p.KMf * M := by ring

/-! ## Axiom audit -- confirm proofs are `sorry`-free (machine-checked).
Each should report only the standard Mathlib axioms `propext`,
`Classical.choice`, `Quot.sound` -- and crucially NOT `sorryAx`. -/

#print axioms production_on_zero_seed
#print axioms dA_on_zero_seed
#print axioms E0_isFixedPoint
#print axioms J0_upper_zero
#print axioms J0_diag_neg
#print axioms two_roots_of_sign_pattern
#print axioms continuousOn_g
#print axioms model_bistable_roots
#print axioms dM_km_zero
#print axioms membrane_decay_unique
#print axioms membrane_decay_tendsto_zero
#print axioms production_le_membrane

end TTProofs
