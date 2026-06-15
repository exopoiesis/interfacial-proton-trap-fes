#!/usr/bin/env python3
"""
DFT NEB: H diffusion in pentlandite (Fe,Ni)9S8, conventional cell (68 atoms).
ABACUS PW GPU + ASE NEB. Vacancy-mediated mechanism.

**K-MESH VERSION (s79):** kpts=(2,2,2) instead of Gamma-only.
Purpose: resolve Gamma-only vs size-effect question.
  - Conv Gamma: E_a = 0.442 eV (s79)
  - Prim k-mesh: E_a = 0.900 (ABACUS LCAO), 1.115 (GPAW PW)
  - If k-mesh conv ~ 0.9-1.1 -> Gamma artifact
  - If k-mesh conv ~ 0.44 -> real size effect

Two-phase NEB:
  Phase 1: climb=False, fmax < 0.3, max 200 steps
  Phase 2: climb=True,  fmax < 0.05, max 300 steps
"""

import json
import sys
import time
import traceback
import numpy as np
from pathlib import Path

from ase import Atom
from ase.io import read, write
from ase.spacegroup import crystal
from ase.geometry import get_distances
from ase.mep import NEB
from ase.optimize import FIRE
from ase.constraints import FixAtoms

# ABACUS PW GPU: abacuslite (ASE interface, идентично v3 / vacancy скриптам)
ABACUS_ASE_PATH = "/opt/abacus-develop-3.9.0.26/interfaces/ASE_interface"
if ABACUS_ASE_PATH not in sys.path:
    sys.path.insert(0, ABACUS_ASE_PATH)

from abacuslite import Abacus, AbacusProfile
print("abacuslite imported OK", flush=True)

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
WORK_DIR = Path("/workspace/neb_pent_conv_kpts")
RESULTS = Path("/workspace/results")
WORK_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

PP_DIR = "/opt/sg15_pp"   # в GPU образе infra-abacus-gpu

# ---------------------------------------------------------------------------
# Параметры NEB
# ---------------------------------------------------------------------------
N_IMAGES = 5
K_SPRING = 0.05           # мягче 0.1 (урок s76)
FMAX_RELAX = 0.05
MAX_STEPS_RELAX = 100
FMAX_PHASE1 = 0.3         # двухфазный: сначала без CI
MAX_STEPS_PHASE1 = 200
FMAX_PHASE2 = 0.05        # потом CI-NEB
MAX_STEPS_PHASE2 = 300

# Timeout для subprocess внутри ABACUS (8ч, урок s76)
SUBPROCESS_TIMEOUT = 43200  # 12h (kpts=(2,2,2) slower than Gamma)

# Cross-verify ссылки
E_A_GPAW = 1.115    # GPAW PBE PW350, примитивная ячейка
E_A_MACE_PRIM = 0.96
E_A_MACE_CONV = 1.43


# ---------------------------------------------------------------------------
# NumpyEncoder (урок s72 -- numpy типы ломают json.dump)
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ---------------------------------------------------------------------------
# Resume: сохраняем позиции images после каждого NEB шага
# ---------------------------------------------------------------------------
def save_images(images, tag="phase1"):
    """Сохраняем все NEB images в XYZ для resume."""
    for k, img in enumerate(images):
        write(str(WORK_DIR / f"neb_{tag}_img{k:02d}.xyz"), img)


def load_images_if_exist(n_images, tag="phase1"):
    """Возвращает список Atoms если все файлы есть, иначе None."""
    files = [WORK_DIR / f"neb_{tag}_img{k:02d}.xyz" for k in range(n_images + 2)]
    if all(f.exists() for f in files):
        imgs = [read(str(f)) for f in files]
        print(f"  Resume: loaded {len(imgs)} images from {tag}")
        return imgs
    return None


# ---------------------------------------------------------------------------
# ABACUS PW GPU профиль и калькулятор
# ---------------------------------------------------------------------------
# Set OMP env globally (AbacusProfile command cannot contain env vars)
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

profile = AbacusProfile(
    # GPU mode: abacus = symlink to abacus_2g (CUDA PW FFT)
    # NOTE: do NOT pass omp_num_threads -- AbacusProfile prepends it
    # to command string causing FileNotFoundError. Use os.environ instead.
    command="abacus",
    pseudo_dir=PP_DIR,
    # PW mode: no orbital files needed
)


def make_calc(label="neb"):
    """ABACUS PW GPU калькулятор.

    basis_type=pw: GPU-ускоренный PW, нет basissets, нет orbital_dir.
    nspin=1: пентландит Паули-парамагнитен при 25°C (T >> T_C ~ 0 K).
    ecutwfc=60 Ry: эквивалент QE 60 Ry (816 eV) для SG15 ONCV.
    SCF: sigma=0.05, scf_nmax=500, mixing_beta=0.2 -- проверены в v3.
    """
    return Abacus(
        profile=profile,
        directory=str(WORK_DIR / label),
        pseudopotentials={
            "Fe": "Fe_ONCV_PBE-1.2.upf",
            "Ni": "Ni_ONCV_PBE-1.2.upf",
            "S":  "S_ONCV_PBE-1.2.upf",
            "H":  "H_ONCV_PBE-1.2.upf",
        },
        # PW mode: basissets НЕ нужны
        kpts={
            'nk': [2, 2, 2],
            'kshift': [0, 0, 0],
            'gamma-centered': True,
            'mode': 'mp-sampling',
        },
        inp={
            'basis_type': 'pw',
            'calculation': 'scf',
            'nspin': 1,            # парамагнитен
            'ecutwfc': 60,         # Ry, эквивалент 816 eV
            'smearing_method': 'gaussian',
            'smearing_sigma': 0.05,
            'scf_thr': 1e-6,
            'scf_nmax': 500,
            'mixing_type': 'broyden',
            'mixing_beta': 0.2,
            'mixing_ndim': 12,
            'cal_force': 1,
            'cal_stress': 0,
            'symmetry': 0,
        },
    )


# ---------------------------------------------------------------------------
# Построение конвенциональной ячейки
# ---------------------------------------------------------------------------
def build_pentlandite_conv():
    """
    Pentlandite (Fe,Ni)9S8, Fm-3m (#225), конвенциональная ячейка 68 атомов.

    Wyckoff:
      4b  (0.5, 0.5, 0.5)           -> 4 Fe  (тетраэдрические ямы)
      32f (0.125, 0.125, 0.125)      -> 32 Fe/Ni (основная сетка)
      8c  (0.25, 0.25, 0.25)         -> 8 S
      24e (0.25, 0.0, 0.0)           -> 24 S
    Итого: 36 (Fe+Ni) + 32 S = 68 атомов.

    Состав Fe20Ni16S32:
      4b  -> 4 Fe
      32f -> первые 16 = Fe, последние 16 = Ni  (случайный порядок)
      8c  -> 8 S
      24e -> 24 S
    """
    a = 10.044  # Å, стандартное значение для (Fe,Ni)9S8
    atoms = crystal(
        symbols=['Fe', 'Fe', 'S', 'S'],
        basis=[
            (0.5, 0.5, 0.5),        # 4b  Fe
            (0.125, 0.125, 0.125),   # 32f Fe (заменим часть на Ni)
            (0.25, 0.25, 0.25),      # 8c  S
            (0.25, 0.0, 0.0),        # 24e S
        ],
        spacegroup=225,
        cellpar=[a, a, a, 90, 90, 90],
        primitive_cell=False,
    )

    syms = atoms.get_chemical_symbols()

    # Разделяем Fe: 4b (ближе к (0.5,0.5,0.5)) и 32f (ближе к (0.125,...))
    fe_all = [i for i, s in enumerate(syms) if s == 'Fe']
    # 4b: 4 атома; 32f: 32 атома -- всего 36 Fe из crystal()
    # crystal генерирует 4b первыми (по Wyckoff множественности: 4 < 32)
    fe_4b = fe_all[:4]
    fe_32f = fe_all[4:]
    assert len(fe_4b) == 4,  f"Expected 4 Fe on 4b, got {len(fe_4b)}"
    assert len(fe_32f) == 32, f"Expected 32 Fe on 32f, got {len(fe_32f)}"

    # Последние 16 из 32f -> Ni (Fe20Ni16S32)
    for i in fe_32f[16:]:
        syms[i] = 'Ni'
    atoms.set_chemical_symbols(syms)

    # Верификация
    syms_final = atoms.get_chemical_symbols()
    n_fe = syms_final.count('Fe')
    n_ni = syms_final.count('Ni')
    n_s  = syms_final.count('S')
    assert n_fe == 20, f"Expected 20 Fe, got {n_fe}"
    assert n_ni == 16, f"Expected 16 Ni, got {n_ni}"
    assert n_s  == 32, f"Expected 32 S, got {n_s}"
    assert len(atoms) == 68, f"Expected 68 atoms, got {len(atoms)}"

    # min_distance
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, 999.0)
    min_d = dists.min()
    assert min_d > 1.5, f"min_distance={min_d:.3f} < 1.5 A -- структура некорректна"

    print(f"  Pentlandite conv: Fe{n_fe}Ni{n_ni}S{n_s}, {len(atoms)} atoms")
    print(f"  Cell: {a:.3f} x {a:.3f} x {a:.3f} A, Fm-3m (#225)")
    print(f"  Min distance: {min_d:.3f} A")
    sys.stdout.flush()
    return atoms


# ---------------------------------------------------------------------------
# Поиск ближайшей S-S пары для вакансионного механизма
# ---------------------------------------------------------------------------
def find_s_pair(atoms):
    """
    Ищет ближайшую S-S пару (расстояние 3.5-5.5 Å).

    Пентландит: S на 8c (~2.51 Å S-S внутри тетраэдра) и 24e.
    Вакансионный механизм: H прыгает между ближайшими S-сайтами.
    Отсекаем слишком близкие (<3.0 Å) -- это не hop-пары.
    """
    s_idx = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'S']
    s_pos = atoms.positions[s_idx]
    _, d_mat = get_distances(s_pos, cell=atoms.cell, pbc=True)

    best = None
    for i in range(len(s_idx)):
        for j in range(i + 1, len(s_idx)):
            d = d_mat[i, j]
            if 3.0 < d < 5.5:
                if best is None or d < best[2]:
                    best = (s_idx[i], s_idx[j], d)

    if best is None:
        raise RuntimeError(
            "Не найдена S-S пара в диапазоне 3.0-5.5 Å. "
            "Проверь структуру или расширь диапазон."
        )

    print(f"  S pair: атомы #{best[0]} и #{best[1]}, dist = {best[2]:.3f} A")
    sys.stdout.flush()
    return (best[0], best[1]), best[2]


# ---------------------------------------------------------------------------
# Построение endpoints: удаляем оба S, ставим H
# ---------------------------------------------------------------------------
def make_endpoints(atoms, s_pair):
    """
    endA: H в позиции S[i], S[j] удалён (пустой сайт -- вакансия).
    endB: H в позиции S[j], S[i] удалён.

    Удаляем оба S из обоих endpoints -- это корректная модель вакансионного
    механизма (H прыгает через двойную вакансию: один сайт занят H,
    второй пустой).
    """
    si, sj = s_pair
    pos_i = atoms.positions[si].copy()
    pos_j = atoms.positions[sj].copy()

    # Удаляем по убыванию индекса, чтобы не сдвинуть второй
    del_order = sorted([si, sj], reverse=True)

    endA = atoms.copy()
    for idx in del_order:
        del endA[idx]
    endA.append(Atom('H', position=pos_i))

    endB = atoms.copy()
    for idx in del_order:
        del endB[idx]
    endB.append(Atom('H', position=pos_j))

    for lbl, ep in [("endA", endA), ("endB", endB)]:
        h = len(ep) - 1
        md = min(ep.get_distance(h, j, mic=True) for j in range(len(ep)) if j != h)
        formula = ep.get_chemical_formula()
        print(f"  {lbl}: {len(ep)} at, {formula}, min H-dist={md:.3f} A")
        if md < 0.8:
            raise RuntimeError(f"{lbl}: H слишком близко к соседу ({md:.3f} A)")

    sys.stdout.flush()
    return endA, endB


# ---------------------------------------------------------------------------
# Релаксация endpoint с resume
# ---------------------------------------------------------------------------
def relax_endpoint(atoms, label, force_rerun=False):
    """Релаксирует H (тяжёлые атомы заморожены). Resume через XYZ checkpoint."""
    xyz_path = WORK_DIR / f"relaxed_{label}.xyz"
    ckpt_path = WORK_DIR / f"checkpoint_{label}.xyz"

    # 1. Полностью сходившийся — skip
    if xyz_path.exists() and not force_rerun:
        relaxed = read(str(xyz_path))
        e = relaxed.get_potential_energy() if relaxed.calc else None
        if e is not None:
            print(f"  {label}: resume из {xyz_path}, E={e:.4f} eV")
            sys.stdout.flush()
            return relaxed, e
        print(f"  {label}: XYZ найден, но энергия не кэширована -- перезапуск")

    # 2. Промежуточный checkpoint — продолжаем с последней геометрии
    if ckpt_path.exists() and not force_rerun:
        atoms = read(str(ckpt_path))
        print(f"  {label}: RESUME из checkpoint ({ckpt_path})")
        sys.stdout.flush()
    else:
        atoms = atoms.copy()

    heavy = [i for i in range(len(atoms)) if atoms[i].symbol != 'H']
    atoms.set_constraint(FixAtoms(indices=heavy))
    atoms.calc = make_calc(f"relax_{label}")

    # Callback: сохраняем геометрию после каждого FIRE шага
    def save_checkpoint():
        write(str(ckpt_path), atoms)

    opt = FIRE(atoms, logfile=str(WORK_DIR / f"relax_{label}.log"))
    opt.attach(save_checkpoint)
    t0 = time.time()
    converged = opt.run(fmax=FMAX_RELAX, steps=MAX_STEPS_RELAX)
    dt = time.time() - t0

    e = atoms.get_potential_energy()
    fmax_final = float(np.max(np.abs(atoms.get_forces())))
    print(f"  {label}: E={e:.4f} eV, fmax={fmax_final:.4f}, "
          f"steps={opt.nsteps}, converged={converged}, t={dt:.0f}s")
    sys.stdout.flush()

    write(str(xyz_path), atoms)
    # Убираем checkpoint после успешной сходимости
    if ckpt_path.exists():
        ckpt_path.unlink()
    return atoms, e


# ---------------------------------------------------------------------------
# Двухфазный NEB (урок s76)
# ---------------------------------------------------------------------------
def _attach_calcs(images, tag):
    """Прикрепляем калькуляторы к промежуточным images."""
    for k in range(1, len(images) - 1):
        images[k].calc = make_calc(f"{tag}_img{k:02d}")
    return images


def _get_barrier(images):
    """Текущая оценка барьера по energies images."""
    energies = [img.get_potential_energy() for img in images]
    e0 = energies[0]
    return max(e - e0 for e in energies), energies


class _NEBStepLogger:
    """Callback: логирует шаг NEB в файл и stdout + checkpoint images."""
    def __init__(self, neb, log_path, save_tag="phase1"):
        self.neb = neb
        self.log_path = log_path
        self.save_tag = save_tag
        self.step = 0
        self.t0 = time.time()

    def __call__(self):
        images = self.neb.images
        forces = self.neb.get_forces()
        fmax = float(np.max(np.abs(forces)))

        energies = []
        for img in images:
            try:
                energies.append(img.get_potential_energy())
            except Exception:
                energies.append(float('nan'))

        e0 = energies[0] if energies else 0.0
        rel_e = [e - e0 for e in energies]
        barrier_est = max(rel_e) if rel_e else float('nan')

        dt = time.time() - self.t0
        line = (f"step={self.step:4d}  barrier_est={barrier_est:.4f} eV  "
                f"fmax={fmax:.4f}  t={dt:.0f}s")
        print(f"  [NEB] {line}", flush=True)

        with open(self.log_path, 'a') as f:
            f.write(line + "\n")

        # SIGKILL-resistant: save images after every step
        save_images(self.neb.images, tag=self.save_tag)

        self.step += 1


def run_two_phase_neb(endA, endB):
    """
    Двухфазный NEB:
      Phase 1: climb=False до fmax < 0.3 (max 200 шагов)
      Phase 2: climb=True  до fmax < 0.05 (max 300 шагов)

    После каждого шага сохраняем позиции для resume.
    """
    t_neb = time.time()
    step_log = WORK_DIR / "neb_step.log"

    # --- Phase 1 ---
    print(f"\n=== Phase 1: NEB climb=False, fmax<{FMAX_PHASE1} ===", flush=True)

    images_p1 = load_images_if_exist(N_IMAGES, tag="phase1")
    if images_p1 is None:
        images_p1 = [endA]
        for i in range(N_IMAGES):
            img = endA.copy()
            heavy = [j for j in range(len(img)) if img[j].symbol != 'H']
            img.set_constraint(FixAtoms(indices=heavy))
            images_p1.append(img)
        images_p1.append(endB)

        neb_p1 = NEB(images_p1, climb=False, k=K_SPRING, method="improvedtangent")
        neb_p1.interpolate("idpp")
        save_images(images_p1, tag="phase1_init")
    else:
        neb_p1 = NEB(images_p1, climb=False, k=K_SPRING, method="improvedtangent")

    images_p1 = _attach_calcs(images_p1, tag="ph1")
    # Endpoints also need calcs (for resume and energy eval)
    images_p1[0].calc = make_calc("ph1_endA")
    images_p1[-1].calc = make_calc("ph1_endB")

    # H позиции
    h_idx = len(images_p1[0]) - 1
    for k, img in enumerate(images_p1):
        p = img.positions[h_idx]
        print(f"  image {k}: H ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")

    logger_p1 = _NEBStepLogger(neb_p1, step_log, save_tag="phase1")
    opt_p1 = FIRE(
        neb_p1,
        logfile=str(WORK_DIR / "neb_phase1.log"),
        trajectory=str(WORK_DIR / "neb_phase1.traj"),
    )
    opt_p1.attach(logger_p1)

    print(f"  Running Phase 1 FIRE (fmax<{FMAX_PHASE1}, max={MAX_STEPS_PHASE1})...",
          flush=True)
    conv_p1 = opt_p1.run(fmax=FMAX_PHASE1, steps=MAX_STEPS_PHASE1)
    save_images(images_p1, tag="phase1")

    barrier_p1, energies_p1 = _get_barrier(images_p1)
    print(f"\n  Phase 1 done: barrier_est={barrier_p1:.4f} eV, "
          f"steps={opt_p1.nsteps}, converged={conv_p1}", flush=True)

    # --- Phase 2 ---
    print(f"\n=== Phase 2: CI-NEB climb=True, fmax<{FMAX_PHASE2} ===", flush=True)

    images_p2 = load_images_if_exist(N_IMAGES, tag="phase2")
    if images_p2 is None:
        # Стартуем из конца phase 1
        images_p2 = [img.copy() for img in images_p1]
        heavy = [j for j in range(len(images_p2[0])) if images_p2[0][j].symbol != 'H']
        for img in images_p2[1:-1]:
            img.set_constraint(FixAtoms(indices=heavy))

    neb_p2 = NEB(images_p2, climb=True, k=K_SPRING, method="improvedtangent")
    images_p2 = _attach_calcs(images_p2, tag="ph2")
    images_p2[0].calc = make_calc("ph2_endA")
    images_p2[-1].calc = make_calc("ph2_endB")

    logger_p2 = _NEBStepLogger(neb_p2, step_log, save_tag="phase2")
    opt_p2 = FIRE(
        neb_p2,
        logfile=str(WORK_DIR / "neb_phase2.log"),
        trajectory=str(WORK_DIR / "neb_phase2.traj"),
    )
    opt_p2.attach(logger_p2)

    print(f"  Running Phase 2 FIRE (fmax<{FMAX_PHASE2}, max={MAX_STEPS_PHASE2})...",
          flush=True)
    conv_p2 = opt_p2.run(fmax=FMAX_PHASE2, steps=MAX_STEPS_PHASE2)
    save_images(images_p2, tag="phase2")

    # Финальный барьер
    barrier_final, energies_final = _get_barrier(images_p2)
    e0 = energies_final[0]
    rel_e = [e - e0 for e in energies_final]

    dt_neb = time.time() - t_neb

    print(f"\n  Phase 2 done: E_a={barrier_final:.4f} eV, "
          f"steps={opt_p2.nsteps}, converged={conv_p2}", flush=True)
    print(f"  Energies: {[f'{e:.4f}' for e in rel_e]}", flush=True)
    print(f"  NEB total time: {dt_neb:.0f}s ({dt_neb/3600:.2f}h)", flush=True)

    # Сохраняем финальные XYZ
    for k, img in enumerate(images_p2):
        write(str(WORK_DIR / f"final_{k:02d}.xyz"), img)

    return (barrier_final, rel_e,
            int(opt_p1.nsteps), int(opt_p2.nsteps),
            bool(conv_p2), dt_neb,
            energies_p1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    t_total = time.time()
    print("=" * 70)
    print("  DFT NEB: pentlandite H diffusion (ABACUS PW GPU, conv. cell)")
    print("  68 atoms, Fm-3m, a=10.044 A, kpts=(2,2,2), ecutwfc=60 Ry")
    print(f"  Two-phase NEB: climb=False->True, k={K_SPRING}, {N_IMAGES} images")
    print(f"  Cross-verify: GPAW={E_A_GPAW} eV, MACE(conv)={E_A_MACE_CONV} eV")
    print("=" * 70)
    sys.stdout.flush()

    # 1. Строим конвенциональную ячейку
    print("\n[1/5] Build pentlandite conv (68 at)")
    pent = build_pentlandite_conv()

    # 2. Находим S-S пару
    print("\n[2/5] Find nearest S-S pair (vacancy hop)")
    s_pair, s_dist = find_s_pair(pent)

    # 3. Строим endpoints
    print("\n[3/5] Build endpoints")
    endA_raw, endB_raw = make_endpoints(pent, s_pair)

    # 4. Релаксируем endpoints
    print("\n[4/5] Relax endpoints (H only, FIRE fmax=0.05)")
    endA_r, e_A = relax_endpoint(endA_raw, "endA")
    endB_r, e_B = relax_endpoint(endB_raw, "endB")
    print(f"  |dE(endA-endB)| = {abs(e_A - e_B):.6f} eV")
    sys.stdout.flush()

    # 5. Двухфазный NEB
    print("\n[5/5] Two-phase NEB")
    (barrier, rel_e,
     steps_p1, steps_p2,
     converged, dt_neb,
     energies_p1) = run_two_phase_neb(endA_r, endB_r)

    dt_total = time.time() - t_total

    # Итог
    print("\n" + "=" * 70)
    print(f"  E_a (ABACUS PW, conv 68at) = {barrier:.4f} eV")
    print(f"  E_a (GPAW PW, prim 17at)   = {E_A_GPAW:.4f} eV")
    print(f"  E_a (MACE, conv 68at)      = {E_A_MACE_CONV:.4f} eV")
    print(f"  dE(ABACUS-GPAW)            = {barrier - E_A_GPAW:+.4f} eV")
    print(f"  Energies: {[f'{e:.4f}' for e in rel_e]}")
    print(f"  Steps: phase1={steps_p1}, phase2={steps_p2}, converged={converged}")
    print(f"  Time: NEB={dt_neb:.0f}s, Total={dt_total:.0f}s "
          f"({dt_total/3600:.2f}h)")
    print("=" * 70)
    sys.stdout.flush()

    # Сохраняем JSON
    result = {
        "system": "pentlandite_conv_68at_vacancy_hop",
        "method": "DFT_ABACUS_PBE_PW_GPU",
        "code": "ABACUS v3.9.0.26",
        "basis": "PW ecutwfc=60 Ry, SG15 ONCV PBE-1.2",
        "cell_type": "conventional",
        "spacegroup": "Fm-3m (#225)",
        "lattice_A": 10.044,
        "n_atoms_bulk": len(pent),
        "n_atoms_neb": len(endA_r),
        "formula_bulk": pent.get_chemical_formula(),
        "formula_neb": endA_r.get_chemical_formula(),
        "cell_A": pent.cell.lengths().tolist(),
        "S_pair_indices": list(s_pair),
        "S_pair_dist_A": float(s_dist),
        "kpts": [2, 2, 2],
        "scf_params": {
            "ecutwfc_Ry": 60,
            "smearing_sigma": 0.05,
            "scf_nmax": 500,
            "mixing_beta": 0.2,
            "mixing_ndim": 12,
        },
        "neb_params": {
            "n_images": N_IMAGES,
            "k_spring": K_SPRING,
            "method": "improvedtangent",
            "phase1_fmax": FMAX_PHASE1,
            "phase2_fmax": FMAX_PHASE2,
        },
        "E_endA_eV": float(e_A),
        "E_endB_eV": float(e_B),
        "dE_endpoints_eV": float(abs(e_A - e_B)),
        "E_a_eV": float(barrier),
        "E_rxn_eV": float(rel_e[-1]),
        "energies_eV": [float(e) for e in rel_e],
        "neb_steps_phase1": steps_p1,
        "neb_steps_phase2": steps_p2,
        "converged": bool(converged),
        "time_neb_s": round(dt_neb, 1),
        "time_total_s": round(dt_total, 1),
        # Cross-verify
        "E_a_GPAW_prim_eV": E_A_GPAW,
        "E_a_MACE_prim_eV": E_A_MACE_PRIM,
        "E_a_MACE_conv_eV": E_A_MACE_CONV,
        "dE_ABACUS_GPAW_eV": float(barrier - E_A_GPAW),
        "NOTE": (
            "Conv cell (68 at) vs GPAW prim (17 at). "
            "kpts=(2,2,2) Gamma-centered. Two-phase NEB: no-CI->CI. "
            "PW GPU (abacus_2g). S-vacancy mechanism."
        ),
    }

    out_json = RESULTS / "neb_pentlandite_abacus_conv_kpts_result.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)

    print(f"\n  JSON: {out_json}")

    # Маркер DONE
    done_marker = RESULTS / "DONE_neb_pent_conv_kpts"
    done_marker.write_text(
        f"E_a={barrier:.4f} eV  completed={time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}\n"
    )
    print(f"  DONE marker: {done_marker}")
    print("DONE")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n*** FATAL: {exc} ***", flush=True)
        traceback.print_exc()
        sys.exit(1)
