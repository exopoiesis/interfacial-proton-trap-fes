#!/usr/bin/env python3
"""
Revision UQ + 2D PMF — Paper #2 (KineticTrap) ответ на вопросы рецензента Stanford.

Задача 1 (Q3): Uncertainty quantification
  - Per-window: autocorrelation time tau_int, N_eff
  - Block-bootstrap 95% CI на WHAM PMF и dF#
  - localized-vs-uniform: разность mean-force MACE vs CHGNet

Задача 2 (Q1): 2D reweighted free energy F(d_FeH, d_OH)
  - WHAM-reweighted 2D гистограмма
  - Вывод: куда идёт d_OH при детачменте

Выход: results/revision_2026-06-14/uq/
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================
# КОНСТАНТЫ
# ============================================================
KB_KJ_MOL = 8.314e-3      # kJ/(mol·K)
EV_PER_KJMOL = 0.0103642697
T_K = 300.0
kT = KB_KJ_MOL * T_K      # ~2.479 kJ/mol
K_KAPPA = 1000.0           # kJ/(mol·Å²)
N_EQ_MD_STEPS = 2000
COLVAR_STRIDE = 10         # каждые 10 MD-шагов = 1 строка colvar
N_EQ_ROWS = N_EQ_MD_STEPS // COLVAR_STRIDE  # 200 строк отбросить
N_BOOTSTRAP = 200          # бутстрэп-реплик
N_BLOCKS_PER_WINDOW = 10   # блоков для блок-бутстрэпа
WHAM_N_ITER = 3000
WHAM_TOL = 1e-6            # kJ/mol (~1e-8 eV) — достаточно для PMF; warm-start ускоряет bootstrap
WHAM_SIGMA_KDE = 0.05      # Å (0.02 was under-smoothed ~2x grid step → argmax noise; 0.05 ~ window std)
OH_COVALENT_A = 1.20       # Å — порог "сформирована O–H связь" (ков. O–H ≈ 0.97, H-bond ≥ 1.8)
SH_COVALENT_A = 1.55       # Å — порог "сформирована S–H связь" (ков. S–H ≈ 1.36)
DETACH_ZONE = [2.0, 2.6]   # Å — зона детачмента для топо-теста носителя
SADDLE_RANGE = [1.5, 3.3]  # Å — достоверный диапазон PMF (edge-артефакт CHGNet начинается ~3.3-3.4:
                           # окна 14+ не дотягиваются до центров, τ_int 60-156; SI плато 0.30-0.36 до 3.3.
                           # [1.5,3.5] захватывал edge-онсет → argmax-седло CHGNet ложно прыгало на 0.63@3.5)
SADDLE_FOCUS = [2.0, 2.6]  # Å — "седловая" зона для анализа localized-vs-uniform
N_GRID = 300

BASE_RESULTS = Path("D:/home/ignat/project-third-matter/results/us_2026-05-06")
OUT_DIR = Path("D:/home/ignat/project-third-matter/results/revision_2026-06-14/uq")
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[init] OUT_DIR = {OUT_DIR.resolve()}", flush=True)


# ============================================================
# УТИЛИТЫ: парсинг
# ============================================================
def parse_colvar(path: Path):
    """Читаем colvar.dat → numpy array shape (N, 7)."""
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                try:
                    rows.append([float(p) for p in parts[:7]])
                except ValueError:
                    pass
    if not rows:
        return np.empty((0, 7))
    return np.array(rows)


def parse_manifest(path: Path):
    """Читаем manifest.txt → список dict с id, center_A, K."""
    windows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 4:
                windows.append({
                    "id": int(parts[0]),
                    "center_A": float(parts[1]),
                    "K_kJ_mol_A2": float(parts[3]),
                })
    return windows


# ============================================================
# УТИЛИТЫ: WHAM
# ============================================================
def wham_solve(samples_list, centers, kappas, n_iter=WHAM_N_ITER, tol=WHAM_TOL, f_init=None):
    """
    Итеративный WHAM. Возвращает f_j (смещения окон, kJ/mol).
    samples_list: list of 1D arrays (cv_FeH production samples per window)
    centers, kappas: arrays of shape (n_windows,)
    f_init: тёплый старт (напр. f_j полных данных) — резко ускоряет bootstrap.
    """
    n_w = len(samples_list)
    all_s = np.concatenate(samples_list)  # (N_total,)
    n_per = np.array([len(s) for s in samples_list])
    c = np.asarray(centers)
    k = np.asarray(kappas)

    # V_ij[i,j] = bias window j at sample i (kJ/mol)
    V_ij = 0.5 * k[None, :] * (all_s[:, None] - c[None, :]) ** 2

    f_j = np.zeros(n_w) if f_init is None else np.asarray(f_init, dtype=float).copy()
    for it in range(n_iter):
        # lw[i,j] = -V_ij/kT + f_j/kT
        lw = -V_ij / kT + f_j[None, :] / kT
        lw_max = lw.max(axis=1)
        log_denom = lw_max + np.log((n_per[None, :] * np.exp(lw - lw_max[:, None])).sum(axis=1))

        # f_j_new = -kT log Σ_i exp(-V_ij/kT - log_denom_i)
        lt = -V_ij / kT - log_denom[:, None]
        lt_max = lt.max(axis=0)
        f_new = -kT * (lt_max + np.log(np.exp(lt - lt_max[None, :]).sum(axis=0)))
        f_new -= f_new[0]

        diff = np.abs(f_new - f_j).max()
        f_j = f_new
        if diff < tol:
            break
    return f_j


def build_pmf(samples_list, centers, kappas, f_j, cv_grid):
    """
    Строим PMF на cv_grid через KDE+WHAM unbiasing.
    Возвращает pmf (kJ/mol), 0 по минимуму в SADDLE_RANGE.
    """
    all_s = np.concatenate(samples_list)
    n_per = np.array([len(s) for s in samples_list])
    c = np.asarray(centers)
    k = np.asarray(kappas)
    n_total = len(all_s)
    sigma = WHAM_SIGMA_KDE

    pmf = np.full(len(cv_grid), np.nan)
    for ki, x in enumerate(cv_grid):
        ker = np.exp(-((all_s - x) ** 2) / (2 * sigma ** 2))
        num = ker.sum()
        V_jx = 0.5 * k * (x - c) ** 2
        denom = (n_per * np.exp(-V_jx / kT + f_j / kT)).sum()
        if denom > 0 and num > 1e-15:
            pmf[ki] = -kT * (
                np.log(num / (n_total * sigma * np.sqrt(2 * np.pi)))
                - np.log(denom)
            )

    # Gauge: 0 at min within reliable range
    mask = (cv_grid >= SADDLE_RANGE[0]) & (cv_grid <= SADDLE_RANGE[1]) & ~np.isnan(pmf)
    if mask.any():
        pmf -= pmf[mask].min()
    else:
        pmf -= np.nanmin(pmf)
    return pmf


def saddle_from_pmf(cv_grid, pmf, x_min=SADDLE_RANGE[0], x_max=SADDLE_RANGE[1]):
    """Ищем максимум PMF в достоверном диапазоне [x_min, x_max]."""
    mask = (cv_grid >= x_min) & (cv_grid <= x_max) & ~np.isnan(pmf)
    if not mask.any():
        return np.nan, np.nan
    idx = np.argmax(pmf[mask])
    x_saddle = cv_grid[mask][idx]
    f_saddle = pmf[mask][idx]
    return x_saddle, f_saddle


# ============================================================
# УТИЛИТЫ: автокорреляционное время
# ============================================================
def integrated_autocorr_time(x):
    """
    Оценка интегрального времени автокорреляции τ_int методом Madras-Sokal.
    Возвращает τ_int в единицах шагов (строк colvar).
    N_eff = N / (2 * τ_int).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return 0.5, n
    x = x - x.mean()
    var = np.dot(x, x) / n
    if var < 1e-30:
        return 0.5, n

    # Нормированный автокорреляционный ряд
    acf = np.correlate(x, x, mode="full")[n - 1:]  # lag 0..n-1
    acf /= (var * n)

    # Правило остановки: накапливаем до lag M, где M < 6*τ (Sokal)
    tau_int = 0.5
    for lag in range(1, n // 4):
        tau_int += acf[lag]
        if lag >= 6 * tau_int:
            break

    tau_int = max(tau_int, 0.5)
    n_eff = n / (2 * tau_int)
    return float(tau_int), float(n_eff)


# ============================================================
# ЗАГРУЗКА ДАННЫХ
# ============================================================
def load_engine_data(engine: str):
    """
    Загружаем все окна для движка ('mace' или 'chgnet').
    Возвращает:
      windows_info: list of dict с ключами id, center_A, K, n_prod,
                    mean_cv, std_cv, tau_int, n_eff,
                    prod_cv (1D array), prod_oh (1D array), prod_sh (1D array)
    """
    base = BASE_RESULTS / engine
    manifest = parse_manifest(base / "manifest.txt")
    windows_info = []

    for w in manifest:
        path = base / "windows" / f"window_{w['id']:02d}" / "colvar.dat"
        if not path.exists():
            print(f"  [{engine}] SKIP window {w['id']:02d} — нет файла", flush=True)
            continue
        data = parse_colvar(path)
        if len(data) <= N_EQ_ROWS:
            print(f"  [{engine}] SKIP window {w['id']:02d} — мало данных ({len(data)} строк)", flush=True)
            continue

        prod = data[N_EQ_ROWS:]
        cv = prod[:, 2]    # cv_FeH
        oh = prod[:, 5]    # d_min_OH
        sh = prod[:, 6]    # d_min_SH

        tau, n_eff = integrated_autocorr_time(cv)
        windows_info.append({
            "id": w["id"],
            "center_A": w["center_A"],
            "K": w["K_kJ_mol_A2"],
            "n_prod": len(cv),
            "mean_cv": float(np.mean(cv)),
            "std_cv": float(np.std(cv)),
            "tau_int": tau,
            "n_eff": n_eff,
            "prod_cv": cv,
            "prod_oh": oh,
            "prod_sh": sh,
        })

    print(f"[{engine}] загружено {len(windows_info)} окон", flush=True)
    return windows_info


# ============================================================
# BLOCK-BOOTSTRAP
# ============================================================
def block_bootstrap_pmf(windows_info, cv_grid, n_bootstrap=N_BOOTSTRAP, n_blocks=N_BLOCKS_PER_WINDOW, f_init=None):
    """
    Block-bootstrap: каждое окно → n_blocks блоков → ресэмплируем с возвращением.
    Возвращаем array (n_bootstrap, len(cv_grid)) — матрицу PMF.
    """
    rng = np.random.default_rng(42)

    centers = np.array([w["center_A"] for w in windows_info])
    kappas = np.array([w["K"] for w in windows_info])

    # Нарезаем каждое окно на блоки
    blocks_per_win = []
    for w in windows_info:
        cv = w["prod_cv"]
        n = len(cv)
        block_size = max(1, n // n_blocks)
        blks = [cv[i * block_size: (i + 1) * block_size] for i in range(n_blocks)]
        # Последний блок может быть короче — оставляем
        blocks_per_win.append(blks)

    pmf_matrix = np.full((n_bootstrap, len(cv_grid)), np.nan)

    for b in range(n_bootstrap):
        if b % 50 == 0:
            print(f"  bootstrap {b}/{n_bootstrap}...", flush=True)
        # Для каждого окна ресэмплируем n_blocks блоков с возвращением
        boot_samples = []
        for blks in blocks_per_win:
            idx = rng.integers(0, len(blks), size=len(blks))
            boot_cv = np.concatenate([blks[i] for i in idx])
            boot_samples.append(boot_cv)

        try:
            f_j = wham_solve(boot_samples, centers, kappas, f_init=f_init)
            pmf = build_pmf(boot_samples, centers, kappas, f_j, cv_grid)
            pmf_matrix[b] = pmf
        except Exception as e:
            print(f"  bootstrap {b} failed: {e}", flush=True)

    return pmf_matrix


# ============================================================
# 2D REWEIGHTED PMF
# ============================================================
def wham_frame_weights(windows_info, f_j, all_cv):
    """Per-frame WHAM unbiasing weights W[k] ∝ 1/Σ_j n_j exp(-V_jk/kT + f_j/kT)."""
    centers = np.array([w["center_A"] for w in windows_info])
    kappas = np.array([w["K"] for w in windows_info])
    n_per = np.array([w["n_prod"] for w in windows_info])
    V_ij = 0.5 * kappas[None, :] * (all_cv[:, None] - centers[None, :]) ** 2  # (N, n_w)
    lw = -V_ij / kT + f_j[None, :] / kT
    lw_max = lw.max(axis=1)
    log_denom = lw_max + np.log((n_per[None, :] * np.exp(lw - lw_max[:, None])).sum(axis=1))
    log_w = -log_denom
    log_w -= log_w.max()
    weights = np.exp(log_w)
    weights /= weights.sum()
    return weights


def build_2d_pmf(windows_info, f_j, coord_key="prod_oh",
                 y_lo=1.0, y_hi=3.5, d_feh_bins=50, d_y_bins=40):
    """
    2D reweighted free energy F(d_FeH, Y), Y = d_OH или d_SH (orthogonal, unbiased).
    Per-frame WHAM weight (см. wham_frame_weights). Только d_FeH был biasing-координатой,
    поэтому это reweight тепловых флуктуаций Y — топология валидна, абс. высота
    ортогонального барьера НЕТ. Возвращает (x_centers, y_centers, F2d, y_at_minF).
    """
    all_cv = np.concatenate([w["prod_cv"] for w in windows_info])
    all_y = np.concatenate([w[coord_key] for w in windows_info])
    weights = wham_frame_weights(windows_info, f_j, all_cv)

    mask_range = (all_cv >= SADDLE_RANGE[0]) & (all_cv <= SADDLE_RANGE[1])
    H_w, xedges, yedges = np.histogram2d(
        all_cv[mask_range], all_y[mask_range],
        bins=[d_feh_bins, d_y_bins],
        range=[[SADDLE_RANGE[0], SADDLE_RANGE[1]], [y_lo, y_hi]],
        weights=weights[mask_range],
    )
    H_w[H_w < 1e-30] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        F2d = -kT * np.log(H_w)
    F2d -= np.nanmin(F2d)

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])

    # Профиль Y при минимуме F вдоль каждого d_FeH (шумный — только для визуализации MEP)
    y_at_min = []
    for ix in range(len(x_centers)):
        col = F2d[ix, :]
        y_at_min.append(np.nan if np.all(np.isnan(col)) else y_centers[np.nanargmin(col)])
    return x_centers, y_centers, F2d, np.array(y_at_min)


def carrier_topology(windows_info, f_j):
    """
    In-flight топо-тест (робастный, без argmin-шума): reweighted СРЕДНИЕ d_OH и d_SH
    в зоне детачмента d_FeH∈DETACH_ZONE. Если оба > ковалентного порога → носитель
    in-flight (не закоммитился ни к O, ни к S). Согласовано с §3.3/SI (без ярлыков
    water-release/assisted). Возвращает dict.
    """
    all_cv = np.concatenate([w["prod_cv"] for w in windows_info])
    all_oh = np.concatenate([w["prod_oh"] for w in windows_info])
    all_sh = np.concatenate([w["prod_sh"] for w in windows_info])
    weights = wham_frame_weights(windows_info, f_j, all_cv)

    zmask = (all_cv >= DETACH_ZONE[0]) & (all_cv <= DETACH_ZONE[1])
    w = weights[zmask]
    wsum = w.sum()
    if wsum <= 0:
        return {"n_frames": 0}
    oh_z, sh_z = all_oh[zmask], all_sh[zmask]
    mean_oh = float(np.sum(w * oh_z) / wsum)
    mean_sh = float(np.sum(w * sh_z) / wsum)
    # медианы (взвеш. приближённо через сортировку весов)
    def wmedian(x, wt):
        o = np.argsort(x); xs, ws = x[o], wt[o]; c = np.cumsum(ws) / ws.sum()
        return float(xs[np.searchsorted(c, 0.5)])
    med_oh = wmedian(oh_z, w)
    med_sh = wmedian(sh_z, w)
    oh_committed = mean_oh < OH_COVALENT_A
    sh_committed = mean_sh < SH_COVALENT_A
    if not oh_committed and not sh_committed:
        verdict = "in-flight (носитель НЕ закоммитился ни к O воды, ни к S — qualitative, consistent §3.3/SI)"
    elif sh_committed and not oh_committed:
        verdict = "S-bound (формируется S–H)"
    elif oh_committed and not sh_committed:
        verdict = "water-bound (формируется O–H)"
    else:
        verdict = "ambiguous (близко и к O, и к S)"
    return {
        "n_frames": int(zmask.sum()),
        "detach_zone_A": DETACH_ZONE,
        "mean_d_OH_A": mean_oh, "median_d_OH_A": med_oh,
        "mean_d_SH_A": mean_sh, "median_d_SH_A": med_sh,
        "OH_covalent_threshold_A": OH_COVALENT_A,
        "SH_covalent_threshold_A": SH_COVALENT_A,
        "verdict": verdict,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60, flush=True)
    print("revision_uq_2d.py — UQ + 2D PMF для Paper #2", flush=True)
    print("=" * 60, flush=True)

    results = {}

    for engine in ["mace", "chgnet"]:
        print(f"\n{'='*40}", flush=True)
        print(f"  Движок: {engine.upper()}", flush=True)
        print(f"{'='*40}", flush=True)

        # --- 1. Загрузка данных ---
        windows_info = load_engine_data(engine)
        n_w = len(windows_info)

        # --- 2. Per-window статистика ---
        print("\n[1] Per-window статистика (τ_int, N_eff):", flush=True)
        print(f"  {'win':>3}  {'d0':>5}  {'⟨cv⟩':>7}  {'std':>6}  {'τ_int':>7}  {'N_eff':>8}", flush=True)
        win_stats = []
        for w in windows_info:
            row = {
                "id": w["id"],
                "center_A": w["center_A"],
                "mean_cv": w["mean_cv"],
                "std_cv": w["std_cv"],
                "tau_int": w["tau_int"],
                "n_eff": w["n_eff"],
                "n_prod": w["n_prod"],
            }
            win_stats.append(row)
            flag = " [edge]" if w["center_A"] > 3.5 else ""
            print(f"  {w['id']:3d}  {w['center_A']:5.2f}  {w['mean_cv']:7.4f}  "
                  f"{w['std_cv']:6.4f}  {w['tau_int']:7.2f}  {w['n_eff']:8.1f}{flag}", flush=True)

        # --- 3. WHAM на полных данных ---
        print("\n[2] Основной WHAM...", flush=True)
        centers = np.array([w["center_A"] for w in windows_info])
        kappas = np.array([w["K"] for w in windows_info])
        samples = [w["prod_cv"] for w in windows_info]

        f_j = wham_solve(samples, centers, kappas)
        cv_grid = np.linspace(1.4, 4.1, N_GRID)
        pmf_full = build_pmf(samples, centers, kappas, f_j, cv_grid)

        x_saddle, f_saddle = saddle_from_pmf(cv_grid, pmf_full)
        f_saddle_ev = f_saddle * EV_PER_KJMOL
        print(f"  Седло: d_FeH = {x_saddle:.3f} Å, ΔF# = {f_saddle:.2f} kJ/mol = {f_saddle_ev:.3f} eV", flush=True)

        # --- 4. Block-bootstrap ---
        print(f"\n[3] Block-bootstrap ({N_BOOTSTRAP} реплик)...", flush=True)
        pmf_matrix = block_bootstrap_pmf(windows_info, cv_grid, f_init=f_j)

        # 95% CI
        valid = ~np.isnan(pmf_matrix).all(axis=1)
        pmf_valid = pmf_matrix[valid]
        print(f"  Валидных реплик: {pmf_valid.shape[0]}", flush=True)

        ci_lo = np.nanpercentile(pmf_valid, 2.5, axis=0)
        ci_hi = np.nanpercentile(pmf_valid, 97.5, axis=0)

        # CI на седло: ФИКСИРУЕМ положение седла по полному PMF и читаем PMF реплик
        # в ЭТОЙ точке. НЕ argmax каждой реплики — иначе получаем распределение
        # max-статистики с положительным смещением (завышает верхнюю границу CI).
        i_sad = int(np.argmin(np.abs(cv_grid - x_saddle)))
        at_boundary = bool(x_saddle >= (SADDLE_RANGE[1] - 0.05))
        saddle_vals = []
        for pmf_rep in pmf_valid:
            v = pmf_rep[i_sad]
            if not np.isnan(v):
                saddle_vals.append(v * EV_PER_KJMOL)

        saddle_vals = np.array(saddle_vals)
        saddle_ci_lo = np.percentile(saddle_vals, 2.5)
        saddle_ci_hi = np.percentile(saddle_vals, 97.5)
        saddle_mean_bs = np.mean(saddle_vals)
        if at_boundary:
            print(f"  ⚠ x_saddle={x_saddle:.2f} Å на границе [1.5,3.5] → вероятно ПЛАТО, "
                  f"ΔF# трактовать как нижнюю границу детачмента, не седло", flush=True)
        print(f"  ΔF# bootstrap (в фикс. точке {x_saddle:.2f} Å): mean={saddle_mean_bs:.3f} eV, "
              f"95% CI [{saddle_ci_lo:.3f}, {saddle_ci_hi:.3f}] eV", flush=True)

        # --- 5. Mean-force per window ---
        mf_per_win = []
        for w in windows_info:
            mf = w["K"] * (w["center_A"] - w["mean_cv"])  # kJ/(mol·Å)
            mf_per_win.append(mf)
        mf_per_win = np.array(mf_per_win)

        # --- 6. 2D PMF: F(d_FeH, d_OH) И F(d_FeH, d_SH) ---
        print(f"\n[4] 2D reweighted PMF F(d_FeH, d_OH) и F(d_FeH, d_SH)...", flush=True)
        x_oh, y_oh, F_oh, oh_at_min = build_2d_pmf(windows_info, f_j, coord_key="prod_oh",
                                                   y_lo=1.0, y_hi=3.5)
        x_sh, y_sh, F_sh, sh_at_min = build_2d_pmf(windows_info, f_j, coord_key="prod_sh",
                                                   y_lo=1.2, y_hi=3.5)

        # In-flight топо-тест (робастный, reweighted средние в зоне детачмента)
        topo = carrier_topology(windows_info, f_j)
        print(f"  Зона детачмента {DETACH_ZONE} Å: ⟨d_OH⟩={topo.get('mean_d_OH_A', float('nan')):.2f} Å, "
              f"⟨d_SH⟩={topo.get('mean_d_SH_A', float('nan')):.2f} Å", flush=True)
        print(f"  Носитель: {topo.get('verdict', 'n/a')}", flush=True)

        # Сохраняем в results
        results[engine] = {
            "n_windows": n_w,
            "window_stats": win_stats,
            "pmf_full": {
                "saddle_d_A": float(x_saddle),
                "saddle_dF_kjmol": float(f_saddle),
                "saddle_dF_eV": float(f_saddle_ev),
            },
            "bootstrap": {
                "n_valid": int(pmf_valid.shape[0]),
                "saddle_dF_eV_mean": float(saddle_mean_bs),
                "saddle_dF_eV_ci95_lo": float(saddle_ci_lo),
                "saddle_dF_eV_ci95_hi": float(saddle_ci_hi),
                "x_saddle_at_boundary": at_boundary,
            },
            "mean_force": {
                "centers_A": centers.tolist(),
                "mf_kJ_mol_A": mf_per_win.tolist(),
            },
            "carrier_topology": topo,
            # Сохраняем для графиков
            "_cv_grid": cv_grid,
            "_pmf_full": pmf_full,
            "_ci_lo": ci_lo,
            "_ci_hi": ci_hi,
            "_centers": centers,
            "_mf": mf_per_win,
            "_x_oh": x_oh, "_y_oh": y_oh, "_F_oh": F_oh, "_oh_at_min": oh_at_min,
            "_x_sh": x_sh, "_y_sh": y_sh, "_F_sh": F_sh, "_sh_at_min": sh_at_min,
        }

    # ============================================================
    # LOCALIZED vs UNIFORM анализ (после обоих движков)
    # ============================================================
    print("\n" + "=" * 60, flush=True)
    print("[5] Localized-vs-uniform анализ MACE vs CHGNet", flush=True)
    print("=" * 60, flush=True)

    c_m = np.array(results["mace"]["mean_force"]["centers_A"])
    mf_m = np.array(results["mace"]["mean_force"]["mf_kJ_mol_A"])
    c_c = np.array(results["chgnet"]["mean_force"]["centers_A"])
    mf_c = np.array(results["chgnet"]["mean_force"]["mf_kJ_mol_A"])

    # Строим на общей сетке центров (пересечение)
    # Интерполируем CHGNet на сетку MACE-центров
    from scipy.interpolate import interp1d
    if len(c_c) >= 4:
        interp_c = interp1d(c_c, mf_c, kind="linear", bounds_error=False, fill_value=np.nan)
        mf_c_on_m = interp_c(c_m)
    else:
        mf_c_on_m = mf_c[:len(c_m)]

    delta_mf = mf_m - mf_c_on_m  # kJ/(mol·Å), MACE−CHGNet

    # Разбиваем на зоны
    mask_saddle = (c_m >= SADDLE_FOCUS[0]) & (c_m <= SADDLE_FOCUS[1])
    mask_rest = (c_m >= SADDLE_RANGE[0]) & (c_m <= SADDLE_RANGE[1]) & ~mask_saddle

    abs_delta_saddle = np.nansum(np.abs(delta_mf[mask_saddle]))
    abs_delta_rest = np.nansum(np.abs(delta_mf[mask_rest]))
    abs_total = abs_delta_saddle + abs_delta_rest
    frac_saddle = abs_delta_saddle / abs_total if abs_total > 0 else np.nan

    # Нормировка на число окон (chemist: голая Σ не учитывает разную плотность точек)
    n_saddle = int(np.sum(mask_saddle))
    n_rest = int(np.sum(mask_rest))
    mean_delta_saddle = abs_delta_saddle / n_saddle if n_saddle else np.nan
    mean_delta_rest = abs_delta_rest / n_rest if n_rest else np.nan

    print(f"\n  [ENERGETIC disagreement — НЕ механистический; механизм см. carrier_topology]", flush=True)
    print(f"  Зона {SADDLE_FOCUS} Å (седловая, {n_saddle} окон):", flush=True)
    print(f"    Σ|ΔMF| = {abs_delta_saddle:.3f}; ⟨|ΔMF|⟩/окно = {mean_delta_saddle:.3f} kJ/(mol·Å)", flush=True)
    print(f"  Остальные окна в [{SADDLE_RANGE[0]}, {SADDLE_RANGE[1]}] Å ({n_rest} окон):", flush=True)
    print(f"    Σ|ΔMF| = {abs_delta_rest:.3f}; ⟨|ΔMF|⟩/окно = {mean_delta_rest:.3f} kJ/(mol·Å)", flush=True)
    print(f"  Доля Σ|ΔMF| в седловой зоне: {100*frac_saddle:.1f}%; "
          f"отношение ⟨|ΔMF|⟩ седло/фланги = {mean_delta_saddle/mean_delta_rest:.2f}×", flush=True)

    # Для ΔF(d) = интеграл ΔMF dd — для PMF-разности
    cv_common = results["mace"]["_cv_grid"]
    pmf_mace = results["mace"]["_pmf_full"]
    pmf_chgnet = results["chgnet"]["_pmf_full"]

    # Интерполируем chgnet PMF на ту же сетку что и mace
    pmf_chgnet_grid = np.full_like(pmf_mace, np.nan)
    cv_c = results["chgnet"]["_cv_grid"]
    pmf_c = results["chgnet"]["_pmf_full"]
    interp_pmf = interp1d(cv_c, pmf_c, kind="linear", bounds_error=False, fill_value=np.nan)
    pmf_chgnet_grid = interp_pmf(cv_common)

    delta_pmf = pmf_mace - pmf_chgnet_grid  # kJ/mol

    mask_saddle_g = (cv_common >= SADDLE_FOCUS[0]) & (cv_common <= SADDLE_FOCUS[1])
    mask_rest_g = (cv_common >= SADDLE_RANGE[0]) & (cv_common <= SADDLE_RANGE[1]) & ~mask_saddle_g

    abs_dpmf_saddle = np.nansum(np.abs(delta_pmf[mask_saddle_g]))
    abs_dpmf_rest = np.nansum(np.abs(delta_pmf[mask_rest_g]))
    abs_dpmf_total = abs_dpmf_saddle + abs_dpmf_rest
    frac_dpmf_saddle = abs_dpmf_saddle / abs_dpmf_total if abs_dpmf_total > 0 else np.nan
    print(f"\n  Вклад в |ΔPMF| (интегр. по cv-сетке):", flush=True)
    print(f"    Седловая зона {SADDLE_FOCUS} Å: {100*frac_dpmf_saddle:.1f}%", flush=True)
    print(f"    Остальные: {100*(1-frac_dpmf_saddle):.1f}%", flush=True)

    results["localized_vs_uniform"] = {
        "centers_A": c_m.tolist(),
        "delta_mf_kJ_mol_A": delta_mf.tolist(),
        "saddle_zone_A": SADDLE_FOCUS,
        "reliable_range_A": SADDLE_RANGE,
        "abs_delta_mf_saddle_zone": float(abs_delta_saddle),
        "abs_delta_mf_rest": float(abs_delta_rest),
        "mean_delta_mf_per_window_saddle": float(mean_delta_saddle),
        "mean_delta_mf_per_window_rest": float(mean_delta_rest),
        "fraction_in_saddle_zone_mf": float(frac_saddle),
        "fraction_in_saddle_zone_pmf": float(frac_dpmf_saddle),
        "note": "ENERGETIC disagreement (PMF gradient along d_FeH), NOT mechanistic; "
                "fraction depends on the [2.0,2.6] bin; lead with pmf-version (uniform grid). "
                "Mechanism (S vs water carrier) = carrier_topology per engine.",
        "_delta_pmf": delta_pmf,
    }

    # ============================================================
    # СОХРАНЕНИЕ JSON
    # ============================================================
    summary = {
        "mace": {
            "saddle_dF_eV": results["mace"]["pmf_full"]["saddle_dF_eV"],
            "saddle_d_A": results["mace"]["pmf_full"]["saddle_d_A"],
            "bootstrap_saddle_dF_eV_mean": results["mace"]["bootstrap"]["saddle_dF_eV_mean"],
            "bootstrap_saddle_dF_eV_ci95": [
                results["mace"]["bootstrap"]["saddle_dF_eV_ci95_lo"],
                results["mace"]["bootstrap"]["saddle_dF_eV_ci95_hi"],
            ],
            "window_stats": results["mace"]["window_stats"],
            "x_saddle_at_boundary": results["mace"]["bootstrap"]["x_saddle_at_boundary"],
            "carrier_topology": results["mace"]["carrier_topology"],
        },
        "chgnet": {
            "saddle_dF_eV": results["chgnet"]["pmf_full"]["saddle_dF_eV"],
            "saddle_d_A": results["chgnet"]["pmf_full"]["saddle_d_A"],
            "bootstrap_saddle_dF_eV_mean": results["chgnet"]["bootstrap"]["saddle_dF_eV_mean"],
            "bootstrap_saddle_dF_eV_ci95": [
                results["chgnet"]["bootstrap"]["saddle_dF_eV_ci95_lo"],
                results["chgnet"]["bootstrap"]["saddle_dF_eV_ci95_hi"],
            ],
            "window_stats": results["chgnet"]["window_stats"],
            "x_saddle_at_boundary": results["chgnet"]["bootstrap"]["x_saddle_at_boundary"],
            "carrier_topology": results["chgnet"]["carrier_topology"],
        },
        "localized_vs_uniform": {
            "fraction_in_saddle_zone_2_0_2_6_A_mf": results["localized_vs_uniform"]["fraction_in_saddle_zone_mf"],
            "fraction_in_saddle_zone_2_0_2_6_A_pmf": results["localized_vs_uniform"]["fraction_in_saddle_zone_pmf"],
            "abs_delta_mf_saddle": results["localized_vs_uniform"]["abs_delta_mf_saddle_zone"],
            "abs_delta_mf_rest": results["localized_vs_uniform"]["abs_delta_mf_rest"],
        },
    }

    with open(OUT_DIR / "uq_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] JSON сохранён: {OUT_DIR / 'uq_summary.json'}", flush=True)

    # ============================================================
    # ГРАФИКИ
    # ============================================================
    print("\n[6] Построение графиков...", flush=True)

    # --- Граф 1: PMF с CI-лентами обоих движков ---
    fig, ax = plt.subplots(figsize=(10, 6))

    cv_m = results["mace"]["_cv_grid"]
    pmf_m = results["mace"]["_pmf_full"]
    ci_lo_m = results["mace"]["_ci_lo"]
    ci_hi_m = results["mace"]["_ci_hi"]

    cv_c2 = results["chgnet"]["_cv_grid"]
    pmf_c2 = results["chgnet"]["_pmf_full"]
    ci_lo_c = results["chgnet"]["_ci_lo"]
    ci_hi_c = results["chgnet"]["_ci_hi"]

    # Маскируем ненадёжный диапазон (>3.5 Å)
    mask_m = (cv_m >= 1.4) & (cv_m <= 3.7)
    mask_c2 = (cv_c2 >= 1.4) & (cv_c2 <= 3.7)

    ax.fill_between(cv_m[mask_m],
                    (ci_lo_m * EV_PER_KJMOL)[mask_m],
                    (ci_hi_m * EV_PER_KJMOL)[mask_m],
                    alpha=0.25, color="steelblue", label="MACE 95% CI")
    ax.plot(cv_m[mask_m], (pmf_m * EV_PER_KJMOL)[mask_m],
            lw=2.5, color="steelblue", label="MACE PMF")

    ax.fill_between(cv_c2[mask_c2],
                    (ci_lo_c * EV_PER_KJMOL)[mask_c2],
                    (ci_hi_c * EV_PER_KJMOL)[mask_c2],
                    alpha=0.25, color="darkorange", label="CHGNet 95% CI")
    ax.plot(cv_c2[mask_c2], (pmf_c2 * EV_PER_KJMOL)[mask_c2],
            lw=2.5, color="darkorange", label="CHGNet PMF")

    # Седловые точки
    xs_m = results["mace"]["pmf_full"]["saddle_d_A"]
    fs_m = results["mace"]["pmf_full"]["saddle_dF_eV"]
    xs_c = results["chgnet"]["pmf_full"]["saddle_d_A"]
    fs_c = results["chgnet"]["pmf_full"]["saddle_dF_eV"]
    ax.plot(xs_m, fs_m, "^", color="steelblue", ms=10, zorder=5,
            label=f"MACE ΔF# = {fs_m:.2f} eV @ {xs_m:.2f} Å")
    ax.plot(xs_c, fs_c, "^", color="darkorange", ms=10, zorder=5,
            label=f"CHGNet ΔF# = {fs_c:.2f} eV @ {xs_c:.2f} Å")

    # Серая зона >3.5 Å (ненадёжно)
    ax.axvspan(3.5, 4.1, color="gray", alpha=0.1, label="edge artifact (>3.5 Å)")
    ax.axvline(3.5, color="gray", lw=1, ls="--")

    ax.set_xlabel("d$_{FeH}$ (Å, smooth-min)", fontsize=13)
    ax.set_ylabel("PMF (eV)", fontsize=13)
    ax.set_xlim(1.4, 4.1)
    ax.set_title("1D PMF с 95% block-bootstrap CI (MACE vs CHGNet)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pmf_ci.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] pmf_ci.png", flush=True)

    # --- Граф 2: разность mean-force ---
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # Верхняя: mean-force обоих
    ax = axes[0]
    ax.plot(c_m, mf_m, "o-", color="steelblue", label="MACE", lw=2)
    ax.plot(c_c, mf_c, "s-", color="darkorange", label="CHGNet", lw=2)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.axvspan(SADDLE_FOCUS[0], SADDLE_FOCUS[1], color="red", alpha=0.1, label="Saddle zone")
    ax.set_ylabel("Mean force K(d₀−⟨cv⟩)\n[kJ/(mol·Å)]", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Per-window mean force и их разность (MACE−CHGNet)", fontsize=12)

    # Нижняя: разность
    ax = axes[1]
    mask_lvu = ~np.isnan(delta_mf) & (c_m >= SADDLE_RANGE[0]) & (c_m <= SADDLE_RANGE[1])
    ax.bar(c_m[mask_lvu], delta_mf[mask_lvu], width=0.13, color=["red" if m else "steelblue"
                                                                   for m in mask_saddle[mask_lvu]],
           alpha=0.7, label="MACE−CHGNet ΔMF")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvspan(SADDLE_FOCUS[0], SADDLE_FOCUS[1], color="red", alpha=0.08, label=f"Saddle zone ({SADDLE_FOCUS[0]}–{SADDLE_FOCUS[1]} Å)")
    ax.set_xlabel("d$_{FeH}$ (Å)", fontsize=12)
    ax.set_ylabel("ΔMACE−CHGNet MF\n[kJ/(mol·Å)]", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    frac_pct = 100 * frac_saddle
    ax.text(0.02, 0.92, f"Седловая зона: {frac_pct:.0f}% суммарного |ΔMF|",
            transform=ax.transAxes, fontsize=10, color="darkred",
            bbox=dict(boxstyle="round", fc="white", ec="darkred", alpha=0.7))

    fig.tight_layout()
    fig.savefig(OUT_DIR / "mean_force_diff.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] mean_force_diff.png", flush=True)

    # --- Граф 3: 2D F(d_FeH, Y) для Y=d_OH и Y=d_SH, оба движка (2×2) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    coord_specs = [
        ("_x_oh", "_y_oh", "_F_oh", "_oh_at_min", "d$_{OH}$ (Å)", OH_COVALENT_A),
        ("_x_sh", "_y_sh", "_F_sh", "_sh_at_min", "d$_{SH}$ (Å)", SH_COVALENT_A),
    ]
    for row, (xk, yk, Fk, mepk, ylab, cov) in enumerate(coord_specs):
        for col, (engine, title) in enumerate(zip(["mace", "chgnet"], ["MACE", "CHGNet"])):
            ax = axes[row, col]
            x2 = results[engine][xk]; y2 = results[engine][yk]
            F2 = results[engine][Fk] * EV_PER_KJMOL
            vmax = min(np.nanmax(F2), 1.5)
            cf = ax.contourf(x2, y2, F2.T, levels=20, cmap="RdYlBu_r", vmin=0, vmax=vmax)
            plt.colorbar(cf, ax=ax, label="F (eV)")
            ax.contour(x2, y2, F2.T, levels=10, colors="k", linewidths=0.5, alpha=0.4)
            ymep = results[engine][mepk]
            mmep = ~np.isnan(ymep) & (x2 >= 1.5) & (x2 <= 3.5)
            ax.plot(x2[mmep], ymep[mmep], "w-", lw=2, label="min-F profile (noisy)")
            ax.axhline(cov, color="lime", lw=1.5, ls="--", alpha=0.8, label=f"covalent {ylab[:4]} ≤ {cov} Å")
            xs_1d = results[engine]["pmf_full"]["saddle_d_A"]
            ax.axvline(xs_1d, color="white", lw=1.5, ls="--", alpha=0.7, label=f"1D saddle {xs_1d:.2f} Å")
            ax.axvspan(DETACH_ZONE[0], DETACH_ZONE[1], color="white", alpha=0.12)
            ax.set_xlabel("d$_{FeH}$ (Å)", fontsize=11)
            ax.set_ylabel(ylab, fontsize=11)
            ax.set_title(f"{title}: F(d_FeH, {ylab[:5]})", fontsize=11)
            ax.legend(fontsize=7, loc="upper right")
            ax.axvline(3.5, color="gray", lw=1, ls=":", alpha=0.5)
            topo = results[engine]["carrier_topology"]
            ax.text(0.02, 0.04,
                    f"⟨d_OH⟩={topo.get('mean_d_OH_A', float('nan')):.2f}, "
                    f"⟨d_SH⟩={topo.get('mean_d_SH_A', float('nan')):.2f} Å\n{topo.get('verdict','')[:34]}",
                    transform=ax.transAxes, fontsize=7, color="white",
                    bbox=dict(boxstyle="round", fc="black", alpha=0.55))
    fig.suptitle("2D reweighted free energy F(d_FeH, d_OH) и F(d_FeH, d_SH) — Paper #2 revision\n"
                 "WARNING: d_OH/d_SH НЕ смещались в US (reweight флуктуаций); топология in-flight валидна, "
                 "абс. высота ортогонального барьера — нет",
                 fontsize=10, color="darkred")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pmf_2d.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] pmf_2d.png", flush=True)

    # --- Граф 4: ΔPMF(d) ---
    fig, ax = plt.subplots(figsize=(9, 5))
    mask_dpmf = (cv_common >= SADDLE_RANGE[0]) & (cv_common <= SADDLE_RANGE[1]) & ~np.isnan(delta_pmf)
    ax.plot(cv_common[mask_dpmf], delta_pmf[mask_dpmf] * EV_PER_KJMOL,
            "k-", lw=2, label="MACE − CHGNet PMF (eV)")
    ax.fill_between(cv_common[mask_dpmf], 0, delta_pmf[mask_dpmf] * EV_PER_KJMOL,
                    where=(cv_common[mask_dpmf] >= SADDLE_FOCUS[0]) & (cv_common[mask_dpmf] <= SADDLE_FOCUS[1]),
                    color="red", alpha=0.25, label=f"Saddle zone {SADDLE_FOCUS}")
    ax.fill_between(cv_common[mask_dpmf], 0, delta_pmf[mask_dpmf] * EV_PER_KJMOL,
                    where=~((cv_common[mask_dpmf] >= SADDLE_FOCUS[0]) & (cv_common[mask_dpmf] <= SADDLE_FOCUS[1])),
                    color="steelblue", alpha=0.2, label="Flanking regions")
    ax.axhline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("d$_{FeH}$ (Å)", fontsize=12)
    ax.set_ylabel("ΔPMF MACE−CHGNet (eV)", fontsize=12)
    ax.set_title(f"PMF difference: {100*frac_dpmf_saddle:.0f}% набирается в седловой зоне {SADDLE_FOCUS} Å",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "delta_pmf.png", dpi=150)
    plt.close(fig)
    print(f"  [OK] delta_pmf.png", flush=True)

    # ============================================================
    # MARKDOWN SUMMARY
    # ============================================================
    mace_bs = results["mace"]["bootstrap"]
    chgnet_bs = results["chgnet"]["bootstrap"]
    lvu = results["localized_vs_uniform"]

    md = f"""# UQ и 2D PMF — revision Paper #2 (KineticTrap)
**Дата:** 2026-06-14 | **Движки:** MACE, CHGNet | **Окна:** 18 × ~9 ps

## Задача 1 — Uncertainty Quantification (Q3)

### Per-window автокорреляция и N_eff

**MACE:**
| win | d0 (Å) | ⟨cv⟩ (Å) | std (Å) | τ_int | N_eff |
|-----|--------|----------|---------|-------|-------|
"""
    for w in results["mace"]["window_stats"]:
        flag = " ⚠" if w["center_A"] > 3.5 else ""
        md += (f"| {w['id']:2d}  | {w['center_A']:.2f} | {w['mean_cv']:.4f} | "
               f"{w['std_cv']:.4f} | {w['tau_int']:.1f} | {w['n_eff']:.0f}{flag} |\n")

    md += f"""
**CHGNet:**
| win | d0 (Å) | ⟨cv⟩ (Å) | std (Å) | τ_int | N_eff |
|-----|--------|----------|---------|-------|-------|
"""
    for w in results["chgnet"]["window_stats"]:
        flag = " ⚠" if w["center_A"] > 3.5 else ""
        md += (f"| {w['id']:2d}  | {w['center_A']:.2f} | {w['mean_cv']:.4f} | "
               f"{w['std_cv']:.4f} | {w['tau_int']:.1f} | {w['n_eff']:.0f}{flag} |\n")

    md += f"""
### Block-bootstrap 95% CI на ΔF# (N={N_BOOTSTRAP} реплик, {N_BLOCKS_PER_WINDOW} блоков/окно)

| Движок | ΔF# (eV) | 95% CI lo (eV) | 95% CI hi (eV) | CI ширина (eV) |
|--------|----------|---------------|---------------|---------------|
| MACE   | {mace_bs['saddle_dF_eV_mean']:.3f} | {mace_bs['saddle_dF_eV_ci95_lo']:.3f} | {mace_bs['saddle_dF_eV_ci95_hi']:.3f} | {mace_bs['saddle_dF_eV_ci95_hi']-mace_bs['saddle_dF_eV_ci95_lo']:.3f} |
| CHGNet | {chgnet_bs['saddle_dF_eV_mean']:.3f} | {chgnet_bs['saddle_dF_eV_ci95_lo']:.3f} | {chgnet_bs['saddle_dF_eV_ci95_hi']:.3f} | {chgnet_bs['saddle_dF_eV_ci95_hi']-chgnet_bs['saddle_dF_eV_ci95_lo']:.3f} |

**Ответ рецензенту Q3 (saddle ΔF# ± 95% CI):**
- MACE: **{mace_bs['saddle_dF_eV_mean']:.2f} eV** [{mace_bs['saddle_dF_eV_ci95_lo']:.2f}, {mace_bs['saddle_dF_eV_ci95_hi']:.2f}] eV
- CHGNet: **{chgnet_bs['saddle_dF_eV_mean']:.2f} eV** [{chgnet_bs['saddle_dF_eV_ci95_lo']:.2f}, {chgnet_bs['saddle_dF_eV_ci95_hi']:.2f}] eV

### Localized vs Uniform — это ENERGETIC расхождение (НЕ механистическое)

Метрика: где по d_FeH расходятся PMF-градиенты MACE и CHGNet. Ведущая (на равномерной cv-сетке) — |ΔPMF|; |ΔMF| на дискретных окнах вторична (зависит от плотности окон).

- Доля |ΔPMF| в седловой зоне [2.0, 2.6] Å: **{100*lvu['fraction_in_saddle_zone_pmf']:.0f}%** (ведущая)
- Доля Σ|ΔMF| в седловой зоне: {100*lvu['fraction_in_saddle_zone_mf']:.0f}% (норм.: ⟨|ΔMF|⟩/окно седло = {lvu['mean_delta_mf_per_window_saddle']:.2f} vs фланги {lvu['mean_delta_mf_per_window_rest']:.2f})
- **Вердикт:** энергетическое расхождение {"КОНЦЕНТРИРУЕТСЯ у седла" if lvu['fraction_in_saddle_zone_pmf'] > 0.5 else "РАСПРЕДЕЛЕНО по координате"} (зависит от границ зоны [2.0,2.6] Å).
- ⚠️ Механизм носителя (S vs water) — НЕ из этой метрики, а из carrier_topology ниже.

## Задача 2 — 2D reweighted PMF F(d_FeH, d_OH) И F(d_FeH, d_SH); in-flight топо-тест

**ПРЕДУПРЕЖДЕНИЕ:** d_OH/d_SH НЕ были biasing-координатами — это reweight тепловых флуктуаций.
In-flight ТОПОЛОГИЯ валидна; абсолютная высота ортогонального барьера — НЕТ. Без ярлыков «water-release/assisted».

In-flight тест: в зоне детачмента d_FeH∈{DETACH_ZONE} Å сравниваем reweighted ⟨d_OH⟩, ⟨d_SH⟩ с ковалентными порогами (O–H<{OH_COVALENT_A} Å, S–H<{SH_COVALENT_A} Å).

| Движок | ⟨d_OH⟩ (Å) | ⟨d_SH⟩ (Å) | Носитель |
|--------|-----------|-----------|----------|
| MACE   | {results['mace']['carrier_topology'].get('mean_d_OH_A', float('nan')):.2f} | {results['mace']['carrier_topology'].get('mean_d_SH_A', float('nan')):.2f} | {results['mace']['carrier_topology'].get('verdict','')} |
| CHGNet | {results['chgnet']['carrier_topology'].get('mean_d_OH_A', float('nan')):.2f} | {results['chgnet']['carrier_topology'].get('mean_d_SH_A', float('nan')):.2f} | {results['chgnet']['carrier_topology'].get('verdict','')} |

## Выходные файлы

- `uq_summary.json` — все числа
- `pmf_ci.png` — PMF с CI-лентой
- `mean_force_diff.png` — per-window ΔMF
- `delta_pmf.png` — ΔPMF(d) MACE−CHGNet
- `pmf_2d.png` — 2D F(d_FeH, d_OH) и F(d_FeH, d_SH), оба движка
"""
    with open(OUT_DIR / "UQ_2D_SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[OK] Markdown: {OUT_DIR / 'UQ_2D_SUMMARY.md'}", flush=True)

    # Финальный вывод чисел
    print("\n" + "=" * 60, flush=True)
    print("ФИНАЛЬНЫЕ ЧИСЛА", flush=True)
    print("=" * 60, flush=True)
    print(f"MACE  ΔF# = {mace_bs['saddle_dF_eV_mean']:.3f} eV "
          f"[{mace_bs['saddle_dF_eV_ci95_lo']:.3f}, {mace_bs['saddle_dF_eV_ci95_hi']:.3f}] eV (95% CI)", flush=True)
    print(f"CHGNet ΔF# = {chgnet_bs['saddle_dF_eV_mean']:.3f} eV "
          f"[{chgnet_bs['saddle_dF_eV_ci95_lo']:.3f}, {chgnet_bs['saddle_dF_eV_ci95_hi']:.3f}] eV (95% CI)", flush=True)
    print(f"Energetic disagreement: {100*lvu['fraction_in_saddle_zone_pmf']:.0f}% |ΔPMF| "
          f"в зоне {SADDLE_FOCUS} Å (ведущая); {100*lvu['fraction_in_saddle_zone_mf']:.0f}% Σ|ΔMF|", flush=True)
    print(f"MACE носитель @детачмент: {results['mace']['carrier_topology'].get('verdict','n/a')} "
          f"(⟨d_OH⟩={results['mace']['carrier_topology'].get('mean_d_OH_A', float('nan')):.2f}, "
          f"⟨d_SH⟩={results['mace']['carrier_topology'].get('mean_d_SH_A', float('nan')):.2f} Å)", flush=True)
    print(f"CHGNet носитель @детачмент: {results['chgnet']['carrier_topology'].get('verdict','n/a')} "
          f"(⟨d_OH⟩={results['chgnet']['carrier_topology'].get('mean_d_OH_A', float('nan')):.2f}, "
          f"⟨d_SH⟩={results['chgnet']['carrier_topology'].get('mean_d_SH_A', float('nan')):.2f} Å)", flush=True)
    print(f"\nВсе файлы в: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
