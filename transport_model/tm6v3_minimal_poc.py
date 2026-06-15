"""
TM6v3 Minimal Proof-of-Concept: 3-переменная модель (A, M, Fe)
================================================================
Задача B1: минимальный proof-of-concept автопоэзиса.

Архитектура: РЕШЕНИЕ-037 (пентландит + макинавит), РЕШЕНИЕ-029 (3 перем.)
Маппинг: A=формиат, M=пентландит+макинавит мембрана, Fe=Fe²⁺
Катализатор R1: макинавит (onset ~100 мВ, РЕШЕНИЕ-034)
Анод: Fe⁰ жертвенный
FE = 8% (Roldan 2015, РЕШЕНИЕ-035)

Переменные:
  A  — формиат HCOO⁻ (мМ)
  M  — мембрана (пентландит+макинавит) (отн. единицы ≈ мМ по массе)
  Fe — Fe²⁺(aq) (мМ)

Реакции:
  R1: CO₂ + 2H⁺ + 2e⁻ →[макинавит] HCOOH
      v = k1 * f1(M) * hill(A)    с FE=0.08
  R3_M: Fe²⁺ → осаждение мембраны M (зависит от A)
      v = km * Fe * A
  R_Fe_supply: → Fe²⁺ (коррозия Fe⁰)
      v = R_Fe (константа)
  R_Fe_gen: A → Fe²⁺ (побочный продукт метаболизма)
      v = k_fe_gen * A
  D_A: A → 0 (разбавление/диффузия)
      v = kd_A * A
  D_M: M → 0 (коррозия мембраны при pH 2-3)
      v = kd_m * M
  D_Fe: Fe²⁺ → 0 (осаждение/окисление)
      v = kd_fe * Fe

Автокаталитическая петля: A →(Fe_gen) Fe →(R3_M) M →(f1) R1 →(более A)

Запуск: python simulations/tm6v3_minimal_poc.py
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import json
import time

_builtin_print = print
def print(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 1. ПАРАМЕТРЫ TM6v3-min
# ============================================================

class TM6v3MinParams:
    """
    Параметры минимальной 3-переменной модели TM6v3.

    Основа: TM6v2-min (Q-009), обновлено для пентландит+макинавит (РЕШЕНИЕ-037).
    """
    def __init__(self):
        # --- Автокатализ R1 (макинавит катализатор) ---
        # k1: базовая скорость CO₂ → HCOOH на макинавите
        # TM6v2-min: k1 = 1e-4 (без PEDOT, alpha=0)
        # FE = 0.08 → k1_eff = k1_raw * FE
        # Но в TM6v2-min alpha=0 уже даёт A*=4.89 мМ → k1 уже включает реальную скорость
        # Макинавит onset ~100 мВ (vs грейгит 400 мВ) → лучше, сохраняем k1
        self.k1 = 1e-4          # с⁻¹         базовый автокатализ

        # Hill n=2 полунасыщение
        self.Ka = 7e-4          # М           (из TM6v1/v2)

        # f1: ресурс CO₂ (растворимость ~33 мМ при 1 атм)
        # Через мембрану доставляется часть → f1 = 5e-3 M (как в TM6v2)
        self.f1 = 5e-3          # М

        # Km_f: полунасыщение по M для транспорта
        # Пентландит 200-500 нм, менее пористый чем SiO₂ → Km_f чуть выше
        # TM6v2: Km_f = 3e-4 (SiO₂+PBA). Пентландит: плотнее но тоньше
        # Нетто: ~2× SiO₂ (более плотный, но e⁻ транспорт встроен)
        self.Km_f = 5e-4        # М           (vs 3e-4 TM6v2)

        # --- Мембрана R3_M: Fe²⁺ + A → M ---
        # TM6v2: km = 3e-2 (с s2 фактором)
        # В TM6v3: km непосредственно связывает Fe и A
        # Пентландит осаждается из Fe²⁺+Ni²⁺+S²⁻ при 25°C
        # A (формиат) как восстановитель/лиганд ускоряет осаждение
        # km * Fe * A ≈ km_v2 * Fe * s2 при A ~ s2
        # s2 (TM6v2) = 5e-3 M → km = km_v2 * s2 / A_ref
        # При A* ~ 5 мМ: km = 3e-2 * 5e-3 / 5e-3 = 3e-2
        self.km = 3e-2          # с⁻¹ М⁻¹    мембранообразование

        # --- Fe²⁺ ---
        self.k_fe_gen = 5e-5    # с⁻¹         A → Fe (побочный продукт)
        self.fe_supply = 5e-8   # М/с         коррозия Fe⁰ жертвенного анода

        # --- Деградация ---
        # kd_A: формиат в открытой системе разбавляется/окисляется
        # TM6v2: kd = 1e-4 для A
        self.kd_A = 1e-4        # с⁻¹         деградация формиата

        # kd_m: коррозия пентландита при pH 2-3
        # Из Q-068: ~0.1 нм/ч при pH 2-4
        # Толщина 200-500 нм → относительная скорость: 0.1/350 ≈ 3e-4 /ч = 8e-8 /с
        # Но в модели M в мМ (не нм), масштаб другой
        # TM6v2: kd_m = 1e-6 (SiO₂, очень стабильный)
        # Пентландит менее стабилен чем SiO₂ при pH 2-3, но стабильнее FeS
        # TM6v1: kd_m = 1e-5 (FeS макинавит)
        # Пентландит — между SiO₂ и FeS: ~3e-6
        self.kd_m = 3e-6        # с⁻¹         деградация мембраны

        # kd_fe: потери Fe²⁺ (осаждение, окисление)
        # TM6v2: kd_fe = 2e-4
        # Без PBA ионного сита → утечки выше
        # Но пентландит непроницаем для Fe²⁺ (кубический, нет каналов)
        # Сохраняем: kd_fe = 3e-4 (чуть хуже чем TM6v2 с PBA)
        self.kd_fe = 3e-4       # с⁻¹         потери Fe²⁺

    def __repr__(self):
        return (f"TM6v3Min(k1={self.k1}, km={self.km}, kd_A={self.kd_A}, "
                f"kd_m={self.kd_m}, Km_f={self.Km_f})")


# ============================================================
# 2. ODE: 3 переменные (A, M, Fe)
# ============================================================

def tm6v3_rhs(t, y, p=None):
    """Правая часть 3-переменной ODE: A, M, Fe."""
    if p is None:
        p = TM6v3MinParams()

    a, m, fe = [max(v, 0) for v in y]

    # Мембранный транспорт ресурсов
    f1_eff = p.f1 * m / (p.Km_f + m)

    # Hill n=2 автокатализ
    hill_a = a**2 / (p.Ka**2 + a**2) if a > 0 else 0.0

    # R1: CO₂ →[макинавит] HCOOH (через мембрану)
    r1 = p.k1 * f1_eff * hill_a

    # R3_M: Fe²⁺ + A → мембрана M
    r3m = p.km * fe * a

    # Fe generation
    r_fe_gen = p.k_fe_gen * a
    r_fe_supply = p.fe_supply

    # Деградация
    d_a = p.kd_A * a
    d_m = p.kd_m * m
    d_fe = p.kd_fe * fe

    # dy/dt
    da = r1 - d_a
    dm = r3m - d_m
    dfe = r_fe_gen + r_fe_supply - r3m - d_fe

    return [da, dm, dfe]


# ============================================================
# 3. ПОИСК СТАЦИОНАРА
# ============================================================

def find_steady_state(p=None, y0=None, t_end=500*3600):
    """Найти стационар интегрированием."""
    if p is None:
        p = TM6v3MinParams()
    if y0 is None:
        y0 = [5e-3, 1e-4, 5e-4]  # A, M, Fe

    sol = solve_ivp(lambda t, y: tm6v3_rhs(t, y, p),
                    [0, t_end], y0, method='LSODA',
                    rtol=1e-10, atol=1e-14, max_step=100)

    final = sol.y[:, -1]
    names = ['a', 'm', 'fe']
    ss = dict(zip(names, final))
    return ss, sol


def check_stability(ss, p=None):
    """Проверить устойчивость через якобиан."""
    if p is None:
        p = TM6v3MinParams()

    y0 = np.array([ss['a'], ss['m'], ss['fe']])
    eps = 1e-8
    n = len(y0)
    J = np.zeros((n, n))
    f0 = np.array(tm6v3_rhs(0, y0, p))

    for j in range(n):
        y_pert = y0.copy()
        y_pert[j] += eps
        f_pert = np.array(tm6v3_rhs(0, y_pert, p))
        J[:, j] = (f_pert - f0) / eps

    eigenvalues = np.linalg.eigvals(J)
    max_real = np.max(np.real(eigenvalues))
    stable = max_real < 0

    return {
        'stable': bool(stable),
        'eigenvalues': [complex(e) for e in eigenvalues],
        'max_real_eigenvalue': float(max_real),
        'lambda_att': float(-max_real) if stable else None,
        'tau_relax_hours': float(1 / (-max_real) / 3600) if stable and max_real < 0 else None
    }


# ============================================================
# 4. ТЕСТ ГОМЕОСТАЗА
# ============================================================

def test_homeostasis(ss, p=None, perturbation=-0.95):
    """Тест: восстановление после -95% A."""
    if p is None:
        p = TM6v3MinParams()

    y0 = [ss['a'] * (1 + perturbation), ss['m'], ss['fe']]

    sol = solve_ivp(lambda t, y: tm6v3_rhs(t, y, p),
                    [0, 72*3600], y0, method='LSODA',
                    rtol=1e-10, atol=1e-14, dense_output=True, max_step=100)

    a_final = sol.y[0, -1]
    recovery = a_final / ss['a'] if ss['a'] > 0 else 0

    t_check = np.linspace(0, 72*3600, 1000)
    a_t = sol.sol(t_check)[0]
    recovered_mask = a_t >= 0.9 * ss['a']
    if np.any(recovered_mask):
        t_recovery_h = t_check[np.argmax(recovered_mask)] / 3600
    else:
        t_recovery_h = float('inf')

    return {
        'perturbation': perturbation,
        'a_final_mM': float(a_final * 1e3),
        'recovery_fraction': float(recovery),
        'recovered': bool(recovery > 0.9),
        't_recovery_90pct_hours': float(t_recovery_h)
    }


# ============================================================
# 5. ТЕСТ АВТОПОЭЗИСА (ключевой!)
# ============================================================

def test_autopoiesis(ss, p=None):
    """
    Ключевой тест автопоэзиса: мембрана необходима И создаётся метаболизмом.

    Три теста:
    1. km=0 (мембранообразование заблокировано) → мембрана деградирует → A должна упасть
       Это показывает: без ВОСПРОИЗВОДСТВА мембраны система умирает.
    2. A(0)=0, M(0)=0 → система остаётся мёртвой (нет seed формиата)
       Это показывает: система не самозапускается из ничего.
    3. Нормальный стационар → жизнь (контроль)

    Дополнительно:
    4. M(0)=0, A(0)>0 → система МОЖЕТ восстановить M из Fe+A
       Это показывает: автопоэзис работает (метаболит строит мембрану).
    """
    if p is None:
        p = TM6v3MinParams()

    results = {}

    # Тест 1: km=0 → мембрана не воспроизводится → смерть
    p_no_km = TM6v3MinParams()
    for attr in vars(p):
        setattr(p_no_km, attr, getattr(p, attr))
    p_no_km.km = 0.0  # блокируем мембранообразование

    # t_end: 5/kd_m ≈ 5/(3e-6) = 1.67e6 с ≈ 463 ч, используем 2000 ч
    t_end_nokm = max(2000 * 3600, 5.0 / p.kd_m)
    y0_nokm = [ss['a'], ss['m'], ss['fe']]
    sol_nokm = solve_ivp(lambda t, y: tm6v3_rhs(t, y, p_no_km),
                         [0, t_end_nokm], y0_nokm, method='LSODA',
                         rtol=1e-10, atol=1e-14, max_step=100)
    a_final_nokm = sol_nokm.y[0, -1]

    results['km_zero'] = {
        'description': f'km=0: мембранообразование заблокировано ({t_end_nokm/3600:.0f} ч)',
        'A_final_mM': float(a_final_nokm * 1e3),
        'M_final_mM': float(sol_nokm.y[1, -1] * 1e3),
        'Fe_final_mM': float(sol_nokm.y[2, -1] * 1e3),
        'alive': bool(a_final_nokm > 1e-4),
        'expected': 'dead (M деградирует → f1_eff→0 → A→0)',
        'sol': sol_nokm
    }

    # Тест 2: A(0)=0, M(0)=0 → должна остаться мёртвой
    y0_zero = [0.0, 0.0, 5e-4]  # только Fe²⁺ из внешнего источника
    sol_zero = solve_ivp(lambda t, y: tm6v3_rhs(t, y, p),
                         [0, 200*3600], y0_zero, method='LSODA',
                         rtol=1e-10, atol=1e-14, max_step=100)
    a_final_zero = sol_zero.y[0, -1]

    results['all_zero'] = {
        'description': 'A(0)=0, M(0)=0: нет seed формиата',
        'A_final_mM': float(a_final_zero * 1e3),
        'M_final_mM': float(sol_zero.y[1, -1] * 1e3),
        'Fe_final_mM': float(sol_zero.y[2, -1] * 1e3),
        'alive': bool(a_final_zero > 1e-4),
        'expected': 'dead (Hill(0)=0 → R1=0)',
        'sol': sol_zero
    }

    # Тест 3: Нормальный стационар → контроль (жизнь)
    y0_alive = [ss['a'], ss['m'], ss['fe']]
    sol_alive = solve_ivp(lambda t, y: tm6v3_rhs(t, y, p),
                          [0, 72*3600], y0_alive, method='LSODA',
                          rtol=1e-10, atol=1e-14, max_step=100)
    a_final_alive = sol_alive.y[0, -1]

    results['control'] = {
        'description': 'Нормальный стационар (контроль)',
        'A_final_mM': float(a_final_alive * 1e3),
        'M_final_mM': float(sol_alive.y[1, -1] * 1e3),
        'Fe_final_mM': float(sol_alive.y[2, -1] * 1e3),
        'alive': bool(a_final_alive > 1e-4),
        'expected': 'alive (A→A*)',
        'sol': sol_alive
    }

    # Тест 4: M(0)=0 но A(0)>0 → самовосстановление мембраны (автопоэзис!)
    y0_noM = [ss['a'], 0.0, ss['fe']]
    sol_noM = solve_ivp(lambda t, y: tm6v3_rhs(t, y, p),
                        [0, 200*3600], y0_noM, method='LSODA',
                        rtol=1e-10, atol=1e-14, max_step=100)
    a_final_noM = sol_noM.y[0, -1]

    results['M0_zero_A_alive'] = {
        'description': 'M(0)=0, A(0)=A*: самовосстановление мембраны',
        'A_final_mM': float(a_final_noM * 1e3),
        'M_final_mM': float(sol_noM.y[1, -1] * 1e3),
        'Fe_final_mM': float(sol_noM.y[2, -1] * 1e3),
        'alive': bool(a_final_noM > 1e-4),
        'expected': 'alive (A→Fe→M→f1→R1→A: автопоэтическая петля)',
        'sol': sol_noM
    }

    # Автопоэзис PASS если:
    # 1. km=0 → мертва (мембрана необходима)
    # 2. A=0,M=0 → мёртва (нужен seed)
    # 3. Контроль → жива
    # 4. M=0,A>0 → жива (мембрана самовосстанавливается)
    results['autopoiesis_pass'] = (
        not results['km_zero']['alive']      # мембрана необходима
        and not results['all_zero']['alive']  # не самозапускается из ничего
        and results['control']['alive']       # контроль жив
        and results['M0_zero_A_alive']['alive']  # мембрана самовосстанавливается
    )

    return results


# ============================================================
# 6. БИСТАБИЛЬНОСТЬ (сканирование f1)
# ============================================================

def scan_bistability(p=None, f1_values=None):
    """Сканирование f1: есть ли два стационара?"""
    if p is None:
        p = TM6v3MinParams()
    if f1_values is None:
        f1_values = np.logspace(-4, -1, 50)

    results = []
    for f1_val in f1_values:
        p_scan = TM6v3MinParams()
        p_scan.f1 = f1_val

        # Из "живого" начального условия
        ss_alive, _ = find_steady_state(p_scan, y0=[5e-3, 1e-3, 5e-4], t_end=200*3600)
        alive = ss_alive['a'] > 1e-4

        # Из "мёртвого" начального условия
        ss_dead, _ = find_steady_state(p_scan, y0=[1e-6, 1e-8, 1e-4], t_end=200*3600)
        dead_alive = ss_dead['a'] > 1e-4

        results.append({
            'f1': float(f1_val),
            'A_alive_mM': float(ss_alive['a'] * 1e3),
            'A_dead_mM': float(ss_dead['a'] * 1e3),
            'bistable': bool(alive and not dead_alive),
            'alive_from_high': bool(alive),
            'alive_from_low': bool(dead_alive)
        })

    return results


# ============================================================
# 7. GILLESPIE SSA (3 вида, 7 реакций)
# ============================================================

# Стехиометрическая матрица (7 реакций × 3 вида: A, M, Fe)
NU_V3 = np.array([
    [+1,  0,  0],   # R1:  → A (автокатализ)
    [ 0, +1, -1],   # R3M: Fe → M (мембранообразование, потребляет Fe)
    [ 0,  0, +1],   # R_Fe_gen: A → Fe (A не потребляется, нетто +Fe)
    [ 0,  0, +1],   # R_Fe_supply: → Fe
    [-1,  0,  0],   # D_A: A → 0
    [ 0, -1,  0],   # D_M: M → 0
    [ 0,  0, -1],   # D_Fe: Fe → 0
], dtype=np.int32)

N_RXN_V3 = 7
N_SP_V3 = 3
SP_NAMES_V3 = ['A', 'M', 'Fe']


def concentration_to_molecules_v3(ss_dict, N_A_target):
    """Конвертация концентраций в молекулы."""
    y_ss = np.array([ss_dict['a'], ss_dict['m'], ss_dict['fe']])
    a_star = y_ss[0]
    Omega = N_A_target / a_star
    n_init = np.round(y_ss * Omega).astype(np.int64)
    n_init[0] = N_A_target
    return n_init, Omega


def compute_propensities_v3(n, Omega, p):
    """7 пропенсити для TM6v3-min."""
    n_A, n_M, n_Fe = n[0], n[1], n[2]
    c_A = n_A / Omega
    c_M = n_M / Omega
    c_Fe = n_Fe / Omega

    # Мембранный транспорт
    f1_eff = p.f1 * c_M / (p.Km_f + c_M) if c_M > 0 else 0.0

    # Hill n=2
    hill_a = c_A**2 / (p.Ka**2 + c_A**2) if c_A > 0 else 0.0

    a = np.zeros(N_RXN_V3)
    a[0] = Omega * p.k1 * f1_eff * hill_a       # R1: автокатализ
    a[1] = p.km * n_Fe * c_A                      # R3M: Fe + A → M (n_Fe * c_A для правильного масштабирования)
    a[2] = p.k_fe_gen * n_A                        # R_Fe_gen: A → A + Fe
    a[3] = Omega * p.fe_supply                     # R_Fe_supply: → Fe
    a[4] = p.kd_A * n_A                            # D_A
    a[5] = p.kd_m * n_M                            # D_M
    a[6] = p.kd_fe * n_Fe                          # D_Fe

    return a


def gillespie_ssa_v3(n_init, Omega, p, t_max, rng,
                     record_trajectory=False, record_interval=100.0):
    """Gillespie SSA для 3-переменной модели."""
    n = n_init.copy()
    t = 0.0
    n_A_star = n_init[0]
    threshold_A = max(1, int(0.1 * n_A_star))

    if record_trajectory:
        t_rec = [0.0]
        n_rec = [n.copy()]
        next_record = record_interval

    while t < t_max:
        a = compute_propensities_v3(n, Omega, p)
        a_total = np.sum(a)

        if a_total <= 0:
            if record_trajectory:
                t_rec.append(t)
                n_rec.append(n.copy())
                return t, False, (np.array(t_rec), np.array(n_rec))
            return t, False, None

        tau = rng.exponential(1.0 / a_total)
        t += tau
        if t > t_max:
            break

        r = rng.random() * a_total
        cumsum = 0.0
        j = 0
        for j in range(N_RXN_V3):
            cumsum += a[j]
            if cumsum >= r:
                break

        n += NU_V3[j]
        np.clip(n, 0, None, out=n)

        if record_trajectory and t >= next_record:
            t_rec.append(t)
            n_rec.append(n.copy())
            next_record += record_interval

        # Смерть: A=0
        if n[0] == 0:
            if record_trajectory:
                t_rec.append(t)
                n_rec.append(n.copy())
                return t, False, (np.array(t_rec), np.array(n_rec))
            return t, False, None

        # Побег из бассейна
        if n[0] < threshold_A:
            if record_trajectory:
                t_rec.append(t)
                n_rec.append(n.copy())
                return t, False, (np.array(t_rec), np.array(n_rec))
            return t, False, None

    if record_trajectory:
        t_rec.append(t_max)
        n_rec.append(n.copy())
        return t_max, True, (np.array(t_rec), np.array(n_rec))
    return t_max, True, None


def run_gillespie_survival(ss, p, N_A=100, n_runs=50, t_max_h=72, seed=42):
    """Запуск множества Gillespie и подсчёт выживания."""
    n_init, Omega = concentration_to_molecules_v3(ss, N_A)
    t_max_s = t_max_h * 3600
    rng = np.random.default_rng(seed)

    survived = 0
    escape_times = []

    for run in range(n_runs):
        _, alive, _ = gillespie_ssa_v3(n_init.copy(), Omega, p, t_max_s, rng)
        if alive:
            survived += 1
        escape_times.append(t_max_s if alive else _)

    return {
        'N_A': N_A,
        'n_runs': n_runs,
        'survived': survived,
        'survival_pct': survived / n_runs * 100,
        'n_init': n_init.tolist(),
        'Omega': float(Omega),
    }


# ============================================================
# 8. SENSITIVITY ANALYSIS (Sobol)
# ============================================================

def run_sensitivity(p_nom=None, N_sobol=512):
    """Sobol sensitivity analysis с SALib."""
    try:
        from SALib.sample import saltelli
        from SALib.analyze import sobol
    except ImportError:
        print("  [SKIP] SALib не установлен")
        return None

    if p_nom is None:
        p_nom = TM6v3MinParams()

    param_names = ['k1', 'km', 'kd_A', 'kd_m', 'kd_fe', 'Km_f',
                   'f1', 'Ka', 'k_fe_gen', 'fe_supply']
    nom_values = [p_nom.k1, p_nom.km, p_nom.kd_A, p_nom.kd_m,
                  p_nom.kd_fe, p_nom.Km_f, p_nom.f1, p_nom.Ka,
                  p_nom.k_fe_gen, p_nom.fe_supply]

    problem = {
        'num_vars': len(param_names),
        'names': param_names,
        'bounds': [[v/5, v*5] for v in nom_values]
    }

    X = saltelli.sample(problem, N_sobol, calc_second_order=False)
    print(f"  Sobol: {len(X)} оценок ODE ({N_sobol} базовых)")

    Y = np.zeros(len(X))
    n_alive = 0

    for i, xi in enumerate(X):
        p_i = TM6v3MinParams()
        for j, name in enumerate(param_names):
            setattr(p_i, name, xi[j])

        try:
            ss_i, _ = find_steady_state(p_i, t_end=100*3600)
            Y[i] = ss_i['a'] * 1e3  # мМ
            if Y[i] > 0.1:
                n_alive += 1
        except:
            Y[i] = 0.0

        if (i + 1) % max(1, len(X) // 10) == 0:
            pct = (i + 1) / len(X) * 100
            print(f"    {i+1}/{len(X)} ({pct:.0f}%)")

    frac_alive = n_alive / len(X)
    print(f"  Доля живых: {frac_alive*100:.1f}%")

    # Анализ
    Si = sobol.analyze(problem, Y, calc_second_order=False)

    results = {
        'param_names': param_names,
        'S1': Si['S1'].tolist(),
        'ST': Si['ST'].tolist(),
        'frac_alive': frac_alive,
        'n_evaluations': len(X),
    }

    # Ранжирование по ST
    ranked = sorted(zip(param_names, Si['S1'], Si['ST']),
                    key=lambda x: -x[2])

    print(f"\n  {'Параметр':<12s} {'S1':>8s} {'ST':>8s}")
    print(f"  {'-'*30}")
    for name, s1, st in ranked:
        marker = " ***" if st > 0.3 else (" **" if st > 0.1 else "")
        print(f"  {name:<12s} {s1:8.3f} {st:8.3f}{marker}")

    results['ranked'] = [{'name': n, 'S1': float(s1), 'ST': float(st)}
                         for n, s1, st in ranked]

    return results


# ============================================================
# 9. ГРАФИКИ
# ============================================================

def plot_ode_trajectory(sol, ss, title='TM6v3 Minimal: ODE Trajectory'):
    """Графики ODE-траектории."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib не доступен")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t_hours = sol.t / 3600
    labels = ['A (формиат)', 'M (мембрана)', 'Fe (Fe²⁺)']
    units = ['мМ', 'мМ', 'мМ']

    for i, (ax, label, unit) in enumerate(zip(axes, labels, units)):
        ax.plot(t_hours, sol.y[i] * 1e3, 'b-', linewidth=1.5)
        ax.axhline(list(ss.values())[i] * 1e3, color='r', linestyle='--',
                   alpha=0.5, label=f'стационар')
        ax.set_xlabel('Время (ч)')
        ax.set_ylabel(f'{label} ({unit})')
        ax.set_title(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'tm6v3_min_ode_trajectory.png', dpi=150)
    plt.close()
    print(f"  -> {OUT_DIR / 'tm6v3_min_ode_trajectory.png'}")


def plot_autopoiesis_test(autopoiesis_results, title='TM6v3: Autopoiesis Test'):
    """Графики теста автопоэзиса."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib не доступен")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    test_names = ['km_zero', 'all_zero', 'control', 'M0_zero_A_alive']
    test_labels = ['km=0 (нет воспр. M)', 'A=0,M=0 (нет seed)',
                   'Контроль (стационар)', 'M=0,A>0 (самовосст.)']

    for ax_idx, (test_name, test_label) in enumerate(
            zip(test_names, test_labels)):
        r = autopoiesis_results[test_name]
        sol = r['sol']
        t_hours = sol.t / 3600

        ax = axes[ax_idx // 2][ax_idx % 2]
        ax.plot(t_hours, sol.y[0] * 1e3, 'b-', linewidth=1.5, label='A (формиат)')
        ax.plot(t_hours, sol.y[1] * 1e3, 'g--', linewidth=1.5, label='M (мембрана)')
        ax.plot(t_hours, sol.y[2] * 1e3, 'r:', linewidth=1.5, label='Fe (Fe2+)')
        ax.set_xlabel('Время (ч)')
        ax.set_ylabel('Концентрация (мМ)')
        ax.set_title(f'{test_label}')
        status = 'ЖИВ' if r['alive'] else 'МЕРТВ'
        ax.annotate(status, xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=14, fontweight='bold',
                    color='green' if 'ЖИВ' in status else 'red',
                    ha='right', va='top')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'tm6v3_min_autopoiesis_test.png', dpi=150)
    plt.close()
    print(f"  -> {OUT_DIR / 'tm6v3_min_autopoiesis_test.png'}")


def plot_bistability(bist_results, title='TM6v3: Bistability Scan'):
    """График бистабильности."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [SKIP] matplotlib не доступен")
        return

    f1_vals = [r['f1'] for r in bist_results]
    a_alive = [r['A_alive_mM'] for r in bist_results]
    a_dead = [r['A_dead_mM'] for r in bist_results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(f1_vals, a_alive, 'b-o', markersize=4, label='A* (из живого IC)')
    ax.plot(f1_vals, a_dead, 'r-s', markersize=4, label='A* (из мёртвого IC)')
    ax.set_xscale('log')
    ax.set_xlabel('f1 (М)')
    ax.set_ylabel('A* (мМ)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Отметить бистабильную область
    bistable_f1 = [r['f1'] for r in bist_results if r['bistable']]
    if bistable_f1:
        ax.axvspan(min(bistable_f1), max(bistable_f1), alpha=0.15, color='yellow',
                   label='бистабильность')
        ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'tm6v3_min_bistability.png', dpi=150)
    plt.close()
    print(f"  -> {OUT_DIR / 'tm6v3_min_bistability.png'}")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    t_start = time.time()

    print("=" * 70)
    print("  TM6v3 MINIMAL PROOF-OF-CONCEPT")
    print("  3 переменные: A (формиат), M (пентландит+макинавит), Fe (Fe²⁺)")
    print("  РЕШЕНИЕ-037 (G3c) + РЕШЕНИЕ-029 (мин. топология)")
    print("=" * 70)

    p = TM6v3MinParams()
    print(f"\n  Параметры: {p}")

    report = {
        'model': 'TM6v3_minimal',
        'task': 'B1',
        'variables': ['A (формиат)', 'M (пентландит+макинавит)', 'Fe (Fe²⁺)'],
        'architecture': 'РЕШЕНИЕ-037 (G3c)',
        'params': {k: getattr(p, k) for k in
                   ['k1', 'Ka', 'f1', 'Km_f', 'km', 'k_fe_gen', 'fe_supply',
                    'kd_A', 'kd_m', 'kd_fe']}
    }

    # ---- 1. ODE стационар ----
    print("\n[1/7] Поиск стационара (500 ч)...")
    ss, sol = find_steady_state(p)

    print(f"  Стационар (мМ):")
    for k, v in ss.items():
        print(f"    {k.upper():3s} = {v*1e3:.4f}")

    alive = ss['a'] > 1e-4
    print(f"  Жив: {'ДА' if alive else 'НЕТ'}")

    report['steady_state'] = {
        'values_mM': {k: round(v*1e3, 6) for k, v in ss.items()},
        'alive': alive
    }

    # Подбор параметров если мёртв
    if not alive:
        print("\n  Система мертва! Подбираю параметры...")
        for k1_mult in [2, 5, 10, 20, 50]:
            p_try = TM6v3MinParams()
            p_try.k1 = p.k1 * k1_mult
            ss_try, _ = find_steady_state(p_try, t_end=200*3600)
            if ss_try['a'] > 1e-4:
                print(f"  -> Живой при k1 x {k1_mult}: A = {ss_try['a']*1e3:.3f} мМ")
                p = p_try
                ss = ss_try
                sol = _
                alive = True
                report['parameter_tuning'] = f'k1 x {k1_mult}'
                break

        if not alive:
            # Пробуем увеличить km
            for km_mult in [2, 5, 10]:
                p_try = TM6v3MinParams()
                p_try.km = p.km * km_mult
                ss_try, _ = find_steady_state(p_try, t_end=200*3600)
                if ss_try['a'] > 1e-4:
                    print(f"  -> Живой при km x {km_mult}: A = {ss_try['a']*1e3:.3f} мМ")
                    p = p_try
                    ss = ss_try
                    sol = _
                    alive = True
                    report['parameter_tuning'] = f'km x {km_mult}'
                    break

    if not alive:
        print("\n  ОШИБКА: не удалось найти живой стационар!")
        print("  Требуется ручная настройка параметров.")
        with open(OUT_DIR / 'tm6v3_min_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        return report

    # ---- 2. Устойчивость ----
    print("\n[2/7] Проверка устойчивости (якобиан)...")
    stab = check_stability(ss, p)
    print(f"  Устойчив: {'ДА' if stab['stable'] else 'НЕТ'}")
    if stab['stable']:
        print(f"  lambda_att = {stab['lambda_att']:.2e} с⁻¹")
        print(f"  tau_relax = {stab['tau_relax_hours']:.1f} ч")
    report['stability'] = {k: v for k, v in stab.items() if k != 'eigenvalues'}

    # ---- 3. Гомеостаз ----
    print("\n[3/7] Тест гомеостаза (-95% A)...")
    homeo = test_homeostasis(ss, p)
    print(f"  A(72ч) = {homeo['a_final_mM']:.3f} мМ")
    print(f"  Восстановление: {homeo['recovery_fraction']:.1%}")
    print(f"  t(90%) = {homeo['t_recovery_90pct_hours']:.1f} ч")
    report['homeostasis'] = homeo

    # ---- 4. Тест автопоэзиса (КЛЮЧЕВОЙ) ----
    print("\n[4/7] ТЕСТ АВТОПОЭЗИСА...")
    autopoiesis = test_autopoiesis(ss, p)

    for key in ['km_zero', 'all_zero', 'control', 'M0_zero_A_alive']:
        r = autopoiesis[key]
        status = 'ЖИВ' if r['alive'] else 'МЕРТВ'
        print(f"  {r['description'][:50]:<50s} -> A={r['A_final_mM']:.4f} мМ [{status}]")

    print(f"\n  АВТОПОЭЗИС: {'PASS' if autopoiesis['autopoiesis_pass'] else 'FAIL'}")
    if autopoiesis['autopoiesis_pass']:
        print(f"    -> Мембрана необходима (km=0 убивает)")
        print(f"    -> Система не самозапускается (A=0,M=0 мертва)")
        print(f"    -> Метаболит восстанавливает мембрану (M=0,A>0 жива)")
        print(f"    -> Автопоэтическая петля: A -> Fe -> M -> f1 -> R1 -> A")

    report['autopoiesis'] = {
        k: {kk: vv for kk, vv in v.items() if kk != 'sol'}
        for k, v in autopoiesis.items() if isinstance(v, dict)
    }
    report['autopoiesis']['pass'] = autopoiesis['autopoiesis_pass']

    # ---- 5. Бистабильность ----
    print("\n[5/7] Скан бистабильности (f1)...")
    bist = scan_bistability(p)
    n_bistable = sum(1 for r in bist if r['bistable'])
    print(f"  Бистабильных точек: {n_bistable}/{len(bist)}")
    if n_bistable > 0:
        bist_f1 = [r['f1'] for r in bist if r['bistable']]
        print(f"  Диапазон бистабильности: f1 = {min(bist_f1):.2e} -- {max(bist_f1):.2e}")
        print(f"  f1 номинал = {p.f1:.2e}, запас = {p.f1 / min(bist_f1):.1f}x")
    report['bistability'] = {
        'n_bistable': n_bistable,
        'scan': bist
    }

    # ---- 6. Gillespie SSA ----
    print("\n[6/7] Gillespie SSA...")
    gillespie_results = {}

    for N_A in [100, 300, 1000]:
        n_runs = 50
        print(f"\n  N_A = {N_A} ({n_runs} прогонов, 72 ч)...")
        t0 = time.time()
        g_res = run_gillespie_survival(ss, p, N_A=N_A, n_runs=n_runs, t_max_h=72)
        dt = time.time() - t0
        print(f"    Выживание: {g_res['survived']}/{n_runs} ({g_res['survival_pct']:.0f}%)")
        print(f"    n_init: {g_res['n_init']}")
        print(f"    Время: {dt:.1f} с")
        gillespie_results[N_A] = g_res

    report['gillespie'] = gillespie_results

    # Длинный прогон N_A=100
    print(f"\n  Длинный прогон: N_A=100, 500 ч, 30 прогонов...")
    t0 = time.time()
    g_long = run_gillespie_survival(ss, p, N_A=100, n_runs=30, t_max_h=500, seed=99)
    dt = time.time() - t0
    print(f"    Выживание 500ч: {g_long['survived']}/30 ({g_long['survival_pct']:.0f}%)")
    print(f"    Время: {dt:.1f} с")
    report['gillespie_long'] = g_long

    # ---- 7. Sensitivity Analysis ----
    print("\n[7/7] Sensitivity Analysis (Sobol)...")
    sa_results = run_sensitivity(p, N_sobol=512)
    if sa_results:
        report['sensitivity'] = sa_results

    # ---- Графики ----
    print("\n" + "=" * 70)
    print("ГРАФИКИ")
    print("=" * 70)
    plot_ode_trajectory(sol, ss)
    plot_autopoiesis_test(autopoiesis)
    plot_bistability(bist)

    # ---- СВОДКА ----
    total_time = time.time() - t_start

    print("\n" + "=" * 70)
    print("  СВОДКА: TM6v3 Minimal PoC (3 переменные: A, M, Fe)")
    print("=" * 70)

    print(f"""
  ODE стационар:
    A*  = {ss['a']*1e3:.4f} мМ (формиат)
    M*  = {ss['m']*1e3:.4f} мМ (мембрана пентландит+макинавит)
    Fe* = {ss['fe']*1e3:.4f} мМ (Fe²⁺)

  Устойчивость: {'ДА' if stab['stable'] else 'НЕТ'}
  tau_relax = {stab['tau_relax_hours']:.1f} ч

  Гомеостаз (-95% A): {'PASS' if homeo['recovered'] else 'FAIL'}
    Восстановление: {homeo['recovery_fraction']:.1%}
    t(90%) = {homeo['t_recovery_90pct_hours']:.1f} ч

  Бистабильность: {'ДА' if n_bistable > 0 else 'НЕТ'}
  {f'  Диапазон: f1 = {min(bist_f1):.2e} -- {max(bist_f1):.2e}' if n_bistable > 0 else ''}

  Gillespie (72 ч):
    N_A=100:  {gillespie_results[100]['survival_pct']:.0f}% выживание ({gillespie_results[100]['survived']}/50)
    N_A=300:  {gillespie_results[300]['survival_pct']:.0f}% выживание ({gillespie_results[300]['survived']}/50)
    N_A=1000: {gillespie_results[1000]['survival_pct']:.0f}% выживание ({gillespie_results[1000]['survived']}/50)

  Gillespie (500 ч, N_A=100): {g_long['survival_pct']:.0f}% выживание ({g_long['survived']}/30)

  Тест автопоэзиса:
    km=0 (нет воспроизв. M)  -> {'МЕРТВ' if not autopoiesis['km_zero']['alive'] else 'ЖИВ!'} (A={autopoiesis['km_zero']['A_final_mM']:.4f} мМ)
    A=0, M=0 (нет seed)      -> {'МЕРТВ' if not autopoiesis['all_zero']['alive'] else 'ЖИВ!'} (A={autopoiesis['all_zero']['A_final_mM']:.4f} мМ)
    Контроль (стационар)      -> {'ЖИВ' if autopoiesis['control']['alive'] else 'МЕРТВ!'} (A={autopoiesis['control']['A_final_mM']:.4f} мМ)
    M=0, A>0 (самовосстановл) -> {'ЖИВ' if autopoiesis['M0_zero_A_alive']['alive'] else 'МЕРТВ!'} (A={autopoiesis['M0_zero_A_alive']['A_final_mM']:.4f} мМ)
    АВТОПОЭЗИС: {'PASS' if autopoiesis['autopoiesis_pass'] else 'FAIL'}""")

    if sa_results:
        print(f"\n  Sensitivity (Sobol ST):")
        for r in sa_results['ranked'][:5]:
            marker = " ***" if r['ST'] > 0.3 else (" **" if r['ST'] > 0.1 else "")
            print(f"    {r['name']:<12s}: ST = {r['ST']:.3f}{marker}")
        print(f"  Доля живых: {sa_results['frac_alive']*100:.1f}%")

    print(f"\n  Время: {total_time:.0f} с ({total_time/60:.1f} мин)")
    print("=" * 70)

    # Сохранение
    report['summary'] = {
        'alive': alive,
        'stable': stab['stable'],
        'homeostasis': homeo['recovered'],
        'autopoiesis': autopoiesis['autopoiesis_pass'],
        'bistability': n_bistable > 0,
        'gillespie_100_pct': gillespie_results[100]['survival_pct'],
        'gillespie_1000_pct': gillespie_results[1000]['survival_pct'],
        'total_time_s': round(total_time, 1),
    }

    report_path = OUT_DIR / 'tm6v3_min_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Отчёт: {report_path}")

    return report


if __name__ == '__main__':
    main()
