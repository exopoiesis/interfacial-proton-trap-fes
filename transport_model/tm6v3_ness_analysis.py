"""
TM6v3-min: NESS (Non-Equilibrium Steady State) Diagnostics
===========================================================
Задача: диагностика NESS для модели TM6v3-min (Paper #2, §3.4 KineticTrap).

Переменные:
  A  -- формиат HCOO- (M)
  M  -- мембрана пентландит+макинавит (M, отн. ед.)
  Fe -- Fe2+(aq) (M)

Уравнения:
  dA/dt  = k1*(f1*M/(KMf+M))*(A^2/(KA^2+A^2)) - kdA*A
  dM/dt  = km*Fe*A - kdM*M
  dFe/dt = kFe_gen*A + RFe_supply - km*Fe*A - kdFe*Fe

Параметры (Table 1, manuscript):
  k1=1e-4 s^-1; KA=7e-4 M; f1=5e-3 M; KMf=5e-4 M
  km=3e-2 s^-1 M^-1; kFe_gen=5e-5 s^-1; RFe_supply=5e-8 M*s^-1
  kdA=1e-4 s^-1; kdM=3e-6 s^-1; kdFe=3e-4 s^-1

Запуск: python -u tmp/tm6v3_ness_analysis.py
"""

import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from pathlib import Path

import builtins

def flush_print(*args, **kwargs):
    builtins.print(*args, **kwargs)
    sys.stdout.flush()

print = flush_print

# ============================================================
# 1. ПАРАМЕТРЫ
# ============================================================

class Params:
    k1       = 1e-4   # s^-1         базовый автокатализ
    KA       = 7e-4   # M            Hill n=2 полунасыщение
    f1       = 5e-3   # M            ресурс CO2 через мембрану
    KMf      = 5e-4   # M            полунасыщение по M
    km       = 3e-2   # s^-1 M^-1    мембранообразование Fe+A->M
    kFe_gen  = 5e-5   # s^-1         A -> Fe (побочный)
    RFe_sup  = 5e-8   # M s^-1       Fe0 коррозия (неравновесный приток)
    kdA      = 1e-4   # s^-1         деградация формиата
    kdM      = 3e-6   # s^-1         деградация мембраны
    kdFe     = 3e-4   # s^-1         потери Fe2+

P = Params()

# ============================================================
# 2. ODE: правая часть + потоки по реакциям
# ============================================================

def reaction_fluxes(A, M, Fe, p=P):
    """
    Все элементарные потоки сети реакций.
    Возвращает словарь скоростей (M/s).
    """
    A  = max(A,  0.0)
    M  = max(M,  0.0)
    Fe = max(Fe, 0.0)

    f1_eff = p.f1 * M / (p.KMf + M)
    hill_a = A**2 / (p.KA**2 + A**2) if A > 0 else 0.0

    R_A_prod    = p.k1 * f1_eff * hill_a          # CO2 -> A (автокатализ)
    R_A_deg     = p.kdA * A                         # A -> 0
    R_M_form    = p.km * Fe * A                     # Fe + A -> M
    R_M_deg     = p.kdM * M                         # M -> 0
    R_Fe_gen    = p.kFe_gen * A                     # A -> A + Fe (побочный)
    R_Fe_supply = p.RFe_sup                         # -> Fe (неравновесный приток)
    R_Fe_consume= p.km * Fe * A                     # Fe потребляется при M-форм.
    R_Fe_deg    = p.kdFe * Fe                       # Fe -> 0

    return {
        'R_A_prod':     R_A_prod,
        'R_A_deg':      R_A_deg,
        'R_M_form':     R_M_form,
        'R_M_deg':      R_M_deg,
        'R_Fe_gen':     R_Fe_gen,
        'R_Fe_supply':  R_Fe_supply,
        'R_Fe_consume': R_Fe_consume,   # == R_M_form (одна и та же реакция)
        'R_Fe_deg':     R_Fe_deg,
    }


def rhs(t, y, p=P):
    A, M, Fe = [max(v, 0.0) for v in y]
    fl = reaction_fluxes(A, M, Fe, p)

    dA  = fl['R_A_prod'] - fl['R_A_deg']
    dM  = fl['R_M_form'] - fl['R_M_deg']
    dFe = fl['R_Fe_gen'] + fl['R_Fe_supply'] - fl['R_Fe_consume'] - fl['R_Fe_deg']

    return [dA, dM, dFe]

# ============================================================
# 3. ПОИСК СТАЦИОНАРОВ
# ============================================================

def integrate_to_ss(y0, t_end=500*3600, p=P):
    """Интегрирование до стационара."""
    sol = solve_ivp(lambda t, y: rhs(t, y, p),
                    [0, t_end], y0, method='LSODA',
                    rtol=1e-10, atol=1e-14, max_step=100)
    return sol.y[:, -1], sol


def check_balance(A, M, Fe, p=P, label=""):
    """Проверка балансов dA/dt, dM/dt, dFe/dt при заданных концентрациях."""
    dA, dM, dFe = rhs(0, [A, M, Fe], p)
    print(f"  [{label}] Баланс: dA/dt={dA:.3e}  dM/dt={dM:.3e}  dFe/dt={dFe:.3e} M/s")
    return abs(dA) + abs(dM) + abs(dFe)


def find_inactive_ss(p=P):
    """
    Искать неактивный стационар (низкий A).
    Пробуем несколько разных начальных условий.
    """
    low_ics = [
        [1e-7, 1e-8, 1e-6],
        [1e-6, 1e-7, 1e-5],
        [1e-5, 1e-8, 1e-4],
        [1e-6, 0.0,  1e-4],
        [0.0,  0.0,  1e-4],
        [1e-8, 1e-8, 5e-7],
    ]
    candidates = []
    for ic in low_ics:
        try:
            ss, _ = integrate_to_ss(ic, t_end=5000*3600, p=p)
            if ss[0] >= 0 and ss[1] >= 0 and ss[2] >= 0:
                candidates.append(ss)
        except Exception:
            pass
    return candidates


# ============================================================
# 4. ЦИКЛОВЫЙ ПОТОК (throughput)
# ============================================================

def compute_throughput(A, M, Fe, p=P):
    """
    Throughput автокаталитического цикла.

    Цикл: A --[R_Fe_gen]--> Fe --[R_M_form]--> M --[boosts R_A_prod]--> A

    Мера throughput = минимальное звено в петле:
    min(R_Fe_gen(A->Fe), R_M_form(Fe->M), R_A_prod(M->A))

    Это консервативная нижняя оценка того, сколько "вещества"
    прокачивается через автокаталитическую петлю в единицу времени.

    Дополнительно: "Fe-driven throughput" = R_Fe_supply (независимый
    внешний приток, питающий петлю).
    """
    fl = reaction_fluxes(A, M, Fe, p)

    # Три ребра петли A->Fe->M->[boosts]A:
    edge_A_to_Fe = fl['R_Fe_gen']      # A продуцирует Fe
    edge_Fe_to_M = fl['R_M_form']      # Fe+A -> M
    edge_M_to_A  = fl['R_A_prod']      # M-опосредованный R1 -> A

    # Cycle flux = min(три ребра) -- Flux Balance нижняя граница
    cycle_flux = min(edge_A_to_Fe, edge_Fe_to_M, edge_M_to_A)

    # Полный throughput через формиат (через сколько M/s прокачивается A)
    A_turnover = fl['R_A_prod']   # скорость производства A (= скорость деградации в SS)

    return {
        'edge_A_to_Fe':    edge_A_to_Fe,
        'edge_Fe_to_M':    edge_Fe_to_M,
        'edge_M_to_A':     edge_M_to_A,
        'cycle_flux_min':  cycle_flux,
        'A_turnover':      A_turnover,
        'Fe_supply_drive': fl['R_Fe_supply'],
        'all_fluxes':      fl,
    }


def detailed_balance_broken(A, M, Fe, p=P):
    """
    Проверить нарушение детального баланса.

    В равновесии: для каждой реакции J+ = J-.
    Здесь все реакции ОДНОСТОРОННИЕ (нет обратных скоростей),
    поэтому детальный баланс нарушен структурно.

    Конкретно: RFe_supply > 0 -- внешний приток без обратного потока.
    Он питает сеть и не может быть сбалансирован обратной реакцией.

    Возвращает количественный показатель "дисбаланса" для каждой реакции.
    """
    fl = reaction_fluxes(A, M, Fe, p)

    # В модели нет обратных скоростей -> J- = 0 для всех реакций.
    # Имеет смысл показать, какие реакции несут ненулевой поток.
    driven_reactions = {
        'R_Fe_supply_driven': fl['R_Fe_supply'],  # чисто движущая реакция
        'R_A_prod_driven':    fl['R_A_prod'],      # автокаталитическая
        'R_M_form_driven':    fl['R_M_form'],      # мембранообразование
        'R_Fe_gen_driven':    fl['R_Fe_gen'],      # побочный Fe-генератор
    }
    return driven_reactions


# ============================================================
# 5. ЯКОБИАН И ВРЕМЯ РЕЛАКСАЦИИ
# ============================================================

def jacobian_numerical(y0, p=P):
    """Численный якобиан в точке y0."""
    eps = 1e-8
    n = len(y0)
    J = np.zeros((n, n))
    f0 = np.array(rhs(0, y0, p))
    for j in range(n):
        yp = np.array(y0, dtype=float)
        yp[j] += eps
        fp = np.array(rhs(0, yp, p))
        J[:, j] = (fp - f0) / eps
    return J


def relaxation_time(y0, p=P):
    """Время релаксации через максимальное действительное собственное значение."""
    J = jacobian_numerical(y0, p)
    eigs = np.linalg.eigvals(J)
    lam_max = np.max(np.real(eigs))
    stable = lam_max < 0
    tau_s  = -1.0 / lam_max if stable else np.inf
    tau_h  = tau_s / 3600.0
    return {
        'stable':   stable,
        'lam_max':  float(lam_max),
        'tau_s':    float(tau_s),
        'tau_h':    float(tau_h),
        'eigenvalues': eigs.tolist(),
    }


# ============================================================
# 6. ИЛЛЮСТРАТИВНАЯ σ (EPR) ПОД ε-допущением
# ============================================================

def illustrative_epr(A, M, Fe, p=P, eps_fwd=1e-6):
    """
    Иллюстративная оценка производства энтропии по Schnakenberg.

    ДИСКЛЕЙМЕР: модель имеет только прямые скорости (J- = 0).
    Schnakenberg sigma = sum_r (J+_r - J-_r) * ln(J+_r / J-_r) расходится
    при J-_r -> 0 (ln -> inf).

    Здесь используется МИНИМАЛЬНОЕ обратное допущение:
        J-_r = eps_fwd * J+_r
    что эквивалентно ln(1/eps_fwd).

    Это НЕ физический расчёт EPR -- только иллюстрация масштаба.
    Результат чувствителен к eps_fwd на порядки (показано ниже).
    """
    fl = reaction_fluxes(A, M, Fe, p)

    # Реакции с ненулевым прямым потоком
    forward_fluxes = {
        'R_A_prod':    fl['R_A_prod'],
        'R_A_deg':     fl['R_A_deg'],
        'R_M_form':    fl['R_M_form'],
        'R_M_deg':     fl['R_M_deg'],
        'R_Fe_gen':    fl['R_Fe_gen'],
        'R_Fe_supply': fl['R_Fe_supply'],
        'R_Fe_deg':    fl['R_Fe_deg'],
    }

    sigma = 0.0
    contributions = {}
    log_factor = np.log(1.0 / eps_fwd)  # ln(J+/J-) = ln(1/eps)

    for name, Jf in forward_fluxes.items():
        if Jf > 0:
            Jb = eps_fwd * Jf
            contrib = (Jf - Jb) * np.log(Jf / Jb)
            sigma += contrib
            contributions[name] = contrib

    return {
        'sigma_illustrative': sigma,
        'eps_fwd': eps_fwd,
        'log_factor': log_factor,
        'contributions': contributions,
        'disclaimer': (
            'ILLUSTRATIVE ONLY: J- = eps*J+. '
            'Physical EPR requires thermodynamically consistent reverse rates.'
        )
    }


# ============================================================
# 7. MAIN
# ============================================================

def main():
    print("=" * 72)
    print("  TM6v3-min: NESS Diagnostics (Paper #2, §3.4)")
    print("=" * 72)

    # -----------------------------------------------------------
    # 7.1 АКТИВНЫЙ СТАЦИОНАР
    # -----------------------------------------------------------
    print("\n[1/5] Поиск АКТИВНОГО стационара (IC: A=5e-3, M=1e-4, Fe=5e-4)...")
    y0_active = [5e-3, 1e-4, 5e-4]
    ss_act, sol_act = integrate_to_ss(y0_active, t_end=500*3600)
    A_act, M_act, Fe_act = ss_act

    print(f"  A*  = {A_act*1e3:.4f} mM  (ref: 4.82 mM)")
    print(f"  M*  = {M_act*1e3:.4f} mM  (ref: 31.4 mM)")
    print(f"  Fe* = {Fe_act*1e3:.4f} mM  (ref: 0.655 mM)")

    # Баланс
    res_act = check_balance(A_act, M_act, Fe_act, label="ACTIVE")

    # Сравнение с референсами
    ref_A  = 4.82e-3
    ref_M  = 31.4e-3
    ref_Fe = 0.655e-3
    dev_A  = abs(A_act - ref_A)  / ref_A  * 100
    dev_M  = abs(M_act - ref_M)  / ref_M  * 100
    dev_Fe = abs(Fe_act - ref_Fe) / ref_Fe * 100
    print(f"  Отклонение от ref: A={dev_A:.1f}%, M={dev_M:.1f}%, Fe={dev_Fe:.1f}%")

    match = (dev_A < 5 and dev_M < 5 and dev_Fe < 5)
    print(f"  Воспроизведение: {'OK (< 5%)' if match else 'РАСХОЖДЕНИЕ > 5%'}")

    # -----------------------------------------------------------
    # 7.2 НЕАКТИВНЫЙ СТАЦИОНАР
    # -----------------------------------------------------------
    print("\n[2/5] Поиск НЕАКТИВНОГО стационара (низкий A, из нескольких IC)...")
    inactive_candidates = find_inactive_ss()

    # Фильтр: неактивный = A < 0.1 mM, не тот же что активный
    inactive_ss_list = []
    for c in inactive_candidates:
        if c[0] * 1e3 < 0.1:
            inactive_ss_list.append(c)

    if inactive_ss_list:
        # Берём "типичный" -- с наименьшим A
        ss_ina = min(inactive_ss_list, key=lambda x: x[0])
        A_ina, M_ina, Fe_ina = ss_ina
        print(f"  Найден неактивный стационар:")
        print(f"  A*  = {A_ina*1e3:.6f} mM")
        print(f"  M*  = {M_ina*1e3:.6f} mM")
        print(f"  Fe* = {Fe_ina*1e3:.6f} mM")
        check_balance(A_ina, M_ina, Fe_ina, label="INACTIVE")
        has_inactive = True
    else:
        print("  Неактивный стационар не найден из стандартных IC.")
        print("  Пробую fsolve от нулевого IC...")
        # Стационар при A->0: dFe/dt = 0 -> Fe* = RFe_sup / kdFe
        Fe_inact = P.RFe_sup / P.kdFe
        # dA/dt|A->0 = k1*f1_eff*0 - kdA*A -> A=0 точно
        # dM/dt|A=0 = 0 - kdM*M -> M=0
        ss_ina = np.array([0.0, 0.0, Fe_inact])
        A_ina, M_ina, Fe_ina = ss_ina
        print(f"  Аналитический тривиальный стационар (A=0, M=0):")
        print(f"  A*  = {A_ina*1e3:.6f} mM")
        print(f"  M*  = {M_ina*1e3:.6f} mM")
        print(f"  Fe* = {Fe_ina*1e3:.6f} mM  (= RFe_sup/kdFe)")
        has_inactive = True

    # -----------------------------------------------------------
    # 7.3 ПОТОКИ ПО РЕАКЦИЯМ
    # -----------------------------------------------------------
    print("\n[3/5] Per-reaction потоки в каждом стационаре...")

    fl_act = reaction_fluxes(A_act, M_act, Fe_act)
    print("\n  --- АКТИВНЫЙ стационар ---")
    for name, val in fl_act.items():
        print(f"    {name:<20s} = {val:.4e} M/s")

    # Проверка баланса через потоки явно
    dA_check  =  fl_act['R_A_prod']   - fl_act['R_A_deg']
    dM_check  =  fl_act['R_M_form']   - fl_act['R_M_deg']
    dFe_check =  fl_act['R_Fe_gen']   + fl_act['R_Fe_supply'] \
              -  fl_act['R_Fe_consume'] - fl_act['R_Fe_deg']
    print(f"    Баланс (должно быть ~0):")
    print(f"      dA/dt   = {dA_check:.4e} M/s")
    print(f"      dM/dt   = {dM_check:.4e} M/s")
    print(f"      dFe/dt  = {dFe_check:.4e} M/s")

    if has_inactive:
        fl_ina = reaction_fluxes(A_ina, M_ina, Fe_ina)
        print("\n  --- НЕАКТИВНЫЙ стационар ---")
        for name, val in fl_ina.items():
            print(f"    {name:<20s} = {val:.4e} M/s")

    # -----------------------------------------------------------
    # 7.4 THROUGHPUT / CYCLE FLUX
    # -----------------------------------------------------------
    print("\n[4/5] Throughput / Cycle flux...")

    tp_act = compute_throughput(A_act, M_act, Fe_act)
    print("\n  --- АКТИВНЫЙ стационар ---")
    print(f"    edge A->Fe (R_Fe_gen):  {tp_act['edge_A_to_Fe']:.4e} M/s")
    print(f"    edge Fe->M (R_M_form):  {tp_act['edge_Fe_to_M']:.4e} M/s")
    print(f"    edge M->A  (R_A_prod):  {tp_act['edge_M_to_A']:.4e} M/s")
    print(f"    cycle_flux (min):       {tp_act['cycle_flux_min']:.4e} M/s")
    print(f"    A_turnover (=R_A_prod): {tp_act['A_turnover']:.4e} M/s")
    print(f"    Fe_supply_drive:        {tp_act['Fe_supply_drive']:.4e} M/s")

    # Отношение cycle_flux к Fe_supply
    ratio_cycle_to_supply = tp_act['cycle_flux_min'] / tp_act['Fe_supply_drive']
    print(f"    cycle_flux / Fe_supply: {ratio_cycle_to_supply:.1f}x")
    print(f"    (показывает amplification неравновесного привода)")

    if has_inactive:
        tp_ina = compute_throughput(A_ina, M_ina, Fe_ina)
        print("\n  --- НЕАКТИВНЫЙ стационар ---")
        print(f"    cycle_flux (min):       {tp_ina['cycle_flux_min']:.4e} M/s")
        print(f"    A_turnover (R_A_prod):  {tp_ina['A_turnover']:.4e} M/s")
        print(f"    Fe_supply_drive:        {tp_ina['Fe_supply_drive']:.4e} M/s")

        if tp_ina['cycle_flux_min'] > 0:
            ratio_act_ina = tp_act['cycle_flux_min'] / tp_ina['cycle_flux_min']
            print(f"    cycle_flux active/inactive: {ratio_act_ina:.2e}x")
        else:
            print(f"    cycle_flux active/inactive: inf (inactive = 0)")

    # -----------------------------------------------------------
    # 7.5 EPR: ЧЕСТНОЕ ОБСУЖДЕНИЕ + ИЛЛЮСТРАТИВНОЕ
    # -----------------------------------------------------------
    print("\n[5/5] Entropy Production Rate (EPR) -- честный анализ...")

    print("""
  Структурный факт:
    Все реакции в TM6v3-min ОДНОСТОРОННИЕ (нет обратных скоростей).
    Schnakenberg: sigma = sum_r (J+_r - J-_r) * ln(J+_r / J-_r)
    При J-_r = 0: ln(J+/0) -> +inf => sigma расходится формально.
    => Абсолютная EPR из данной феноменологической модели НЕ определена.

  Что надёжно:
    (a) RFe_supply > 0 -- ненулевой материальный приток без обратного потока.
        Это структурный признак NESS (неравновесного привода).
    (b) В активном стационаре: ненулевой устойчивый цикловый поток
        (cycle_flux ~ A_turnover). Детальный баланс нарушен.
    (c) В неактивном (A=0, M=0): cycle_flux = 0, только supply-drain.
    => Активный стационар = driven NESS; неактивный = near-equilibrium drain.
    """)

    print("  Иллюстративная sigma (eps_fwd = J-/J+ = 1e-6, 1e-8, 1e-10):")
    for eps in [1e-6, 1e-8, 1e-10]:
        epr_act = illustrative_epr(A_act, M_act, Fe_act, eps_fwd=eps)
        epr_ina = illustrative_epr(A_ina, M_ina, Fe_ina, eps_fwd=eps)
        print(f"    eps={eps:.0e}: sigma_active={epr_act['sigma_illustrative']:.3e} M/s"
              f"  sigma_inactive={epr_ina['sigma_illustrative']:.3e} M/s"
              f"  ratio={epr_act['sigma_illustrative']/max(epr_ina['sigma_illustrative'], 1e-300):.2e}x")

    print("  [ДИСКЛЕЙМЕР] Значения sigma выше -- ИЛЛЮСТРАТИВНЫЕ.")
    print("  Реальный EPR требует термодинамически согласованных обратных скоростей.")
    print("  Надёжная наблюдаемая NESS = cycle_flux и Fe_supply (см. п.4).")

    # -----------------------------------------------------------
    # 7.6 ВРЕМЯ РЕЛАКСАЦИИ (активный стационар)
    # -----------------------------------------------------------
    print("\n[+] Время релаксации активного стационара (якобиан)...")
    stab = relaxation_time([A_act, M_act, Fe_act])
    print(f"  Устойчив: {stab['stable']}")
    print(f"  lam_max = {stab['lam_max']:.4e} s^-1")
    print(f"  tau_relax = {stab['tau_s']:.1f} s = {stab['tau_h']:.1f} h  (ref: ~95 h)")

    # -----------------------------------------------------------
    # 7.7 ИТОГОВАЯ ТАБЛИЦА
    # -----------------------------------------------------------
    print("\n" + "=" * 72)
    print("  ИТОГОВАЯ ТАБЛИЦА NESS-ДИАГНОСТИКИ")
    print("=" * 72)

    print(f"""
  Активный стационар:
    A*  = {A_act*1e3:.4f} mM  (ref: 4.82)
    M*  = {M_act*1e3:.4f} mM  (ref: 31.4)
    Fe* = {Fe_act*1e3:.4f} mM  (ref: 0.655)
    tau_relax = {stab['tau_h']:.1f} h  (ref: ~95 h)

  Неактивный стационар:
    A*  = {A_ina*1e3:.6f} mM
    M*  = {M_ina*1e3:.6f} mM
    Fe* = {Fe_ina*1e3:.6f} mM

  Per-reaction потоки (активный):
    R_A_prod     = {fl_act['R_A_prod']:.4e} M/s   (autocatalytic production)
    R_A_deg      = {fl_act['R_A_deg']:.4e} M/s   (formate degradation)
    R_M_form     = {fl_act['R_M_form']:.4e} M/s   (membrane formation)
    R_M_deg      = {fl_act['R_M_deg']:.4e} M/s   (membrane degradation)
    R_Fe_gen     = {fl_act['R_Fe_gen']:.4e} M/s   (Fe2+ byproduct)
    R_Fe_supply  = {fl_act['R_Fe_supply']:.4e} M/s   (external Fe0 drive)
    R_Fe_consume = {fl_act['R_Fe_consume']:.4e} M/s   (Fe into membrane)
    R_Fe_deg     = {fl_act['R_Fe_deg']:.4e} M/s   (Fe2+ losses)

  Throughput (цикловый поток):
    cycle_flux_active   = {tp_act['cycle_flux_min']:.4e} M/s
    cycle_flux_inactive = {tp_ina['cycle_flux_min']:.4e} M/s
    A_turnover_active   = {tp_act['A_turnover']:.4e} M/s
    A_turnover_inactive = {tp_ina['A_turnover']:.4e} M/s
    Fe_supply_drive     = {tp_act['Fe_supply_drive']:.4e} M/s
    amplification       = {tp_act['cycle_flux_min'] / tp_act['Fe_supply_drive']:.1f}x

  EPR:
    Формальная sigma РАСХОДИТСЯ (нет обратных скоростей в модели).
    Иллюстративно (eps=1e-8): sigma_active / sigma_inactive >> 1.
    Надёжная NESS-наблюдаемая = cycle_flux + Fe_supply (broken detailed balance).
    """)

    # -----------------------------------------------------------
    # 7.8 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # -----------------------------------------------------------
    out_dir = Path(__file__).parent
    out_path = out_dir / "tm6v3_ness_results_2026-06-11.md"

    epr_act_1e8 = illustrative_epr(A_act, M_act, Fe_act, eps_fwd=1e-8)
    epr_ina_1e8 = illustrative_epr(A_ina, M_ina, Fe_ina, eps_fwd=1e-8)

    md_content = f"""# TM6v3-min: NESS Diagnostics Results
*Дата: 2026-06-11*

## 1. Стационарные состояния

| Переменная | Активный NESS | Неактивный | ref (manuscript) |
|---|---|---|---|
| A* (mM) | {A_act*1e3:.4f} | {A_ina*1e3:.6f} | 4.82 |
| M* (mM) | {M_act*1e3:.4f} | {M_ina*1e3:.6f} | 31.4 |
| Fe* (mM) | {Fe_act*1e3:.4f} | {Fe_ina*1e3:.6f} | 0.655 |
| tau_relax (h) | {stab['tau_h']:.1f} | -- | ~95 |

Отклонение от ref: A={dev_A:.1f}%, M={dev_M:.1f}%, Fe={dev_Fe:.1f}%
Воспроизведение: {'OK (< 5%)' if match else 'РАСХОЖДЕНИЕ > 5%'}

## 2. Per-reaction потоки (активный стационар)

| Реакция | Скорость (M/s) | Описание |
|---|---|---|
| R_A_prod | {fl_act['R_A_prod']:.4e} | Autocatalytic CO2->A |
| R_A_deg | {fl_act['R_A_deg']:.4e} | Formate degradation |
| R_M_form | {fl_act['R_M_form']:.4e} | Fe+A -> membrane M |
| R_M_deg | {fl_act['R_M_deg']:.4e} | Membrane degradation |
| R_Fe_gen | {fl_act['R_Fe_gen']:.4e} | A -> A + Fe2+ (byproduct) |
| R_Fe_supply | {fl_act['R_Fe_supply']:.4e} | External Fe0 drive (RFe_sup) |
| R_Fe_consume | {fl_act['R_Fe_consume']:.4e} | Fe2+ consumed in R_M_form |
| R_Fe_deg | {fl_act['R_Fe_deg']:.4e} | Fe2+ losses |

Балансы (проверка SS):
- dA/dt = {dA_check:.4e} M/s (должно ~0)
- dM/dt = {dM_check:.4e} M/s (должно ~0)
- dFe/dt = {dFe_check:.4e} M/s (должно ~0)

Per-reaction потоки (неактивный стационар):

| Реакция | Скорость (M/s) |
|---|---|
| R_A_prod | {fl_ina['R_A_prod']:.4e} |
| R_M_form | {fl_ina['R_M_form']:.4e} |
| R_Fe_supply | {fl_ina['R_Fe_supply']:.4e} |
| R_Fe_deg | {fl_ina['R_Fe_deg']:.4e} |

## 3. Sustained cycle-flux / Throughput

| Показатель | Активный NESS | Неактивный |
|---|---|---|
| edge A->Fe (R_Fe_gen) | {tp_act['edge_A_to_Fe']:.4e} M/s | {tp_ina['edge_A_to_Fe']:.4e} M/s |
| edge Fe->M (R_M_form) | {tp_act['edge_Fe_to_M']:.4e} M/s | {tp_ina['edge_Fe_to_M']:.4e} M/s |
| edge M->A (R_A_prod) | {tp_act['edge_M_to_A']:.4e} M/s | {tp_ina['edge_M_to_A']:.4e} M/s |
| cycle_flux (min ребро) | {tp_act['cycle_flux_min']:.4e} M/s | {tp_ina['cycle_flux_min']:.4e} M/s |
| A_turnover | {tp_act['A_turnover']:.4e} M/s | {tp_ina['A_turnover']:.4e} M/s |
| Fe_supply_drive | {tp_act['Fe_supply_drive']:.4e} M/s | {tp_ina['Fe_supply_drive']:.4e} M/s |
| amplification cycle/supply | {tp_act['cycle_flux_min']/tp_act['Fe_supply_drive']:.1f}x | -- |

**Вывод:** В активном NESS цикловый поток ({tp_act['cycle_flux_min']:.2e} M/s) строго
больше нуля (неактивный = 0). Это driven NESS, а не равновесие с детальным балансом.

## 4. Entropy Production Rate -- честный анализ

**Формальная sigma расходится:** все реакции в TM6v3-min односторонние (J- = 0).
Schnakenberg sigma = sum(J+ - J-)*ln(J+/J-) -> +inf при J- -> 0.
Абсолютный EPR из данной феноменологической модели НЕ определён.

**Надёжные наблюдаемые NESS:**
1. RFe_supply = {P.RFe_sup:.2e} M/s -- ненулевой материальный приток (нарушение детального баланса)
2. cycle_flux_active = {tp_act['cycle_flux_min']:.2e} M/s >> cycle_flux_inactive ≈ 0
3. A_turnover = {tp_act['A_turnover']:.2e} M/s (устойчивый оборот формиата через сеть)

**Иллюстративная sigma (eps_fwd = J-/J+, ТОЛЬКО для масштаба, НЕ для цитирования):**

| eps_fwd | sigma_active (M/s) | sigma_inactive (M/s) | ratio |
|---|---|---|---|
| 1e-6 | {illustrative_epr(A_act,M_act,Fe_act,eps_fwd=1e-6)['sigma_illustrative']:.3e} | {illustrative_epr(A_ina,M_ina,Fe_ina,eps_fwd=1e-6)['sigma_illustrative']:.3e} | {illustrative_epr(A_act,M_act,Fe_act,eps_fwd=1e-6)['sigma_illustrative']/max(illustrative_epr(A_ina,M_ina,Fe_ina,eps_fwd=1e-6)['sigma_illustrative'],1e-300):.2e} |
| 1e-8 | {epr_act_1e8['sigma_illustrative']:.3e} | {epr_ina_1e8['sigma_illustrative']:.3e} | {epr_act_1e8['sigma_illustrative']/max(epr_ina_1e8['sigma_illustrative'],1e-300):.2e} |
| 1e-10 | {illustrative_epr(A_act,M_act,Fe_act,eps_fwd=1e-10)['sigma_illustrative']:.3e} | {illustrative_epr(A_ina,M_ina,Fe_ina,eps_fwd=1e-10)['sigma_illustrative']:.3e} | {illustrative_epr(A_act,M_act,Fe_act,eps_fwd=1e-10)['sigma_illustrative']/max(illustrative_epr(A_ina,M_ina,Fe_ina,eps_fwd=1e-10)['sigma_illustrative'],1e-300):.2e} |

Иллюстративная sigma масштабируется как ~ln(1/eps) * sum(J+) -- зависимость логарифмическая.
При eps=1e-8: log-factor = {np.log(1/1e-8):.1f} (натуральный логарифм).
Вывод: ratio active/inactive устойчив к выбору eps (определяется разностью sum(J+)).

## 5. Маппинг на привод (RFe_supply, Deltaμ)

- **Материальный привод:** RFe_supply = {P.RFe_sup:.2e} M/s (Fe0 коррозия)
  поддерживает ненулевой Fe2+ приток -> питает автокаталитическую петлю.
  Без RFe_supply: Fe* = RFe_gen*A / (kdFe+km*A); при малом A цепь разрывается.

- **Термодинамический привод:** Deltaμ_H = 0.355 эВ (§3.5) задаёт энергетический
  уклон, который делает реакции необратимыми (J- << J+) и обосновывает
  односторонность эффективных скоростей в феноменологической модели.
  Deltaμ соответствует ln(J+/J-) >> 1, качественно совместимо с eps << 1
  в иллюстративном расчёте выше.

## 6. Manuscript-ready абзац для §3.4

The active steady state of the TM6v3-min network is a driven non-equilibrium
steady state (NESS) sustained by a continuous Fe2+ influx (R_Fe_supply =
{P.RFe_sup:.1e} M s^-1) that acts as the non-equilibrium drive. In this state,
all three edges of the autocatalytic cycle (A->Fe, Fe->M, M->A via R1) carry
non-vanishing concurrent fluxes -- quantified by a cycle flux of
{tp_act['cycle_flux_min']:.2e} M s^-1, approximately {tp_act['cycle_flux_min']/tp_act['Fe_supply_drive']:.1f}-fold above the bare Fe2+ supply rate -- whereas
the inactive fixed point (A* ≈ 0, M* ≈ 0) exhibits zero cycle flux and
near-equilibrium drain dynamics. This non-zero cycle flux constitutes a broken
detailed balance condition and is the dynamical signature of a self-maintaining
boundary: the system continuously converts the incoming Fe2+ flux into membrane
material (M) and formate (A), dissipating free energy available from the
electrochemical gradient (Deltaμ_H = 0.355 eV, §3.5). We note that computing
the absolute entropy production rate from the phenomenological rate equations
requires thermodynamically consistent reverse rate constants, which are outside
the scope of the present effective-kinetics model; the cycle-flux throughput
therefore serves as the primary quantitative NESS observable reported here.
"""

    out_path.write_text(md_content, encoding='utf-8')
    print(f"\n  Результаты сохранены: {out_path}")
    print("=" * 72)
    print("  DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()
